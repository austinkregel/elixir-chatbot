defmodule ChatBot.Learning.DocumentIngestor do
  @moduledoc """
  Processes large documents for entity discovery and learning.

  Handles:
  - Chunking large files into manageable pieces
  - Streaming processing to avoid memory issues
  - Progress tracking and reporting
  - Batch entity discovery
  """

  require Logger

  alias ChatBot.ML.{Tokenizer, POSTagger}
  alias ChatBot.Learning.{WorldManager, WorldMetrics, EntityDiscoverer, TypeInferrer}

  @type ingest_opts :: [
          chunk_size: pos_integer(),
          overlap: non_neg_integer(),
          progress_callback: (map() -> any()) | nil,
          learn_types: boolean()
        ]

  @type ingest_result :: %{
          documents_processed: non_neg_integer(),
          total_chunks: non_neg_integer(),
          total_tokens: non_neg_integer(),
          entities_discovered: non_neg_integer(),
          processing_time_ms: non_neg_integer()
        }

  # Default chunk size in characters
  @default_chunk_size 5000
  # Default overlap between chunks
  @default_overlap 200

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Ingests a single file into a training world.

  ## Options
    - `:chunk_size` - Characters per chunk (default: 5000)
    - `:overlap` - Overlap between chunks (default: 200)
    - `:progress_callback` - Function called with progress updates
    - `:learn_types` - Whether to learn type patterns from known entities (default: true)
  """
  def ingest_file(world_id, file_path, opts \\ [])
      when is_binary(world_id) and is_binary(file_path) do
    start_time = System.monotonic_time(:millisecond)

    case File.read(file_path) do
      {:ok, content} ->
        result = ingest_text(world_id, content, opts)
        duration = System.monotonic_time(:millisecond) - start_time

        # Record document processed event
        WorldManager.record_event(world_id, :document_processed, %{
          file_path: file_path,
          chunks: result.total_chunks,
          entities: result.entities_discovered,
          duration_ms: duration
        })

        {:ok, result}

      {:error, reason} ->
        {:error, {:file_read_failed, reason}}
    end
  end

  @doc """
  Ingests multiple files into a training world.

  Processes files sequentially and aggregates results.
  """
  def ingest_files(world_id, file_paths, opts \\ [])
      when is_binary(world_id) and is_list(file_paths) do
    start_time = System.monotonic_time(:millisecond)
    total_files = length(file_paths)
    progress_callback = Keyword.get(opts, :progress_callback)

    results =
      file_paths
      |> Enum.with_index(1)
      |> Enum.map(fn {file_path, idx} ->
        # Report progress
        if progress_callback do
          progress_callback.(%{
            type: :file_started,
            file: file_path,
            current: idx,
            total: total_files
          })
        end

        case ingest_file(world_id, file_path, opts) do
          {:ok, result} ->
            if progress_callback do
              progress_callback.(%{
                type: :file_completed,
                file: file_path,
                current: idx,
                total: total_files,
                result: result
              })
            end

            {:ok, file_path, result}

          {:error, reason} ->
            Logger.warning("Failed to ingest file", %{file: file_path, reason: reason})

            if progress_callback do
              progress_callback.(%{
                type: :file_failed,
                file: file_path,
                current: idx,
                total: total_files,
                error: reason
              })
            end

            {:error, file_path, reason}
        end
      end)

    # Aggregate results
    successful = Enum.filter(results, fn r -> elem(r, 0) == :ok end)
    failed = Enum.filter(results, fn r -> elem(r, 0) == :error end)

    aggregated = %{
      documents_processed: length(successful),
      documents_failed: length(failed),
      total_chunks: Enum.sum(Enum.map(successful, fn {:ok, _, r} -> r.total_chunks end)),
      total_tokens: Enum.sum(Enum.map(successful, fn {:ok, _, r} -> r.total_tokens end)),
      entities_discovered:
        Enum.sum(Enum.map(successful, fn {:ok, _, r} -> r.entities_discovered end)),
      processing_time_ms: System.monotonic_time(:millisecond) - start_time,
      failed_files: Enum.map(failed, fn {:error, path, _} -> path end)
    }

    # Record batch completion
    WorldManager.record_event(world_id, :batch_complete, aggregated)

    {:ok, aggregated}
  end

  @doc """
  Ingests a directory of files matching a pattern.

  Uses Path.wildcard for pattern matching.
  """
  def ingest_directory(world_id, dir_path, pattern \\ "*.txt", opts \\ []) do
    full_pattern = Path.join(dir_path, pattern)
    files = Path.wildcard(full_pattern)

    if length(files) == 0 do
      {:error, :no_files_found}
    else
      Logger.info("Found files to ingest", %{count: length(files), pattern: full_pattern})
      ingest_files(world_id, files, opts)
    end
  end

  @doc """
  Ingests raw text content into a training world.
  """
  def ingest_text(world_id, text, opts \\ []) when is_binary(world_id) and is_binary(text) do
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    overlap = Keyword.get(opts, :overlap, @default_overlap)
    learn_types = Keyword.get(opts, :learn_types, true)
    progress_callback = Keyword.get(opts, :progress_callback)

    start_time = System.monotonic_time(:millisecond)

    # Load POS model once for all chunks
    pos_model =
      case POSTagger.load_model() do
        {:ok, model} -> model
        {:error, _} -> nil
      end

    # Split into chunks
    chunks = chunk_text(text, chunk_size, overlap)
    total_chunks = length(chunks)

    # Process each chunk
    {total_tokens, total_entities} =
      chunks
      |> Enum.with_index(1)
      |> Enum.reduce({0, 0}, fn {chunk, idx}, {tokens_acc, entities_acc} ->
        # Report progress
        if progress_callback do
          progress_callback.(%{
            type: :chunk_processed,
            current: idx,
            total: total_chunks
          })
        end

        # Process chunk
        {chunk_tokens, chunk_entities} =
          process_chunk(chunk, world_id, pos_model, learn_types)

        {tokens_acc + chunk_tokens, entities_acc + chunk_entities}
      end)

    duration = System.monotonic_time(:millisecond) - start_time

    # Update world metrics
    WorldManager.update_metrics(world_id, fn metrics ->
      WorldMetrics.record_document(metrics, total_tokens, total_chunks, duration)
    end)

    %{
      documents_processed: 1,
      total_chunks: total_chunks,
      total_tokens: total_tokens,
      entities_discovered: total_entities,
      processing_time_ms: duration
    }
  end

  @doc """
  Streams a large file for processing without loading it entirely into memory.
  """
  def stream_file(world_id, file_path, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    learn_types = Keyword.get(opts, :learn_types, true)

    # Load POS model once
    pos_model =
      case POSTagger.load_model() do
        {:ok, model} -> model
        {:error, _} -> nil
      end

    start_time = System.monotonic_time(:millisecond)

    result =
      try do
        file_path
        |> File.stream!([], chunk_size)
        |> Stream.with_index(1)
        |> Enum.reduce({0, 0, 0}, fn {chunk, _idx}, {chunks_acc, tokens_acc, entities_acc} ->
          {chunk_tokens, chunk_entities} =
            process_chunk(chunk, world_id, pos_model, learn_types)

          {chunks_acc + 1, tokens_acc + chunk_tokens, entities_acc + chunk_entities}
        end)
      rescue
        e ->
          Logger.error("Stream processing failed", %{error: inspect(e)})
          {:error, e}
      end

    case result do
      {total_chunks, total_tokens, total_entities} ->
        duration = System.monotonic_time(:millisecond) - start_time

        WorldManager.update_metrics(world_id, fn metrics ->
          WorldMetrics.record_document(metrics, total_tokens, total_chunks, duration)
        end)

        {:ok,
         %{
           documents_processed: 1,
           total_chunks: total_chunks,
           total_tokens: total_tokens,
           entities_discovered: total_entities,
           processing_time_ms: duration
         }}

      {:error, _} = error ->
        error
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp chunk_text(text, chunk_size, _overlap) when byte_size(text) <= chunk_size do
    [text]
  end

  defp chunk_text(text, chunk_size, overlap) do
    # Split into chunks with overlap
    # Try to break at sentence boundaries when possible

    do_chunk(text, chunk_size, overlap, [])
  end

  defp do_chunk("", _chunk_size, _overlap, acc), do: Enum.reverse(acc)

  defp do_chunk(text, chunk_size, _overlap, acc) when byte_size(text) <= chunk_size do
    Enum.reverse([text | acc])
  end

  defp do_chunk(text, chunk_size, overlap, acc) do
    # Take chunk_size characters
    chunk = String.slice(text, 0, chunk_size)

    # Try to find a sentence boundary near the end
    chunk = adjust_to_sentence_boundary(chunk)

    # Calculate where to start next chunk (with overlap)
    actual_chunk_size = String.length(chunk)
    next_start = max(0, actual_chunk_size - overlap)

    # Get remaining text
    remaining = String.slice(text, next_start..-1//1)

    do_chunk(remaining, chunk_size, overlap, [chunk | acc])
  end

  defp adjust_to_sentence_boundary(chunk) do
    # Look for sentence-ending punctuation in the last portion
    # This uses character analysis, not regex

    chunk_length = String.length(chunk)
    search_window = min(200, div(chunk_length, 4))
    search_start = chunk_length - search_window

    # Get the portion to search
    end_portion = String.slice(chunk, search_start..-1//1)

    # Find the last sentence boundary
    last_boundary = find_last_sentence_boundary(end_portion)

    case last_boundary do
      nil ->
        chunk

      boundary_offset ->
        # Cut at the boundary
        cut_point = search_start + boundary_offset + 1
        String.slice(chunk, 0, cut_point)
    end
  end

  defp find_last_sentence_boundary(text) do
    # Scan for sentence-ending punctuation followed by space
    graphemes = String.graphemes(text)

    graphemes
    |> Enum.with_index()
    |> Enum.reduce(nil, fn {g, idx}, last_boundary ->
      if sentence_ender?(g) and followed_by_space_or_end?(graphemes, idx) do
        idx
      else
        last_boundary
      end
    end)
  end

  defp sentence_ender?(grapheme), do: grapheme in [".", "!", "?"]

  defp followed_by_space_or_end?(graphemes, idx) do
    case Enum.at(graphemes, idx + 1) do
      nil -> true
      " " -> true
      "\n" -> true
      "\t" -> true
      _ -> false
    end
  end

  defp process_chunk(chunk, world_id, pos_model, learn_types) do
    # Tokenize
    tokens = Tokenizer.tokenize(chunk)
    token_count = length(tokens)

    # Discover entities
    discoveries =
      if pos_model do
        EntityDiscoverer.discover_entities(chunk, world_id, model: pos_model)
      else
        []
      end

    entity_count = length(discoveries)

    # Learn type patterns from known entities if enabled
    if learn_types and pos_model do
      learn_from_known_entities(chunk, world_id, pos_model, tokens)
    end

    {token_count, entity_count}
  end

  defp learn_from_known_entities(_chunk, world_id, pos_model, tokens) do
    token_texts = Enum.map(tokens, & &1.text)
    pos_predictions = POSTagger.predict(token_texts, pos_model)

    # Find known entities (not PROPN, but in gazetteer)
    pos_predictions
    |> Enum.with_index()
    |> Enum.each(fn {{token_text, _tag}, idx} ->
      # Check if this token is a known entity
      known_types = ChatBot.ML.Gazetteer.lookup_all_types(token_text, world_id)

      if length(known_types) == 1 do
        # Single known type - learn from this context
        entity_type = Map.get(hd(known_types), :entity_type) || Map.get(hd(known_types), :type)

        # Extract context
        context_window = 5
        start_idx = max(0, idx - context_window)
        end_idx = min(length(tokens) - 1, idx + context_window)

        context_tokens = Enum.slice(tokens, start_idx..end_idx)

        context_tags =
          Enum.slice(pos_predictions, start_idx..end_idx) |> Enum.map(fn {_, tag} -> tag end)

        TypeInferrer.learn_from_known_entity(entity_type, context_tokens, context_tags, world_id)
      end
    end)
  end
end
