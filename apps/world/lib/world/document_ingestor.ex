defmodule World.DocumentIngestor do
  @moduledoc "Processes large documents for entity discovery and learning.\n\nHandles:\n- Chunking large files into manageable pieces\n- Streaming processing to avoid memory issues\n- Progress tracking and reporting\n- Batch entity discovery\n- **Code file analysis** (routes to Brain.Code.Pipeline)\n\n## Code File Support\n\nWhen processing files with code extensions (.py, .ex, .go, etc.), the\ningestor automatically routes them to `Brain.Code.Pipeline` for proper\nAST-based analysis instead of NLP-based processing.\n\nSupported code extensions: .c, .h, .cpp, .cc, .java, .cs, .php, .py, .rb, .ex, .exs, .go\n"

  alias Brain.ML.Gazetteer
  alias Brain.Code.Pipeline
  alias Brain.ML
  require Logger

  alias ML.{Tokenizer, POSTagger}
  alias World.{EntityDiscoverer, TypeInferrer}
  alias World.Manager, as: WorldManager
  alias World.Metrics, as: WorldMetrics
  @code_extensions ~w(.c .h .cpp .cc .cxx .hpp .java .cs .php .py .pyw .rb .ex .exs .go)

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
  @default_chunk_size 5000
  @default_overlap 200

  @doc "Ingests a single file into a training world.\n\nFor code files (.py, .ex, .go, etc.), automatically routes to\n`Brain.Code.Pipeline` for AST-based analysis.\n\nFor text/document files, uses NLP-based chunking and entity discovery.\n\n## Options\n  - `:chunk_size` - Characters per chunk (default: 5000)\n  - `:overlap` - Overlap between chunks (default: 200)\n  - `:progress_callback` - Function called with progress updates\n  - `:learn_types` - Whether to learn type patterns from known entities (default: true)\n  - `:force_text` - Force text processing even for code files (default: false)\n"
  def ingest_file(world_id, file_path, opts \\ [])
      when is_binary(world_id) and is_binary(file_path) do
    force_text = Keyword.get(opts, :force_text, false)

    if is_code_file?(file_path) and not force_text do
      ingest_code_file(world_id, file_path, opts)
    else
      ingest_text_file(world_id, file_path, opts)
    end
  end

  @doc "Checks if a file is a code file based on its extension.\n"
  @spec is_code_file?(String.t()) :: boolean()
  def is_code_file?(file_path) do
    ext = Path.extname(file_path) |> String.downcase()
    ext in @code_extensions
  end

  defp ingest_code_file(world_id, file_path, opts) do
    start_time = System.monotonic_time(:millisecond)
    progress_callback = Keyword.get(opts, :progress_callback)

    if progress_callback do
      progress_callback.(%{
        type: :code_analysis_started,
        file: file_path
      })
    end

    case Pipeline.process_file(file_path, world_id: world_id, store: true) do
      {:ok, result} ->
        duration = System.monotonic_time(:millisecond) - start_time

        WorldManager.record_event(world_id, :code_file_processed, %{
          file_path: file_path,
          language: result.language,
          symbols: result.stats.symbol_count,
          relations: result.stats.relation_count,
          duration_ms: duration
        })

        if progress_callback do
          progress_callback.(%{
            type: :code_analysis_completed,
            file: file_path,
            result: result
          })
        end

        {:ok,
         %{
           documents_processed: 1,
           total_chunks: 1,
           total_tokens: result.stats.symbol_count,
           entities_discovered: result.stats.symbol_count,
           processing_time_ms: duration,
           code_analysis: result
         }}

      {:error, reason} ->
        Logger.warning("Code file analysis failed", %{file: file_path, reason: reason})

        if progress_callback do
          progress_callback.(%{
            type: :code_analysis_failed,
            file: file_path,
            error: reason
          })
        end

        {:error, {:code_analysis_failed, reason}}
    end
  end

  defp ingest_text_file(world_id, file_path, opts) do
    start_time = System.monotonic_time(:millisecond)

    case File.read(file_path) do
      {:ok, content} ->
        result = ingest_text(world_id, content, opts)
        duration = System.monotonic_time(:millisecond) - start_time

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

  @doc "Ingests multiple files into a training world.\n\nProcesses files sequentially and aggregates results.\n"
  def ingest_files(world_id, file_paths, opts \\ [])
      when is_binary(world_id) and is_list(file_paths) do
    start_time = System.monotonic_time(:millisecond)
    total_files = length(file_paths)
    progress_callback = Keyword.get(opts, :progress_callback)

    results =
      file_paths
      |> Enum.with_index(1)
      |> Enum.map(fn {file_path, idx} ->
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

    WorldManager.record_event(world_id, :batch_complete, aggregated)

    {:ok, aggregated}
  end

  @doc "Ingests a directory of files matching a pattern.\n\nUses Path.wildcard for pattern matching.\n\n## Options\n  - `:include_code` - Include code files in processing (default: true)\n  - `:recursive` - Process subdirectories (default: false for pattern, true for code)\n  - All other options are passed to `ingest_file/3`\n"
  def ingest_directory(world_id, dir_path, pattern \\ "*.txt", opts \\ []) do
    include_code = Keyword.get(opts, :include_code, true)
    recursive = Keyword.get(opts, :recursive, false)

    full_pattern =
      if recursive do
        Path.join([dir_path, "**", pattern])
      else
        Path.join(dir_path, pattern)
      end

    text_files = Path.wildcard(full_pattern)

    code_files =
      if include_code do
        find_code_files(dir_path, recursive)
      else
        []
      end

    all_files = Enum.uniq(text_files ++ code_files)

    if all_files == [] do
      {:error, :no_files_found}
    else
      text_count = length(text_files)
      code_count = length(code_files)

      Logger.info("Found files to ingest", %{
        total: length(all_files),
        text_files: text_count,
        code_files: code_count,
        pattern: full_pattern
      })

      ingest_files(world_id, all_files, opts)
    end
  end

  @doc "Ingests only code files from a directory.\n\nThis is a convenience function for analyzing codebases.\n\n## Options\n  - `:recursive` - Process subdirectories (default: true)\n  - `:extensions` - Code extensions to include (default: all supported)\n  - `:exclude` - Patterns to exclude (default: [\"node_modules\", \".git\", \"_build\"])\n"
  def ingest_codebase(world_id, dir_path, opts \\ []) do
    recursive = Keyword.get(opts, :recursive, true)
    extensions = Keyword.get(opts, :extensions, @code_extensions)

    exclude =
      Keyword.get(opts, :exclude, [
        "node_modules",
        ".git",
        "_build",
        "deps",
        "__pycache__",
        "vendor"
      ])

    files = find_code_files(dir_path, recursive, extensions, exclude)

    if files == [] do
      {:error, :no_code_files_found}
    else
      Logger.info("Found code files to analyze", %{count: length(files), dir: dir_path})
      ingest_files(world_id, files, opts)
    end
  end

  defp find_code_files(dir_path, recursive, extensions \\ @code_extensions, exclude \\ []) do
    pattern =
      if recursive do
        Path.join(dir_path, "**/*")
      else
        Path.join(dir_path, "*")
      end

    Path.wildcard(pattern)
    |> Enum.filter(fn path ->
      File.regular?(path) and
        Path.extname(path) in extensions and
        not excluded?(path, exclude)
    end)
    |> Enum.sort()
  end

  defp excluded?(path, exclude_patterns) do
    Enum.any?(exclude_patterns, fn pattern ->
      String.contains?(path, pattern)
    end)
  end

  @doc "Ingests raw text content into a training world.\n"
  def ingest_text(world_id, text, opts \\ []) when is_binary(world_id) and is_binary(text) do
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    overlap = Keyword.get(opts, :overlap, @default_overlap)
    learn_types = Keyword.get(opts, :learn_types, true)
    progress_callback = Keyword.get(opts, :progress_callback)

    start_time = System.monotonic_time(:millisecond)

    pos_model =
      case POSTagger.load_model() do
        {:ok, model} -> model
        {:error, _} -> nil
      end

    chunks = chunk_text(text, chunk_size, overlap)
    total_chunks = length(chunks)

    {total_tokens, total_entities} =
      chunks
      |> Enum.with_index(1)
      |> Enum.reduce({0, 0}, fn {chunk, idx}, {tokens_acc, entities_acc} ->
        if progress_callback do
          progress_callback.(%{
            type: :chunk_processed,
            current: idx,
            total: total_chunks
          })
        end

        {chunk_tokens, chunk_entities} =
          process_chunk(chunk, world_id, pos_model, learn_types)

        {tokens_acc + chunk_tokens, entities_acc + chunk_entities}
      end)

    duration = System.monotonic_time(:millisecond) - start_time

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

  @doc "Streams a large file for processing without loading it entirely into memory.\n"
  def stream_file(world_id, file_path, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    learn_types = Keyword.get(opts, :learn_types, true)

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

  defp chunk_text(text, chunk_size, _overlap) when byte_size(text) <= chunk_size do
    [text]
  end

  defp chunk_text(text, chunk_size, overlap) do
    do_chunk(text, chunk_size, overlap, [])
  end

  defp do_chunk("", _chunk_size, _overlap, acc) do
    Enum.reverse(acc)
  end

  defp do_chunk(text, chunk_size, _overlap, acc) when byte_size(text) <= chunk_size do
    Enum.reverse([text | acc])
  end

  defp do_chunk(text, chunk_size, overlap, acc) do
    chunk = String.slice(text, 0, chunk_size)
    chunk = adjust_to_sentence_boundary(chunk)
    actual_chunk_size = String.length(chunk)
    next_start = max(0, actual_chunk_size - overlap)
    remaining = String.slice(text, next_start..-1//1)

    do_chunk(remaining, chunk_size, overlap, [chunk | acc])
  end

  defp adjust_to_sentence_boundary(chunk) do
    chunk_length = String.length(chunk)
    search_window = min(200, div(chunk_length, 4))
    search_start = chunk_length - search_window
    end_portion = String.slice(chunk, search_start..-1//1)
    last_boundary = find_last_sentence_boundary(end_portion)

    case last_boundary do
      nil ->
        chunk

      boundary_offset ->
        cut_point = search_start + boundary_offset + 1
        String.slice(chunk, 0, cut_point)
    end
  end

  defp find_last_sentence_boundary(text) do
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

  defp sentence_ender?(grapheme) do
    grapheme in [".", "!", "?"]
  end

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
    tokens = Tokenizer.tokenize(chunk)
    token_count = length(tokens)

    discoveries =
      if pos_model do
        EntityDiscoverer.discover_entities(chunk, world_id, model: pos_model)
      else
        []
      end

    entity_count = length(discoveries)

    if learn_types and pos_model do
      learn_from_known_entities(chunk, world_id, pos_model, tokens)
    end

    {token_count, entity_count}
  end

  defp learn_from_known_entities(_chunk, world_id, pos_model, tokens) do
    token_texts = Enum.map(tokens, & &1.text)
    pos_predictions = POSTagger.predict(token_texts, pos_model)

    pos_predictions
    |> Enum.with_index()
    |> Enum.each(fn {{token_text, _tag}, idx} ->
      known_types = Gazetteer.lookup_all_types(token_text, world_id)

      if length(known_types) == 1 do
        entity_type = Map.get(hd(known_types), :entity_type) || Map.get(hd(known_types), :type)
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
