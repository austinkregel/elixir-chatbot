defmodule ChatBot.Memory.Embedder do
  @moduledoc """
  Embedding utilities for the cognitive memory system.

  Produces TF-IDF based embeddings for text that can be used in
  similarity search. This replaces the byte frequency approach
  from the Rust implementation with a more semantically meaningful
  TF-IDF vectorization.

  The embedder maintains a vocabulary and IDF weights that are
  built from training data and used to consistently embed new text.

  ## World Scoping

  This module provides both global and world-scoped embedding:
  - `embed/1` - Uses the global/default vocabulary
  - `embed/2` - Uses world-specific vocabulary via WorldEmbedder

  World-specific embeddings are managed by `ChatBot.Learning.WorldEmbedder`
  and are built lazily from world-specific episodes.
  """

  use GenServer

  alias ChatBot.ML.Tokenizer
  alias ChatBot.Learning.WorldEmbedder

  require Logger

  @default_vocab_size 2000
  @min_word_frequency 2
  @default_world_id "default"
  @pubsub ChatBot.PubSub

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Build vocabulary and IDF weights from a list of texts.
  This should be called during training for the default/global vocabulary.
  """
  def build_vocabulary(texts) when is_list(texts) do
    GenServer.call(__MODULE__, {:build_vocabulary, texts}, :infinity)
  end

  @doc """
  Build vocabulary for a specific world.
  For non-default worlds, this triggers the WorldEmbedder to build from episodes.
  For the default world, uses the provided texts directly.
  """
  def build_vocabulary(texts, world_id) when is_list(texts) and is_binary(world_id) do
    if world_id == @default_world_id do
      build_vocabulary(texts)
    else
      # WorldEmbedder builds from episodes, so just trigger a rebuild
      WorldEmbedder.build_vocabulary(world_id, force: true)
    end
  end

  @doc """
  Embed text into a TF-IDF vector using the global/default vocabulary.
  Returns a list of floats representing the embedding.
  """
  def embed(text) when is_binary(text) do
    # Wrap with telemetry span for async, non-blocking metrics
    ChatBot.Telemetry.span(:memory_embed, %{text_length: byte_size(text)}, fn ->
      GenServer.call(__MODULE__, {:embed, text})
    end)
  end

  @doc """
  Embed text using a world-specific vocabulary.
  Falls back to global vocabulary if world vocabulary is not available.
  """
  def embed(text, world_id) when is_binary(text) and is_binary(world_id) do
    if world_id == @default_world_id do
      embed(text)
    else
      case WorldEmbedder.embed(world_id, text) do
        {:ok, embedding} ->
          {:ok, embedding}

        {:error, :not_ready} ->
          # Fall back to global embedder
          embed(text)

        {:error, :no_training_data} ->
          # Fall back to global embedder
          embed(text)

        error ->
          error
      end
    end
  end

  @doc """
  Compute cosine similarity between two embedding vectors.
  Returns a float in [-1, 1].
  """
  def cosine_similarity(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b) do
    compute_cosine_similarity(vec_a, vec_b)
  end

  @doc """
  Check if the embedder is initialized with vocabulary.
  Uses a short timeout to avoid blocking if embedder is busy.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc """
  Check if a world-specific embedder is ready.
  """
  def ready?(world_id) when is_binary(world_id) do
    if world_id == @default_world_id do
      ready?()
    else
      WorldEmbedder.ready?(world_id)
    end
  end

  @doc """
  Get the current vocabulary size.
  """
  def vocabulary_size do
    GenServer.call(__MODULE__, :vocabulary_size)
  end

  @doc """
  Get vocabulary size for a specific world.
  """
  def vocabulary_size(world_id) when is_binary(world_id) do
    if world_id == @default_world_id do
      vocabulary_size()
    else
      case WorldEmbedder.get_status(world_id) do
        %{vocabulary_size: size} -> size
        _ -> 0
      end
    end
  end

  @doc """
  Load a pre-built vocabulary and IDF weights.
  """
  def load_model(model) when is_map(model) do
    GenServer.call(__MODULE__, {:load_model, model})
  end

  @doc """
  Load a pre-built vocabulary for a specific world.
  """
  def load_model(model, world_id) when is_map(model) and is_binary(world_id) do
    if world_id == @default_world_id do
      load_model(model)
    else
      WorldEmbedder.load_model(world_id, model)
    end
  end

  @doc """
  Export the current vocabulary and IDF weights for persistence.
  """
  def export_model do
    GenServer.call(__MODULE__, :export_model)
  end

  @doc """
  Export vocabulary for a specific world.
  """
  def export_model(world_id) when is_binary(world_id) do
    if world_id == @default_world_id do
      export_model()
    else
      WorldEmbedder.export_model(world_id)
    end
  end

  @doc """
  Get detailed status of the embedder including initialization progress.
  Uses a short timeout to avoid blocking.
  """
  def get_status do
    try do
      GenServer.call(__MODULE__, :get_status, 100)
    catch
      :exit, {:timeout, _} ->
        %{
          ready: false,
          phase: :busy,
          phase_label: "Processing (not responding)",
          progress: nil,
          vocabulary_size: 0,
          started_at: nil
        }

      :exit, {:noproc, _} ->
        %{
          ready: false,
          phase: :not_started,
          phase_label: "Not started",
          progress: nil,
          vocabulary_size: 0,
          started_at: nil
        }
    end
  end

  @doc """
  Get status for a specific world's embedder.
  """
  def get_status(world_id) when is_binary(world_id) do
    if world_id == @default_world_id do
      get_status()
    else
      WorldEmbedder.get_status(world_id)
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Subscribe to world model events
    Phoenix.PubSub.subscribe(@pubsub, "world_models:status")

    {:ok,
     %{
       vocabulary: %{},
       idf_weights: %{},
       ready: false,
       # Initialization progress tracking
       phase: :idle,
       phase_label: "Idle",
       progress: nil,
       total_texts: 0,
       processed_texts: 0,
       started_at: nil
     }}
  end

  # Handle world model events - could trigger vocabulary reload
  @impl true
  def handle_info({:world_models_loaded, _world_id, _status}, state) do
    # World models were loaded - WorldEmbedder handles its own state
    {:noreply, state}
  end

  @impl true
  def handle_info({:world_models_loading, _world_id}, state) do
    {:noreply, state}
  end

  @impl true
  def handle_info({:world_models_error, _world_id, _reason}, state) do
    {:noreply, state}
  end

  @impl true
  def handle_call({:build_vocabulary, texts}, _from, state) do
    total_texts = length(texts)
    started_at = System.monotonic_time(:millisecond)

    Logger.info("Building embedder vocabulary from #{total_texts} texts")

    # Phase 1: Tokenizing texts
    state = %{
      state
      | phase: :tokenizing,
        phase_label: "Tokenizing texts",
        total_texts: total_texts,
        processed_texts: 0,
        started_at: started_at
    }

    # Tokenize all texts with progress tracking
    {all_tokens, tokenized_texts} = tokenize_with_progress(texts, state)

    # Phase 2: Building frequency table
    state = %{state | phase: :building_frequencies, phase_label: "Building frequency table"}

    token_frequencies =
      all_tokens
      |> Enum.frequencies()
      |> Enum.filter(fn {_word, count} -> count >= @min_word_frequency end)
      |> Enum.sort_by(fn {_word, count} -> -count end)
      |> Enum.take(@default_vocab_size)
      |> Enum.map(fn {word, _count} -> word end)

    vocabulary =
      token_frequencies
      |> Enum.with_index()
      |> Enum.into(%{})

    vocab_size = map_size(vocabulary)

    # Phase 3: Calculating IDF weights
    state = %{
      state
      | phase: :calculating_idf,
        phase_label: "Calculating IDF weights",
        progress: %{current: 0, total: vocab_size}
    }

    idf_weights = calculate_idf_weights(vocabulary, tokenized_texts, total_texts)

    elapsed_ms = System.monotonic_time(:millisecond) - started_at
    Logger.info("Embedder vocabulary built: #{vocab_size} words in #{elapsed_ms}ms")

    new_state = %{
      state
      | vocabulary: vocabulary,
        idf_weights: idf_weights,
        ready: true,
        phase: :ready,
        phase_label: "Ready",
        progress: nil
    }

    {:reply, {:ok, vocab_size}, new_state}
  end

  @impl true
  def handle_call({:embed, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:embed, text}, _from, state) do
    vector = vectorize(text, state.vocabulary, state.idf_weights)
    {:reply, {:ok, vector}, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  @impl true
  def handle_call(:vocabulary_size, _from, state) do
    {:reply, map_size(state.vocabulary), state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    elapsed_ms =
      if state.started_at do
        System.monotonic_time(:millisecond) - state.started_at
      else
        nil
      end

    status = %{
      ready: state.ready,
      phase: state.phase,
      phase_label: state.phase_label,
      progress: build_progress_info(state),
      vocabulary_size: map_size(state.vocabulary),
      elapsed_ms: elapsed_ms
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call({:load_model, model}, _from, _state) do
    new_state = %{
      vocabulary: Map.get(model, :vocabulary, %{}),
      idf_weights: Map.get(model, :idf_weights, %{}),
      ready: true,
      phase: :ready,
      phase_label: "Ready (loaded from model)",
      progress: nil,
      total_texts: 0,
      processed_texts: 0,
      started_at: nil
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:export_model, _from, state) do
    model = %{
      vocabulary: state.vocabulary,
      idf_weights: state.idf_weights
    }

    {:reply, {:ok, model}, state}
  end

  # ============================================================================
  # Private Functions - Progress Tracking
  # ============================================================================

  defp tokenize_with_progress(texts, _state) do
    # Tokenize all texts and keep the tokenized versions for IDF calculation
    tokenized =
      texts
      |> Enum.map(&tokenize/1)

    all_tokens = List.flatten(tokenized)
    {all_tokens, tokenized}
  end

  defp calculate_idf_weights(vocabulary, tokenized_texts, num_docs) do
    # Convert tokenized texts to sets for faster membership testing
    text_sets = Enum.map(tokenized_texts, &MapSet.new/1)

    vocabulary
    |> Enum.map(fn {word, _idx} ->
      doc_freq = Enum.count(text_sets, fn set -> MapSet.member?(set, word) end)
      idf = :math.log(num_docs / max(doc_freq, 1))
      {word, idf}
    end)
    |> Enum.into(%{})
  end

  defp build_progress_info(%{phase: :tokenizing} = state) do
    %{
      current: state.processed_texts,
      total: state.total_texts,
      percent:
        if(state.total_texts > 0,
          do: round(state.processed_texts / state.total_texts * 100),
          else: 0
        ),
      detail: "Processing #{state.total_texts} documents"
    }
  end

  defp build_progress_info(%{phase: :building_frequencies}) do
    %{
      current: nil,
      total: nil,
      percent: nil,
      detail: "Analyzing token frequencies"
    }
  end

  defp build_progress_info(%{phase: :calculating_idf, progress: progress})
       when is_map(progress) do
    %{
      current: progress.current,
      total: progress.total,
      percent:
        if(progress.total > 0, do: round(progress.current / progress.total * 100), else: 0),
      detail: "Computing IDF for #{progress.total} vocabulary terms"
    }
  end

  defp build_progress_info(_state), do: nil

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp tokenize(text) do
    Tokenizer.tokenize_normalized(text, min_length: 2)
  end

  defp vectorize(text, vocabulary, idf_weights) do
    tokens = tokenize(text)
    token_freq = Enum.frequencies(tokens)

    # Build TF-IDF vector
    vector =
      vocabulary
      |> Enum.sort_by(fn {_word, idx} -> idx end)
      |> Enum.map(fn {word, _idx} ->
        tf = Map.get(token_freq, word, 0)
        idf = Map.get(idf_weights, word, 0.0)
        tf * idf
      end)

    # Normalize the vector
    normalize_vector(vector)
  end

  defp normalize_vector(vector) do
    magnitude = :math.sqrt(Enum.reduce(vector, 0, fn val, acc -> acc + val * val end))

    if magnitude > 0 do
      Enum.map(vector, &(&1 / magnitude))
    else
      vector
    end
  end

  defp compute_cosine_similarity(vec1, vec2) do
    if length(vec1) != length(vec2) or length(vec1) == 0 do
      0.0
    else
      dot_product =
        Enum.zip(vec1, vec2)
        |> Enum.reduce(0, fn {a, b}, acc -> acc + a * b end)

      mag1 = :math.sqrt(Enum.reduce(vec1, 0, fn val, acc -> acc + val * val end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0, fn val, acc -> acc + val * val end))

      if mag1 > 0 and mag2 > 0 do
        dot_product / (mag1 * mag2)
      else
        0.0
      end
    end
  end
end
