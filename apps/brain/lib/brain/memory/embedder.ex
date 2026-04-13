defmodule Brain.Memory.Embedder do
  @moduledoc "Embedding utilities for the cognitive memory system.\n\nProduces TF-IDF based embeddings for text that can be used in\nsimilarity search. This replaces the byte frequency approach\nfrom the Rust implementation with a more semantically meaningful\nTF-IDF vectorization.\n\nThe embedder maintains a vocabulary and IDF weights that are\nbuilt from training data and used to consistently embed new text.\n\n## World Scoping\n\nThis module provides both global and world-scoped embedding:\n- `embed/1` - Uses the global/default vocabulary\n- `embed/2` - Uses world-specific vocabulary via WorldEmbedder\n\nWorld-specific embeddings are managed by `World.Embedder`\nand are built lazily from world-specific episodes.\n"

  # World.Embedder is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Embedder}

  alias Phoenix.PubSub
  alias Brain.Telemetry
  use GenServer

  alias Brain.ML.Tokenizer
  alias World.Embedder, as: WorldEmbedder

  require Logger

  @default_vocab_size 2000
  @min_word_frequency 2
  @default_world_id "default"
  @pubsub Brain.PubSub

  @doc """
  Starts the Embedder GenServer.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Build vocabulary and IDF weights from a list of texts.\nThis should be called during training for the default/global vocabulary.\n"
  def build_vocabulary(texts) when is_list(texts) do
    GenServer.call(__MODULE__, {:build_vocabulary, texts}, :infinity)
  end

  @doc "Build vocabulary for a specific world.\nFor non-default worlds, this triggers the WorldEmbedder to build from episodes.\nFor the default world, uses the provided texts directly.\n"
  def build_vocabulary(texts, world_id) when is_list(texts) and is_binary(world_id) do
    if world_id == @default_world_id do
      build_vocabulary(texts)
    else
      WorldEmbedder.build_vocabulary(world_id, force: true)
    end
  end

  @doc "Embed text into a TF-IDF vector using the global/default vocabulary.\nReturns a list of floats representing the embedding.\n"
  def embed(text) when is_binary(text) do
    Telemetry.span(:memory_embed, %{text_length: byte_size(text)}, fn ->
      GenServer.call(__MODULE__, {:embed, text})
    end)
  end

  @doc "Embed text using a world-specific vocabulary.\nFalls back to global vocabulary if world vocabulary is not available.\n"
  def embed(text, world_id) when is_binary(text) and is_binary(world_id) do
    if world_id == @default_world_id do
      embed(text)
    else
      case WorldEmbedder.embed(world_id, text) do
        {:ok, embedding} ->
          {:ok, embedding}

        {:error, reason}
        when reason in [
               :no_training_data,
               :vocabulary_building,
               :not_initialized,
               :table_not_ready
             ] ->
          embed(text)

        error ->
          error
      end
    end
  end

  @doc "Compute cosine similarity between two embedding vectors.\nReturns a float in [-1, 1].\n"
  def cosine_similarity(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b) do
    FourthWall.Math.cosine_similarity(vec_a, vec_b)
  end

  @doc "Check if the embedder is initialized with vocabulary.\nUses a short timeout to avoid blocking if embedder is busy.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc "Check if a world-specific embedder is ready.\n"
  def ready?(world_id) when is_binary(world_id) do
    if world_id == @default_world_id do
      ready?()
    else
      WorldEmbedder.ready?(world_id)
    end
  end

  @doc "Get the current vocabulary size.\n"
  def vocabulary_size do
    GenServer.call(__MODULE__, :vocabulary_size)
  end

  @doc "Get vocabulary size for a specific world.\n"
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

  @doc "Load a pre-built vocabulary and IDF weights.\n"
  def load_model(model) when is_map(model) do
    GenServer.call(__MODULE__, {:load_model, model})
  end

  @doc "Load a pre-built vocabulary for a specific world.\n"
  def load_model(model, world_id) when is_map(model) and is_binary(world_id) do
    if world_id == @default_world_id do
      load_model(model)
    else
      WorldEmbedder.load_model(world_id, model)
    end
  end

  @doc "Export the current vocabulary and IDF weights for persistence.\n"
  def export_model do
    GenServer.call(__MODULE__, :export_model)
  end

  @doc "Export vocabulary for a specific world.\n"
  def export_model(world_id) when is_binary(world_id) do
    if world_id == @default_world_id do
      export_model()
    else
      WorldEmbedder.export_model(world_id)
    end
  end

  @doc "Get detailed status of the embedder including initialization progress.\nUses a short timeout to avoid blocking.\n"
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

  @doc "Get status for a specific world's embedder.\n"
  def get_status(world_id) when is_binary(world_id) do
    if world_id == @default_world_id do
      get_status()
    else
      WorldEmbedder.get_status(world_id)
    end
  end

  @impl true
  def init(_opts) do
    PubSub.subscribe(@pubsub, "world_models:status")

    state = %{
      vocabulary: %{},
      idf_weights: %{},
      ready: false,
      phase: :idle,
      phase_label: "Idle",
      progress: nil,
      total_texts: 0,
      processed_texts: 0,
      started_at: nil
    }

    send(self(), :try_auto_load)

    {:ok, state}
  end

  @impl true
  def handle_info(:try_auto_load, %{ready: true} = state) do
    {:noreply, state}
  end

  def handle_info(:try_auto_load, state) do
    embedder_path =
      case Application.get_env(:brain, :ml, [])[:models_path] do
        nil -> Path.join(:code.priv_dir(:brain), "ml_models/embedder.term")
        path -> Path.join(path, "embedder.term")
      end

    Brain.ML.ModelStore.ensure_local("embedder.term", embedder_path)

    state =
      if File.exists?(embedder_path) do
        case File.read(embedder_path) do
          {:ok, binary} ->
            model = :erlang.binary_to_term(binary)

            Logger.info("Embedder vocabulary auto-loaded from #{embedder_path}")

            %{
              state
              | vocabulary: Map.get(model, :vocabulary, %{}),
                idf_weights: Map.get(model, :idf_weights, %{}),
                ready: true,
                phase: :ready,
                phase_label: "Ready (auto-loaded)"
            }

          {:error, reason} ->
            Logger.debug("Failed to auto-load embedder model: #{inspect(reason)}")
            state
        end
      else
        Logger.debug("No embedder model found at #{embedder_path} for auto-load")
        state
      end

    {:noreply, state}
  end

  @impl true
  def handle_info({:world_models_loaded, _world_id, _status}, state) do
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

    state = %{
      state
      | phase: :tokenizing,
        phase_label: "Tokenizing texts",
        total_texts: total_texts,
        processed_texts: 0,
        started_at: started_at
    }

    {all_tokens, tokenized_texts} = tokenize_with_progress(texts, state)
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

  defp tokenize_with_progress(texts, _state) do
    tokenized =
      texts
      |> Enum.map(&tokenize/1)

    all_tokens = List.flatten(tokenized)
    {all_tokens, tokenized}
  end

  defp calculate_idf_weights(vocabulary, tokenized_texts, num_docs) do
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
        if(state.total_texts > 0) do
          round(state.processed_texts / state.total_texts * 100)
        else
          0
        end,
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
        if(progress.total > 0) do
          round(progress.current / progress.total * 100)
        else
          0
        end,
      detail: "Computing IDF for #{progress.total} vocabulary terms"
    }
  end

  defp build_progress_info(_state) do
    nil
  end

  defp tokenize(text) do
    Tokenizer.tokenize_normalized(text, min_length: 2)
  end

  defp vectorize(text, vocabulary, idf_weights) do
    tokens = tokenize(text)
    expanded = expand_tokens_with_lexicon(tokens, vocabulary)
    token_freq = Enum.frequencies(expanded)

    vector =
      vocabulary
      |> Enum.sort_by(fn {_word, idx} -> idx end)
      |> Enum.map(fn {word, _idx} ->
        tf = Map.get(token_freq, word, 0)
        idf = Map.get(idf_weights, word, 0.0)
        tf * idf
      end)

    normalize_vector(vector)
  end

  defp expand_tokens_with_lexicon(tokens, vocabulary) do
    if Process.whereis(Brain.ML.Lexicon) do
      Brain.ML.Lexicon.expand_with_synonyms(tokens, vocabulary)
    else
      tokens
    end
  end

  defp normalize_vector(vector) do
    magnitude = :math.sqrt(Enum.reduce(vector, 0, fn val, acc -> acc + val * val end))

    if magnitude > 0 do
      Enum.map(vector, &(&1 / magnitude))
    else
      vector
    end
  end

end
