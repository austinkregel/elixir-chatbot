defmodule World.Embedder do
  @moduledoc "Manages world-specific TF-IDF embeddings.\n\nEach training world can have its own vocabulary and IDF weights, built from\nthe episodes stored in that world. This provides domain-specific semantic\nsimilarity within each world's context.\n\nEmbeddings are built lazily on first use and stored in ETS for performance.\nVocabulary can be rebuilt when significant new data is added.\n"

  alias Brain.Memory.Store
  require Logger

  alias Brain.ML.Tokenizer

  @ets_table :world_embedders
  @default_vocab_size 2000
  @min_word_frequency 2

  @doc "Ensures the ETS table exists. Called during application startup.\n"
  def init do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
    end

    :ok
  end

  @doc "Checks if embeddings are ready for a world.\nReturns true if the world has a built vocabulary.\n"
  def ready?(world_id) do
    case get_state(world_id) do
      {:ok, state} -> state.ready
      _ -> false
    end
  end

  @doc "Gets the current status of a world's embedder.\nReturns detailed information for dashboard display.\n"
  def get_status(world_id) do
    case get_state(world_id) do
      {:ok, state} ->
        %{
          ready: state.ready,
          phase: state.phase,
          phase_label: phase_label(state.phase),
          vocabulary_size: map_size(state.vocabulary),
          built_at: state.built_at,
          episode_count: state.episode_count
        }

      {:error, :table_not_ready} ->
        %{
          ready: false,
          phase: :table_not_ready,
          phase_label: "Table not ready",
          vocabulary_size: 0,
          built_at: nil,
          episode_count: 0
        }

      {:error, :not_found} ->
        %{
          ready: false,
          phase: :not_initialized,
          phase_label: "Not initialized",
          vocabulary_size: 0,
          built_at: nil,
          episode_count: 0
        }
    end
  end

  @doc "Embeds text using the world's vocabulary.\nDoes NOT auto-build vocabulary to avoid deadlocks - call build_vocabulary first.\n\nReturns {:ok, embedding} or {:error, reason}\n"
  def embed(world_id, text) when is_binary(world_id) and is_binary(text) do
    case get_state(world_id) do
      {:ok, %{ready: true} = state} ->
        embedding = vectorize(text, state.vocabulary, state.idf_weights)
        {:ok, embedding}

      {:ok, %{phase: :no_data}} ->
        {:error, :no_training_data}

      {:ok, _state} ->
        {:error, :vocabulary_building}

      {:error, :not_found} ->
        {:error, :not_initialized}

      {:error, :table_not_ready} ->
        {:error, :table_not_ready}

      error ->
        error
    end
  end

  @doc "Builds or rebuilds the vocabulary for a world from its episodes.\n"
  def build_vocabulary(world_id, opts \\ []) do
    force = Keyword.get(opts, :force, false)

    case get_state(world_id) do
      {:ok, %{ready: true}} when not force ->
        {:ok, :already_built}

      _ ->
        do_build_vocabulary(world_id)
    end
  end

  @doc "Computes cosine similarity between two embeddings.\n"
  def cosine_similarity(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b) do
    if length(vec_a) != length(vec_b) or vec_a == [] do
      0.0
    else
      dot_product =
        Enum.zip(vec_a, vec_b)
        |> Enum.reduce(0, fn {a, b}, acc -> acc + a * b end)

      mag1 = :math.sqrt(Enum.reduce(vec_a, 0, fn val, acc -> acc + val * val end))
      mag2 = :math.sqrt(Enum.reduce(vec_b, 0, fn val, acc -> acc + val * val end))

      if mag1 > 0 and mag2 > 0 do
        dot_product / (mag1 * mag2)
      else
        0.0
      end
    end
  end

  @doc "Clears the embedder state for a world (used when world is destroyed).\n"
  def clear(world_id) do
    :ets.delete(@ets_table, world_id)
    :ok
  rescue
    ArgumentError -> :ok
  end

  @doc "Loads a pre-built vocabulary and IDF weights for a world.\nUsed when loading world models from disk.\n"
  def load_model(world_id, model) when is_binary(world_id) and is_map(model) do
    state = %{
      ready: true,
      phase: :ready,
      vocabulary: Map.get(model, :vocabulary, %{}),
      idf_weights: Map.get(model, :idf_weights, %{}),
      built_at: Map.get(model, :built_at, DateTime.utc_now()),
      episode_count: Map.get(model, :episode_count, 0)
    }

    case set_state(world_id, state) do
      :ok ->
        Logger.info(
          "Loaded embedder model for world #{world_id}: #{map_size(state.vocabulary)} terms"
        )

        :ok

      error ->
        error
    end
  end

  @doc "Exports the vocabulary and IDF weights for a world for persistence.\n"
  def export_model(world_id) when is_binary(world_id) do
    case get_state(world_id) do
      {:ok, state} ->
        {:ok,
         %{
           vocabulary: state.vocabulary,
           idf_weights: state.idf_weights,
           built_at: state.built_at,
           episode_count: state.episode_count
         }}

      error ->
        error
    end
  end

  @doc "Preloads/builds vocabulary for a world if not already ready.\nCalled by WorldModelRegistry when activating a world.\n"
  def preload(world_id) when is_binary(world_id) do
    case get_state(world_id) do
      {:ok, %{ready: true}} ->
        {:ok, :already_ready}

      _ ->
        build_vocabulary(world_id)
    end
  end

  defp get_state(world_id) do
    try do
      case :ets.lookup(@ets_table, world_id) do
        [{^world_id, state}] -> {:ok, state}
        [] -> {:error, :not_found}
      end
    rescue
      ArgumentError -> {:error, :table_not_ready}
    end
  end

  defp set_state(world_id, state) do
    try do
      :ets.insert(@ets_table, {world_id, state})
      :ok
    rescue
      ArgumentError -> {:error, :table_not_ready}
    end
  end

  defp do_build_vocabulary(world_id) do
    Logger.info("Building embedder vocabulary for world: #{world_id}")

    set_state(world_id, %{
      ready: false,
      phase: :loading_episodes,
      vocabulary: %{},
      idf_weights: %{},
      built_at: nil,
      episode_count: 0
    })

    episodes = get_world_episodes(world_id)

    if Enum.empty?(episodes) do
      Logger.debug("No episodes found for world #{world_id}, embedder will use fallback")

      set_state(world_id, %{
        ready: false,
        phase: :no_data,
        vocabulary: %{},
        idf_weights: %{},
        built_at: nil,
        episode_count: 0
      })

      {:error, :no_data}
    else
      texts = Enum.map(episodes, & &1.state)
      episode_count = length(texts)

      set_state(world_id, %{
        ready: false,
        phase: :tokenizing,
        vocabulary: %{},
        idf_weights: %{},
        built_at: nil,
        episode_count: episode_count
      })

      tokenized = Enum.map(texts, &tokenize/1)
      all_tokens = List.flatten(tokenized)

      set_state(world_id, %{
        ready: false,
        phase: :building_frequencies,
        vocabulary: %{},
        idf_weights: %{},
        built_at: nil,
        episode_count: episode_count
      })

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

      set_state(world_id, %{
        ready: false,
        phase: :calculating_idf,
        vocabulary: vocabulary,
        idf_weights: %{},
        built_at: nil,
        episode_count: episode_count
      })

      text_sets = Enum.map(tokenized, &MapSet.new/1)
      num_docs = length(texts)

      idf_weights =
        vocabulary
        |> Enum.map(fn {word, _idx} ->
          doc_freq = Enum.count(text_sets, fn set -> MapSet.member?(set, word) end)
          idf = :math.log(num_docs / max(doc_freq, 1))
          {word, idf}
        end)
        |> Enum.into(%{})

      vocab_size = map_size(vocabulary)

      set_state(world_id, %{
        ready: true,
        phase: :ready,
        vocabulary: vocabulary,
        idf_weights: idf_weights,
        built_at: DateTime.utc_now(),
        episode_count: episode_count
      })

      Logger.info(
        "World embedder built for #{world_id}: #{vocab_size} terms from #{episode_count} episodes"
      )

      {:ok, vocab_size}
    end
  end

  defp get_world_episodes(world_id) do
    case Store.all_episodes(world_id: world_id) do
      {:ok, episodes} -> episodes
      _ -> []
    end
  end

  defp tokenize(text) do
    Tokenizer.tokenize_normalized(text, min_length: 2)
  end

  defp vectorize(text, vocabulary, idf_weights) do
    tokens = tokenize(text)
    token_freq = Enum.frequencies(tokens)

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

  defp normalize_vector(vector) do
    magnitude = :math.sqrt(Enum.reduce(vector, 0, fn val, acc -> acc + val * val end))

    if magnitude > 0 do
      Enum.map(vector, &(&1 / magnitude))
    else
      vector
    end
  end

  defp phase_label(:loading_episodes) do
    "Loading episodes"
  end

  defp phase_label(:tokenizing) do
    "Tokenizing texts"
  end

  defp phase_label(:building_frequencies) do
    "Building frequencies"
  end

  defp phase_label(:calculating_idf) do
    "Calculating IDF weights"
  end

  defp phase_label(:ready) do
    "Ready"
  end

  defp phase_label(:no_data) do
    "No training data"
  end

  defp phase_label(:not_initialized) do
    "Not initialized"
  end

  defp phase_label(_) do
    "Unknown"
  end
end