defmodule World.Embedder do
  @moduledoc """
  Each world's TF-IDF embedder: its vocabulary and IDF weights.

  The embedder's vocabulary is a world's vocabulary statistics, so it belongs to
  a world. A world holds its own copy at `<world>/models/embedder.term`
  (`path/1`), seeded by copying the trained vocabulary (`trained_path/0`, where
  `mix train_models` writes it):

  - a persistent world is seeded when `World.Manager` creates it, and an existing
    world, or every world after a retrain, with `mix world.seed_model embedder`;
  - an ephemeral world has no directory, so it is seeded in memory when it is
    created (`seed_in_memory!/1`) and cleared with it (`clear/1`).

  `loaded!/1` reads a world's copy the first time the world embeds and keeps it
  in ETS. A world with no copy raises naming the world: there is no fallback to
  the trained vocabulary or to another world.

  `build_vocabulary/2` rebuilds a world's vocabulary from the episodes stored in
  it, replacing the loaded one. A world with no episodes keeps the vocabulary it
  has (its seeded copy, or one built earlier). While the vocabulary is rebuilt
  the world cannot embed, so `Brain.Memory.Store` stores an episode written in
  that time without an embedding; when the world's vocabulary is ready again,
  the rebuild forgets every cache of vectors embedded with the world's
  vocabulary (`Brain.EmbeddingCaches.forget_world/1`), each built again on its
  next use, and tells the store (`Brain.Memory.Store.vocabulary_rebuilt/1`),
  which re-embeds and re-indexes every episode and semantic fact of the world
  with it, those episodes included.

  A caller that needs an embedding while the world's vocabulary is being
  rebuilt waits for the rebuild to finish (`await_ready/2`). It is woken when
  the whole rebuild has finished (the vocabulary built, the caches forgotten,
  and the world's memory re-embedded), and raises when the rebuild fails or
  does not finish in time. A rebuild that raises leaves the world's embedder
  in the `:rebuild_failed` phase, which `embed/2` returns as
  `{:error, :rebuild_failed}` and `await_ready/2` raises on, until the world is
  rebuilt or reseeded.

  A process that a rebuild calls (`Brain.Memory.Store` and the caches
  `Brain.EmbeddingCaches` forgets) must not wait inside its own callbacks: the
  rebuild would wait on it in turn. Such a process returns
  `{:error, :vocabulary_building}` to its caller, which waits and calls again.
  """

  alias Brain.EmbeddingCaches
  alias Brain.Memory.Store
  require Logger

  alias Brain.ML.{Tokenizer, WorldModelFile}

  @ets_table :world_embedders
  # The callers waiting for a world's rebuild to finish: `{world_id, pid, ref}`.
  @waiters_table :world_embedder_waiters
  @default_vocab_size 2000
  @min_word_frequency 2
  @file_name "embedder.term"
  # How long `await_ready/1` waits for a rebuild to finish.
  @rebuild_wait_ms 60_000

  @doc """
  Ensures the ETS tables exist: the embedders, and the callers waiting for a
  rebuild. `World.Manager` calls it when it starts, so the tables live as long
  as the worlds whose embedders they hold.
  """
  def init do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
    end

    if :ets.whereis(@waiters_table) == :undefined do
      :ets.new(@waiters_table, [:bag, :public, :named_table])
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

  @doc """
  Embeds text using the world's vocabulary, reading the world's copy
  (`loaded!/1`) the first time the world embeds. Raises when the world has no
  copy.

  Returns {:ok, embedding} or {:error, reason}
  """
  def embed(world_id, text) when is_binary(world_id) and is_binary(text) do
    case get_state(world_id) do
      {:ok, %{ready: true} = state} ->
        embedding = vectorize(text, state.vocabulary, state.idf_weights)
        {:ok, embedding}

      {:ok, %{phase: :no_data}} ->
        {:error, :no_training_data}

      {:ok, %{phase: :rebuild_failed}} ->
        {:error, :rebuild_failed}

      {:ok, _state} ->
        {:error, :vocabulary_building}

      {:error, :not_found} ->
        loaded!(world_id)
        embed(world_id, text)

      {:error, :table_not_ready} ->
        {:error, :table_not_ready}

      error ->
        error
    end
  end

  @doc """
  Builds or rebuilds the vocabulary for a world from its episodes.

  Returns `{:ok, vocabulary_size}` after a rebuild, `{:ok, :already_built}`
  when the world's vocabulary is ready and `force: true` was not given, and
  `{:ok, :kept}` when the world has no episodes to rebuild from and keeps the
  vocabulary it had. A world with no episodes and no vocabulary returns
  `{:error, :no_data}`; one whose vocabulary was never loaded reads its seeded
  copy, raising when it has none (`loaded!/1`).

  When the vocabulary is ready again, the world's embedding caches are
  forgotten (`Brain.EmbeddingCaches.forget_world/1`), and every episode and
  semantic fact of the world, the episodes stored while it was being rebuilt
  included, is re-embedded with it and re-indexed
  (`Brain.Memory.Store.vocabulary_rebuilt/1`) before this returns; a failure
  there raises.

  The callers waiting for the rebuild (`await_ready/2`) are woken when it
  finishes. A rebuild that raises leaves the world in the `:rebuild_failed`
  phase, wakes its waiters with the failure, and raises.
  """
  def build_vocabulary(world_id, opts \\ []) do
    force = Keyword.get(opts, :force, false)

    case get_state(world_id) do
      {:ok, %{ready: true}} when not force ->
        {:ok, :already_built}

      _ ->
        rebuild!(world_id)
    end
  end

  # A rebuild that raises, or exits (a call it makes times out), is failed:
  # the world is left in the `:rebuild_failed` phase, its waiters are woken
  # with the failure, and the failure goes on to the caller unchanged.
  defp rebuild!(world_id) do
    do_build_vocabulary(world_id)
  catch
    kind, reason ->
      stacktrace = __STACKTRACE__

      case get_state(world_id) do
        {:ok, state} -> set_state(world_id, %{state | ready: false, phase: :rebuild_failed})
        _ -> :ok
      end

      notify_waiters(world_id, {:failed, Exception.format_banner(kind, reason, stacktrace)})
      :erlang.raise(kind, reason, stacktrace)
  end

  @doc """
  Blocks the caller until `world_id`'s embedder can embed: at once when it can,
  or, while its vocabulary is being rebuilt, until the rebuild has finished
  (the vocabulary built, the world's embedding caches forgotten, and its
  memory re-embedded). A world whose vocabulary was never loaded reads its
  seeded copy (`loaded!/1`).

  Returns `:ok`. Raises naming the world when the rebuild does not finish
  within `timeout` milliseconds, when it fails or ends with no vocabulary, when
  the world has no vocabulary and none is being built, and when the world has
  no copy to read.

  A process that a rebuild calls must not wait in its own callbacks (see the
  moduledoc): it returns `{:error, :vocabulary_building}` to its caller, which
  waits here.
  """
  @spec await_ready(String.t(), non_neg_integer()) :: :ok
  def await_ready(world_id, timeout \\ @rebuild_wait_ms)
      when is_binary(world_id) and is_integer(timeout) and timeout >= 0 do
    ref = make_ref()
    waiter = {world_id, self(), ref}

    # Registered before the state is read, so a rebuild that finishes between
    # the two still wakes this caller.
    add_waiter!(waiter)

    try do
      case get_state(world_id) do
        {:ok, %{ready: true, phase: :ready}} ->
          :ok

        {:ok, %{phase: :no_data}} ->
          raise "World.Embedder: world #{inspect(world_id)} has no vocabulary: it had no episodes to " <>
                  "build one from and none to keep"

        {:ok, %{phase: :rebuild_failed}} ->
          raise "World.Embedder: world #{inspect(world_id)}'s last vocabulary rebuild failed; rebuild " <>
                  "or reseed it before embedding in it"

        {:ok, %{phase: phase}} ->
          wait_for_rebuild(world_id, ref, timeout, phase)

        {:error, :not_found} ->
          loaded!(world_id)
          :ok

        {:error, :table_not_ready} ->
          raise table_not_ready_message(world_id)
      end
    after
      :ets.delete_object(@waiters_table, waiter)

      # A wake-up sent after this caller stopped waiting.
      receive do
        {__MODULE__, ^ref, _outcome} -> :ok
      after
        0 -> :ok
      end
    end
  end

  defp wait_for_rebuild(world_id, ref, timeout, phase) do
    receive do
      {__MODULE__, ^ref, :ready} ->
        :ok

      {__MODULE__, ^ref, {:failed, message}} ->
        raise "World.Embedder: world #{inspect(world_id)}'s vocabulary rebuild failed: #{message}"
    after
      timeout ->
        raise "World.Embedder: world #{inspect(world_id)}'s vocabulary rebuild did not finish within " <>
                "#{timeout} ms (it was #{phase_label(phase)})"
    end
  end

  defp add_waiter!(waiter) do
    if :ets.whereis(@waiters_table) == :undefined do
      raise table_not_ready_message(elem(waiter, 0))
    end

    :ets.insert(@waiters_table, waiter)
  end

  # Wakes every caller waiting for the world's rebuild with its outcome.
  defp notify_waiters(world_id, outcome) do
    for {^world_id, pid, ref} <- :ets.take(@waiters_table, world_id) do
      send(pid, {__MODULE__, ref, outcome})
    end

    :ok
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

  @doc "Clears the embedder loaded for a world, for a world that is destroyed or reseeded.\n"
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

  @doc "Where a world keeps its copy of the embedder vocabulary."
  @spec path(String.t()) :: Path.t()
  def path(world_id) when is_binary(world_id), do: WorldModelFile.path(world_id, @file_name)

  @doc "Where training writes the embedder vocabulary that worlds are seeded from."
  @spec trained_path() :: Path.t()
  def trained_path, do: WorldModelFile.trained_path(@file_name)

  @doc """
  The world's embedder vocabulary: the one loaded for it, or its copy read from
  disk the first time it is asked for. Raises when the world has no copy, or
  when its vocabulary is being rebuilt from its episodes.
  """
  @spec loaded!(String.t()) :: map()
  def loaded!(world_id) when is_binary(world_id) do
    case get_state(world_id) do
      {:ok, %{ready: true} = state} ->
        Map.take(state, [:vocabulary, :idf_weights, :built_at, :episode_count])

      {:ok, %{phase: phase}} ->
        raise "World.Embedder: world #{inspect(world_id)}'s embedder is not ready (#{phase_label(phase)})"

      {:error, :not_found} ->
        model = read!(path(world_id), world_id)
        load!(world_id, model)
        model

      {:error, :table_not_ready} ->
        raise table_not_ready_message(world_id)
    end
  end

  @doc """
  Copies the trained vocabulary into the world's models directory, checks the
  copy byte for byte, and clears the vocabulary loaded for the world so its
  next embedding reads the new copy. Replaces any copy the world held, which is
  how a retrained vocabulary reaches a world.
  """
  @spec seed!(String.t()) :: :ok
  def seed!(world_id) when is_binary(world_id) do
    read!(trained_path(), :trained)
    :ok = WorldModelFile.copy!(world_id, @file_name)
    clear(world_id)
  end

  @doc "Makes a copy of the trained vocabulary the loaded vocabulary of a world that has no directory."
  @spec seed_in_memory!(String.t()) :: :ok
  def seed_in_memory!(world_id) when is_binary(world_id) do
    load!(world_id, read!(trained_path(), :trained))
  end

  @doc """
  Reads and checks an embedder vocabulary file. Raises when it is missing, or
  does not hold a vocabulary and IDF weights.
  """
  @spec read!(Path.t(), String.t() | :trained) :: map()
  def read!(file, owner) do
    Brain.Telemetry.measure_model_load("world_embedder", fn -> read_checked!(file, owner) end)
  end

  defp read_checked!(file, owner) do
    unless File.regular?(file) do
      raise missing_message(file, owner)
    end

    case file |> File.read!() |> :erlang.binary_to_term() do
      %{vocabulary: vocabulary, idf_weights: idf_weights} = model
      when is_map(vocabulary) and is_map(idf_weights) ->
        model

      other ->
        raise "World.Embedder: #{file} does not hold an embedder vocabulary, got #{inspect(other, limit: 3)}"
    end
  end

  defp load!(world_id, model) do
    case load_model(world_id, model) do
      :ok -> :ok
      {:error, :table_not_ready} -> raise table_not_ready_message(world_id)
    end
  end

  defp missing_message(file, :trained),
    do: "World.Embedder: no trained embedder vocabulary at #{file}; train one with `mix train_models`"

  defp missing_message(file, world_id),
    do:
      "World.Embedder: world #{inspect(world_id)} has no embedder vocabulary at #{file}; " <>
        "seed it with `mix world.seed_model embedder --world #{world_id}`"

  defp table_not_ready_message(world_id),
    do:
      "World.Embedder: the #{inspect(@ets_table)} table does not exist, so world #{inspect(world_id)}'s " <>
        "embedder cannot be held; World.Manager creates it when it starts"

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

    previous = get_state(world_id)

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
      keep_vocabulary(world_id, previous)
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
        phase: :reembedding,
        vocabulary: vocabulary,
        idf_weights: idf_weights,
        built_at: DateTime.utc_now(),
        episode_count: episode_count
      })

      Logger.info(
        "World embedder built for #{world_id}: #{vocab_size} terms from #{episode_count} episodes"
      )

      vocabulary_ready!(world_id)
      {:ok, vocab_size}
    end
  end

  # Nothing to rebuild from: the world keeps the vocabulary it had. One that
  # was loaded is put back; one never loaded is read from the world's seeded
  # copy (raising when it has none). Only a world that had no vocabulary at all
  # is left without one.
  defp keep_vocabulary(world_id, {:ok, %{ready: true} = previous_state}) do
    set_state(world_id, %{previous_state | phase: :reembedding})
    Logger.info("No episodes in world #{world_id} to rebuild its embedder from; kept its vocabulary")
    vocabulary_ready!(world_id)
    {:ok, :kept}
  end

  defp keep_vocabulary(world_id, {:error, :not_found}) do
    clear(world_id)
    loaded!(world_id)
    {:ok, seeded_state} = get_state(world_id)
    set_state(world_id, %{seeded_state | phase: :reembedding})
    Logger.info("No episodes in world #{world_id} to rebuild its embedder from; kept its seeded vocabulary")
    vocabulary_ready!(world_id)
    {:ok, :kept}
  end

  defp keep_vocabulary(world_id, _previous) do
    Logger.info("No episodes in world #{world_id} and no vocabulary to keep; its embedder has none")

    set_state(world_id, %{
      ready: false,
      phase: :no_data,
      vocabulary: %{},
      idf_weights: %{},
      built_at: nil,
      episode_count: 0
    })

    notify_waiters(world_id, {:failed, "it had no episodes to build a vocabulary from and none to keep"})
    {:error, :no_data}
  end

  # The world's vocabulary is ready again after a rebuild. Every cache of
  # vectors embedded with the world's vocabulary is forgotten first, so none
  # is compared with what the vocabulary now embeds while the episodes and
  # semantic facts are re-embedded; then they are re-embedded. Only then is
  # the rebuild finished, and its waiters woken.
  defp vocabulary_ready!(world_id) do
    :ok = EmbeddingCaches.forget_world(world_id)
    reembed_world!(world_id)
    {:ok, state} = get_state(world_id)
    set_state(world_id, %{state | phase: :ready})
    notify_waiters(world_id, :ready)
  end

  defp reembed_world!(world_id) do
    case Store.vocabulary_rebuilt(world_id) do
      {:ok, _reembedded} ->
        :ok

      {:error, reason} ->
        raise "World.Embedder: world #{inspect(world_id)}'s vocabulary is ready, but its episodes " <>
                "and semantic facts were not all re-embedded with it: #{inspect(reason)}"
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

  defp phase_label(:reembedding) do
    "Re-embedding the world's memory"
  end

  defp phase_label(:ready) do
    "Ready"
  end

  defp phase_label(:rebuild_failed) do
    "Rebuild failed"
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