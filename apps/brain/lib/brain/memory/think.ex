defmodule Brain.Memory.Think do
  @moduledoc """
  High-level API for the cognitive memory system.

  Ported from the Rust cognitive_memory_system think module.

  This module exposes a `think` function that orchestrates memory
  insertion, retrieval, and consolidation. It wraps lower-level
  components in a convenient interface.
  """

  alias Brain.Memory.{Store, Embedder, Consolidation}
  alias Brain.Memory.Types.Episode

  require Logger

  @type think_input ::
          {:add_episode, map()}
          | {:query_chat, map()}
          | {:query_semantic, map()}
          | {:consolidate, map()}

  @type think_output ::
          {:episode_added, String.t()}
          | {:chat_results, [{Episode.t(), float()}]}
          | {:semantic_results, [{SemanticFact.t(), float()}]}
          | {:consolidated, non_neg_integer()}

  @doc """
  High-level entry point for interacting with the memory system.

  ## Examples

      # Add a new episode
      Think.think(:add_episode, %{
        state: "Hello there!",
        action: "greeting",
        outcome: "responded with hello",
        tags: ["greeting", "smalltalk"]
      })

      # Query for similar episodes
      Think.think(:query_chat, %{input: "Hi!", k: 5})

      # Consolidate similar episodes into semantic facts
      Think.think(:consolidate, %{threshold: 0.8, min_size: 2})
  """
  def think(operation, params \\ %{})

  def think(:add_episode, params) do
    # Check if Store is available before attempting to add
    if Process.whereis(Store) do
      state = Map.get(params, :state, "")
      action = Map.get(params, :action, "")
      outcome = Map.get(params, :outcome, "")
      tags = Map.get(params, :tags, [])
      world_id = Map.get(params, :world_id, "default")

      # Pass world_id option to the store
      opts = [world_id: world_id]

      case Store.add_episode(state, action, outcome, tags, opts) do
        {:ok, id} ->
          Logger.debug("Episode added", id: id, world_id: world_id)
          {:ok, {:episode_added, id}}

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:error, :store_not_available}
    end
  end

  def think(:query_chat, params) do
    input = Map.get(params, :input, "")
    k = Map.get(params, :k, 5)
    world_id = Map.get(params, :world_id, "default")

    opts = [world_id: world_id]

    case Store.query_similar(input, k, opts) do
      {:ok, results} ->
        formatted =
          Enum.map(results, fn {episode, similarity} ->
            {episode.id, similarity}
          end)

        {:ok, {:chat_results, formatted}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def think(:query_semantic, params) do
    input = Map.get(params, :input, "")
    k = Map.get(params, :k, 5)
    world_id = Map.get(params, :world_id, "default")

    opts = [world_id: world_id]

    case Store.query_semantic(input, k, opts) do
      {:ok, results} ->
        formatted =
          Enum.map(results, fn {semantic, similarity} ->
            {semantic.id, similarity}
          end)

        {:ok, {:semantic_results, formatted}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def think(:consolidate, params) do
    threshold = Map.get(params, :threshold, 0.8)
    min_size = Map.get(params, :min_size, 2)
    world_id = Map.get(params, :world_id, "default")

    {:ok, count} =
      Consolidation.consolidate(
        threshold: threshold,
        min_cluster_size: min_size,
        world_id: world_id
      )

    {:ok, {:consolidated, count}}
  end

  def think(:stats, _params) do
    stats = Store.stats()
    {:ok, {:stats, stats}}
  end

  def think(:persist, _params) do
    case Store.persist() do
      :ok -> {:ok, :persisted}
      error -> error
    end
  end

  def think(:clear, _params) do
    Store.clear()
    {:ok, :cleared}
  end

  @doc """
  Initialize the memory system by starting required processes.
  Should be called at application startup.
  """
  def init do
    Logger.info("Initializing cognitive memory system...")

    # Ensure Embedder is started
    case Process.whereis(Embedder) do
      nil ->
        case Embedder.start_link() do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
          error -> error
        end

      _pid ->
        :ok
    end

    # Ensure Store is started
    case Process.whereis(Store) do
      nil ->
        case Store.start_link() do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
          error -> error
        end

      _pid ->
        :ok
    end

    Logger.info("Cognitive memory system initialized")
    :ok
  end

  @doc """
  Load training data into the memory system as episodes.
  This populates the episodic memory from intent training data.
  """
  def load_training_data do
    Logger.info("Loading training data into memory system...")

    case Brain.ML.DataLoaders.load_all_intents() do
      {:ok, examples} ->
        # First build vocabulary from all texts
        texts = Enum.map(examples, & &1.text)
        Embedder.build_vocabulary(texts)

        # Then add each example as an episode
        count =
          Enum.reduce(examples, 0, fn example, acc ->
            case think(:add_episode, %{
                   state: example.text,
                   action: example.intent,
                   outcome: "",
                   tags: [example.intent | extract_entity_types(example.entities)]
                 }) do
              {:ok, _} -> acc + 1
              {:error, _} -> acc
            end
          end)

        Logger.info("Loaded training data as episodes", count: count)
        {:ok, count}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp extract_entity_types(entities) when is_list(entities) do
    entities
    |> Enum.map(fn e -> Map.get(e, :type) || Map.get(e, "type") end)
    |> Enum.filter(&is_binary/1)
    |> Enum.uniq()
  end

  defp extract_entity_types(_), do: []
end
