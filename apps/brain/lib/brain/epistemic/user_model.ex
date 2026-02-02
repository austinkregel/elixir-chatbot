defmodule Brain.Epistemic.UserModelStore do
  @moduledoc """
  GenServer managing per-user knowledge models.

  The UserModelStore maintains explicit, inspectable models of what
  the system knows about each user. This enables:

  - Self-referential responses ("From what I remember, you...")
  - Epistemic reasoning (knowing what we know vs. don't know)
  - Appropriate disclosure (sharing facts with proper hedging)

  Each UserModel tracks:
  - Facts with confidence and provenance
  - Interaction patterns
  - Disclosure history
  """

  use GenServer

  alias Brain.Epistemic.Types.{UserModel, Belief, Config}
  alias Brain.Epistemic.BeliefStore

  require Logger

  # Default persistence path resolved at runtime
  defp default_persistence_path, do: Brain.priv_path("data/user_models.term")

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Gets or creates a UserModel for the given user ID.
  """
  def get_or_create(user_id) do
    GenServer.call(__MODULE__, {:get_or_create, user_id})
  end

  @doc """
  Gets a UserModel by user ID, returns nil if not found.
  """
  def get(user_id) do
    GenServer.call(__MODULE__, {:get, user_id})
  end

  @doc """
  Updates a fact in the user's model.

  Parameters:
  - user_id: The user identifier
  - key: Fact key (atom or string)
  - value: The fact value
  - source: How the fact was learned (:explicit | :inferred | :assumed | :learned)
  - confidence: Confidence level (0.0 - 1.0)
  """
  def update_fact(user_id, key, value, source, confidence) do
    if Config.enabled?() do
      GenServer.call(__MODULE__, {:update_fact, user_id, key, value, source, confidence})
    else
      :ok
    end
  end

  @doc """
  Gets a specific fact with its confidence and provenance.
  """
  def get_fact(user_id, key) do
    GenServer.call(__MODULE__, {:get_fact, user_id, key})
  end

  @doc """
  Gets all facts above a confidence threshold.
  """
  def get_facts_with_confidence(user_id, min_confidence \\ 0.0) do
    GenServer.call(__MODULE__, {:get_facts_with_confidence, user_id, min_confidence})
  end

  @doc """
  Gets the epistemic bounds (confidence levels) for all facts.
  """
  def get_epistemic_bounds(user_id) do
    GenServer.call(__MODULE__, {:get_epistemic_bounds, user_id})
  end

  @doc """
  Records an interaction pattern.
  """
  def record_interaction_pattern(user_id, pattern_type, data) do
    GenServer.call(__MODULE__, {:record_pattern, user_id, pattern_type, data})
  end

  @doc """
  Records that facts were disclosed to the user.
  """
  def record_disclosure(user_id, disclosed_keys, context) do
    GenServer.call(__MODULE__, {:record_disclosure, user_id, disclosed_keys, context})
  end

  @doc """
  Gets disclosure history for a user.
  """
  def get_disclosure_history(user_id, limit \\ 20) do
    GenServer.call(__MODULE__, {:get_disclosure_history, user_id, limit})
  end

  @doc """
  Extracts beliefs from the user model and syncs to BeliefStore.
  """
  def sync_to_beliefs(user_id) do
    GenServer.call(__MODULE__, {:sync_to_beliefs, user_id})
  end

  @doc """
  Updates the user model from beliefs in the BeliefStore.
  """
  def sync_from_beliefs(user_id) do
    GenServer.call(__MODULE__, {:sync_from_beliefs, user_id})
  end

  @doc """
  Gets store statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Lists all user IDs with stored models.
  """
  def list_all_users do
    GenServer.call(__MODULE__, :list_all_users)
  end

  @doc """
  Persists all user models to disk.
  """
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc """
  Clears a specific user's model.
  """
  def clear_user(user_id) do
    GenServer.call(__MODULE__, {:clear_user, user_id})
  end

  @doc """
  Clears all user models (for testing).
  """
  def clear_all do
    GenServer.call(__MODULE__, :clear_all)
  end

  @doc """
  Checks if the store is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    persistence_path = Keyword.get(opts, :persistence_path, default_persistence_path())

    state = %{
      models: %{},
      persistence_path: persistence_path
    }

    # Try to load from disk
    state = maybe_load_from_disk(state)

    Logger.info("UserModelStore initialized", user_count: map_size(state.models))

    {:ok, state}
  end

  @impl true
  def handle_call({:get_or_create, user_id}, _from, state) do
    case Map.get(state.models, user_id) do
      nil ->
        model = UserModel.new(user_id)
        new_models = Map.put(state.models, user_id, model)
        new_state = %{state | models: new_models}
        {:reply, {:ok, model}, new_state}

      model ->
        {:reply, {:ok, model}, state}
    end
  end

  @impl true
  def handle_call({:get, user_id}, _from, state) do
    {:reply, Map.get(state.models, user_id), state}
  end

  @impl true
  def handle_call({:update_fact, user_id, key, value, source, confidence}, _from, state) do
    model = Map.get(state.models, user_id) || UserModel.new(user_id)
    updated = UserModel.update_fact(model, key, value, source, confidence)
    new_models = Map.put(state.models, user_id, updated)
    new_state = %{state | models: new_models}

    Logger.debug("User fact updated",
      user_id: user_id,
      key: key,
      source: source,
      confidence: confidence
    )

    {:reply, {:ok, updated}, new_state}
  end

  @impl true
  def handle_call({:get_fact, user_id, key}, _from, state) do
    case Map.get(state.models, user_id) do
      nil -> {:reply, nil, state}
      model -> {:reply, UserModel.get_fact(model, key), state}
    end
  end

  @impl true
  def handle_call({:get_facts_with_confidence, user_id, min_confidence}, _from, state) do
    case Map.get(state.models, user_id) do
      nil ->
        {:reply, [], state}

      model ->
        facts = UserModel.get_facts_above_confidence(model, min_confidence)
        {:reply, facts, state}
    end
  end

  @impl true
  def handle_call({:get_epistemic_bounds, user_id}, _from, state) do
    case Map.get(state.models, user_id) do
      nil -> {:reply, %{}, state}
      model -> {:reply, model.epistemic_bounds, state}
    end
  end

  @impl true
  def handle_call({:record_pattern, user_id, pattern_type, data}, _from, state) do
    model = Map.get(state.models, user_id) || UserModel.new(user_id)
    updated = UserModel.record_pattern(model, pattern_type, data)
    new_models = Map.put(state.models, user_id, updated)
    new_state = %{state | models: new_models}

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:record_disclosure, user_id, disclosed_keys, context}, _from, state) do
    model = Map.get(state.models, user_id) || UserModel.new(user_id)
    updated = UserModel.record_disclosure(model, disclosed_keys, context)
    new_models = Map.put(state.models, user_id, updated)
    new_state = %{state | models: new_models}

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:get_disclosure_history, user_id, limit}, _from, state) do
    case Map.get(state.models, user_id) do
      nil ->
        {:reply, [], state}

      model ->
        history = Enum.take(model.disclosure_history, limit)
        {:reply, history, state}
    end
  end

  @impl true
  def handle_call({:sync_to_beliefs, user_id}, _from, state) do
    case Map.get(state.models, user_id) do
      nil ->
        {:reply, {:ok, 0}, state}

      model ->
        # Create beliefs from user model facts
        created =
          model.facts
          |> Enum.map(fn {key, value} ->
            source = Map.get(model.provenance_map, key, :inferred)
            confidence = Map.get(model.epistemic_bounds, key, 0.5)

            belief =
              Belief.new(:user, key, value,
                source: source,
                confidence: confidence,
                user_id: user_id
              )

            BeliefStore.add_belief(belief)
            1
          end)
          |> Enum.sum()

        {:reply, {:ok, created}, state}
    end
  end

  @impl true
  def handle_call({:sync_from_beliefs, user_id}, _from, state) do
    model = Map.get(state.models, user_id) || UserModel.new(user_id)

    # Get all beliefs about this user
    case BeliefStore.get_beliefs_for_user(user_id) do
      {:ok, beliefs} ->
        updated =
          Enum.reduce(beliefs, model, fn belief, acc ->
            UserModel.update_fact(
              acc,
              belief.predicate,
              belief.object,
              belief.source,
              belief.confidence
            )
          end)

        new_models = Map.put(state.models, user_id, updated)
        new_state = %{state | models: new_models}

        {:reply, {:ok, length(beliefs)}, new_state}

      _ ->
        {:reply, {:ok, 0}, state}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    total_facts =
      state.models
      |> Map.values()
      |> Enum.map(fn m -> map_size(m.facts) end)
      |> Enum.sum()

    stats = %{
      total_users: map_size(state.models),
      total_facts: total_facts
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:list_all_users, _from, state) do
    user_ids = Map.keys(state.models)
    {:reply, {:ok, user_ids}, state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    result = persist_to_disk(state)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:clear_user, user_id}, _from, state) do
    new_models = Map.delete(state.models, user_id)
    {:reply, :ok, %{state | models: new_models}}
  end

  @impl true
  def handle_call(:clear_all, _from, state) do
    {:reply, :ok, %{state | models: %{}}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp maybe_load_from_disk(state) do
    path = state.persistence_path

    if File.exists?(path) do
      case File.read(path) do
        {:ok, binary} ->
          try do
            models = :erlang.binary_to_term(binary)

            Logger.info("Loaded UserModels from disk", user_count: map_size(models))

            %{state | models: models}
          rescue
            e ->
              Logger.warning("Failed to load UserModels: #{inspect(e)}")
              state
          end

        {:error, reason} ->
          Logger.warning("Could not read UserModels file: #{inspect(reason)}")
          state
      end
    else
      state
    end
  end

  defp persist_to_disk(state) do
    path = state.persistence_path

    # Ensure directory exists
    path |> Path.dirname() |> File.mkdir_p!()

    binary = :erlang.term_to_binary(state.models)

    case File.write(path, binary) do
      :ok ->
        Logger.info("UserModels persisted to disk", user_count: map_size(state.models))
        :ok

      {:error, reason} ->
        Logger.error("Failed to persist UserModels: #{inspect(reason)}")
        {:error, reason}
    end
  end
end
