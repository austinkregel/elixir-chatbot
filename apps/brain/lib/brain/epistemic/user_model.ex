defmodule Brain.Epistemic.UserModelStore do
  @moduledoc "GenServer managing per-user knowledge models.\n\nThe UserModelStore maintains explicit, inspectable models of what\nthe system knows about each user. This enables:\n\n- Self-referential responses (\"From what I remember, you...\")\n- Epistemic reasoning (knowing what we know vs. don't know)\n- Appropriate disclosure (sharing facts with proper hedging)\n\nEach UserModel tracks:\n- Facts with confidence and provenance\n- Interaction patterns\n- Disclosure history\n"

  alias Brain.Epistemic.Types
  use GenServer

  alias Types.{UserModel, Belief, Config}
  alias Brain.Epistemic.BeliefStore

  require Logger


  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Gets or creates a UserModel for the given user ID.\n"
  def get_or_create(user_id) do
    GenServer.call(__MODULE__, {:get_or_create, user_id})
  end

  @doc "Gets a UserModel by user ID, returns nil if not found.\n"
  def get(user_id) do
    GenServer.call(__MODULE__, {:get, user_id})
  end

  @doc "Updates a fact in the user's model.\n\nParameters:\n- user_id: The user identifier\n- key: Fact key (atom or string)\n- value: The fact value\n- source: How the fact was learned (:explicit | :inferred | :assumed | :learned)\n- confidence: Confidence level (0.0 - 1.0)\n"
  def update_fact(user_id, key, value, source, confidence) do
    if Config.enabled?() do
      GenServer.call(__MODULE__, {:update_fact, user_id, key, value, source, confidence})
    else
      :ok
    end
  end

  @doc "Gets a specific fact with its confidence and provenance.\n"
  def get_fact(user_id, key) do
    GenServer.call(__MODULE__, {:get_fact, user_id, key})
  end

  @doc "Gets all facts above a confidence threshold.\n"
  def get_facts_with_confidence(user_id, min_confidence \\ 0.0) do
    GenServer.call(__MODULE__, {:get_facts_with_confidence, user_id, min_confidence})
  end

  @doc "Gets the epistemic bounds (confidence levels) for all facts.\n"
  def get_epistemic_bounds(user_id) do
    GenServer.call(__MODULE__, {:get_epistemic_bounds, user_id})
  end

  @doc "Records an interaction pattern.\n"
  def record_interaction_pattern(user_id, pattern_type, data) do
    GenServer.call(__MODULE__, {:record_pattern, user_id, pattern_type, data})
  end

  @doc "Records that facts were disclosed to the user.\n"
  def record_disclosure(user_id, disclosed_keys, context) do
    GenServer.call(__MODULE__, {:record_disclosure, user_id, disclosed_keys, context})
  end

  @doc "Gets disclosure history for a user.\n"
  def get_disclosure_history(user_id, limit \\ 20) do
    GenServer.call(__MODULE__, {:get_disclosure_history, user_id, limit})
  end

  @doc "Extracts beliefs from the user model and syncs to BeliefStore.\n"
  def sync_to_beliefs(user_id) do
    GenServer.call(__MODULE__, {:sync_to_beliefs, user_id})
  end

  @doc "Updates the user model from beliefs in the BeliefStore.\n"
  def sync_from_beliefs(user_id) do
    GenServer.call(__MODULE__, {:sync_from_beliefs, user_id})
  end

  @doc "Gets store statistics.\n"
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc "Lists all user IDs with stored models.\n"
  def list_all_users do
    GenServer.call(__MODULE__, :list_all_users)
  end

  @doc "Persists all user models to disk.\n"
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc "Clears a specific user's model.\n"
  def clear_user(user_id) do
    GenServer.call(__MODULE__, {:clear_user, user_id})
  end

  @doc "Clears all user models (for testing).\n"
  def clear_all do
    GenServer.call(__MODULE__, :clear_all)
  end

  @doc "Checks if the store is ready.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @impl true
  def init(_opts) do
    state = %{models: %{}}
    state = load_from_atlas(state)

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

    Brain.AtlasIntegration.persist_user_model(updated)

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

    Brain.AtlasIntegration.persist_user_model(updated)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:record_disclosure, user_id, disclosed_keys, context}, _from, state) do
    model = Map.get(state.models, user_id) || UserModel.new(user_id)
    updated = UserModel.record_disclosure(model, disclosed_keys, context)
    new_models = Map.put(state.models, user_id, updated)
    new_state = %{state | models: new_models}

    Brain.AtlasIntegration.persist_user_model(updated)

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
    {:reply, :ok, state}
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

  defp load_from_atlas(state) do
    case Brain.AtlasIntegration.load_user_models() do
      {:ok, models} when models != %{} ->
        Logger.info("Loaded UserModels from Atlas", user_count: map_size(models))
        %{state | models: models}

      _ ->
        Logger.debug("No user models in Atlas, starting with empty store")
        state
    end
  rescue
    e ->
      Logger.warning("Failed to load UserModels from Atlas: #{inspect(e)}")
      state
  end
end
