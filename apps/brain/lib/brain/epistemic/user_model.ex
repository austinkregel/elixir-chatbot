defmodule Brain.Epistemic.UserModelStore do
  @moduledoc "GenServer managing per-user knowledge models.\n\nThe UserModelStore maintains explicit, inspectable models of what\nthe system knows about each user. This enables:\n\n- Self-referential responses (\"From what I remember, you...\")\n- Epistemic reasoning (knowing what we know vs. don't know)\n- Appropriate disclosure (sharing facts with proper hedging)\n\nEach UserModel tracks:\n- Facts with confidence and provenance\n- Interaction patterns\n- Disclosure history\n\n## Whose model, and where it is kept\n\nA user model belongs to a web user (`Atlas.Schemas.WebUser`) and is keyed by\nthat user's id, the `user_id:` the web interface passes to `Brain.evaluate/3`.\nIt has no world: the same model is used in every world the user talks in.\nOnly `sync_to_beliefs/2` and `sync_from_beliefs/2` take a world, because they\nmove facts into or out of that world's beliefs.\n\nAtlas (`atlas_user_models`) holds every model. The store loads them all when\nit starts, and every change is written to Atlas before it is kept here, so\nnothing is held only in this process. A change that cannot be written raises\nin the caller, naming the user and the reason (a user id that is no web\nuser's, or Atlas being unreachable); a store that cannot load does not start.\n"

  alias Brain.Epistemic.Types
  use GenServer

  alias Types.{UserModel, Belief, Config}
  alias Brain.Epistemic.BeliefStore

  require Logger


  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Gets or creates a UserModel for the given web user's id. A created model is written to Atlas.\n"
  def get_or_create(user_id) do
    call!({:get_or_create, user_id!(user_id)})
  end

  @doc "Gets a UserModel by user ID, returns nil if not found.\n"
  def get(user_id) do
    GenServer.call(__MODULE__, {:get, user_id!(user_id)})
  end

  @doc "Updates a fact in the user's model.\n\nParameters:\n- user_id: The web user's id\n- key: Fact key (atom or string)\n- value: The fact value\n- source: How the fact was learned (:explicit | :inferred | :assumed | :learned)\n- confidence: Confidence level (0.0 - 1.0)\n"
  def update_fact(user_id, key, value, source, confidence) do
    if Config.enabled?() do
      call!({:update_fact, user_id!(user_id), key, value, source, confidence})
    else
      :ok
    end
  end

  @doc "Gets a specific fact with its confidence and provenance.\n"
  def get_fact(user_id, key) do
    GenServer.call(__MODULE__, {:get_fact, user_id!(user_id), key})
  end

  @doc "Gets all facts above a confidence threshold.\n"
  def get_facts_with_confidence(user_id, min_confidence \\ 0.0) do
    GenServer.call(__MODULE__, {:get_facts_with_confidence, user_id!(user_id), min_confidence})
  end

  @doc "Gets the epistemic bounds (confidence levels) for all facts.\n"
  def get_epistemic_bounds(user_id) do
    GenServer.call(__MODULE__, {:get_epistemic_bounds, user_id!(user_id)})
  end

  @doc "Records an interaction pattern.\n"
  def record_interaction_pattern(user_id, pattern_type, data) do
    call!({:record_pattern, user_id!(user_id), pattern_type, data})
  end

  @doc "Records that facts were disclosed to the user.\n"
  def record_disclosure(user_id, disclosed_keys, context) do
    call!({:record_disclosure, user_id!(user_id), disclosed_keys, context})
  end

  @doc "Gets disclosure history for a user.\n"
  def get_disclosure_history(user_id, limit \\ 20) do
    GenServer.call(__MODULE__, {:get_disclosure_history, user_id!(user_id), limit})
  end

  @doc "Extracts beliefs from the user model and syncs them to the world's BeliefStore.\n"
  def sync_to_beliefs(world_id, user_id) do
    GenServer.call(__MODULE__, {:sync_to_beliefs, belief_world_id!(world_id), user_id!(user_id)})
  end

  @doc "Updates the user model from the beliefs held in the world's BeliefStore.\n"
  def sync_from_beliefs(world_id, user_id) do
    call!({:sync_from_beliefs, belief_world_id!(world_id), user_id!(user_id)})
  end

  # Beliefs are held per world; checked in the caller's process.
  defp belief_world_id!(world_id) when is_binary(world_id) and world_id != "", do: world_id

  defp belief_world_id!(other) do
    raise ArgumentError, "UserModel needs the world whose beliefs it syncs with, got: #{inspect(other)}"
  end

  # A user model is a web user's. A turn or caller with no user has no model
  # to read or write; it raises here, in the caller's process.
  defp user_id!(user_id) when is_binary(user_id) and user_id != "", do: user_id

  defp user_id!(other) do
    raise ArgumentError, "UserModelStore needs the web user's id, got: #{inspect(other)}"
  end

  # A change that was not written to Atlas is not kept, and raises here.
  defp call!(request) do
    case GenServer.call(__MODULE__, request) do
      {:error, {:not_persisted, user_id, reason}} ->
        raise "UserModelStore: the user model of #{inspect(user_id)} was not written to Atlas: " <>
                describe_error(reason)

      reply ->
        reply
    end
  end

  defp describe_error(%Ecto.Changeset{} = changeset) do
    changeset
    |> Ecto.Changeset.traverse_errors(fn {message, opts} ->
      Enum.reduce(opts, message, fn {key, value}, acc -> String.replace(acc, "%{#{key}}", to_string(value)) end)
    end)
    |> inspect()
  end

  defp describe_error(reason), do: inspect(reason)

  @doc "Gets store statistics.\n"
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc "Lists all user IDs with stored models.\n"
  def list_all_users do
    GenServer.call(__MODULE__, :list_all_users)
  end

  @doc "Clears all user models held in this process (for testing). Atlas is not touched.\n"
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
    case Brain.AtlasIntegration.load_user_models() do
      {:ok, models} ->
        Logger.info("UserModelStore initialized", user_count: map_size(models))
        {:ok, %{models: models}}

      {:error, reason} ->
        {:stop, {:user_models_not_loaded, reason}}
    end
  end

  @impl true
  def handle_call({:get_or_create, user_id}, _from, state) do
    case Map.get(state.models, user_id) do
      nil -> write_through(state, UserModel.new(user_id), {:ok, :model})
      model -> {:reply, {:ok, model}, state}
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

    Logger.debug("User fact updated",
      user_id: user_id,
      key: key,
      source: source,
      confidence: confidence
    )

    write_through(state, updated, {:ok, :model})
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
    write_through(state, UserModel.record_pattern(model, pattern_type, data), :ok)
  end

  @impl true
  def handle_call({:record_disclosure, user_id, disclosed_keys, context}, _from, state) do
    model = Map.get(state.models, user_id) || UserModel.new(user_id)
    write_through(state, UserModel.record_disclosure(model, disclosed_keys, context), :ok)
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
  def handle_call({:sync_to_beliefs, world_id, user_id}, _from, state) do
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

            BeliefStore.add_belief(world_id, belief)
            1
          end)
          |> Enum.sum()

        {:reply, {:ok, created}, state}
    end
  end

  @impl true
  def handle_call({:sync_from_beliefs, world_id, user_id}, _from, state) do
    model = Map.get(state.models, user_id) || UserModel.new(user_id)
    {:ok, beliefs} = BeliefStore.get_beliefs_for_user(world_id, user_id)

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

    write_through(state, updated, {:ok, length(beliefs)})
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
  def handle_call(:clear_all, _from, state) do
    {:reply, :ok, %{state | models: %{}}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # Writes the model to Atlas, then keeps it. `reply` is the caller's answer on
  # success, with `:model` standing for the model kept. A model that was not
  # written is not kept.
  defp write_through(state, %UserModel{user_id: user_id} = model, reply) do
    case Brain.AtlasIntegration.persist_user_model(model) do
      {:ok, _row} ->
        reply = if reply == {:ok, :model}, do: {:ok, model}, else: reply
        {:reply, reply, %{state | models: Map.put(state.models, user_id, model)}}

      {:error, reason} ->
        {:reply, {:error, {:not_persisted, user_id, reason}}, state}
    end
  end
end
