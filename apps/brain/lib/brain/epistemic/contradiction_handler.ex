defmodule Brain.Epistemic.ContradictionHandler do
  @moduledoc """
  Handles contradictions detected by the JTMS.

  When the JTMS detects that a contradiction node has become IN,
  this module is responsible for:

  1. Identifying the minimal set of assumptions causing the contradiction
  2. Determining resolution strategies
  3. Presenting options or auto-resolving based on configuration
  4. Tracking resolution history for learning

  Resolution strategies:
  - Retract the least confident assumption
  - Retract the most recent assumption
  - Present options to the user
  - Apply domain-specific rules
  """

  use GenServer

  alias Brain.Epistemic.JTMS

  require Logger

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Handles a contradiction notification from the JTMS.

  Returns a resolution decision or {:needs_user_input, options}.
  """
  def handle_contradiction(node_id, supporting_assumptions) do
    GenServer.call(__MODULE__, {:handle_contradiction, node_id, supporting_assumptions})
  end

  @doc """
  Resolves a contradiction by retracting the specified assumption.
  """
  def resolve_by_retraction(assumption_id) do
    GenServer.call(__MODULE__, {:resolve_by_retraction, assumption_id})
  end

  @doc """
  Gets all pending contradictions awaiting user resolution.
  """
  def get_pending do
    GenServer.call(__MODULE__, :get_pending)
  end

  @doc """
  Gets resolution history.
  """
  def get_history(opts \\ []) do
    GenServer.call(__MODULE__, {:get_history, opts})
  end

  @doc """
  Sets the resolution strategy.

  Strategies:
  - :auto_least_confident - Automatically retract least confident assumption
  - :auto_most_recent - Automatically retract most recent assumption
  - :manual - Always require user input
  - :hybrid - Auto for low stakes, manual for high stakes
  """
  def set_strategy(strategy) do
    GenServer.call(__MODULE__, {:set_strategy, strategy})
  end

  @doc """
  Registers a domain-specific resolution rule.

  Rules are functions that take (node_id, assumptions) and return
  {:resolve, assumption_to_retract} or :no_match.
  """
  def register_rule(name, rule_fn) when is_function(rule_fn, 2) do
    GenServer.call(__MODULE__, {:register_rule, name, rule_fn})
  end

  @doc """
  Checks if the handler is ready.
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
  Gets statistics about the contradiction handler.
  """
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Register as JTMS contradiction handler
    if Process.whereis(JTMS) do
      JTMS.set_contradiction_handler(&handle_jtms_callback/1)
    end

    state = %{
      pending: %{},
      history: [],
      strategy: :hybrid,
      rules: %{},
      assumption_metadata: %{}
    }

    # Register built-in rules after state is created
    # We'll register the knowledge expansion rule via a message to self
    send(self(), :register_builtin_rules)

    Logger.info("ContradictionHandler initialized")

    {:ok, state}
  end

  @impl true
  def handle_info(:register_builtin_rules, state) do
    # Register the knowledge expansion rule
    new_rules =
      Map.put(state.rules, :knowledge_expansion, &handle_knowledge_expansion_conflict/2)

    {:noreply, %{state | rules: new_rules}}
  end

  @impl true
  def handle_info({:register_with_jtms}, state) do
    # Delayed registration in case JTMS starts after us
    if Process.whereis(JTMS) do
      JTMS.set_contradiction_handler(&handle_jtms_callback/1)
    end

    {:noreply, state}
  end

  @impl true
  def handle_call({:handle_contradiction, node_id, assumptions}, _from, state) do
    Logger.info("Handling contradiction",
      node_id: node_id,
      assumptions: assumptions
    )

    # Try domain-specific rules first
    case try_rules(state.rules, node_id, assumptions) do
      {:resolve, assumption_id} ->
        # Auto-resolve using the rule
        result = do_resolution(assumption_id, node_id, :rule)
        new_state = record_resolution(state, node_id, assumption_id, :rule)
        {:reply, {:resolved, result}, new_state}

      :no_match ->
        # Apply strategy
        case apply_strategy(state.strategy, assumptions, state.assumption_metadata) do
          {:auto_resolve, assumption_id, reason} ->
            result = do_resolution(assumption_id, node_id, reason)
            new_state = record_resolution(state, node_id, assumption_id, reason)
            {:reply, {:resolved, result}, new_state}

          :needs_user_input ->
            # Add to pending
            pending_entry = %{
              node_id: node_id,
              assumptions: assumptions,
              detected_at: DateTime.utc_now(),
              options: build_resolution_options(assumptions, state.assumption_metadata)
            }

            new_pending = Map.put(state.pending, node_id, pending_entry)
            new_state = %{state | pending: new_pending}

            {:reply, {:needs_user_input, pending_entry.options}, new_state}
        end
    end
  end

  @impl true
  def handle_call({:resolve_by_retraction, assumption_id}, _from, state) do
    # Find which pending contradiction this resolves
    {resolved_node, remaining} =
      Enum.split_with(state.pending, fn {_node_id, entry} ->
        assumption_id in entry.assumptions
      end)

    case resolved_node do
      [{node_id, _entry} | _] ->
        result = do_resolution(assumption_id, node_id, :user_choice)
        new_state = record_resolution(state, node_id, assumption_id, :user_choice)
        new_state = %{new_state | pending: Map.new(remaining)}
        {:reply, {:ok, result}, new_state}

      [] ->
        {:reply, {:error, :no_pending_contradiction}, state}
    end
  end

  @impl true
  def handle_call(:get_pending, _from, state) do
    {:reply, Map.values(state.pending), state}
  end

  @impl true
  def handle_call({:get_history, opts}, _from, state) do
    limit = Keyword.get(opts, :limit, 50)
    history = Enum.take(state.history, limit)
    {:reply, history, state}
  end

  @impl true
  def handle_call({:set_strategy, strategy}, _from, state)
      when strategy in [:auto_least_confident, :auto_most_recent, :manual, :hybrid] do
    {:reply, :ok, %{state | strategy: strategy}}
  end

  @impl true
  def handle_call({:register_rule, name, rule_fn}, _from, state) do
    new_rules = Map.put(state.rules, name, rule_fn)
    {:reply, :ok, %{state | rules: new_rules}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      pending_count: map_size(state.pending),
      resolution_history_count: length(state.history),
      strategy: state.strategy,
      rules_count: map_size(state.rules)
    }

    {:reply, stats, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp handle_jtms_callback({:contradiction, node_id, assumptions}) do
    # Called by JTMS when contradiction detected
    # We need to handle this async to avoid deadlock
    spawn(fn ->
      handle_contradiction(node_id, assumptions)
    end)
  end

  # Knowledge Expansion rule: when a new fact from knowledge expansion
  # conflicts with an existing belief, queue it for admin review instead
  # of auto-resolving.
  defp handle_knowledge_expansion_conflict(node_id, _assumptions) do
    # Check if this is a knowledge expansion conflict
    case get_conflict_context(node_id) do
      {:knowledge_expansion, new_fact, existing_belief} ->
        # Queue for admin review instead of auto-resolving
        if Process.whereis(Brain.Knowledge.ReviewQueue) do
          Brain.Knowledge.ReviewQueue.add_contradiction(new_fact, existing_belief)
          Logger.info("Knowledge expansion conflict queued for review",
            node_id: node_id,
            new_fact: inspect(new_fact)
          )
        end

        # Return :no_match so it goes to :needs_user_input
        :no_match

      _ ->
        :no_match
    end
  end

  # Check if a contradiction node is related to knowledge expansion
  defp get_conflict_context(node_id) do
    case JTMS.get_node(node_id) do
      {:ok, node} ->
        metadata = node.metadata || %{}

        if Map.get(metadata, :source) == :knowledge_expansion do
          new_fact = Map.get(metadata, :new_fact, %{})
          existing_belief = Map.get(metadata, :existing_belief, %{})
          {:knowledge_expansion, new_fact, existing_belief}
        else
          :not_knowledge_expansion
        end

      _ ->
        :not_knowledge_expansion
    end
  end

  defp try_rules(rules, node_id, assumptions) do
    rules
    |> Enum.find_value(:no_match, fn {_name, rule_fn} ->
      case rule_fn.(node_id, assumptions) do
        {:resolve, assumption_id} -> {:resolve, assumption_id}
        _ -> nil
      end
    end)
  end

  defp apply_strategy(:auto_least_confident, assumptions, metadata) do
    # Find assumption with lowest confidence
    assumption_with_confidence =
      assumptions
      |> Enum.map(fn id ->
        conf = get_in(metadata, [id, :confidence]) || 0.5
        {id, conf}
      end)
      |> Enum.min_by(fn {_id, conf} -> conf end, fn -> {nil, 0} end)

    case assumption_with_confidence do
      {nil, _} -> :needs_user_input
      {id, _conf} -> {:auto_resolve, id, :least_confident}
    end
  end

  defp apply_strategy(:auto_most_recent, assumptions, metadata) do
    # Find most recently added assumption
    assumption_with_time =
      assumptions
      |> Enum.map(fn id ->
        time = get_in(metadata, [id, :created_at]) || ~U[2000-01-01 00:00:00Z]
        {id, time}
      end)
      |> Enum.max_by(fn {_id, time} -> DateTime.to_unix(time) end, fn -> {nil, nil} end)

    case assumption_with_time do
      {nil, _} -> :needs_user_input
      {id, _time} -> {:auto_resolve, id, :most_recent}
    end
  end

  defp apply_strategy(:manual, _assumptions, _metadata) do
    :needs_user_input
  end

  defp apply_strategy(:hybrid, assumptions, metadata) do
    # Auto for single assumption or clear confidence difference
    if length(assumptions) == 1 do
      {:auto_resolve, hd(assumptions), :only_option}
    else
      # Check confidence spread
      confidences =
        Enum.map(assumptions, fn id ->
          get_in(metadata, [id, :confidence]) || 0.5
        end)

      min_conf = Enum.min(confidences)
      max_conf = Enum.max(confidences)

      if max_conf - min_conf > 0.3 do
        # Clear winner - retract least confident
        apply_strategy(:auto_least_confident, assumptions, metadata)
      else
        # Too close to call automatically
        :needs_user_input
      end
    end
  end

  defp do_resolution(assumption_id, _node_id, _reason) do
    # Retract the assumption in JTMS
    case JTMS.retract_assumption(assumption_id) do
      :ok ->
        Logger.info("Contradiction resolved by retracting assumption", id: assumption_id)
        :ok

      {:error, reason} ->
        Logger.warning("Failed to retract assumption", id: assumption_id, reason: reason)
        {:error, reason}
    end
  end

  defp record_resolution(state, node_id, assumption_id, reason) do
    entry = %{
      node_id: node_id,
      retracted_assumption: assumption_id,
      reason: reason,
      resolved_at: DateTime.utc_now()
    }

    history = [entry | state.history] |> Enum.take(100)
    %{state | history: history}
  end

  defp build_resolution_options(assumptions, metadata) do
    Enum.map(assumptions, fn id ->
      meta = Map.get(metadata, id, %{})

      %{
        assumption_id: id,
        description: Map.get(meta, :description, "Unknown assumption"),
        confidence: Map.get(meta, :confidence, 0.5),
        created_at: Map.get(meta, :created_at),
        impact: estimate_impact(id)
      }
    end)
  end

  defp estimate_impact(assumption_id) do
    # Estimate how many nodes would be affected by retracting this assumption
    case JTMS.consequences_of(assumption_id) do
      {:ok, consequences} -> length(consequences)
      _ -> 0
    end
  end
end
