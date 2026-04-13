defmodule Brain.Epistemic.ContradictionHandler do
  @moduledoc "Handles contradictions detected by the JTMS.\n\nWhen the JTMS detects that a contradiction node has become IN,\nthis module is responsible for:\n\n1. Identifying the minimal set of assumptions causing the contradiction\n2. Determining resolution strategies\n3. Presenting options or auto-resolving based on configuration\n4. Tracking resolution history for learning\n\nResolution strategies:\n- Retract the least confident assumption\n- Retract the most recent assumption\n- Present options to the user\n- Apply domain-specific rules\n"

  alias Brain.Knowledge.ReviewQueue
  use GenServer

  alias Brain.Epistemic.JTMS

  require Logger

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Handles a contradiction notification from the JTMS.\n\nReturns a resolution decision or {:needs_user_input, options}.\n"
  def handle_contradiction(node_id, supporting_assumptions) do
    GenServer.call(__MODULE__, {:handle_contradiction, node_id, supporting_assumptions}, 5_000)
  end

  @doc "Resolves a contradiction by retracting the specified assumption.\n"
  def resolve_by_retraction(assumption_id) do
    GenServer.call(__MODULE__, {:resolve_by_retraction, assumption_id}, 5_000)
  end

  @doc "Gets all pending contradictions awaiting user resolution.\n"
  def get_pending do
    GenServer.call(__MODULE__, :get_pending, 5_000)
  end

  @doc "Gets resolution history.\n"
  def get_history(opts \\ []) do
    GenServer.call(__MODULE__, {:get_history, opts}, 5_000)
  end

  @doc "Sets the resolution strategy.\n\nStrategies:\n- :auto_least_confident - Automatically retract least confident assumption\n- :auto_most_recent - Automatically retract most recent assumption\n- :manual - Always require user input\n- :hybrid - Auto for low stakes, manual for high stakes\n"
  def set_strategy(strategy) do
    GenServer.call(__MODULE__, {:set_strategy, strategy}, 5_000)
  end

  @doc "Registers a domain-specific resolution rule.\n\nRules are functions that take (node_id, assumptions) and return\n{:resolve, assumption_to_retract} or :no_match.\n"
  def register_rule(name, rule_fn) when is_function(rule_fn, 2) do
    GenServer.call(__MODULE__, {:register_rule, name, rule_fn}, 5_000)
  end

  @doc "Checks if the handler is ready.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc "Gets statistics about the contradiction handler.\n"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats, 5_000)
  end

  @impl true
  def init(_opts) do
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

    send(self(), :register_builtin_rules)

    Logger.info("ContradictionHandler initialized")

    {:ok, state}
  end

  @impl true
  def handle_info(:register_builtin_rules, state) do
    new_rules =
      state.rules
      |> Map.put(:knowledge_expansion, &handle_knowledge_expansion_conflict/2)
      |> Map.put(:academic_papers, &handle_academic_paper_conflict/2)

    {:noreply, %{state | rules: new_rules}}
  end

  @impl true
  def handle_info({:register_with_jtms}, state) do
    if Process.whereis(JTMS) do
      JTMS.set_contradiction_handler(&handle_jtms_callback/1)
    end

    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def handle_call({:handle_contradiction, node_id, assumptions}, _from, state) do
    Logger.info("Handling contradiction", node_id: node_id, assumptions: assumptions)

    case try_rules(state.rules, node_id, assumptions) do
      {:resolve, assumption_id} ->
        result = do_resolution(assumption_id, node_id, :rule)
        new_state = record_resolution(state, node_id, assumption_id, :rule)
        {:reply, {:resolved, result}, new_state}

      :no_match ->
        case apply_strategy(state.strategy, assumptions, state.assumption_metadata) do
          {:auto_resolve, assumption_id, reason} ->
            result = do_resolution(assumption_id, node_id, reason)
            new_state = record_resolution(state, node_id, assumption_id, reason)
            {:reply, {:resolved, result}, new_state}

          :needs_user_input ->
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

  defp handle_jtms_callback({:contradiction, node_id, assumptions}) do
    spawn(fn ->
      handle_contradiction(node_id, assumptions)
    end)
  end

  defp handle_knowledge_expansion_conflict(node_id, _assumptions) do
    case get_conflict_context(node_id) do
      {:knowledge_expansion, new_fact, existing_belief} ->
        if Process.whereis(Brain.Knowledge.ReviewQueue) do
          ReviewQueue.add_contradiction(new_fact, existing_belief)

          Logger.info("Knowledge expansion conflict queued for review",
            node_id: node_id,
            new_fact: inspect(new_fact)
          )
        end

        :no_match

      _ ->
        :no_match
    end
  end

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

  defp handle_academic_paper_conflict(_node_id, assumptions) do
    papers =
      assumptions
      |> Enum.map(fn id ->
        case JTMS.get_node(id) do
          {:ok, node} ->
            metadata = node.metadata || %{}

            if Map.get(metadata, :source) == :academic do
              {id, metadata}
            else
              nil
            end

          _ ->
            nil
        end
      end)
      |> Enum.reject(&is_nil/1)

    if length(papers) >= 2 do
      sorted =
        papers
        |> Enum.sort_by(
          fn {_, meta} ->
            Map.get(meta, :citation_count, 0)
          end,
          :desc
        )

      case sorted do
        [{_winner_id, winner_meta}, {loser_id, loser_meta} | _] ->
          winner_cites = Map.get(winner_meta, :citation_count, 0)
          loser_cites = Map.get(loser_meta, :citation_count, 0)

          if winner_cites > 0 and winner_cites > loser_cites * 2 do
            Logger.info("Resolving academic conflict by citation count",
              winner_citations: winner_cites,
              loser_citations: loser_cites,
              retracted_id: loser_id
            )

            {:resolve, loser_id}
          else
            Logger.info("Academic conflict too close to auto-resolve",
              winner_citations: winner_cites,
              loser_citations: loser_cites
            )

            :no_match
          end

        _ ->
          :no_match
      end
    else
      :no_match
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
    if length(assumptions) == 1 do
      {:auto_resolve, hd(assumptions), :only_option}
    else
      confidences =
        Enum.map(assumptions, fn id ->
          get_in(metadata, [id, :confidence]) || 0.5
        end)

      min_conf = Enum.min(confidences)
      max_conf = Enum.max(confidences)

      if max_conf - min_conf > 0.3 do
        apply_strategy(:auto_least_confident, assumptions, metadata)
      else
        :needs_user_input
      end
    end
  end

  defp do_resolution(assumption_id, _node_id, _reason) do
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
    case JTMS.consequences_of(assumption_id) do
      {:ok, consequences} -> length(consequences)
      _ -> 0
    end
  end
end
