defmodule Brain.Epistemic.JTMS do
  @moduledoc "Justification-Based Truth Maintenance System (JTMS).\n\nA JTMS maintains a dependency network where:\n- Nodes represent beliefs/sentences\n- Justifications link premise nodes to conclusion nodes\n- Labels (IN/OUT) propagate based on justification validity\n\nThis implementation follows the classic JTMS design from Forbus & de Kleer,\nadapted for Elixir with GenServer-based state management.\n\nKey operations:\n- create_node: Add a belief node to the network\n- justify_node: Add a justification linking premises to conclusion\n- enable_assumption/retract_assumption: Toggle assumption nodes\n- Label propagation: Automatic when justifications change\n\nThe system supports:\n- Premise nodes (always IN)\n- Assumption nodes (can be enabled/retracted)\n- Derived nodes (IN if any valid justification)\n- Contradiction nodes (trigger handler when IN)\n"

  alias Brain.Epistemic.Types
  use GenServer

  alias Types.{Node, Justification, Config}
  alias Brain.Telemetry

  require Logger

  @default_world "default"

  def start_link(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world)
    GenServer.start_link(__MODULE__, opts, name: via(world_id))
  end

  @doc false
  def child_spec(opts) do
    world_id = Keyword.get(opts, :world_id, @default_world)

    %{
      id: {__MODULE__, world_id},
      start: {__MODULE__, :start_link, [opts]},
      restart: :transient
    }
  end

  # Per-world addressing. Each agent's mind-world has its OWN truth-maintenance
  # web (isolated nodes/justifications/contradictions) — no shared belief soup.
  defp via(world_id),
    do: {:via, Registry, {Brain.Epistemic.JTMSRegistry, {:jtms, world_id}}}

  @doc "Lazily start the JTMS web for a world; idempotent."
  def ensure(world_id \\ @default_world) do
    case Registry.lookup(Brain.Epistemic.JTMSRegistry, {:jtms, world_id}) do
      [{pid, _}] ->
        {:ok, pid}

      [] ->
        case DynamicSupervisor.start_child(Brain.Epistemic.JTMS.Supervisor, {__MODULE__, world_id: world_id}) do
          {:ok, pid} -> {:ok, pid}
          {:error, {:already_started, pid}} -> {:ok, pid}
          other -> other
        end
    end
  end

  defp call(world_id, message, timeout \\ 5_000) do
    ensure(world_id)
    GenServer.call(via(world_id), message, timeout)
  end

  # ── Public API (each takes a leading world_id; a compat clause defaults to the
  # global "default" web so existing callers keep working unchanged) ──────────

  @doc "Creates a new node in the dependency network. `opts[:world_id]` scopes it."
  def create_node(datum, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world)

    if Config.enabled?() do
      call(world_id, {:create_node, datum, opts})
    else
      {:ok, generate_id()}
    end
  end

  def create_premise(datum, opts \\ []),
    do: create_node(datum, Keyword.put(opts, :node_type, :premise))

  def create_assumption(datum, enabled? \\ false, opts \\ []) do
    opts = opts |> Keyword.put(:node_type, :assumption) |> Keyword.put(:assumption_enabled, enabled?)
    create_node(datum, opts)
  end

  def create_contradiction(datum, opts \\ []),
    do: create_node(datum, Keyword.put(opts, :node_type, :contradiction))

  def justify_node(in_list, out_list, conclusion_id, informant),
    do: justify_node(@default_world, in_list, out_list, conclusion_id, informant)

  def justify_node(world_id, in_list, out_list, conclusion_id, informant)
      when is_binary(world_id) and is_list(in_list) and is_list(out_list) do
    Telemetry.span(:jtms_justify, %{conclusion_id: conclusion_id, informant: informant}, fn ->
      if Config.enabled?() do
        call(world_id, {:justify_node, in_list, out_list, conclusion_id, informant})
      else
        {:ok, generate_id()}
      end
    end)
  end

  def justify_node(premise_ids, conclusion_id, informant) when is_list(premise_ids),
    do: justify_node(@default_world, premise_ids, [], conclusion_id, informant)

  def enable_assumption(node_id), do: enable_assumption(@default_world, node_id)
  def enable_assumption(world_id, node_id), do: call(world_id, {:enable_assumption, node_id})

  def retract_assumption(node_id), do: retract_assumption(@default_world, node_id)
  def retract_assumption(world_id, node_id), do: call(world_id, {:retract_assumption, node_id})

  def is_in?(node_id), do: is_in?(@default_world, node_id)
  def is_in?(world_id, node_id), do: call(world_id, {:is_in?, node_id})

  def get_label(node_id), do: get_label(@default_world, node_id)
  def get_label(world_id, node_id), do: call(world_id, {:get_label, node_id})

  def get_node(node_id), do: get_node(@default_world, node_id)
  def get_node(world_id, node_id), do: call(world_id, {:get_node, node_id})

  def why_node(node_id), do: why_node(@default_world, node_id)
  def why_node(world_id, node_id), do: call(world_id, {:why_node, node_id})

  def consequences_of(node_id), do: consequences_of(@default_world, node_id)
  def consequences_of(world_id, node_id), do: call(world_id, {:consequences_of, node_id})

  def antecedents_of(node_id), do: antecedents_of(@default_world, node_id)
  def antecedents_of(world_id, node_id), do: call(world_id, {:antecedents_of, node_id})

  def register_contradiction(node_ids, informant \\ "contradiction_rule"),
    do: register_contradiction(@default_world, node_ids, informant)

  def register_contradiction(world_id, node_ids, informant) when is_binary(world_id),
    do: call(world_id, {:register_contradiction, node_ids, informant})

  def check_consistency(world_id \\ @default_world), do: call(world_id, :check_consistency)

  def get_contradictions(world_id \\ @default_world), do: call(world_id, :get_contradictions)

  def set_contradiction_handler(handler_fn) when is_function(handler_fn, 1),
    do: set_contradiction_handler(@default_world, handler_fn)

  def set_contradiction_handler(world_id, handler_fn) when is_function(handler_fn, 1),
    do: call(world_id, {:set_handler, handler_fn})

  def stats(world_id \\ @default_world), do: call(world_id, :stats)

  def clear(world_id \\ @default_world), do: call(world_id, :clear, 30_000)

  @doc "Checks if the JTMS web for a world is ready."
  def ready?(world_id \\ @default_world) do
    {:ok, _} = ensure(world_id)
    GenServer.call(via(world_id), :ready?, 100)
  catch
    :exit, {:timeout, _} -> false
    :exit, {:noproc, _} -> false
  end

  @impl true
  def init(opts) do
    state = %{
      world_id: Keyword.get(opts, :world_id, @default_world),
      nodes: %{},
      justifications: %{},
      node_to_justifications: %{},
      justification_to_conclusion: %{},
      contradiction_nodes: MapSet.new(),
      # Every world's JTMS is born wired to the real handler. Registering from
      # the handler's side is impossible: JTMS processes live under
      # `{:via, Registry, {JTMSRegistry, {:jtms, world_id}}}`, so no process ever
      # holds the bare `Brain.Epistemic.JTMS` atom for a `Process.whereis/1` to
      # find — and `set_contradiction_handler/1` would only ever reach the
      # "default" world anyway, which is wrong for per-officer mind-worlds.
      contradiction_handler: &Brain.Epistemic.ContradictionHandler.handle_jtms_callback/1
    }

    Logger.info("JTMS initialized", world_id: state.world_id)

    {:ok, state}
  end

  @impl true
  def handle_call({:create_node, datum, opts}, _from, state) do
    node = Node.new(datum, opts)

    new_nodes = Map.put(state.nodes, node.id, node)
    new_state = %{state | nodes: new_nodes}

    new_state =
      if node.node_type == :contradiction do
        %{new_state | contradiction_nodes: MapSet.put(state.contradiction_nodes, node.id)}
      else
        new_state
      end

    Logger.debug("Node created", id: node.id, type: node.node_type, label: node.label)

    Brain.Graph.Writer.write_jtms_node(node)

    {:reply, {:ok, node.id}, new_state}
  end

  @impl true
  def handle_call({:justify_node, in_list, out_list, conclusion_id, informant}, _from, state) do
    all_node_ids = [conclusion_id | in_list ++ out_list]

    missing = Enum.filter(all_node_ids, fn id -> not Map.has_key?(state.nodes, id) end)

    if missing != [] do
      {:reply, {:error, {:nodes_not_found, missing}}, state}
    else
      justification = Justification.new(in_list, out_list, conclusion_id, informant)
      new_justifications = Map.put(state.justifications, justification.id, justification)

      new_node_to_justs =
        Enum.reduce(
          [conclusion_id | in_list ++ out_list],
          state.node_to_justifications,
          fn node_id, acc ->
            current = Map.get(acc, node_id, [])
            Map.put(acc, node_id, [justification.id | current])
          end
        )

      new_just_to_conc =
        Map.put(state.justification_to_conclusion, justification.id, conclusion_id)

      new_state = %{
        state
        | justifications: new_justifications,
          node_to_justifications: new_node_to_justs,
          justification_to_conclusion: new_just_to_conc
      }

      new_state = propagate_from_justification(new_state, justification)

      Logger.debug("Justification added",
        id: justification.id,
        in_list: in_list,
        out_list: out_list,
        conclusion: conclusion_id
      )

      Brain.Graph.Writer.write_justification(%{
        id: justification.id,
        informant: informant,
        conclusion_id: conclusion_id,
        in_list: in_list,
        out_list: out_list
      })

      {:reply, {:ok, justification.id}, new_state}
    end
  end

  @impl true
  def handle_call({:enable_assumption, node_id}, _from, state) do
    case Map.get(state.nodes, node_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      %Node{node_type: :assumption} = node ->
        updated = %{node | assumption_enabled: true, label: :in}
        new_nodes = Map.put(state.nodes, node_id, updated)
        new_state = %{state | nodes: new_nodes}
        new_state = propagate_from_node(new_state, node_id)

        Brain.Graph.Writer.update_jtms_label(node_id, :in)

        {:reply, :ok, new_state}

      _ ->
        {:reply, {:error, :not_an_assumption}, state}
    end
  end

  @impl true
  def handle_call({:retract_assumption, node_id}, _from, state) do
    case Map.get(state.nodes, node_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      %Node{node_type: :assumption} = node ->
        updated = %{node | assumption_enabled: false, label: :out}
        new_nodes = Map.put(state.nodes, node_id, updated)
        new_state = %{state | nodes: new_nodes}
        new_state = propagate_from_node(new_state, node_id)

        Brain.Graph.Writer.update_jtms_label(node_id, :out)

        {:reply, :ok, new_state}

      _ ->
        {:reply, {:error, :not_an_assumption}, state}
    end
  end

  @impl true
  def handle_call({:is_in?, node_id}, _from, state) do
    case Map.get(state.nodes, node_id) do
      nil -> {:reply, {:error, :not_found}, state}
      node -> {:reply, node.label == :in, state}
    end
  end

  @impl true
  def handle_call({:get_label, node_id}, _from, state) do
    case Map.get(state.nodes, node_id) do
      nil -> {:reply, {:error, :not_found}, state}
      node -> {:reply, {:ok, node.label}, state}
    end
  end

  @impl true
  def handle_call({:get_node, node_id}, _from, state) do
    case Map.get(state.nodes, node_id) do
      nil -> {:reply, {:error, :not_found}, state}
      node -> {:reply, {:ok, node}, state}
    end
  end

  @impl true
  def handle_call({:why_node, node_id}, _from, state) do
    case do_why_node(node_id, state) do
      {:ok, result} -> {:reply, {:ok, result}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:consequences_of, node_id}, _from, state) do
    consequences =
      state.justifications
      |> Map.values()
      |> Enum.filter(fn j -> node_id in j.in_list or node_id in j.out_list end)
      |> Enum.map(fn j -> j.conclusion_id end)
      |> Enum.uniq()
      |> Enum.map(&Map.get(state.nodes, &1))
      |> Enum.filter(&(&1 != nil))

    {:reply, {:ok, consequences}, state}
  end

  @impl true
  def handle_call({:antecedents_of, node_id}, _from, state) do
    just_ids = Map.get(state.node_to_justifications, node_id, [])

    antecedents =
      just_ids
      |> Enum.map(&Map.get(state.justifications, &1))
      |> Enum.filter(fn j -> j != nil and j.conclusion_id == node_id end)
      |> Enum.flat_map(fn j -> j.in_list ++ j.out_list end)
      |> Enum.uniq()
      |> Enum.map(&Map.get(state.nodes, &1))
      |> Enum.filter(&(&1 != nil))

    {:reply, {:ok, antecedents}, state}
  end

  @impl true
  def handle_call({:register_contradiction, node_ids, informant}, _from, state) do
    contra_node = Node.contradiction("contradiction_of_#{Enum.join(node_ids, "_")}")
    new_nodes = Map.put(state.nodes, contra_node.id, contra_node)

    new_state = %{
      state
      | nodes: new_nodes,
        contradiction_nodes: MapSet.put(state.contradiction_nodes, contra_node.id)
    }

    justification = Justification.new(node_ids, [], contra_node.id, informant)
    new_justifications = Map.put(new_state.justifications, justification.id, justification)

    new_node_to_justs =
      Enum.reduce([contra_node.id | node_ids], new_state.node_to_justifications, fn nid, acc ->
        current = Map.get(acc, nid, [])
        Map.put(acc, nid, [justification.id | current])
      end)

    new_just_to_conc =
      Map.put(new_state.justification_to_conclusion, justification.id, contra_node.id)

    new_state = %{
      new_state
      | justifications: new_justifications,
        node_to_justifications: new_node_to_justs,
        justification_to_conclusion: new_just_to_conc
    }

    new_state = propagate_from_justification(new_state, justification)

    Brain.Graph.Writer.write_contradiction(node_ids)

    {:reply, {:ok, contra_node.id}, new_state}
  end

  @impl true
  def handle_call(:check_consistency, _from, state) do
    active_contradictions =
      state.contradiction_nodes
      |> Enum.map(&Map.get(state.nodes, &1))
      |> Enum.filter(fn node -> node != nil and node.label == :in end)

    result =
      case active_contradictions do
        [] -> {:ok, :consistent}
        [first | _] -> {:error, {:contradiction, first.id}}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call(:get_contradictions, _from, state) do
    contradictions =
      state.contradiction_nodes
      |> Enum.map(&Map.get(state.nodes, &1))
      |> Enum.filter(fn node -> node != nil and node.label == :in end)

    {:reply, contradictions, state}
  end

  @impl true
  def handle_call({:set_handler, handler_fn}, _from, state) do
    {:reply, :ok, %{state | contradiction_handler: handler_fn}}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    in_nodes = Enum.count(state.nodes, fn {_, n} -> n.label == :in end)

    stats = %{
      total_nodes: map_size(state.nodes),
      in_nodes: in_nodes,
      out_nodes: map_size(state.nodes) - in_nodes,
      justifications: map_size(state.justifications),
      contradiction_nodes: MapSet.size(state.contradiction_nodes)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    # `world_id` must survive a clear: it identifies which world's web this is,
    # and `check_contradiction_triggered/2` stamps it into the handler callback.
    new_state = %{
      world_id: state.world_id,
      nodes: %{},
      justifications: %{},
      node_to_justifications: %{},
      justification_to_conclusion: %{},
      contradiction_nodes: MapSet.new(),
      contradiction_handler: &Brain.Epistemic.ContradictionHandler.handle_jtms_callback/1
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  defp propagate_from_justification(state, justification) do
    if justification_valid?(justification, state) do
      updated_just = %{justification | label: :in}
      new_justifications = Map.put(state.justifications, justification.id, updated_just)
      state = %{state | justifications: new_justifications}
      conclusion = Map.get(state.nodes, justification.conclusion_id)

      if conclusion && conclusion.label == :out do
        updated_node = %{conclusion | label: :in}
        new_nodes = Map.put(state.nodes, conclusion.id, updated_node)
        state = %{state | nodes: new_nodes}
        Brain.Graph.Writer.update_jtms_label(conclusion.id, :in)
        state = check_contradiction_triggered(state, conclusion.id)
        propagate_from_node(state, conclusion.id)
      else
        state
      end
    else
      updated_just = %{justification | label: :out}
      new_justifications = Map.put(state.justifications, justification.id, updated_just)
      %{state | justifications: new_justifications}
    end
  end

  defp propagate_from_node(state, node_id) do
    just_ids = Map.get(state.node_to_justifications, node_id, [])

    Enum.reduce(just_ids, state, fn just_id, acc_state ->
      case Map.get(acc_state.justifications, just_id) do
        nil ->
          acc_state

        justification ->
          was_valid = justification.label == :in
          now_valid = justification_valid?(justification, acc_state)

          cond do
            was_valid and not now_valid ->
              handle_justification_invalidated(acc_state, justification)

            not was_valid and now_valid ->
              propagate_from_justification(acc_state, justification)

            true ->
              acc_state
          end
      end
    end)
  end

  defp handle_justification_invalidated(state, justification) do
    updated_just = %{justification | label: :out}
    new_justifications = Map.put(state.justifications, justification.id, updated_just)
    state = %{state | justifications: new_justifications}
    conclusion_id = justification.conclusion_id
    conclusion = Map.get(state.nodes, conclusion_id)

    if conclusion && conclusion.node_type == :derived do
      has_valid = has_valid_justification?(state, conclusion_id)

      if not has_valid and conclusion.label == :in do
        updated_node = %{conclusion | label: :out}
        new_nodes = Map.put(state.nodes, conclusion_id, updated_node)
        state = %{state | nodes: new_nodes}
        Brain.Graph.Writer.update_jtms_label(conclusion_id, :out)
        propagate_from_node(state, conclusion_id)
      else
        state
      end
    else
      state
    end
  end

  defp justification_valid?(justification, state) do
    node_labels =
      state.nodes
      |> Enum.map(fn {id, node} -> {id, node.label} end)
      |> Map.new()

    Justification.valid?(justification, node_labels)
  end

  defp has_valid_justification?(state, node_id) do
    just_ids = Map.get(state.node_to_justifications, node_id, [])

    Enum.any?(just_ids, fn just_id ->
      case Map.get(state.justifications, just_id) do
        nil -> false
        j -> j.conclusion_id == node_id and justification_valid?(j, state)
      end
    end)
  end

  defp check_contradiction_triggered(state, node_id) do
    if MapSet.member?(state.contradiction_nodes, node_id) do
      node = Map.get(state.nodes, node_id)

      if node && node.label == :in do
        assumptions = find_supporting_assumptions(state, node_id)
        # The world travels with the notification: without it a handler cannot
        # know which world's web to retract in, and would silently act on
        # "default" — a web belonging to no one under Fleet mind-worlds.
        state.contradiction_handler.({:contradiction, state.world_id, node_id, assumptions})
      end
    end

    state
  end

  defp find_supporting_assumptions(state, node_id) do
    find_assumptions_recursive(state, [node_id], MapSet.new(), [])
  end

  defp find_assumptions_recursive(_state, [], _visited, assumptions) do
    assumptions
  end

  defp find_assumptions_recursive(state, [node_id | rest], visited, assumptions) do
    if MapSet.member?(visited, node_id) do
      find_assumptions_recursive(state, rest, visited, assumptions)
    else
      visited = MapSet.put(visited, node_id)
      node = Map.get(state.nodes, node_id)

      cond do
        node == nil ->
          find_assumptions_recursive(state, rest, visited, assumptions)

        node.node_type == :assumption and node.label == :in ->
          find_assumptions_recursive(state, rest, visited, [node_id | assumptions])

        node.node_type == :premise ->
          find_assumptions_recursive(state, rest, visited, assumptions)

        true ->
          case do_why_node(node_id, state) do
            {:ok, result} ->
              antecedent_ids =
                result.supporting_justifications
                |> Enum.flat_map(fn j -> j.in_list end)

              find_assumptions_recursive(state, antecedent_ids ++ rest, visited, assumptions)

            _ ->
              find_assumptions_recursive(state, rest, visited, assumptions)
          end
      end
    end
  end

  defp do_why_node(node_id, state) do
    case Map.get(state.nodes, node_id) do
      nil ->
        {:error, :not_found}

      node ->
        just_ids = Map.get(state.node_to_justifications, node_id, [])

        supporting =
          just_ids
          |> Enum.map(&Map.get(state.justifications, &1))
          |> Enum.filter(fn j -> j != nil and j.conclusion_id == node_id end)
          |> Enum.filter(fn j -> justification_valid?(j, state) end)

        result = %{
          node: node,
          supporting_justifications: supporting
        }

        {:ok, result}
    end
  end

  defp generate_id do
    FourthWall.ID.generate()
  end
end
