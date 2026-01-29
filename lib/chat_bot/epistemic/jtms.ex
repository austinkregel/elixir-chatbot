defmodule ChatBot.Epistemic.JTMS do
  @moduledoc """
  Justification-Based Truth Maintenance System (JTMS).

  A JTMS maintains a dependency network where:
  - Nodes represent beliefs/sentences
  - Justifications link premise nodes to conclusion nodes
  - Labels (IN/OUT) propagate based on justification validity

  This implementation follows the classic JTMS design from Forbus & de Kleer,
  adapted for Elixir with GenServer-based state management.

  Key operations:
  - create_node: Add a belief node to the network
  - justify_node: Add a justification linking premises to conclusion
  - enable_assumption/retract_assumption: Toggle assumption nodes
  - Label propagation: Automatic when justifications change

  The system supports:
  - Premise nodes (always IN)
  - Assumption nodes (can be enabled/retracted)
  - Derived nodes (IN if any valid justification)
  - Contradiction nodes (trigger handler when IN)
  """

  use GenServer

  alias ChatBot.Epistemic.Types.{Node, Justification, Config}
  alias ChatBot.Telemetry

  require Logger

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Creates a new node in the dependency network.

  Options:
  - :node_type - :premise | :assumption | :derived | :contradiction
  - :assumption_enabled - For assumptions, whether initially enabled
  - :metadata - Additional metadata

  Returns {:ok, node_id}
  """
  def create_node(datum, opts \\ []) do
    if Config.enabled?() do
      GenServer.call(__MODULE__, {:create_node, datum, opts})
    else
      {:ok, generate_id()}
    end
  end

  @doc """
  Creates a premise node (always IN).
  """
  def create_premise(datum, opts \\ []) do
    create_node(datum, Keyword.put(opts, :node_type, :premise))
  end

  @doc """
  Creates an assumption node.
  """
  def create_assumption(datum, enabled? \\ false, opts \\ []) do
    opts =
      opts |> Keyword.put(:node_type, :assumption) |> Keyword.put(:assumption_enabled, enabled?)

    create_node(datum, opts)
  end

  @doc """
  Creates a contradiction node.
  When this node becomes IN, the contradiction handler is triggered.
  """
  def create_contradiction(datum, opts \\ []) do
    create_node(datum, Keyword.put(opts, :node_type, :contradiction))
  end

  @doc """
  Adds a justification linking premises to a conclusion.

  - in_list: Node IDs that must be IN
  - out_list: Node IDs that must be OUT
  - conclusion_id: The node this supports
  - informant: What created this justification

  Returns {:ok, justification_id}
  """
  def justify_node(in_list, out_list, conclusion_id, informant) do
    Telemetry.span(:jtms_justify, %{conclusion_id: conclusion_id, informant: informant}, fn ->
      if Config.enabled?() do
        GenServer.call(__MODULE__, {:justify_node, in_list, out_list, conclusion_id, informant})
      else
        {:ok, generate_id()}
      end
    end)
  end

  @doc """
  Adds a simple justification (no out_list).
  """
  def justify_node(premise_ids, conclusion_id, informant) when is_list(premise_ids) do
    justify_node(premise_ids, [], conclusion_id, informant)
  end

  @doc """
  Enables an assumption node, making it IN.
  Triggers label propagation.
  """
  def enable_assumption(node_id) do
    GenServer.call(__MODULE__, {:enable_assumption, node_id})
  end

  @doc """
  Retracts an assumption node, making it OUT.
  Triggers label propagation.
  """
  def retract_assumption(node_id) do
    GenServer.call(__MODULE__, {:retract_assumption, node_id})
  end

  @doc """
  Checks if a node is currently IN.
  """
  def is_in?(node_id) do
    GenServer.call(__MODULE__, {:is_in?, node_id})
  end

  @doc """
  Gets the current label of a node.
  """
  def get_label(node_id) do
    GenServer.call(__MODULE__, {:get_label, node_id})
  end

  @doc """
  Gets a node by ID.
  """
  def get_node(node_id) do
    GenServer.call(__MODULE__, {:get_node, node_id})
  end

  @doc """
  Gets the justification chain explaining why a node is IN.
  Returns the list of justifications supporting the node.
  """
  def why_node(node_id) do
    GenServer.call(__MODULE__, {:why_node, node_id})
  end

  @doc """
  Gets all nodes that depend on the given node (forward chaining).
  """
  def consequences_of(node_id) do
    GenServer.call(__MODULE__, {:consequences_of, node_id})
  end

  @doc """
  Gets all nodes that the given node depends on (backward chaining).
  """
  def antecedents_of(node_id) do
    GenServer.call(__MODULE__, {:antecedents_of, node_id})
  end

  @doc """
  Registers a set of node IDs as mutually contradictory.
  When all are IN, triggers contradiction handling.
  """
  def register_contradiction(node_ids, informant \\ "contradiction_rule") do
    GenServer.call(__MODULE__, {:register_contradiction, node_ids, informant})
  end

  @doc """
  Checks consistency of the network.
  Returns {:ok, :consistent} or {:error, {:contradiction, node_id}}.
  """
  def check_consistency do
    GenServer.call(__MODULE__, :check_consistency)
  end

  @doc """
  Gets all current contradictions (contradiction nodes that are IN).
  """
  def get_contradictions do
    GenServer.call(__MODULE__, :get_contradictions)
  end

  @doc """
  Sets the contradiction handler callback.
  The callback receives {:contradiction, node_id, supporting_assumptions}.
  """
  def set_contradiction_handler(handler_fn) when is_function(handler_fn, 1) do
    GenServer.call(__MODULE__, {:set_handler, handler_fn})
  end

  @doc """
  Gets network statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Clears the entire network (for testing).
  """
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  @doc """
  Checks if JTMS is ready.
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
  def init(_opts) do
    state = %{
      nodes: %{},
      justifications: %{},
      node_to_justifications: %{},
      justification_to_conclusion: %{},
      contradiction_nodes: MapSet.new(),
      contradiction_handler: &default_contradiction_handler/1
    }

    Logger.info("JTMS initialized")

    {:ok, state}
  end

  @impl true
  def handle_call({:create_node, datum, opts}, _from, state) do
    node = Node.new(datum, opts)

    new_nodes = Map.put(state.nodes, node.id, node)
    new_state = %{state | nodes: new_nodes}

    # Track contradiction nodes
    new_state =
      if node.node_type == :contradiction do
        %{new_state | contradiction_nodes: MapSet.put(state.contradiction_nodes, node.id)}
      else
        new_state
      end

    Logger.debug("Node created", id: node.id, type: node.node_type, label: node.label)

    {:reply, {:ok, node.id}, new_state}
  end

  @impl true
  def handle_call({:justify_node, in_list, out_list, conclusion_id, informant}, _from, state) do
    # Verify all referenced nodes exist
    all_node_ids = [conclusion_id | in_list ++ out_list]

    missing = Enum.filter(all_node_ids, fn id -> not Map.has_key?(state.nodes, id) end)

    if length(missing) > 0 do
      {:reply, {:error, {:nodes_not_found, missing}}, state}
    else
      justification = Justification.new(in_list, out_list, conclusion_id, informant)

      # Store justification
      new_justifications = Map.put(state.justifications, justification.id, justification)

      # Update node-to-justifications mapping
      new_node_to_justs =
        Enum.reduce(
          [conclusion_id | in_list ++ out_list],
          state.node_to_justifications,
          fn node_id, acc ->
            current = Map.get(acc, node_id, [])
            Map.put(acc, node_id, [justification.id | current])
          end
        )

      # Update justification-to-conclusion mapping
      new_just_to_conc =
        Map.put(state.justification_to_conclusion, justification.id, conclusion_id)

      new_state = %{
        state
        | justifications: new_justifications,
          node_to_justifications: new_node_to_justs,
          justification_to_conclusion: new_just_to_conc
      }

      # Propagate labels
      new_state = propagate_from_justification(new_state, justification)

      Logger.debug("Justification added",
        id: justification.id,
        in_list: in_list,
        out_list: out_list,
        conclusion: conclusion_id
      )

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

        # Propagate label changes
        new_state = propagate_from_node(new_state, node_id)

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

        # Propagate label changes
        new_state = propagate_from_node(new_state, node_id)

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
    case Map.get(state.nodes, node_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      node ->
        # Get all justifications that support this node
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

        {:reply, {:ok, result}, state}
    end
  end

  @impl true
  def handle_call({:consequences_of, node_id}, _from, state) do
    # Find all justifications where this node is in in_list or out_list
    # Then find their conclusions
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
    # Find justifications for this node, then their premises
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
    # Create a contradiction node
    contra_node = Node.contradiction("contradiction_of_#{Enum.join(node_ids, "_")}")
    new_nodes = Map.put(state.nodes, contra_node.id, contra_node)

    new_state = %{
      state
      | nodes: new_nodes,
        contradiction_nodes: MapSet.put(state.contradiction_nodes, contra_node.id)
    }

    # Create justification: if all node_ids are IN, contradiction is IN
    justification = Justification.new(node_ids, [], contra_node.id, informant)
    new_justifications = Map.put(new_state.justifications, justification.id, justification)

    # Update mappings
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

    # Propagate to check if contradiction is already triggered
    new_state = propagate_from_justification(new_state, justification)

    {:reply, {:ok, contra_node.id}, new_state}
  end

  @impl true
  def handle_call(:check_consistency, _from, state) do
    # Find any contradiction nodes that are IN
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
  def handle_call(:clear, _from, _state) do
    new_state = %{
      nodes: %{},
      justifications: %{},
      node_to_justifications: %{},
      justification_to_conclusion: %{},
      contradiction_nodes: MapSet.new(),
      contradiction_handler: &default_contradiction_handler/1
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # ============================================================================
  # Label Propagation
  # ============================================================================

  defp propagate_from_justification(state, justification) do
    # Check if justification is now valid
    if justification_valid?(justification, state) do
      # Update justification label
      updated_just = %{justification | label: :in}
      new_justifications = Map.put(state.justifications, justification.id, updated_just)
      state = %{state | justifications: new_justifications}

      # Update conclusion node to IN if it isn't already
      conclusion = Map.get(state.nodes, justification.conclusion_id)

      if conclusion && conclusion.label == :out do
        updated_node = %{conclusion | label: :in}
        new_nodes = Map.put(state.nodes, conclusion.id, updated_node)
        state = %{state | nodes: new_nodes}

        # Check for contradiction
        state = check_contradiction_triggered(state, conclusion.id)

        # Continue propagation
        propagate_from_node(state, conclusion.id)
      else
        state
      end
    else
      # Justification not valid, ensure its label is OUT
      updated_just = %{justification | label: :out}
      new_justifications = Map.put(state.justifications, justification.id, updated_just)
      %{state | justifications: new_justifications}
    end
  end

  defp propagate_from_node(state, node_id) do
    # Find all justifications that reference this node
    just_ids = Map.get(state.node_to_justifications, node_id, [])

    Enum.reduce(just_ids, state, fn just_id, acc_state ->
      case Map.get(acc_state.justifications, just_id) do
        nil ->
          acc_state

        justification ->
          # Recompute validity
          was_valid = justification.label == :in
          now_valid = justification_valid?(justification, acc_state)

          cond do
            was_valid and not now_valid ->
              # Justification became invalid, may need to update conclusion to OUT
              handle_justification_invalidated(acc_state, justification)

            not was_valid and now_valid ->
              # Justification became valid
              propagate_from_justification(acc_state, justification)

            true ->
              acc_state
          end
      end
    end)
  end

  defp handle_justification_invalidated(state, justification) do
    # Update justification label to OUT
    updated_just = %{justification | label: :out}
    new_justifications = Map.put(state.justifications, justification.id, updated_just)
    state = %{state | justifications: new_justifications}

    # Check if conclusion has any other valid justifications
    conclusion_id = justification.conclusion_id
    conclusion = Map.get(state.nodes, conclusion_id)

    if conclusion && conclusion.node_type == :derived do
      has_valid = has_valid_justification?(state, conclusion_id)

      if not has_valid and conclusion.label == :in do
        # No valid justifications, set to OUT
        updated_node = %{conclusion | label: :out}
        new_nodes = Map.put(state.nodes, conclusion_id, updated_node)
        state = %{state | nodes: new_nodes}

        # Continue propagation
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
        # Find supporting assumptions
        assumptions = find_supporting_assumptions(state, node_id)

        # Call contradiction handler
        state.contradiction_handler.({:contradiction, node_id, assumptions})
      end
    end

    state
  end

  defp find_supporting_assumptions(state, node_id) do
    # BFS to find all assumptions that support this node
    find_assumptions_recursive(state, [node_id], MapSet.new(), [])
  end

  defp find_assumptions_recursive(_state, [], _visited, assumptions), do: assumptions

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
          # Find justifications that support this node
          case handle_call({:why_node, node_id}, nil, state) do
            {:reply, {:ok, result}, _} ->
              antecedent_ids =
                result.supporting_justifications
                |> Enum.flat_map(fn j -> j.in_list end)

              find_assumptions_recursive(state, antecedent_ids ++ rest, visited, assumptions)

            _ ->
              # Node not found or error, skip it
              find_assumptions_recursive(state, rest, visited, assumptions)
          end
      end
    end
  end

  defp default_contradiction_handler({:contradiction, node_id, assumptions}) do
    Logger.warning("Contradiction detected",
      node_id: node_id,
      supporting_assumptions: assumptions
    )
  end

  defp generate_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
end
