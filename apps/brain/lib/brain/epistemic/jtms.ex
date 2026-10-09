defmodule Brain.Epistemic.JTMS do
  @moduledoc """
  Justification-Based Truth Maintenance System (JTMS).

  A JTMS maintains a dependency network where:
  - Nodes represent beliefs/sentences
  - Justifications link premise nodes to conclusion nodes
  - Labels (IN/OUT) propagate based on justification validity

  This implementation follows the classic JTMS design from Forbus & de Kleer,
  adapted for Elixir with GenServer-based state management.

  Each world has its own network. Every function that reads or changes a
  network takes the world first: nodes, justifications, labels and
  contradictions live in that world's network, and are written to that
  world's epistemic graph. A justification can only connect nodes of one
  world, and a node id from another world is not found. A missing or
  malformed world raises `ArgumentError` in the caller's process.

  Key operations:
  - create_node: Add a belief node to the world's network
  - justify_node: Add a justification linking premises to conclusion
  - enable_assumption/retract_assumption: Toggle assumption nodes
  - Label propagation: Automatic when justifications change

  The system supports:
  - Premise nodes (always IN)
  - Assumption nodes (can be enabled/retracted)
  - Derived nodes (IN if any valid justification)
  - Contradiction nodes (trigger handler when IN)

  The contradiction handler is one callback for every world; it is called
  with `{:contradiction, world_id, node_id, supporting_assumptions}`, naming
  the world whose network holds the contradiction.
  """

  alias Brain.Epistemic.Types
  use GenServer

  alias Types.{Node, Justification, Config}
  alias Brain.Telemetry

  require Logger

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Creates a new node in the world's dependency network.

  Options:
  - :node_type - :premise | :assumption | :derived | :contradiction
  - :assumption_enabled - For assumptions, whether initially enabled
  - :metadata - Additional metadata

  Returns {:ok, node_id}
  """
  def create_node(world_id, datum, opts \\ []) do
    world_id = world_id!(world_id, "create_node")

    if Config.enabled?() do
      GenServer.call(__MODULE__, {:create_node, world_id, datum, opts}, 5_000)
    else
      {:ok, generate_id()}
    end
  end

  @doc "Creates a premise node (always IN) in the world's network.\n"
  def create_premise(world_id, datum, opts \\ []) do
    create_node(world_id, datum, Keyword.put(opts, :node_type, :premise))
  end

  @doc "Creates an assumption node in the world's network.\n"
  def create_assumption(world_id, datum, enabled? \\ false, opts \\ []) do
    opts =
      opts |> Keyword.put(:node_type, :assumption) |> Keyword.put(:assumption_enabled, enabled?)

    create_node(world_id, datum, opts)
  end

  @doc """
  Creates a contradiction node in the world's network.
  When this node becomes IN, the contradiction handler is triggered.
  """
  def create_contradiction(world_id, datum, opts \\ []) do
    create_node(world_id, datum, Keyword.put(opts, :node_type, :contradiction))
  end

  @doc """
  Adds a justification linking premises to a conclusion, in the world's
  network.

  - in_list: Node IDs that must be IN
  - out_list: Node IDs that must be OUT
  - conclusion_id: The node this supports
  - informant: What created this justification

  Every node must be in the world's network. Returns {:ok, justification_id}
  """
  def justify_node(world_id, in_list, out_list, conclusion_id, informant) do
    world_id = world_id!(world_id, "justify_node")

    Telemetry.span(:jtms_justify, %{conclusion_id: conclusion_id, informant: informant}, fn ->
      if Config.enabled?() do
        GenServer.call(__MODULE__, {:justify_node, world_id, in_list, out_list, conclusion_id, informant}, 5_000)
      else
        {:ok, generate_id()}
      end
    end)
  end

  @doc "Adds a simple justification (no out_list) in the world's network.\n"
  def justify_node(world_id, premise_ids, conclusion_id, informant) when is_list(premise_ids) do
    justify_node(world_id, premise_ids, [], conclusion_id, informant)
  end

  @doc """
  Enables an assumption node of the world's network, making it IN.
  Triggers label propagation.
  """
  def enable_assumption(world_id, node_id) do
    world_id = world_id!(world_id, "enable_assumption")
    GenServer.call(__MODULE__, {:enable_assumption, world_id, node_id}, 5_000)
  end

  @doc """
  Retracts an assumption node of the world's network, making it OUT.
  Triggers label propagation.
  """
  def retract_assumption(world_id, node_id) do
    world_id = world_id!(world_id, "retract_assumption")
    GenServer.call(__MODULE__, {:retract_assumption, world_id, node_id}, 5_000)
  end

  @doc "Checks if a node of the world's network is currently IN.\n"
  def is_in?(world_id, node_id) do
    world_id = world_id!(world_id, "is_in?")
    GenServer.call(__MODULE__, {:is_in?, world_id, node_id}, 5_000)
  end

  @doc "Gets the current label of a node of the world's network.\n"
  def get_label(world_id, node_id) do
    world_id = world_id!(world_id, "get_label")
    GenServer.call(__MODULE__, {:get_label, world_id, node_id}, 5_000)
  end

  @doc "Gets a node of the world's network by ID.\n"
  def get_node(world_id, node_id) do
    world_id = world_id!(world_id, "get_node")
    GenServer.call(__MODULE__, {:get_node, world_id, node_id}, 5_000)
  end

  @doc """
  Gets the justification chain explaining why a node of the world's network
  is IN. Returns the list of justifications supporting the node.
  """
  def why_node(world_id, node_id) do
    world_id = world_id!(world_id, "why_node")
    GenServer.call(__MODULE__, {:why_node, world_id, node_id}, 5_000)
  end

  @doc "Gets all nodes of the world's network that depend on the given node (forward chaining).\n"
  def consequences_of(world_id, node_id) do
    world_id = world_id!(world_id, "consequences_of")
    GenServer.call(__MODULE__, {:consequences_of, world_id, node_id}, 5_000)
  end

  @doc "Gets all nodes of the world's network that the given node depends on (backward chaining).\n"
  def antecedents_of(world_id, node_id) do
    world_id = world_id!(world_id, "antecedents_of")
    GenServer.call(__MODULE__, {:antecedents_of, world_id, node_id}, 5_000)
  end

  @doc """
  Registers a set of node IDs of the world's network as mutually
  contradictory. When all are IN, triggers contradiction handling.
  """
  def register_contradiction(world_id, node_ids, informant \\ "contradiction_rule") do
    world_id = world_id!(world_id, "register_contradiction")
    GenServer.call(__MODULE__, {:register_contradiction, world_id, node_ids, informant}, 5_000)
  end

  @doc """
  Checks consistency of the world's network.
  Returns {:ok, :consistent} or {:error, {:contradiction, node_id}}.
  """
  def check_consistency(world_id) do
    world_id = world_id!(world_id, "check_consistency")
    GenServer.call(__MODULE__, {:check_consistency, world_id}, 5_000)
  end

  @doc "Gets all current contradictions (contradiction nodes that are IN) of the world's network.\n"
  def get_contradictions(world_id) do
    world_id = world_id!(world_id, "get_contradictions")
    GenServer.call(__MODULE__, {:get_contradictions, world_id}, 5_000)
  end

  @doc """
  Sets the contradiction handler callback, for every world.
  The callback receives {:contradiction, world_id, node_id, supporting_assumptions}.
  """
  def set_contradiction_handler(handler_fn) when is_function(handler_fn, 1) do
    GenServer.call(__MODULE__, {:set_handler, handler_fn}, 5_000)
  end

  @doc "Gets the statistics of the world's network.\n"
  def stats(world_id) do
    world_id = world_id!(world_id, "stats")
    GenServer.call(__MODULE__, {:stats, world_id}, 5_000)
  end

  @doc "Clears the world's network (for testing). Other worlds keep theirs.\n"
  def clear(world_id) do
    world_id = world_id!(world_id, "clear")
    GenServer.call(__MODULE__, {:clear, world_id}, 30_000)
  end

  @doc "Checks if JTMS is ready.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  defp world_id!(world_id, _function) when is_binary(world_id) and world_id != "", do: world_id

  defp world_id!(other, function) do
    raise ArgumentError,
          "JTMS.#{function} needs the world whose network it works on, got: #{inspect(other)}"
  end

  defp empty_network do
    %{
      nodes: %{},
      justifications: %{},
      node_to_justifications: %{},
      justification_to_conclusion: %{},
      contradiction_nodes: MapSet.new()
    }
  end

  # A world's network, with the world and the contradiction handler beside it
  # so propagation can name the world and call the handler.
  defp network(state, world_id) do
    state.worlds
    |> Map.get(world_id, empty_network())
    |> Map.merge(%{world_id: world_id, contradiction_handler: state.contradiction_handler})
  end

  defp put_network(state, network) do
    stored = Map.drop(network, [:world_id, :contradiction_handler])
    %{state | worlds: Map.put(state.worlds, network.world_id, stored)}
  end

  @impl true
  def init(_opts) do
    state = %{
      worlds: %{},
      contradiction_handler: &default_contradiction_handler/1
    }

    Logger.info("JTMS initialized")

    {:ok, state}
  end

  @impl true
  def handle_call({:create_node, world_id, datum, opts}, _from, state) do
    net = network(state, world_id)
    node = Node.new(datum, opts)

    net = %{net | nodes: Map.put(net.nodes, node.id, node)}

    net =
      if node.node_type == :contradiction do
        %{net | contradiction_nodes: MapSet.put(net.contradiction_nodes, node.id)}
      else
        net
      end

    Logger.debug("Node created", id: node.id, world_id: world_id, type: node.node_type, label: node.label)

    Brain.Graph.Writer.write_jtms_node(node, world_id)

    {:reply, {:ok, node.id}, put_network(state, net)}
  end

  @impl true
  def handle_call({:justify_node, world_id, in_list, out_list, conclusion_id, informant}, _from, state) do
    net = network(state, world_id)
    all_node_ids = [conclusion_id | in_list ++ out_list]

    missing = Enum.filter(all_node_ids, fn id -> not Map.has_key?(net.nodes, id) end)

    if missing != [] do
      {:reply, {:error, {:nodes_not_found, missing}}, state}
    else
      justification = Justification.new(in_list, out_list, conclusion_id, informant)
      net = add_justification(net, justification, all_node_ids)
      net = propagate_from_justification(net, justification)

      Logger.debug("Justification added",
        id: justification.id,
        world_id: world_id,
        in_list: in_list,
        out_list: out_list,
        conclusion: conclusion_id
      )

      Brain.Graph.Writer.write_justification(
        %{
          id: justification.id,
          informant: informant,
          conclusion_id: conclusion_id,
          in_list: in_list,
          out_list: out_list
        },
        world_id
      )

      {:reply, {:ok, justification.id}, put_network(state, net)}
    end
  end

  @impl true
  def handle_call({:enable_assumption, world_id, node_id}, _from, state) do
    net = network(state, world_id)

    case Map.get(net.nodes, node_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      %Node{node_type: :assumption} = node ->
        updated = %{node | assumption_enabled: true, label: :in}
        net = %{net | nodes: Map.put(net.nodes, node_id, updated)}
        net = propagate_from_node(net, node_id)

        Brain.Graph.Writer.update_jtms_label(node_id, :in, world_id)

        {:reply, :ok, put_network(state, net)}

      _ ->
        {:reply, {:error, :not_an_assumption}, state}
    end
  end

  @impl true
  def handle_call({:retract_assumption, world_id, node_id}, _from, state) do
    net = network(state, world_id)

    case Map.get(net.nodes, node_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      %Node{node_type: :assumption} = node ->
        updated = %{node | assumption_enabled: false, label: :out}
        net = %{net | nodes: Map.put(net.nodes, node_id, updated)}
        net = propagate_from_node(net, node_id)

        Brain.Graph.Writer.update_jtms_label(node_id, :out, world_id)

        {:reply, :ok, put_network(state, net)}

      _ ->
        {:reply, {:error, :not_an_assumption}, state}
    end
  end

  @impl true
  def handle_call({:is_in?, world_id, node_id}, _from, state) do
    case Map.get(network(state, world_id).nodes, node_id) do
      nil -> {:reply, {:error, :not_found}, state}
      node -> {:reply, node.label == :in, state}
    end
  end

  @impl true
  def handle_call({:get_label, world_id, node_id}, _from, state) do
    case Map.get(network(state, world_id).nodes, node_id) do
      nil -> {:reply, {:error, :not_found}, state}
      node -> {:reply, {:ok, node.label}, state}
    end
  end

  @impl true
  def handle_call({:get_node, world_id, node_id}, _from, state) do
    case Map.get(network(state, world_id).nodes, node_id) do
      nil -> {:reply, {:error, :not_found}, state}
      node -> {:reply, {:ok, node}, state}
    end
  end

  @impl true
  def handle_call({:why_node, world_id, node_id}, _from, state) do
    case do_why_node(node_id, network(state, world_id)) do
      {:ok, result} -> {:reply, {:ok, result}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:consequences_of, world_id, node_id}, _from, state) do
    net = network(state, world_id)

    consequences =
      net.justifications
      |> Map.values()
      |> Enum.filter(fn j -> node_id in j.in_list or node_id in j.out_list end)
      |> Enum.map(fn j -> j.conclusion_id end)
      |> Enum.uniq()
      |> Enum.map(&Map.get(net.nodes, &1))
      |> Enum.filter(&(&1 != nil))

    {:reply, {:ok, consequences}, state}
  end

  @impl true
  def handle_call({:antecedents_of, world_id, node_id}, _from, state) do
    net = network(state, world_id)
    just_ids = Map.get(net.node_to_justifications, node_id, [])

    antecedents =
      just_ids
      |> Enum.map(&Map.get(net.justifications, &1))
      |> Enum.filter(fn j -> j != nil and j.conclusion_id == node_id end)
      |> Enum.flat_map(fn j -> j.in_list ++ j.out_list end)
      |> Enum.uniq()
      |> Enum.map(&Map.get(net.nodes, &1))
      |> Enum.filter(&(&1 != nil))

    {:reply, {:ok, antecedents}, state}
  end

  @impl true
  def handle_call({:register_contradiction, world_id, node_ids, informant}, _from, state) do
    net = network(state, world_id)
    contra_node = Node.contradiction("contradiction_of_#{Enum.join(node_ids, "_")}")

    net = %{
      net
      | nodes: Map.put(net.nodes, contra_node.id, contra_node),
        contradiction_nodes: MapSet.put(net.contradiction_nodes, contra_node.id)
    }

    justification = Justification.new(node_ids, [], contra_node.id, informant)
    net = add_justification(net, justification, [contra_node.id | node_ids])
    net = propagate_from_justification(net, justification)

    Brain.Graph.Writer.write_contradiction(node_ids, world_id)

    {:reply, {:ok, contra_node.id}, put_network(state, net)}
  end

  @impl true
  def handle_call({:check_consistency, world_id}, _from, state) do
    result =
      case active_contradictions(network(state, world_id)) do
        [] -> {:ok, :consistent}
        [first | _] -> {:error, {:contradiction, first.id}}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_contradictions, world_id}, _from, state) do
    {:reply, active_contradictions(network(state, world_id)), state}
  end

  @impl true
  def handle_call({:set_handler, handler_fn}, _from, state) do
    {:reply, :ok, %{state | contradiction_handler: handler_fn}}
  end

  @impl true
  def handle_call({:stats, world_id}, _from, state) do
    net = network(state, world_id)
    in_nodes = Enum.count(net.nodes, fn {_, n} -> n.label == :in end)

    stats = %{
      total_nodes: map_size(net.nodes),
      in_nodes: in_nodes,
      out_nodes: map_size(net.nodes) - in_nodes,
      justifications: map_size(net.justifications),
      contradiction_nodes: MapSet.size(net.contradiction_nodes)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call({:clear, world_id}, _from, state) do
    {:reply, :ok, %{state | worlds: Map.delete(state.worlds, world_id)}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  defp add_justification(net, justification, node_ids) do
    new_node_to_justs =
      Enum.reduce(node_ids, net.node_to_justifications, fn node_id, acc ->
        current = Map.get(acc, node_id, [])
        Map.put(acc, node_id, [justification.id | current])
      end)

    %{
      net
      | justifications: Map.put(net.justifications, justification.id, justification),
        node_to_justifications: new_node_to_justs,
        justification_to_conclusion:
          Map.put(net.justification_to_conclusion, justification.id, justification.conclusion_id)
    }
  end

  defp active_contradictions(net) do
    net.contradiction_nodes
    |> Enum.map(&Map.get(net.nodes, &1))
    |> Enum.filter(fn node -> node != nil and node.label == :in end)
  end

  defp propagate_from_justification(net, justification) do
    if justification_valid?(justification, net) do
      updated_just = %{justification | label: :in}
      net = %{net | justifications: Map.put(net.justifications, justification.id, updated_just)}
      conclusion = Map.get(net.nodes, justification.conclusion_id)

      if conclusion && conclusion.label == :out do
        updated_node = %{conclusion | label: :in}
        net = %{net | nodes: Map.put(net.nodes, conclusion.id, updated_node)}
        Brain.Graph.Writer.update_jtms_label(conclusion.id, :in, net.world_id)
        net = check_contradiction_triggered(net, conclusion.id)
        propagate_from_node(net, conclusion.id)
      else
        net
      end
    else
      updated_just = %{justification | label: :out}
      %{net | justifications: Map.put(net.justifications, justification.id, updated_just)}
    end
  end

  defp propagate_from_node(net, node_id) do
    just_ids = Map.get(net.node_to_justifications, node_id, [])

    Enum.reduce(just_ids, net, fn just_id, acc_net ->
      case Map.get(acc_net.justifications, just_id) do
        nil ->
          acc_net

        justification ->
          was_valid = justification.label == :in
          now_valid = justification_valid?(justification, acc_net)

          cond do
            was_valid and not now_valid ->
              handle_justification_invalidated(acc_net, justification)

            not was_valid and now_valid ->
              propagate_from_justification(acc_net, justification)

            true ->
              acc_net
          end
      end
    end)
  end

  defp handle_justification_invalidated(net, justification) do
    updated_just = %{justification | label: :out}
    net = %{net | justifications: Map.put(net.justifications, justification.id, updated_just)}
    conclusion_id = justification.conclusion_id
    conclusion = Map.get(net.nodes, conclusion_id)

    if conclusion && conclusion.node_type == :derived do
      has_valid = has_valid_justification?(net, conclusion_id)

      if not has_valid and conclusion.label == :in do
        updated_node = %{conclusion | label: :out}
        net = %{net | nodes: Map.put(net.nodes, conclusion_id, updated_node)}
        Brain.Graph.Writer.update_jtms_label(conclusion_id, :out, net.world_id)
        propagate_from_node(net, conclusion_id)
      else
        net
      end
    else
      net
    end
  end

  defp justification_valid?(justification, net) do
    node_labels =
      net.nodes
      |> Enum.map(fn {id, node} -> {id, node.label} end)
      |> Map.new()

    Justification.valid?(justification, node_labels)
  end

  defp has_valid_justification?(net, node_id) do
    just_ids = Map.get(net.node_to_justifications, node_id, [])

    Enum.any?(just_ids, fn just_id ->
      case Map.get(net.justifications, just_id) do
        nil -> false
        j -> j.conclusion_id == node_id and justification_valid?(j, net)
      end
    end)
  end

  defp check_contradiction_triggered(net, node_id) do
    if MapSet.member?(net.contradiction_nodes, node_id) do
      node = Map.get(net.nodes, node_id)

      if node && node.label == :in do
        assumptions = find_supporting_assumptions(net, node_id)
        net.contradiction_handler.({:contradiction, net.world_id, node_id, assumptions})
      end
    end

    net
  end

  defp find_supporting_assumptions(net, node_id) do
    find_assumptions_recursive(net, [node_id], MapSet.new(), [])
  end

  defp find_assumptions_recursive(_net, [], _visited, assumptions) do
    assumptions
  end

  defp find_assumptions_recursive(net, [node_id | rest], visited, assumptions) do
    if MapSet.member?(visited, node_id) do
      find_assumptions_recursive(net, rest, visited, assumptions)
    else
      visited = MapSet.put(visited, node_id)
      node = Map.get(net.nodes, node_id)

      cond do
        node == nil ->
          find_assumptions_recursive(net, rest, visited, assumptions)

        node.node_type == :assumption and node.label == :in ->
          find_assumptions_recursive(net, rest, visited, [node_id | assumptions])

        node.node_type == :premise ->
          find_assumptions_recursive(net, rest, visited, assumptions)

        true ->
          case do_why_node(node_id, net) do
            {:ok, result} ->
              antecedent_ids =
                result.supporting_justifications
                |> Enum.flat_map(fn j -> j.in_list end)

              find_assumptions_recursive(net, antecedent_ids ++ rest, visited, assumptions)

            _ ->
              find_assumptions_recursive(net, rest, visited, assumptions)
          end
      end
    end
  end

  defp do_why_node(node_id, net) do
    case Map.get(net.nodes, node_id) do
      nil ->
        {:error, :not_found}

      node ->
        just_ids = Map.get(net.node_to_justifications, node_id, [])

        supporting =
          just_ids
          |> Enum.map(&Map.get(net.justifications, &1))
          |> Enum.filter(fn j -> j != nil and j.conclusion_id == node_id end)
          |> Enum.filter(fn j -> justification_valid?(j, net) end)

        result = %{
          node: node,
          supporting_justifications: supporting
        }

        {:ok, result}
    end
  end

  defp default_contradiction_handler({:contradiction, world_id, node_id, assumptions}) do
    Logger.warning("Contradiction detected",
      world_id: world_id,
      node_id: node_id,
      supporting_assumptions: assumptions
    )
  end

  defp generate_id do
    FourthWall.ID.generate()
  end
end
