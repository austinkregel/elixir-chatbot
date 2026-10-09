defmodule Atlas.Graph do
  @moduledoc """
  High-level interface for graph operations using Apache AGE.

  Provides CRUD operations on graph nodes and edges, plus extraction
  functions designed to feed Brain's NLP pipeline and GNN models.

  ## Graphs

  A world is a scoped dataset, and the graphs that hold what a world knows
  and remembers belong to it. Each world has its own AGE graph of each of
  these kinds, named by `world_graph/2`:

  - `knowledge_graph` -- entities, events, beliefs, entity types
  - `user_graph` -- user preferences
  - `semantic_graph` -- consolidated memory: semantic facts and their episodes
  - `conversation_graph` -- conversations, messages, topics
  - `epistemic_graph` -- JTMS nodes, justifications, contradictions

  The expression-level graphs describe the language rather than any world's
  data, and are shared by every world under their own names:

  - `pos_graph` -- part-of-speech tags and tag transitions

  A per-world kind has no graph of its own name: passing `"knowledge_graph"`
  where a graph is expected raises, so nothing reads or writes a world's data
  without naming the world.

  ## Usage

      graph = Atlas.Graph.world_graph("knowledge_graph", world_id)

      # Add a node
      Atlas.Graph.add_node(graph, "Person", %{name: "John"})

      # Add an edge
      Atlas.Graph.add_edge(graph, from_id, to_id, "KNOWS", %{since: "2024"})

      # Run a raw Cypher query
      Atlas.Graph.cypher(graph, "MATCH (n:Person) RETURN n LIMIT 5")

      # Extract adjacency matrix for GNN input
      Atlas.Graph.to_adjacency(graph)
  """

  alias Atlas.Graph.Cypher

  @world_graph_kinds ~w(knowledge_graph user_graph semantic_graph conversation_graph epistemic_graph)
  @shared_graphs ~w(pos_graph)

  # A world's graph is named `<kind>__<world id>`. The kinds are fixed and none
  # contains the separator, so a name belongs to exactly one kind and world.
  @world_separator "__"

  # AGE's limits on a graph name: at most 63 bytes (it is also the name of the
  # graph's Postgres schema), starting with a letter or underscore, then
  # letters, digits, underscores, dots and hyphens, ending in a letter, digit
  # or underscore. A world graph name starts with its kind, so the world id
  # only has to fit the rest; it is used as is, never rewritten, so two worlds
  # can never share a graph.
  @max_graph_name_bytes 63
  @world_id_format ~r/\A[A-Za-z0-9_-]*[A-Za-z0-9_]\z/

  @doc "The kinds of graph each world has its own of."
  def world_graph_kinds, do: @world_graph_kinds

  @doc "The expression-level graphs every world shares, by name."
  def shared_graphs, do: @shared_graphs

  @doc """
  The name of a world's graph of the given kind.

  Raises `ArgumentError` for a kind that is not per world (a shared graph is
  used by its own name), for a missing world, and for a world id that cannot
  be part of an AGE graph name: anything but ASCII letters, digits,
  underscores and hyphens, ending in a hyphen, or too long for the name to fit
  in 63 bytes.
  """
  def world_graph(kind, world_id) when kind in @world_graph_kinds do
    name = kind <> @world_separator <> graph_world_id!(world_id)

    if byte_size(name) > @max_graph_name_bytes do
      raise ArgumentError,
            "world id #{inspect(world_id)} is too long for a graph name: #{inspect(name)} is " <>
              "#{byte_size(name)} bytes, AGE allows #{@max_graph_name_bytes}"
    end

    name
  end

  def world_graph(kind, _world_id) when kind in @shared_graphs do
    raise ArgumentError, "#{inspect(kind)} is shared by every world; use it by its own name"
  end

  def world_graph(kind, _world_id) do
    raise ArgumentError,
          "#{inspect(kind)} is not a graph kind; per-world kinds are #{inspect(@world_graph_kinds)}"
  end

  @doc "The names of all of a world's graphs, one per kind."
  def world_graphs(world_id), do: Enum.map(@world_graph_kinds, &world_graph(&1, world_id))

  defp graph_world_id!(world_id) when is_binary(world_id) and world_id != "" do
    if Regex.match?(@world_id_format, world_id) do
      world_id
    else
      raise ArgumentError,
            "world id #{inspect(world_id)} cannot name a graph: it must be ASCII letters, digits, " <>
              "underscores and hyphens, and not end in a hyphen"
    end
  end

  defp graph_world_id!(nil) do
    raise ArgumentError, "a world's graph needs the world; got no world id"
  end

  defp graph_world_id!(other) do
    raise ArgumentError, "a world's graph needs a world id, got: #{inspect(other)}"
  end

  @doc """
  Create every graph of a world that does not exist yet. Idempotent.

  Refuses while an unscoped graph of a per-world kind still exists: that is
  data from before graphs were per world, and it has to be moved into a
  world with `move_unscoped_graphs_to_world!/1` (`mix
  atlas.move_graphs_to_world`) before worlds get graphs beside it.
  """
  def ensure_world_graphs!(world_id) do
    names = world_graphs(world_id)
    refuse_unscoped_graphs!()

    Enum.each(names, fn name ->
      unless graph_exists?(name) do
        Atlas.Repo.query!("SELECT ag_catalog.create_graph($1)", [name])
      end
    end)

    names
  end

  @doc """
  Drop every graph of a world, with everything in it, for a world that is
  purged. A kind the world has no graph of is left alone, so dropping is
  idempotent. Returns the names of the graphs dropped.
  """
  def drop_world_graphs!(world_id) do
    world_id
    |> world_graphs()
    |> Enum.filter(&graph_exists?/1)
    |> Enum.map(fn name ->
      Atlas.Repo.query!("SELECT ag_catalog.drop_graph($1, true)", [name])
      name
    end)
  end

  @doc """
  Move the unscoped graph of each per-world kind (`knowledge_graph`, ...) into
  the given world, by renaming it to that world's graph of the kind. The data
  is not copied: the same graph changes its name.

  A kind with no unscoped graph is left alone. Raises, before renaming
  anything, when the world already has a graph of a kind that also has an
  unscoped one, because both would hold data and renaming cannot merge them.

  Returns `[{kind, :moved | :no_unscoped_graph}]`.
  """
  def move_unscoped_graphs_to_world!(world_id) do
    plan =
      Enum.map(@world_graph_kinds, fn kind ->
        target = world_graph(kind, world_id)
        {kind, target, graph_exists?(kind), graph_exists?(target)}
      end)

    conflicts = for {kind, target, true, true} <- plan, do: {kind, target}

    if conflicts != [] do
      raise ArgumentError,
            "cannot move unscoped graphs into world #{inspect(world_id)}: the world already has " <>
              Enum.map_join(conflicts, ", ", fn {kind, target} -> "#{target} (and #{kind} still exists)" end)
    end

    Enum.map(plan, fn
      {kind, target, true, false} ->
        Atlas.Repo.query!("SELECT ag_catalog.alter_graph($1, 'RENAME', $2)", [kind, target])
        {kind, :moved}

      {kind, _target, false, _} ->
        {kind, :no_unscoped_graph}
    end)
  end

  @doc "Whether an AGE graph of this name exists."
  def graph_exists?(name) when is_binary(name) do
    %{rows: [[count]]} =
      Atlas.Repo.query!("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = $1::name", [name])

    count > 0
  end

  @doc "The names of the AGE graphs that exist, sorted."
  def existing_graphs do
    %{rows: rows} = Atlas.Repo.query!("SELECT name::text FROM ag_catalog.ag_graph ORDER BY 1", [])
    Enum.map(rows, fn [name] -> name end)
  end

  @doc """
  Raise while an unscoped graph of a per-world kind (`knowledge_graph`, ...)
  still exists, naming the graphs and the task that moves them into a world.
  `ensure_world_graphs!/1` refuses with it, and `:world` checks it before it
  starts, so a database still holding them fails the start with this message.
  """
  def refuse_unscoped_graphs! do
    case Enum.filter(@world_graph_kinds, &graph_exists?/1) do
      [] ->
        :ok

      unscoped ->
        raise ArgumentError,
              "unscoped graphs #{inspect(unscoped)} hold data from before graphs were per world; " <>
                "move them into a world first: mix atlas.move_graphs_to_world --world default"
    end
  end

  @doc """
  Execute a raw Cypher query against a named graph.

  Wraps the query in AGE's SQL function and returns parsed results. A
  per-world kind's bare name raises: name the world's graph with
  `world_graph/2`.
  """
  def cypher(graph_name, query, params \\ %{})

  def cypher(graph_name, _query, _params) when graph_name in @world_graph_kinds do
    raise ArgumentError,
          "#{inspect(graph_name)} is per world; name the world's graph with Atlas.Graph.world_graph/2"
  end

  def cypher(graph_name, query, params) do
    Atlas.Telemetry.span(:graph_query, %{graph: graph_name}, fn ->
      Cypher.execute(Atlas.Repo, graph_name, query, params)
    end)
  end

  @doc """
  Add a labeled node to a graph with properties.

  Returns `{:ok, vertex}` or `{:error, reason}`.
  """
  def add_node(graph, label, properties \\ %{}) do
    props_str = encode_properties(properties)

    result = cypher(graph, "CREATE (n:#{label} #{props_str}) RETURN n")

    case result do
      {:ok, [[vertex]]} ->
        Atlas.Telemetry.emit_node_added(graph, label)
        {:ok, vertex}

      {:ok, rows} ->
        Atlas.Telemetry.emit_node_added(graph, label)
        {:ok, rows}

      error ->
        error
    end
  end

  @doc """
  Add a directed edge between two nodes identified by AGE internal IDs.

  Returns `{:ok, edge}` or `{:error, reason}`.
  """
  def add_edge(graph, from_id, to_id, rel_type, properties \\ %{}) do
    props_str = encode_properties(properties)

    result =
      cypher(graph, """
        MATCH (a), (b)
        WHERE id(a) = #{from_id} AND id(b) = #{to_id}
        CREATE (a)-[r:#{rel_type} #{props_str}]->(b)
        RETURN r
      """)

    case result do
      {:ok, [[edge]]} ->
        Atlas.Telemetry.emit_edge_added(graph, rel_type)
        {:ok, edge}

      {:ok, rows} ->
        Atlas.Telemetry.emit_edge_added(graph, rel_type)
        {:ok, rows}

      error ->
        error
    end
  end

  @doc """
  Count all nodes in a graph, optionally filtered by label.
  """
  def count_nodes(graph, label \\ nil) do
    query =
      if label do
        "MATCH (n:#{label}) RETURN count(n)"
      else
        "MATCH (n) RETURN count(n)"
      end

    case cypher(graph, query) do
      {:ok, [[count]]} -> {:ok, count}
      {:ok, _} -> {:ok, 0}
      error -> error
    end
  end

  @doc """
  Count all edges in a graph, optionally filtered by relationship type.
  """
  def count_edges(graph, rel_type \\ nil) do
    query =
      if rel_type do
        "MATCH ()-[r:#{rel_type}]->() RETURN count(r)"
      else
        "MATCH ()-[r]->() RETURN count(r)"
      end

    case cypher(graph, query) do
      {:ok, [[count]]} -> {:ok, count}
      {:ok, _} -> {:ok, 0}
      error -> error
    end
  end

  @doc """
  Find the shortest path between two nodes.
  """
  def shortest_path(graph, from_id, to_id) do
    cypher(graph, """
      MATCH p = shortestPath((a)-[*]-(b))
      WHERE id(a) = #{from_id} AND id(b) = #{to_id}
      RETURN p
    """)
  end

  @doc """
  Get all neighbors of a node within N hops.
  """
  def neighborhood(graph, node_id, hops \\ 2) do
    cypher(graph, """
      MATCH (a)-[*1..#{hops}]-(b)
      WHERE id(a) = #{node_id}
      RETURN DISTINCT b
    """)
  end

  @doc """
  Extract adjacency data for GNN input.

  Returns `{:ok, %{node_ids: [...], edges: [{from, to}, ...], features: %{id => props}}}`.
  This format is ready for conversion to Nx tensors for Graph Convolutional Networks.
  """
  def to_adjacency(graph, label \\ nil) do
    nodes_query =
      if label,
        do: "MATCH (n:#{label}) RETURN id(n), properties(n)",
        else: "MATCH (n) RETURN id(n), properties(n)"

    edges_query = "MATCH (a)-[r]->(b) RETURN id(a), id(b), type(r)"

    with {:ok, node_rows} <- cypher(graph, nodes_query),
         {:ok, edge_rows} <- cypher(graph, edges_query) do
      node_ids = Enum.map(node_rows, fn [id | _] -> id end)
      features = Map.new(node_rows, fn [id, props] -> {id, props} end)
      edges = Enum.map(edge_rows, fn [from, to | _] -> {from, to} end)

      {:ok, %{node_ids: node_ids, edges: edges, features: features}}
    end
  end

  @doc """
  Export (subject, relation, object) triples for KG-BERT-style embedding injection.
  """
  def to_triples(graph) do
    cypher(graph, """
      MATCH (a)-[r]->(b)
      RETURN properties(a), type(r), properties(b)
    """)
  end

  @doc """
  Export hierarchical IS_A relations for Poincare embedding training.
  """
  def poincare_data(graph) do
    cypher(graph, """
      MATCH (child)-[:#{Atlas.Graph.EdgeLabels.is_a()}]->(parent)
      RETURN properties(child).name, properties(parent).name
    """)
  end

  @doc """
  Retrieve a sentence's AMR subgraph for semantic role analysis.
  """
  def amr_subgraph(graph, event_id) do
    cypher(graph, """
      MATCH (e)-[r]->(arg)
      WHERE id(e) = #{event_id}
      RETURN e, type(r), arg
    """)
  end

  # Encode a map of properties into AGE's Cypher property syntax.
  defp encode_properties(props) when map_size(props) == 0, do: ""

  defp encode_properties(props) do
    pairs =
      Enum.map(props, fn {k, v} ->
        "#{k}: #{encode_value(v)}"
      end)

    "{" <> Enum.join(pairs, ", ") <> "}"
  end

  @doc "Encode an Elixir value for use in a Cypher property literal."
  def encode_value(v) when is_binary(v), do: "'#{String.replace(v, "'", "\\'")}'"
  def encode_value(v) when is_number(v), do: to_string(v)
  def encode_value(v) when is_boolean(v), do: to_string(v)
  def encode_value(v) when is_atom(v), do: "'#{v}'"
  def encode_value(v) when is_list(v), do: "[" <> Enum.map_join(v, ", ", &encode_value/1) <> "]"
  def encode_value(v), do: "'#{inspect(v)}'"
end
