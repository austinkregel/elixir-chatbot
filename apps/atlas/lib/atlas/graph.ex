defmodule Atlas.Graph do
  @moduledoc """
  High-level interface for graph operations using Apache AGE.

  Provides CRUD operations on graph nodes and edges, plus extraction
  functions designed to feed Brain's NLP pipeline and GNN models.

  ## Graph Schemas

  - `knowledge_graph` -- entity-relation triples for GNN reasoning
  - `user_graph` -- conversational memory and personalization
  - `semantic_graph` -- AMR-style semantic role representations

  ## Usage

      # Add a node
      Atlas.Graph.add_node("knowledge_graph", "Person", %{name: "John"})

      # Add an edge
      Atlas.Graph.add_edge("knowledge_graph", from_id, to_id, "KNOWS", %{since: "2024"})

      # Run a raw Cypher query
      Atlas.Graph.cypher("knowledge_graph", "MATCH (n:Person) RETURN n LIMIT 5")

      # Extract adjacency matrix for GNN input
      Atlas.Graph.to_adjacency("knowledge_graph")
  """

  alias Atlas.Graph.Cypher

  @graphs ~w(knowledge_graph user_graph semantic_graph conversation_graph epistemic_graph pos_graph)

  @doc "List all configured graph schema names."
  def graphs, do: @graphs

  @doc """
  Execute a raw Cypher query against a named graph.

  Wraps the query in AGE's SQL function and returns parsed results.
  """
  def cypher(graph_name, query, params \\ %{}) do
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
  Export hierarchical is-a relations for Poincare embedding training.
  """
  def poincare_data(graph) do
    cypher(graph, """
      MATCH (child)-[:is_a]->(parent)
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

  defp encode_value(v) when is_binary(v), do: "'#{String.replace(v, "'", "\\'")}'"
  defp encode_value(v) when is_number(v), do: to_string(v)
  defp encode_value(v) when is_boolean(v), do: to_string(v)
  defp encode_value(v) when is_atom(v), do: "'#{v}'"
  defp encode_value(v), do: "'#{inspect(v)}'"
end
