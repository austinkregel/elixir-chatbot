defmodule Brain.Graph.Reader do
  @moduledoc """
  Reads from Atlas graphs to enrich response generation context.

  All functions have graceful fallbacks that return empty results when
  Atlas is unavailable or queries time out. In production this prevents
  response generation from blocking on graph queries. In tests, Atlas
  is required infrastructure and these functions use real graph data.

  ## Graphs Read From

  - `knowledge_graph` -- entity neighborhoods, relationship paths
  - `user_graph` -- user preferences
  - `semantic_graph` -- evidence chains
  - `conversation_graph` -- conversation topics, topic transitions
  - `epistemic_graph` -- belief justification chains
  - `pos_graph` -- POS tag patterns and transitions
  """

  alias Brain.AtlasIntegration
  alias Atlas.Graph
  alias Atlas.Graph.Types.Vertex

  require Logger

  # ============================================================================
  # knowledge_graph: Entity Context
  # ============================================================================

  @doc """
  Look up each entity in knowledge_graph and return its neighborhood.

  Returns a list of maps, one per entity:
    `%{entity: entity, neighbors: [%Vertex{}], node: %Vertex{} | nil}`
  """
  def entity_context(entities, opts \\ []) when is_list(entities) do
    depth = Keyword.get(opts, :depth, 2)

    Enum.map(entities, fn entity ->
      entity_type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type") || "Entity"
      value = Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :text, "")

      label = normalize_label(entity_type)

      case AtlasIntegration.find_node("knowledge_graph", label, value) do
        {:ok, node} ->
          neighbors = get_neighbors("knowledge_graph", node.id, depth)
          %{entity: entity, neighbors: neighbors, node: node}

        _ ->
          %{entity: entity, neighbors: [], node: nil}
      end
    end)
  rescue
    _ -> Enum.map(entities, fn e -> %{entity: e, neighbors: [], node: nil} end)
  end

  @doc """
  Find a relationship path between two entities in knowledge_graph.

  Uses a simple 1-3 hop search since Apache AGE doesn't support
  the Cypher `shortestPath()` function.

  Returns `{:ok, path_nodes}` or `{:error, :not_found}`.
  """
  def relationship_path(entity_a, entity_b) do
    name_a = Map.get(entity_a, :value) || Map.get(entity_a, "value", "")
    name_b = Map.get(entity_b, :value) || Map.get(entity_b, "value", "")
    label_a = normalize_label(Map.get(entity_a, :entity_type, "Entity"))
    label_b = normalize_label(Map.get(entity_b, :entity_type, "Entity"))

    with {:ok, node_a} <- AtlasIntegration.find_node("knowledge_graph", label_a, name_a),
         {:ok, node_b} <- AtlasIntegration.find_node("knowledge_graph", label_b, name_b) do
      find_path("knowledge_graph", node_a, node_b)
    else
      _ -> {:error, :not_found}
    end
  rescue
    _ -> {:error, :not_found}
  end

  defp find_path(graph, node_a, node_b) do
    query = """
    MATCH (a)-[r]->(b)
    WHERE id(a) = #{node_a.id} AND id(b) = #{node_b.id}
    RETURN a, r, b
    """

    case Graph.cypher(graph, query) do
      {:ok, [row | _]} ->
        {:ok, row}

      _ ->
        query_reverse = """
        MATCH (b)-[r]->(a)
        WHERE id(a) = #{node_a.id} AND id(b) = #{node_b.id}
        RETURN b, r, a
        """

        case Graph.cypher(graph, query_reverse) do
          {:ok, [row | _]} -> {:ok, row}
          _ -> {:error, :not_found}
        end
    end
  rescue
    _ -> {:error, :not_found}
  end

  @doc """
  Expand a query with graph-derived related concept names.

  Returns `{query_text, [related_name]}`.
  """
  def expand_query(query_text, entities) when is_list(entities) do
    graph_names =
      entities
      |> Enum.flat_map(fn e ->
        case entity_context([e], depth: 1) do
          [%{neighbors: neighbors}] when neighbors != [] ->
            neighbors
            |> Enum.map(fn v -> Map.get(v.properties, "name", "") end)
            |> Enum.reject(&(&1 == ""))

          _ ->
            []
        end
      end)
      |> Enum.uniq()

    synonym_names = lexicon_query_synonyms(query_text)

    {query_text, Enum.uniq(graph_names ++ synonym_names)}
  rescue
    _ -> {query_text, []}
  end

  defp lexicon_query_synonyms(query_text) do
    if Process.whereis(Brain.ML.Lexicon) do
      query_text
      |> Brain.ML.Tokenizer.tokenize_normalized(min_length: 3)
      |> Enum.flat_map(fn token ->
        Brain.ML.Lexicon.synonyms(token)
        |> Enum.take(2)
      end)
      |> Enum.uniq()
    else
      []
    end
  end

  # ============================================================================
  # user_graph: User Preferences
  # ============================================================================

  @doc """
  Get all preference edges for a user from user_graph.

  Returns a list of `%{topic: name, rel_type: type, properties: map}`.
  """
  def user_preferences(user_id) do
    escaped = String.replace(to_string(user_id), "'", "\\'")
    # AGE doesn't support type(r) or properties(r) as Cypher functions.
    # Query specific relationship types individually.
    rel_types = ["LIKES", "WANTS", "INTERESTED_IN", "DISLIKES", "ASKED_ABOUT", "NEEDS"]

    Enum.flat_map(rel_types, fn rel_type ->
      query = """
      MATCH (u:User)-[r:#{rel_type}]->(t:Topic)
      WHERE u.id = '#{escaped}' OR u.name = '#{escaped}'
      RETURN t, r
      """

      case Graph.cypher("user_graph", query) do
        {:ok, rows} when is_list(rows) ->
          Enum.map(rows, fn
            [%Vertex{properties: props}, %Atlas.Graph.Types.Edge{properties: r_props}] ->
              %{
                topic: Map.get(props, "name", ""),
                rel_type: rel_type,
                properties: r_props || %{}
              }

            _ ->
              nil
          end)
          |> Enum.reject(&is_nil/1)

        _ ->
          []
      end
    end)
  rescue
    _ -> []
  end

  # ============================================================================
  # semantic_graph: Evidence Chains
  # ============================================================================

  @doc """
  Get the episode evidence chain for a semantic fact.

  Returns a list of episode vertex maps.
  """
  def evidence_chain(semantic_fact_name) do
    escaped = String.replace(to_string(semantic_fact_name), "'", "\\'")
    query = """
    MATCH (e:Episode)-[:EVIDENCE_FOR]->(sf:SemanticFact)
    WHERE sf.name = '#{escaped}'
    RETURN e
    """

    case Graph.cypher("semantic_graph", query) do
      {:ok, rows} ->
        Enum.map(rows, fn [%Vertex{} = v] -> v; _ -> nil end)
        |> Enum.reject(&is_nil/1)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  # ============================================================================
  # conversation_graph: Conversations + Topics
  # ============================================================================

  @doc "Get all topics discussed in a conversation."
  def conversation_topics(conversation_id) do
    escaped = String.replace(to_string(conversation_id), "'", "\\'")
    query = """
    MATCH (c:Conversation)-[:CONTAINS]->(m:Message)-[:HAS_TOPIC]->(t:Topic)
    WHERE c.name = '#{escaped}'
    RETURN DISTINCT t
    """

    case Graph.cypher("conversation_graph", query) do
      {:ok, rows} ->
        Enum.map(rows, fn [%Vertex{properties: props}] -> Map.get(props, "name", "") end)
        |> Enum.reject(&(&1 == ""))

      _ ->
        []
    end
  rescue
    _ -> []
  end

  @doc "Get the most common topic-to-topic transitions."
  def topic_transitions(limit \\ 20) do
    query = """
    MATCH (a:Topic)-[r:TOPIC_TRANSITION]->(b:Topic)
    RETURN a, b, r
    """

    case Graph.cypher("conversation_graph", query) do
      {:ok, rows} ->
        rows
        |> Enum.map(fn
          [%Vertex{properties: a_props}, %Vertex{properties: b_props}, %Atlas.Graph.Types.Edge{properties: r_props}] ->
            %{
              from: Map.get(a_props, "name", ""),
              to: Map.get(b_props, "name", ""),
              count: Map.get(r_props, "count", 1)
            }

          _ ->
            nil
        end)
        |> Enum.reject(&is_nil/1)
        |> Enum.sort_by(& &1.count, :desc)
        |> Enum.take(limit)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  @doc "Get the last N messages in a conversation with their topics."
  def recent_context(conversation_id, n \\ 5) do
    escaped = String.replace(to_string(conversation_id), "'", "\\'")
    query = """
    MATCH (c:Conversation)-[:CONTAINS]->(m:Message)
    WHERE c.name = '#{escaped}'
    OPTIONAL MATCH (m)-[:HAS_TOPIC]->(t:Topic)
    RETURN m, t
    ORDER BY id(m) DESC
    LIMIT #{n}
    """

    case Graph.cypher("conversation_graph", query) do
      {:ok, rows} ->
        Enum.map(rows, fn
          [%Vertex{} = msg, %Vertex{properties: tp}] ->
            %{message: msg.properties, topic: Map.get(tp, "name")}

          [%Vertex{} = msg, _] ->
            %{message: msg.properties, topic: nil}

          _ ->
            nil
        end)
        |> Enum.reject(&is_nil/1)
        |> Enum.reverse()

      _ ->
        []
    end
  rescue
    _ -> []
  end

  # ============================================================================
  # epistemic_graph: Belief Justification Chains
  # ============================================================================

  @doc """
  Get the full justification chain for a JTMS node.

  Returns the node, its supporting justifications, and their required nodes.
  """
  def belief_justification_chain(node_name) do
    escaped = String.replace(to_string(node_name), "'", "\\'")
    query = """
    MATCH (j:Justification)-[:SUPPORTS]->(n:JTMSNode)
    WHERE n.name = '#{escaped}'
    OPTIONAL MATCH (j)-[:REQUIRES_IN]->(req:JTMSNode)
    RETURN n, j, collect(req)
    """

    case Graph.cypher("epistemic_graph", query) do
      {:ok, rows} ->
        Enum.map(rows, fn
          [%Vertex{} = node, %Vertex{} = just, requirements] ->
            reqs = if is_list(requirements), do: Enum.filter(requirements, &match?(%Vertex{}, &1)), else: []
            %{node: node, justification: just, requirements: reqs}

          _ ->
            nil
        end)
        |> Enum.reject(&is_nil/1)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  @doc """
  Get what would break if an assumption was retracted.

  Returns nodes that depend on this assumption through justification chains.
  """
  def assumption_consequences(node_name) do
    escaped = String.replace(to_string(node_name), "'", "\\'")
    query = """
    MATCH (j:Justification)-[:REQUIRES_IN]->(a:JTMSNode)
    WHERE a.name = '#{escaped}'
    MATCH (j)-[:SUPPORTS]->(derived:JTMSNode)
    RETURN DISTINCT derived
    """

    case Graph.cypher("epistemic_graph", query) do
      {:ok, rows} ->
        Enum.map(rows, fn [%Vertex{} = v] -> v; _ -> nil end)
        |> Enum.reject(&is_nil/1)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  # ============================================================================
  # pos_graph: POS Patterns
  # ============================================================================

  @doc "Get what POS tags commonly follow a given tag, with frequencies."
  def tag_transitions(from_tag) do
    escaped = String.replace(to_string(from_tag), "'", "\\'")
    query = """
    MATCH (a:POSTag)-[r:FOLLOWED_BY]->(b:POSTag)
    WHERE a.name = '#{escaped}'
    RETURN b, r
    """

    case Graph.cypher("pos_graph", query) do
      {:ok, rows} ->
        rows
        |> Enum.map(fn
          [%Vertex{properties: b_props}, %Atlas.Graph.Types.Edge{properties: r_props}] ->
            %{to_tag: Map.get(b_props, "name", ""), frequency: Map.get(r_props, "frequency", 0)}

          _ ->
            nil
        end)
        |> Enum.reject(&is_nil/1)
        |> Enum.sort_by(& &1.frequency, :desc)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  @doc "Find tokens that have multiple POS tags (ambiguous words)."
  def ambiguous_tokens(min_tag_count \\ 2) do
    query = """
    MATCH (t:Token)-[r:HAS_TAG]->(tag:POSTag)
    RETURN t, tag
    """

    case Graph.cypher("pos_graph", query) do
      {:ok, rows} ->
        rows
        |> Enum.map(fn
          [%Vertex{properties: t_props}, %Vertex{properties: tag_props}] ->
            {Map.get(t_props, "name", ""), Map.get(tag_props, "name", "")}

          _ ->
            nil
        end)
        |> Enum.reject(&is_nil/1)
        |> Enum.group_by(fn {token, _} -> token end, fn {_, tag} -> tag end)
        |> Enum.filter(fn {_token, tags} -> length(tags) >= min_tag_count end)
        |> Enum.map(fn {token, tags} ->
          %{token: token, tags: Enum.uniq(tags), tag_count: length(Enum.uniq(tags))}
        end)
        |> Enum.sort_by(& &1.tag_count, :desc)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  @doc "Find matching POS tag patterns."
  def pos_patterns(tag_sequence) when is_list(tag_sequence) do
    pattern_name = Enum.join(tag_sequence, "_")
    escaped = String.replace(pattern_name, "'", "\\'")
    query = """
    MATCH (p:Pattern)
    WHERE p.name = '#{escaped}'
    RETURN p
    """

    case Graph.cypher("pos_graph", query) do
      {:ok, [[%Vertex{} = v] | _]} -> [v]
      _ -> []
    end
  rescue
    _ -> []
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp get_neighbors(graph, node_id, depth) do
    case Graph.neighborhood(graph, node_id, depth) do
      {:ok, rows} ->
        rows
        |> List.flatten()
        |> Enum.filter(&match?(%Vertex{}, &1))
        |> Enum.reject(&(&1.id == node_id))
        |> Enum.uniq_by(& &1.id)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp normalize_label(type) when is_binary(type) do
    type
    |> String.replace("-", "_")
    |> String.split("_")
    |> Enum.map(&String.capitalize/1)
    |> Enum.join("")
  end

  defp normalize_label(type) when is_atom(type), do: normalize_label(to_string(type))
  defp normalize_label(_), do: "Entity"
end
