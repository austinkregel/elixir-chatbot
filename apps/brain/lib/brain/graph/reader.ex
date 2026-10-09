defmodule Brain.Graph.Reader do
  @moduledoc """
  Reads from Atlas graphs to enrich response generation context.

  All functions have graceful fallbacks that return empty results when
  Atlas is unavailable or queries time out. In production this prevents
  response generation from blocking on graph queries. In tests, Atlas
  is required infrastructure and these functions use real graph data.

  ## Graphs Read From

  Each world has its own graph of these kinds (`Atlas.Graph.world_graph/2`).
  Every function reading one takes the world as the `:world_id` option and
  reads only that world's graph. The world is checked before any fallback
  applies, so a missing or malformed world raises instead of reading as an
  empty result:

  - `knowledge_graph` -- entity neighborhoods, relationship paths
  - `user_graph` -- user preferences
  - `semantic_graph` -- evidence chains
  - `conversation_graph` -- conversation topics, topic transitions
  - `epistemic_graph` -- belief justification chains

  The expression-level graph is shared by every world, so its readers take
  no world:

  - `pos_graph` -- POS tag patterns and transitions
  """

  alias Brain.AtlasIntegration
  alias Brain.Graph.ContextCache
  alias Atlas.Graph
  alias Atlas.Graph.EdgeLabels
  alias Atlas.Graph.Types.Vertex

  require Logger

  # ============================================================================
  # knowledge_graph: Entity Context
  # ============================================================================

  @doc """
  Look up each entity in the world's knowledge_graph and return its
  neighborhood.

  Returns a list of maps, one per entity:
    `%{entity: entity, neighbors: [%Vertex{}], node: %Vertex{} | nil}`

  ## Options
    - `:world_id` (required) - the world whose graph is read and whose lookups are cached
    - `:depth` - neighborhood depth (default: 2)
  """
  def entity_context(entities, opts) when is_list(entities) do
    {world_id, graph} = world_graph!(opts, "knowledge_graph", "entity_context")
    depth = Keyword.get(opts, :depth, 2)
    lookup_entity_contexts(entities, world_id, graph, depth)
  end

  defp lookup_entity_contexts(entities, world_id, graph, depth) do
    Enum.map(entities, fn entity ->
      entity_type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type") || "Entity"
      value = Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :text, "")

      label = normalize_label(entity_type)

      case ContextCache.get(world_id, "knowledge_graph", label, value) do
        {:ok, {node, neighbors}} ->
          %{entity: entity, neighbors: neighbors, node: node}

        :miss ->
          case fetch_entity_context(graph, label, value, depth) do
            {node, neighbors} when not is_nil(node) ->
              ContextCache.put(world_id, "knowledge_graph", label, value, {node, neighbors})
              %{entity: entity, neighbors: neighbors, node: node}

            # A miss is not cached. `fetch_entity_context/4` cannot tell "no
            # such entity" from a lookup that failed -- an unavailable graph,
            # a connection error -- so caching it records a failure as a fact,
            # for every later reader, until the process that cached it
            # remembers to purge. Re-reading a genuinely unknown entity costs
            # one query.
            _ ->
              %{entity: entity, neighbors: [], node: nil}
          end
      end
    end)
  rescue
    _ -> Enum.map(entities, fn e -> %{entity: e, neighbors: [], node: nil} end)
  end

  @doc """
  Look up a single entity name across multiple candidate labels in one batch,
  in the world's knowledge_graph.

  Issues a single Cypher query that matches any of the provided labels,
  then fetches the neighborhood once for the first match. Returns a map
  keyed by normalized label: `%{label => %{node: vertex, neighbors: [vertex]}}`.

  ## Options
    - `:world_id` (required) - the world whose graph is read and whose lookups are cached
    - `:depth` - neighborhood depth (default: 2)
  """
  def entity_context_multi_label(name, labels, opts) when is_binary(name) and is_list(labels) do
    {world_id, graph} = world_graph!(opts, "knowledge_graph", "entity_context_multi_label")
    depth = Keyword.get(opts, :depth, 2)
    lookup_multi_label(name, labels, world_id, graph, depth)
  end

  defp lookup_multi_label(name, labels, world_id, graph, depth) do
    escaped = String.replace(to_string(name), "'", "\\'")
    normalized_labels = Enum.map(labels, &normalize_label/1)

    label_or =
      normalized_labels
      |> Enum.map(fn l -> "(n:#{l})" end)
      |> Enum.join(" OR ")

    query = "MATCH (n) WHERE (#{label_or}) AND n.name = '#{escaped}' RETURN n"

    nodes =
      case Graph.cypher(graph, query) do
        {:ok, rows} when is_list(rows) ->
          rows
          |> Enum.flat_map(fn
            [%Vertex{} = v] -> [v]
            _ -> []
          end)

        _ ->
          []
      end

    case nodes do
      [] ->
        %{}

      [node | _] ->
        neighbors = get_neighbors(graph, node.id, depth)
        ContextCache.put(world_id, "knowledge_graph", hd(normalized_labels), name, {node, neighbors})

        Map.new(normalized_labels, fn label ->
          {label, %{node: node, neighbors: neighbors}}
        end)
    end
  rescue
    _ -> %{}
  end

  # Every world-scoped read names its world; there is no default. Returns the
  # world and its graph of the kind, raising for a missing or malformed world.
  defp world_graph!(opts, kind, function) do
    case Keyword.fetch(opts, :world_id) do
      {:ok, world_id} when is_binary(world_id) and world_id != "" ->
        {world_id, Graph.world_graph(kind, world_id)}

      {:ok, other} ->
        raise ArgumentError, "Reader.#{function} option :world_id must be a world id, got: #{inspect(other)}"

      :error ->
        raise ArgumentError, "Reader.#{function} needs :world_id, the world whose graph is read"
    end
  end

  defp fetch_entity_context(graph, label, value, depth) do
    case AtlasIntegration.find_node(graph, label, value) do
      {:ok, node} ->
        neighbors = get_neighbors(graph, node.id, depth)
        {node, neighbors}

      :not_found ->
        {nil, []}

      # Still degrades — response generation must not block on the graph — but
      # says so. "The graph is unreachable" and "we have never heard of this
      # entity" produce the same empty result here and the same hedged reply
      # downstream, so the difference has to be visible somewhere.
      {:error, reason} ->
        Logger.warning("Reader: lookup of #{label} #{inspect(value)} failed: #{inspect(reason)}")
        {nil, []}
    end
  rescue
    e ->
      Logger.warning("Reader: lookup of #{label} #{inspect(value)} raised: #{Exception.message(e)}")
      {nil, []}
  end

  @doc """
  Find a relationship path between two entities in the world's
  knowledge_graph.

  Uses a simple 1-3 hop search since Apache AGE doesn't support
  the Cypher `shortestPath()` function.

  Returns `{:ok, path_nodes}` or `{:error, :not_found}`.

  ## Options
    - `:world_id` (required) - the world whose graph is read
  """
  def relationship_path(entity_a, entity_b, opts) do
    {_world_id, graph} = world_graph!(opts, "knowledge_graph", "relationship_path")
    find_relationship_path(graph, entity_a, entity_b)
  end

  defp find_relationship_path(graph, entity_a, entity_b) do
    name_a = Map.get(entity_a, :value) || Map.get(entity_a, "value", "")
    name_b = Map.get(entity_b, :value) || Map.get(entity_b, "value", "")
    label_a = normalize_label(Map.get(entity_a, :entity_type, "Entity"))
    label_b = normalize_label(Map.get(entity_b, :entity_type, "Entity"))

    with {:ok, node_a} <- AtlasIntegration.find_node(graph, label_a, name_a),
         {:ok, node_b} <- AtlasIntegration.find_node(graph, label_b, name_b) do
      find_path(graph, node_a, node_b)
    else
      :not_found ->
        {:error, :not_found}

      {:error, reason} ->
        Logger.warning("Reader: relationship path lookup failed: #{inspect(reason)}")
        {:error, :not_found}
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
  Expand a query with related concept names from the world's
  knowledge_graph.

  Returns `{query_text, [related_name]}`.

  ## Options
    - `:world_id` (required) - the world whose graph is read and whose entity lookups are cached
  """
  def expand_query(query_text, entities, opts) when is_list(entities) do
    {world_id, _graph} = world_graph!(opts, "knowledge_graph", "expand_query")
    expand_query_in_world(query_text, entities, world_id)
  end

  defp expand_query_in_world(query_text, entities, world_id) do
    graph_names =
      entities
      |> Enum.flat_map(fn e ->
        case entity_context([e], depth: 1, world_id: world_id) do
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
    query_text
    |> Brain.ML.Tokenizer.tokenize_normalized(min_length: 3)
    |> Enum.flat_map(fn token ->
      Brain.Lexicon.synonyms(token)
      |> Enum.take(2)
    end)
    |> Enum.uniq()
  end

  # ============================================================================
  # user_graph: User Preferences
  # ============================================================================

  @doc """
  Get all preference edges for a user from the world's user_graph.

  Returns a list of `%{topic: name, rel_type: type, properties: map}`.

  ## Options
    - `:world_id` (required) - the world whose graph is read
  """
  def user_preferences(user_id, opts) do
    {_world_id, graph} = world_graph!(opts, "user_graph", "user_preferences")
    read_user_preferences(graph, user_id)
  end

  defp read_user_preferences(graph, user_id) do
    escaped = String.replace(to_string(user_id), "'", "\\'")
    # AGE doesn't support type(r) or properties(r) as Cypher functions.
    # Query specific relationship types individually.
    rel_types = [EdgeLabels.likes(), EdgeLabels.wants(), EdgeLabels.interested_in(),
                  EdgeLabels.dislikes(), EdgeLabels.asked_about(), EdgeLabels.needs()]

    Enum.flat_map(rel_types, fn rel_type ->
      query = """
      MATCH (u:User)-[r:#{rel_type}]->(t:Topic)
      WHERE u.id = '#{escaped}' OR u.name = '#{escaped}'
      RETURN t, r
      """

      case Graph.cypher(graph, query) do
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
  Get the episode evidence chain for a semantic fact in the world's
  semantic_graph.

  Returns a list of episode vertex maps.

  ## Options
    - `:world_id` (required) - the world whose graph is read
  """
  def evidence_chain(semantic_fact_name, opts) do
    {_world_id, graph} = world_graph!(opts, "semantic_graph", "evidence_chain")
    read_evidence_chain(graph, semantic_fact_name)
  end

  defp read_evidence_chain(graph, semantic_fact_name) do
    escaped = String.replace(to_string(semantic_fact_name), "'", "\\'")
    query = """
    MATCH (e:Episode)-[:#{EdgeLabels.evidence_for()}]->(sf:SemanticFact)
    WHERE sf.name = '#{escaped}'
    RETURN e
    """

    case Graph.cypher(graph, query) do
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

  @doc """
  Get all topics discussed in a conversation, from the world's
  conversation_graph.

  ## Options
    - `:world_id` (required) - the world whose graph is read
  """
  def conversation_topics(conversation_id, opts) do
    {_world_id, graph} = world_graph!(opts, "conversation_graph", "conversation_topics")
    read_conversation_topics(graph, conversation_id)
  end

  defp read_conversation_topics(graph, conversation_id) do
    escaped = String.replace(to_string(conversation_id), "'", "\\'")
    query = """
    MATCH (c:Conversation)-[:#{EdgeLabels.contains()}]->(m:Message)-[:#{EdgeLabels.has_topic()}]->(t:Topic)
    WHERE c.name = '#{escaped}'
    RETURN DISTINCT t
    """

    case Graph.cypher(graph, query) do
      {:ok, rows} ->
        Enum.map(rows, fn [%Vertex{properties: props}] -> Map.get(props, "name", "") end)
        |> Enum.reject(&(&1 == ""))

      _ ->
        []
    end
  rescue
    _ -> []
  end

  @doc """
  Get the most common topic-to-topic transitions in the world's
  conversation_graph.

  ## Options
    - `:world_id` (required) - the world whose graph is read
    - `:limit` - how many transitions (default: 20)
  """
  def topic_transitions(opts) do
    {_world_id, graph} = world_graph!(opts, "conversation_graph", "topic_transitions")
    read_topic_transitions(graph, Keyword.get(opts, :limit, 20))
  end

  defp read_topic_transitions(graph, limit) do
    query = """
    MATCH (a:Topic)-[r:#{EdgeLabels.topic_transition()}]->(b:Topic)
    RETURN a, b, r
    """

    case Graph.cypher(graph, query) do
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

  @doc """
  Get the last N messages in a conversation with their topics, from the
  world's conversation_graph.

  ## Options
    - `:world_id` (required) - the world whose graph is read
    - `:n` - how many messages (default: 5)
  """
  def recent_context(conversation_id, opts) do
    {_world_id, graph} = world_graph!(opts, "conversation_graph", "recent_context")
    read_recent_context(graph, conversation_id, Keyword.get(opts, :n, 5))
  end

  defp read_recent_context(graph, conversation_id, n) do
    escaped = String.replace(to_string(conversation_id), "'", "\\'")
    query = """
    MATCH (c:Conversation)-[:#{EdgeLabels.contains()}]->(m:Message)
    WHERE c.name = '#{escaped}'
    OPTIONAL MATCH (m)-[:#{EdgeLabels.has_topic()}]->(t:Topic)
    RETURN m, t
    ORDER BY id(m) DESC
    LIMIT #{n}
    """

    case Graph.cypher(graph, query) do
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
  Get the full justification chain for a JTMS node in the world's
  epistemic_graph.

  Returns the node, its supporting justifications, and their required nodes.

  ## Options
    - `:world_id` (required) - the world whose graph is read
  """
  def belief_justification_chain(node_name, opts) do
    {_world_id, graph} = world_graph!(opts, "epistemic_graph", "belief_justification_chain")
    read_belief_justification_chain(graph, node_name)
  end

  defp read_belief_justification_chain(graph, node_name) do
    escaped = String.replace(to_string(node_name), "'", "\\'")
    query = """
    MATCH (j:Justification)-[:#{EdgeLabels.supports()}]->(n:JTMSNode)
    WHERE n.name = '#{escaped}'
    OPTIONAL MATCH (j)-[:#{EdgeLabels.requires_in()}]->(req:JTMSNode)
    RETURN n, j, collect(req)
    """

    case Graph.cypher(graph, query) do
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
  Get what would break if an assumption was retracted, from the world's
  epistemic_graph.

  Returns nodes that depend on this assumption through justification chains.

  ## Options
    - `:world_id` (required) - the world whose graph is read
  """
  def assumption_consequences(node_name, opts) do
    {_world_id, graph} = world_graph!(opts, "epistemic_graph", "assumption_consequences")
    read_assumption_consequences(graph, node_name)
  end

  defp read_assumption_consequences(graph, node_name) do
    escaped = String.replace(to_string(node_name), "'", "\\'")
    query = """
    MATCH (j:Justification)-[:#{EdgeLabels.requires_in()}]->(a:JTMSNode)
    WHERE a.name = '#{escaped}'
    MATCH (j)-[:#{EdgeLabels.supports()}]->(derived:JTMSNode)
    RETURN DISTINCT derived
    """

    case Graph.cypher(graph, query) do
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

  @doc "Get what POS tags commonly follow a given tag, with frequencies, from the shared pos_graph."
  def tag_transitions(from_tag) do
    escaped = String.replace(to_string(from_tag), "'", "\\'")
    query = """
    MATCH (a:POSTag)-[r:#{EdgeLabels.followed_by()}]->(b:POSTag)
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

  @doc "Find tokens that have multiple POS tags (ambiguous words), in the shared pos_graph."
  def ambiguous_tokens(min_tag_count \\ 2) do
    query = """
    MATCH (t:Token)-[r:#{EdgeLabels.has_tag()}]->(tag:POSTag)
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

  @doc "Find matching POS tag patterns in the shared pos_graph."
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
