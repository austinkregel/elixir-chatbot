defmodule Brain.Graph.Writer do
  @moduledoc """
  Writes structured data from the analysis pipeline into Atlas graphs.

  All write operations are asynchronous via `AtlasIntegration.async/1`
  to avoid blocking the pipeline. Each public function returns `:ok`
  immediately.

  ## Graphs Written To

  - `knowledge_graph` -- entities, events, beliefs
  - `user_graph` -- user preferences
  - `semantic_graph` -- semantic facts, episode clusters
  - `conversation_graph` -- conversations, messages, topics
  - `epistemic_graph` -- JTMS nodes, justifications, contradictions
  - `pos_graph` -- POS tag patterns and transitions
  """

  alias Brain.AtlasIntegration
  alias Atlas.Graph.EdgeLabels
  require Logger

  # ============================================================================
  # knowledge_graph: Entities
  # ============================================================================

  @doc """
  Write extracted entities as nodes in knowledge_graph.

  Creates one node per entity (deduped by label+name). When multiple
  entities appear in the same call, creates co_occurs_with edges.
  """
  def write_entities(entities) when is_list(entities) do
    AtlasIntegration.async(fn ->
      nodes =
        entities
        |> Enum.reject(fn e -> entity_value(e) in [nil, ""] end)
        |> Enum.map(fn entity ->
          label = normalize_label(entity_type(entity))
          name = entity_value(entity)
          props = %{name: name, source: "entity_extraction"}

          case AtlasIntegration.ensure_node("knowledge_graph", label, props) do
            {:ok, vertex} -> {entity, vertex}
            _ -> nil
          end
        end)
        |> Enum.reject(&is_nil/1)

      if length(nodes) > 1 do
        write_co_occurrences(nodes)
      end
    end)
  end

  def write_entities(_), do: :ok

  # ============================================================================
  # knowledge_graph: Events
  # ============================================================================

  @doc """
  Write extracted events as subgraphs in knowledge_graph.

  Each event creates an Event node with edges to actor and object.
  """
  def write_events(events) when is_list(events) do
    AtlasIntegration.async(fn ->
      Enum.each(events, &write_single_event/1)
    end)
  end

  def write_events(_), do: :ok

  # ============================================================================
  # knowledge_graph + user_graph: Beliefs
  # ============================================================================

  @doc """
  Write a belief as a node in knowledge_graph.

  For user preference predicates (:likes, :wants, :interested_in),
  also creates edges in user_graph.
  """
  def write_belief(belief) do
    AtlasIntegration.async(fn ->
      subject = to_string(Map.get(belief, :subject, ""))
      predicate = to_string(Map.get(belief, :predicate, ""))
      object = to_string(Map.get(belief, :object, ""))
      confidence = Map.get(belief, :confidence, 0.5)

      props = %{
        name: "#{subject}_#{predicate}_#{object}" |> String.slice(0, 100),
        subject: subject,
        predicate: predicate,
        object: object,
        confidence: confidence
      }

      case Atlas.Graph.add_node("knowledge_graph", "Belief", props) do
        {:ok, vertex} ->
          belief_id = Map.get(belief, :id)

          if belief_id && function_exported?(Brain.Epistemic.BeliefStore, :link_to_node, 2) do
            try do
              Brain.Epistemic.BeliefStore.link_to_node(belief_id, vertex.id)
            rescue
              _ -> :ok
            end
          end

          maybe_write_user_preference(belief, confidence)

        _ ->
          :ok
      end
    end)
  end

  # ============================================================================
  # user_graph: User Preferences
  # ============================================================================

  @doc """
  Write a user preference edge in user_graph.
  """
  def write_user_preference(user_id, predicate, object, confidence \\ 0.5) do
    AtlasIntegration.async(fn ->
      {:ok, user_node} = AtlasIntegration.ensure_node("user_graph", "User", %{name: to_string(user_id), id: to_string(user_id)})
      {:ok, topic_node} = AtlasIntegration.ensure_node("user_graph", "Topic", %{name: to_string(object)})

      rel_type = preference_rel_type(predicate)
      AtlasIntegration.find_or_create_edge("user_graph", user_node.id, topic_node.id, rel_type, %{confidence: confidence})
    end)
  end

  # ============================================================================
  # semantic_graph: Semantic Clusters
  # ============================================================================

  @doc """
  Write a semantic fact and its episode evidence as a subgraph in semantic_graph.
  """
  def write_semantic_cluster(semantic_fact, episode_ids) when is_list(episode_ids) do
    AtlasIntegration.async(fn ->
      representation = Map.get(semantic_fact, :representation, "")

      props = %{
        name: "sf_#{:crypto.strong_rand_bytes(4) |> Base.encode16(case: :lower)}",
        representation: representation,
        evidence_count: length(episode_ids)
      }

      case Atlas.Graph.add_node("semantic_graph", "SemanticFact", props) do
        {:ok, fact_node} ->
          Enum.each(episode_ids, fn ep_id ->
            {:ok, ep_node} = AtlasIntegration.ensure_node("semantic_graph", "Episode", %{name: to_string(ep_id), id: to_string(ep_id)})
            Atlas.Graph.add_edge("semantic_graph", ep_node.id, fact_node.id, EdgeLabels.evidence_for())
          end)

        _ ->
          :ok
      end
    end)
  end

  def write_semantic_cluster(_, _), do: :ok

  # ============================================================================
  # conversation_graph: Conversations + Messages
  # ============================================================================

  @doc "Write a new conversation node."
  def write_conversation(conversation) do
    AtlasIntegration.async(fn ->
      conv_id = Map.get(conversation, :id) || Map.get(conversation, "id", "")
      world_id = Map.get(conversation, :world_id) || Map.get(conversation, "world_id", "default")

      Atlas.Graph.add_node("conversation_graph", "Conversation", %{
        name: to_string(conv_id),
        world_id: to_string(world_id),
        created_at: System.system_time(:millisecond)
      })
    end)
  end

  @doc """
  Write a message node linked to its conversation.

  Creates CONTAINS edge from conversation, FOLLOWS edge from previous message,
  and HAS_TOPIC edge if an intent was classified.
  """
  def write_message(conversation_id, message, analysis \\ nil) do
    AtlasIntegration.async(fn ->
      msg_id = Map.get(message, :id) || Map.get(message, "id", "msg_#{System.unique_integer([:positive])}")
      role = Map.get(message, :role) || Map.get(message, "role", "user")
      content = Map.get(message, :content) || Map.get(message, "content", "")

      props = %{
        name: to_string(msg_id),
        role: to_string(role),
        content: String.slice(to_string(content), 0, 500)
      }

      case Atlas.Graph.add_node("conversation_graph", "Message", props) do
        {:ok, msg_node} ->
          link_message_to_conversation(conversation_id, msg_node)
          link_message_to_previous(conversation_id, msg_node)
          link_message_to_topic(msg_node, analysis)

        _ ->
          :ok
      end
    end)
  end

  # ============================================================================
  # epistemic_graph: JTMS Nodes + Justifications
  # ============================================================================

  @doc "Write a JTMS node to epistemic_graph."
  def write_jtms_node(node) do
    AtlasIntegration.async(fn ->
      node_id = Map.get(node, :id, "")
      datum = Map.get(node, :datum, "")
      node_type = Map.get(node, :node_type, :derived)
      label = Map.get(node, :label, :out)

      Atlas.Graph.add_node("epistemic_graph", "JTMSNode", %{
        name: to_string(node_id),
        datum: inspect(datum),
        node_type: to_string(node_type),
        label: to_string(label)
      })
    end)
  end

  @doc "Write a JTMS justification with SUPPORTS and REQUIRES edges."
  def write_justification(justification) do
    AtlasIntegration.async(fn ->
      just_id = Map.get(justification, :id, "")
      informant = Map.get(justification, :informant, "")
      conclusion_id = Map.get(justification, :conclusion_id, "")
      in_list = Map.get(justification, :in_list, [])
      out_list = Map.get(justification, :out_list, [])

      props = %{name: to_string(just_id), informant: to_string(informant)}

      case Atlas.Graph.add_node("epistemic_graph", "Justification", props) do
        {:ok, just_node} ->
          link_justification_to_conclusion(just_node, conclusion_id)
          link_justification_requirements(just_node, in_list, EdgeLabels.requires_in())
          link_justification_requirements(just_node, out_list, EdgeLabels.requires_out())

        _ ->
          :ok
      end
    end)
  end

  @doc "Write CONTRADICTS edges between contradiction-registered nodes."
  def write_contradiction(node_ids) when is_list(node_ids) do
    AtlasIntegration.async(fn ->
      graph_nodes =
        Enum.map(node_ids, fn nid ->
          case AtlasIntegration.find_node("epistemic_graph", "JTMSNode", to_string(nid)) do
            {:ok, v} -> v
            _ -> nil
          end
        end)
        |> Enum.reject(&is_nil/1)

      for a <- graph_nodes, b <- graph_nodes, a.id != b.id do
        AtlasIntegration.find_or_create_edge("epistemic_graph", a.id, b.id, EdgeLabels.contradicts())
      end
    end)
  end

  def write_contradiction(_), do: :ok

  @doc "Update the label property of a JTMS node in epistemic_graph."
  def update_jtms_label(node_id, new_label) do
    AtlasIntegration.async(fn ->
      escaped_name = String.replace(to_string(node_id), "'", "\\'")
      query = "MATCH (n:JTMSNode) WHERE n.name = '#{escaped_name}' SET n.label = '#{new_label}' RETURN n"

      Atlas.Graph.cypher("epistemic_graph", query)
    end)
  end

  # ============================================================================
  # pos_graph: POS Tags + Transitions
  # ============================================================================

  @doc """
  Write token-tag associations to pos_graph.

  Expects a list of `{token, tag}` tuples. Creates Token nodes,
  POSTag nodes, and HAS_TAG edges with frequency counts.
  """
  def write_pos_tags(token_tag_pairs) when is_list(token_tag_pairs) do
    AtlasIntegration.async(fn ->
      Enum.each(token_tag_pairs, fn
        {token, tag} when is_binary(token) and is_binary(tag) ->
          {:ok, token_node} = AtlasIntegration.ensure_node("pos_graph", "Token", %{name: String.downcase(token)})
          {:ok, tag_node} = AtlasIntegration.ensure_node("pos_graph", "POSTag", %{name: tag})
          increment_edge_count("pos_graph", token_node.id, tag_node.id, EdgeLabels.has_tag(), "count")

        _ ->
          :ok
      end)
    end)
  end

  def write_pos_tags(_), do: :ok

  @doc """
  Write POS tag transitions to pos_graph.

  Expects a list of `{token, tag}` tuples (tag sequence). Creates
  FOLLOWED_BY edges between consecutive POSTag nodes with frequency counts.
  """
  def write_pos_transitions(token_tag_pairs) when is_list(token_tag_pairs) do
    AtlasIntegration.async(fn ->
      tags = Enum.map(token_tag_pairs, fn {_token, tag} -> tag end)

      tags
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [from_tag, to_tag] ->
        {:ok, from_node} = AtlasIntegration.ensure_node("pos_graph", "POSTag", %{name: from_tag})
        {:ok, to_node} = AtlasIntegration.ensure_node("pos_graph", "POSTag", %{name: to_tag})
        increment_edge_count("pos_graph", from_node.id, to_node.id, EdgeLabels.followed_by(), "frequency")
      end)
    end)
  end

  def write_pos_transitions(_), do: :ok

  # ============================================================================
  # Convenience: Write full analysis model
  # ============================================================================

  @doc """
  Write all graph-relevant data from a pipeline InternalModel.

  Extracts entities, events, and POS tags from all chunk analyses
  and writes them to the appropriate graphs.
  """
  def write_analysis(model) do
    analyses = Map.get(model, :analyses, [])

    all_entities = Enum.flat_map(analyses, fn a -> Map.get(a, :entities, []) end)
    all_events = Enum.flat_map(analyses, fn a -> Map.get(a, :events, []) end)
    all_srl_frames = Enum.flat_map(analyses, fn a -> Map.get(a, :srl_frames, []) end)
    all_event_frames = Enum.flat_map(analyses, fn a -> Map.get(a, :event_frames, []) end)

    write_entities(all_entities)
    write_events(all_events)
    write_srl_triples(all_srl_frames)
    write_event_frames(all_event_frames)

    Enum.each(analyses, fn analysis ->
      pos_tags = Map.get(analysis, :pos_tags, [])

      if is_list(pos_tags) and pos_tags != [] do
        write_pos_tags(pos_tags)
        write_pos_transitions(pos_tags)
      end
    end)

    :ok
  end

  @doc "Write SRL-derived triples to the semantic_graph with KG scoring metadata."
  def write_srl_triples(srl_frames) when is_list(srl_frames) do
    triples = Brain.Analysis.SemanticRoleLabeler.to_triples(srl_frames)
    if triples == [], do: :ok, else: do_write_srl_async(triples)
  rescue
    _ -> :ok
  end

  def write_srl_triples(_), do: :ok

  defp do_write_srl_async(triples) do
    AtlasIntegration.async(fn ->
      start_time = System.monotonic_time(:millisecond)

      normalized =
        Enum.map(triples, fn {s, p, o} ->
          if srl_gating_enabled?() do
            case Brain.ML.KnowledgeGraph.PredicateNormalizer.normalize(p) do
              {:ok, canon, kind} -> {:scorable, s, canon, o, kind, p}
              {:error, :oov} -> {:unscorable, s, p, o, :oov_predicate, p}
              {:error, _} -> {:unscorable, s, p, o, :oov_predicate, p}
            end
          else
            {:unscorable, s, p, o, :oov_predicate, p}
          end
        end)

      {scores, model_version} = score_normalized_triples(normalized)
      write_scored_edges(normalized, scores, model_version)
      record_predicate_frequency(normalized)

      elapsed = System.monotonic_time(:millisecond) - start_time
      :telemetry.execute([:brain, :kg_signal, :srl_batch], %{duration_ms: elapsed, count: length(triples)}, %{})
    end)

    :ok
  end

  defp score_normalized_triples(normalized) do
    scorable =
      Enum.filter(normalized, &match?({:scorable, _, _, _, _, _}, &1))

    if scorable == [] or not srl_gating_enabled?() do
      {%{}, nil}
    else
      model_version =
        case Brain.ML.KnowledgeGraph.TripleScorer.current_model_version() do
          {:ok, v} -> v
          _ -> nil
        end

      if Brain.ML.KnowledgeGraph.TripleScorer.ready?() do
        batch = Enum.map(scorable, fn {:scorable, s, canon, o, _kind, _raw} -> {s, canon, o} end)

        case Brain.ML.KnowledgeGraph.TripleScorer.score_batch(batch) do
          {:ok, score_list} ->
            score_map =
              Enum.zip(scorable, score_list)
              |> Enum.into(%{}, fn {{:scorable, s, canon, o, _kind, _raw}, score} ->
                {{s, canon, o}, score}
              end)

            :telemetry.execute([:brain, :kg_signal, :scored], %{count: length(score_list)}, %{})
            {score_map, model_version}

          _ ->
            {%{}, model_version}
        end
      else
        {%{}, model_version}
      end
    end
  end

  defp write_scored_edges(normalized, scores, model_version) do
    Enum.each(normalized, fn entry ->
      {s, edge_label, o, props} =
        case entry do
          {:scorable, s, canon, o, kind, raw} ->
            score = Map.get(scores, {s, canon, o})

            props = %{
              kg_score: score,
              kg_score_model: model_version,
              predicate_kind: to_string(kind),
              raw_predicate: raw
            }

            {s, canon, o, props}

          {:unscorable, s, p, o, _kind, raw} ->
            props = %{
              kg_score: nil,
              kg_score_model: nil,
              predicate_kind: "oov_predicate",
              raw_predicate: raw
            }

            {s, p, o, props}
        end

      with {:ok, subj_node} <-
             AtlasIntegration.ensure_node(
               "semantic_graph",
               "SrlEntity",
               %{name: s, type: "srl_entity"}
             ),
           {:ok, obj_node} <-
             AtlasIntegration.ensure_node(
               "semantic_graph",
               "SrlEntity",
               %{name: o, type: "srl_entity"}
             ) do
        AtlasIntegration.find_or_create_edge(
          "semantic_graph",
          subj_node.id,
          obj_node.id,
          edge_label,
          props
        )
      end
    end)
  end

  @predicate_freq_counter :srl_predicate_freq_counter

  defp record_predicate_frequency(normalized) do
    freq_path = srl_frequencies_path()

    existing =
      case :persistent_term.get(@predicate_freq_counter, nil) do
        nil -> %{count: 0, freqs: %{}}
        val -> val
      end

    updated_freqs =
      Enum.reduce(normalized, existing.freqs, fn entry, acc ->
        raw =
          case entry do
            {:scorable, _, _, _, _, raw} -> raw
            {:unscorable, _, _, _, _, raw} -> raw
          end

        Map.update(acc, raw, 1, &(&1 + 1))
      end)

    new_count = existing.count + length(normalized)
    :persistent_term.put(@predicate_freq_counter, %{count: new_count, freqs: updated_freqs})

    if rem(new_count, 1_000) < length(normalized) do
      tmp_path = freq_path <> ".tmp"
      File.mkdir_p!(Path.dirname(freq_path))
      File.write!(tmp_path, :erlang.term_to_binary(updated_freqs))
      File.rename!(tmp_path, freq_path)
    end
  rescue
    _ -> :ok
  end

  defp srl_gating_enabled? do
    config = Application.get_env(:brain, :kg_signals, [])
    Keyword.get(config, :enabled, true) and Keyword.get(config, :srl_gating, true)
  end

  defp srl_frequencies_path do
    priv = :code.priv_dir(:brain) |> to_string()
    Path.join([priv, "analysis", "srl_predicate_frequencies.term"])
  end

  @doc "Write enriched event frames (from EventLinker) to the knowledge_graph."
  def write_event_frames(event_frames) when is_list(event_frames) do
    if event_frames != [] do
      AtlasIntegration.async(fn ->
        Enum.each(event_frames, fn frame ->
          trigger = Map.get(frame, :trigger, "unknown")

          case AtlasIntegration.ensure_node(
            "knowledge_graph", "LinkedEvent",
            %{name: "event_#{trigger}", type: "LinkedEvent", trigger: trigger}
          ) do
            {:ok, event_node} ->
              args = Map.get(frame, :arguments, [])

              Enum.each(args, fn arg ->
                arg_text = Map.get(arg, :text, "")
                arg_role = Map.get(arg, :role, "ARG")

                case AtlasIntegration.ensure_node(
                  "knowledge_graph", "EventArgument",
                  %{name: arg_text, type: "EventArgument"}
                ) do
                  {:ok, arg_node} ->
                    AtlasIntegration.find_or_create_edge(
                      "knowledge_graph",
                      event_node.id,
                      arg_node.id,
                      "has_#{arg_role}"
                    )

                  _ -> :ok
                end
              end)

            _ -> :ok
          end
        end)
      end)
    end

    :ok
  rescue
    _ -> :ok
  end

  def write_event_frames(_), do: :ok

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp entity_type(e), do: Map.get(e, :entity_type) || Map.get(e, "entity_type") || "Entity"
  defp entity_value(e), do: Map.get(e, :value) || Map.get(e, "value") || Map.get(e, :text)

  defp normalize_label(type) when is_binary(type) do
    type
    |> String.replace("-", "_")
    |> String.split("_")
    |> Enum.map(&String.capitalize/1)
    |> Enum.join("")
  end

  defp normalize_label(type) when is_atom(type), do: normalize_label(to_string(type))
  defp normalize_label(_), do: "Entity"

  defp write_co_occurrences(nodes) do
    for {_e1, v1} <- nodes, {_e2, v2} <- nodes, v1.id < v2.id do
      AtlasIntegration.find_or_create_edge("knowledge_graph", v1.id, v2.id, EdgeLabels.co_occurs_with())
    end
  end

  defp write_single_event(event) do
    action = Map.get(event, :action, %{})
    lemma = Map.get(action, :lemma) || Map.get(action, "lemma", "unknown")
    tense = Map.get(action, :tense) || Map.get(action, "tense", "unknown")

    props = %{
      name: "evt_#{:crypto.strong_rand_bytes(4) |> Base.encode16(case: :lower)}",
      lemma: to_string(lemma),
      tense: to_string(tense)
    }

    case Atlas.Graph.add_node("knowledge_graph", "Event", props) do
      {:ok, event_node} ->
        actor = Map.get(event, :actor)
        object = Map.get(event, :object)

        if actor do
          actor_text = Map.get(actor, :text) || Map.get(actor, "text", "")
          actor_type = Map.get(actor, :type) || Map.get(actor, "type", "Entity")

          case AtlasIntegration.ensure_node("knowledge_graph", normalize_label(actor_type), %{name: actor_text}) do
            {:ok, actor_node} ->
              Atlas.Graph.add_edge("knowledge_graph", actor_node.id, event_node.id, EdgeLabels.actor())

            _ ->
              :ok
          end
        end

        if object do
          obj_text = Map.get(object, :text) || Map.get(object, "text", "")
          obj_type = Map.get(object, :type) || Map.get(object, "type", "Entity")

          case AtlasIntegration.ensure_node("knowledge_graph", normalize_label(obj_type), %{name: obj_text}) do
            {:ok, obj_node} ->
              Atlas.Graph.add_edge("knowledge_graph", event_node.id, obj_node.id, EdgeLabels.acts_on())

            _ ->
              :ok
          end
        end

      _ ->
        :ok
    end
  end

  defp maybe_write_user_preference(belief, confidence) do
    predicate = Map.get(belief, :predicate)
    subject = Map.get(belief, :subject)
    object = Map.get(belief, :object)

    if predicate in [:likes, :wants, :interested_in, :needs] and object do
      user_id = if subject in [:user, "user"], do: "default_user", else: to_string(subject)
      rel_type = preference_rel_type(predicate)
      {:ok, user_node} = AtlasIntegration.ensure_node("user_graph", "User", %{name: user_id, id: user_id})
      {:ok, topic_node} = AtlasIntegration.ensure_node("user_graph", "Topic", %{name: to_string(object)})
      AtlasIntegration.find_or_create_edge("user_graph", user_node.id, topic_node.id, rel_type, %{confidence: confidence})
    end
  end

  defp preference_rel_type(predicate) do
    case predicate do
      :likes -> EdgeLabels.likes()
      :wants -> EdgeLabels.wants()
      :interested_in -> EdgeLabels.interested_in()
      :needs -> EdgeLabels.needs()
      :dislikes -> EdgeLabels.dislikes()
      other -> String.upcase(to_string(other))
    end
  end

  defp link_message_to_conversation(conversation_id, msg_node) do
    case AtlasIntegration.find_node("conversation_graph", "Conversation", to_string(conversation_id)) do
      {:ok, conv_node} ->
        Atlas.Graph.add_edge("conversation_graph", conv_node.id, msg_node.id, EdgeLabels.contains())

      _ ->
        :ok
    end
  end

  defp link_message_to_previous(conversation_id, msg_node) do
    escaped_conv = String.replace(to_string(conversation_id), "'", "\\'")
    query = """
    MATCH (c:Conversation)-[:#{EdgeLabels.contains()}]->(m:Message)
    WHERE c.name = '#{escaped_conv}' AND id(m) <> #{msg_node.id}
    RETURN m ORDER BY id(m) DESC LIMIT 1
    """

    case Atlas.Graph.cypher("conversation_graph", query) do
      {:ok, [[%Atlas.Graph.Types.Vertex{} = prev] | _]} ->
        Atlas.Graph.add_edge("conversation_graph", prev.id, msg_node.id, EdgeLabels.follows())

      _ ->
        :ok
    end
  end

  defp link_message_to_topic(msg_node, nil), do: msg_node

  defp link_message_to_topic(msg_node, analysis) do
    intent =
      cond do
        is_map(analysis) and Map.has_key?(analysis, :intent) -> analysis.intent
        is_map(analysis) and Map.has_key?(analysis, :analyses) ->
          analysis.analyses
          |> Enum.max_by(fn a -> Map.get(a, :confidence, 0) end, fn -> %{} end)
          |> Map.get(:intent)
        true -> nil
      end

    if intent do
      {:ok, topic_node} = AtlasIntegration.ensure_node("conversation_graph", "Topic", %{name: to_string(intent)})
      Atlas.Graph.add_edge("conversation_graph", msg_node.id, topic_node.id, EdgeLabels.has_topic())

      create_topic_transition(msg_node, topic_node)
    end

    msg_node
  end

  defp create_topic_transition(msg_node, current_topic) do
    query = """
    MATCH (prev:Message)-[:#{EdgeLabels.follows()}]->(msg:Message)-[:#{EdgeLabels.has_topic()}]->(current:Topic)
    WHERE id(msg) = #{msg_node.id}
    MATCH (prev)-[:#{EdgeLabels.has_topic()}]->(prev_topic:Topic)
    WHERE id(prev_topic) <> id(current)
    RETURN prev_topic
    LIMIT 1
    """

    case Atlas.Graph.cypher("conversation_graph", query) do
      {:ok, [[%Atlas.Graph.Types.Vertex{} = prev_topic] | _]} ->
        increment_edge_count("conversation_graph", prev_topic.id, current_topic.id, EdgeLabels.topic_transition(), "count")

      _ ->
        :ok
    end
  end

  defp link_justification_to_conclusion(just_node, conclusion_id) do
    case AtlasIntegration.find_node("epistemic_graph", "JTMSNode", to_string(conclusion_id)) do
      {:ok, conclusion_node} ->
        Atlas.Graph.add_edge("epistemic_graph", just_node.id, conclusion_node.id, EdgeLabels.supports())

      _ ->
        :ok
    end
  end

  defp link_justification_requirements(just_node, node_ids, rel_type) do
    Enum.each(node_ids, fn nid ->
      case AtlasIntegration.find_node("epistemic_graph", "JTMSNode", to_string(nid)) do
        {:ok, target} ->
          Atlas.Graph.add_edge("epistemic_graph", just_node.id, target.id, rel_type)

        _ ->
          :ok
      end
    end)
  end

  defp increment_edge_count(graph, from_id, to_id, rel_type, count_field) do
    query = "MATCH (a)-[r:#{rel_type}]->(b) WHERE id(a) = #{from_id} AND id(b) = #{to_id} RETURN r"

    case Atlas.Graph.cypher(graph, query) do
      {:ok, [[%Atlas.Graph.Types.Edge{properties: props} = _edge] | _]} ->
        current = Map.get(props, count_field, 0) || 0
        new_count = if is_number(current), do: current + 1, else: 1

        new_props =
          props
          |> Map.put(count_field, new_count)
          |> Map.drop(["id", "start_id", "end_id", "label"])
          |> Enum.map(fn {k, v} -> {String.to_atom(k), v} end)
          |> Map.new()

        delete_query = """
        MATCH (a)-[r:#{rel_type}]->(b)
        WHERE id(a) = #{from_id} AND id(b) = #{to_id}
        DELETE r
        """

        Atlas.Graph.cypher(graph, delete_query)
        Atlas.Graph.add_edge(graph, from_id, to_id, rel_type, new_props)

      _ ->
        Atlas.Graph.add_edge(graph, from_id, to_id, rel_type, %{String.to_atom(count_field) => 1})
    end
  end
end
