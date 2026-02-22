defmodule Brain.Graph.WriterTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Graph.Writer

  describe "write_entities/1" do
    test "creates entity nodes in knowledge_graph" do
      entities = [%{entity_type: "location", value: "Tokyo", confidence: 0.95}]
      :ok = Writer.write_entities(entities)
      Process.sleep(200)
      assert_node_exists("knowledge_graph", "Location", "Tokyo")
    end

    test "deduplicates entity nodes across calls" do
      entities = [%{entity_type: "location", value: "Berlin", confidence: 0.9}]
      :ok = Writer.write_entities(entities)
      Process.sleep(200)
      :ok = Writer.write_entities(entities)
      Process.sleep(200)

      query = "MATCH (n:Location) WHERE n.name = 'Berlin' RETURN count(n)"

      case Graph.cypher("knowledge_graph", query) do
        {:ok, [[count]]} -> assert count == 1
        _ -> flunk("Expected to find exactly one Berlin node")
      end
    end

    test "creates co-occurrence edges for entities in same context" do
      entities = [
        %{entity_type: "location", value: "Madrid", confidence: 0.9},
        %{entity_type: "person", value: "Carlos", confidence: 0.8}
      ]

      :ok = Writer.write_entities(entities)
      Process.sleep(200)

      assert_node_exists("knowledge_graph", "Location", "Madrid")
      assert_node_exists("knowledge_graph", "Person", "Carlos")
      assert_edge_exists("knowledge_graph", "co_occurs_with")
    end

    test "handles empty entity list" do
      assert :ok = Writer.write_entities([])
    end

    test "handles nil entities" do
      assert :ok = Writer.write_entities(nil)
    end
  end

  describe "write_events/1" do
    test "creates event subgraph with actor and object" do
      events = [
        %{
          action: %{lemma: "play", tense: :present},
          actor: %{text: "user", type: "pronoun"},
          object: %{text: "jazz", type: "noun"},
          confidence: 0.8
        }
      ]

      :ok = Writer.write_events(events)
      Process.sleep(200)

      assert_edge_exists("knowledge_graph", "ACTOR")
      assert_edge_exists("knowledge_graph", "ACTS_ON")
    end

    test "handles events without actor or object" do
      events = [%{action: %{lemma: "rain", tense: :present}, confidence: 0.7}]
      :ok = Writer.write_events(events)
      Process.sleep(200)

      {:ok, count} = count_nodes("knowledge_graph", "Event")
      assert count >= 1
    end
  end

  describe "write_belief/1" do
    test "creates belief node in knowledge_graph" do
      belief = %{
        id: "test_belief_1",
        subject: :user,
        predicate: :knows,
        object: "Elixir is great",
        confidence: 0.9
      }

      :ok = Writer.write_belief(belief)
      Process.sleep(200)

      {:ok, count} = count_nodes("knowledge_graph", "Belief")
      assert count >= 1
    end

    test "creates user preference edge for :likes predicate" do
      belief = %{
        id: "test_pref_1",
        subject: :user,
        predicate: :likes,
        object: "jazz",
        confidence: 0.85
      }

      :ok = Writer.write_belief(belief)
      Process.sleep(300)

      assert_node_exists("user_graph", "Topic", "jazz")
      assert_edge_exists("user_graph", "LIKES")
    end
  end

  describe "write_user_preference/4" do
    test "creates user and topic nodes with preference edge" do
      :ok = Writer.write_user_preference("user_42", :wants, "pizza", 0.95)
      Process.sleep(200)

      assert_node_exists("user_graph", "User", "user_42")
      assert_node_exists("user_graph", "Topic", "pizza")
      assert_edge_exists("user_graph", "WANTS")
    end
  end

  describe "write_semantic_cluster/2" do
    test "creates semantic fact node with evidence edges" do
      semantic = %{representation: "Action: query | Examples: weather in London"}
      episode_ids = ["ep_1", "ep_2", "ep_3"]

      :ok = Writer.write_semantic_cluster(semantic, episode_ids)
      Process.sleep(300)

      {:ok, fact_count} = count_nodes("semantic_graph", "SemanticFact")
      assert fact_count >= 1

      {:ok, ep_count} = count_nodes("semantic_graph", "Episode")
      assert ep_count >= 3

      {:ok, edge_count} = count_edges("semantic_graph", "EVIDENCE_FOR")
      assert edge_count >= 3
    end
  end

  describe "write_conversation/1" do
    test "creates conversation node" do
      :ok = Writer.write_conversation(%{id: "conv_test_1", world_id: "default"})
      Process.sleep(200)

      assert_node_exists("conversation_graph", "Conversation", "conv_test_1")
    end
  end

  describe "write_message/3" do
    test "creates message node linked to conversation" do
      :ok = Writer.write_conversation(%{id: "conv_msg_test", world_id: "default"})
      Process.sleep(200)

      message = %{id: "msg_test_1", role: "user", content: "Hello there!"}
      analysis = %{intent: "smalltalk.greetings.hello"}
      :ok = Writer.write_message("conv_msg_test", message, analysis)
      Process.sleep(300)

      assert_node_exists("conversation_graph", "Message", "msg_test_1")
      assert_edge_exists("conversation_graph", "CONTAINS")
      assert_edge_exists("conversation_graph", "HAS_TOPIC")
    end

    test "creates FOLLOWS edge between sequential messages" do
      :ok = Writer.write_conversation(%{id: "conv_seq_test", world_id: "default"})
      Process.sleep(200)

      :ok = Writer.write_message("conv_seq_test", %{id: "seq_1", role: "user", content: "Hi"}, %{intent: "smalltalk.greetings.hello"})
      Process.sleep(200)
      :ok = Writer.write_message("conv_seq_test", %{id: "seq_2", role: "assistant", content: "Hello!"}, nil)
      Process.sleep(300)

      assert_edge_exists("conversation_graph", "FOLLOWS")
    end
  end

  describe "write_jtms_node/1" do
    test "creates JTMSNode in epistemic_graph" do
      node = %{id: "jtms_test_1", datum: "test belief", node_type: :premise, label: :in}
      :ok = Writer.write_jtms_node(node)
      Process.sleep(200)

      assert_node_exists("epistemic_graph", "JTMSNode", "jtms_test_1")
    end
  end

  describe "write_justification/1" do
    test "creates justification with SUPPORTS edge" do
      node1 = %{id: "just_premise", datum: "A", node_type: :assumption, label: :in}
      node2 = %{id: "just_conclusion", datum: "B", node_type: :derived, label: :in}
      :ok = Writer.write_jtms_node(node1)
      :ok = Writer.write_jtms_node(node2)
      Process.sleep(200)

      just = %{id: "j_test_1", informant: "test_rule", conclusion_id: "just_conclusion", in_list: ["just_premise"], out_list: []}
      :ok = Writer.write_justification(just)
      Process.sleep(300)

      assert_node_exists("epistemic_graph", "Justification", "j_test_1")
      assert_edge_exists("epistemic_graph", "SUPPORTS")
      assert_edge_exists("epistemic_graph", "REQUIRES_IN")
    end
  end

  describe "write_pos_tags/1" do
    test "creates token-tag edges" do
      pairs = [{"hello", "INTJ"}, {"world", "NOUN"}]
      :ok = Writer.write_pos_tags(pairs)
      Process.sleep(200)

      assert_node_exists("pos_graph", "Token", "hello")
      assert_node_exists("pos_graph", "Token", "world")
      assert_edge_exists("pos_graph", "HAS_TAG")
    end
  end

  describe "write_pos_transitions/1" do
    test "creates FOLLOWED_BY edges between consecutive tags" do
      pairs = [{"the", "DET"}, {"big", "ADJ"}, {"cat", "NOUN"}]
      :ok = Writer.write_pos_transitions(pairs)
      Process.sleep(200)

      assert_edge_exists("pos_graph", "FOLLOWED_BY")
    end
  end

  describe "write_analysis/1" do
    test "writes entities, events, and POS tags from a full analysis model" do
      model = %{
        analyses: [
          %{
            entities: [%{entity_type: "location", value: "Sydney", confidence: 0.9}],
            events: [%{action: %{lemma: "visit", tense: :future}, actor: %{text: "I", type: "pronoun"}, object: %{text: "Sydney", type: "proper_noun"}, confidence: 0.8}],
            pos_tags: [{"I", "PRON"}, {"will", "AUX"}, {"visit", "VERB"}, {"Sydney", "PROPN"}]
          }
        ]
      }

      assert :ok = Writer.write_analysis(model)
      Process.sleep(400)

      assert_node_exists("knowledge_graph", "Location", "Sydney")
      {:ok, event_count} = count_nodes("knowledge_graph", "Event")
      assert event_count >= 1
    end
  end
end
