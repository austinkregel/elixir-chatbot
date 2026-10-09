defmodule Brain.Graph.WriterTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Graph.Writer

  # Each test writes into its own world's graphs (`GraphCase` creates them);
  # `g/2` names that world's graph of a kind.
  setup %{graph_world_id: world_id} do
    %{world: world_id}
  end

  defp g(kind, world), do: Graph.world_graph(kind, world)

  defp other_world do
    world = "test_graph_other_#{System.unique_integer([:positive])}"
    Graph.ensure_world_graphs!(world)
    world
  end

  defp count_named(graph, label, name) do
    {:ok, [[count]]} = Graph.cypher(graph, "MATCH (n:#{label}) WHERE n.name = '#{name}' RETURN count(n)")
    count
  end

  describe "a world's writes" do
    test "every world-scoped writer raises without a world, before writing anything" do
      for write <- [
            fn -> Writer.write_entities([%{entity_type: "location", value: "Oslo"}], nil) end,
            fn -> Writer.write_events([], nil) end,
            fn -> Writer.write_belief(%{subject: :user, predicate: :likes, object: "x"}, nil) end,
            fn -> Writer.write_user_preference("u", :likes, "x", 0.5, nil) end,
            fn -> Writer.write_semantic_cluster(%{representation: "x"}, ["ep_1"], nil) end,
            fn -> Writer.write_message("c", %{id: "m"}, nil, nil) end,
            fn -> Writer.write_jtms_node(%{id: "n"}, nil) end,
            fn -> Writer.write_justification(%{id: "j"}, nil) end,
            fn -> Writer.write_contradiction(["a", "b"], nil) end,
            fn -> Writer.update_jtms_label("n", :in, nil) end,
            fn -> Writer.write_analysis(%{analyses: []}, nil) end,
            fn -> Writer.write_srl_triples([], nil) end,
            fn -> Writer.write_event_frames([], nil) end,
            fn -> Writer.write_conversation(%{id: "c"}) end
          ] do
        assert_raise ArgumentError, write
      end
    end

    test "a world id that cannot name a graph raises", %{world: _world} do
      assert_raise ArgumentError, ~r/cannot name a graph/, fn ->
        Writer.write_entities([%{entity_type: "location", value: "Oslo"}], "not a world")
      end
    end

    test "entities written in one world are invisible in another, and each world dedupes its own", %{world: world} do
      other = other_world()
      entities = [%{entity_type: "location", value: "Lisbon", confidence: 0.9}]

      :ok = Writer.write_entities(entities, world)
      Process.sleep(200)

      assert count_named(g("knowledge_graph", world), "Location", "Lisbon") == 1
      assert count_named(g("knowledge_graph", other), "Location", "Lisbon") == 0

      # The same entity in the other world is a node of the other world's own,
      # not the first world's node reused.
      :ok = Writer.write_entities(entities, other)
      Process.sleep(200)

      assert count_named(g("knowledge_graph", world), "Location", "Lisbon") == 1
      assert count_named(g("knowledge_graph", other), "Location", "Lisbon") == 1
    end

    test "conversations, JTMS nodes and semantic facts stay in their world", %{world: world} do
      other = other_world()

      :ok = Writer.write_conversation(%{id: "conv_scoped", world_id: world})
      :ok = Writer.write_jtms_node(%{id: "jtms_scoped", datum: "d", node_type: :premise, label: :in}, world)
      :ok = Writer.write_semantic_cluster(%{representation: "r"}, ["ep_scoped"], world)
      Process.sleep(300)

      assert count_named(g("conversation_graph", world), "Conversation", "conv_scoped") == 1
      assert count_named(g("conversation_graph", other), "Conversation", "conv_scoped") == 0
      assert count_named(g("epistemic_graph", world), "JTMSNode", "jtms_scoped") == 1
      assert count_named(g("epistemic_graph", other), "JTMSNode", "jtms_scoped") == 0
      assert count_named(g("semantic_graph", world), "Episode", "ep_scoped") == 1
      assert count_named(g("semantic_graph", other), "Episode", "ep_scoped") == 0
    end

    test "POS tags go to the one pos_graph every world shares" do
      :ok = Writer.write_pos_tags([{"sharedtoken", "NOUN"}])
      Process.sleep(200)

      assert count_named("pos_graph", "Token", "sharedtoken") == 1
    end
  end

  describe "write_entities/2" do
    test "creates entity nodes in the world's knowledge_graph", %{world: world} do
      entities = [%{entity_type: "location", value: "Tokyo", confidence: 0.95}]
      :ok = Writer.write_entities(entities, world)
      Process.sleep(200)
      assert_node_exists(g("knowledge_graph", world), "Location", "Tokyo")
    end

    test "deduplicates entity nodes across calls", %{world: world} do
      entities = [%{entity_type: "location", value: "Berlin", confidence: 0.9}]
      :ok = Writer.write_entities(entities, world)
      Process.sleep(200)
      :ok = Writer.write_entities(entities, world)
      Process.sleep(200)

      assert count_named(g("knowledge_graph", world), "Location", "Berlin") == 1
    end

    test "creates co-occurrence edges for entities in same context", %{world: world} do
      entities = [
        %{entity_type: "location", value: "Madrid", confidence: 0.9},
        %{entity_type: "person", value: "Carlos", confidence: 0.8}
      ]

      :ok = Writer.write_entities(entities, world)
      Process.sleep(200)

      assert_node_exists(g("knowledge_graph", world), "Location", "Madrid")
      assert_node_exists(g("knowledge_graph", world), "Person", "Carlos")
      assert_edge_exists(g("knowledge_graph", world), Atlas.Graph.EdgeLabels.co_occurs_with())
    end

    test "handles empty entity list", %{world: world} do
      assert :ok = Writer.write_entities([], world)
    end

    test "handles nil entities", %{world: world} do
      assert :ok = Writer.write_entities(nil, world)
    end
  end

  describe "write_events/2" do
    test "creates event subgraph with actor and object", %{world: world} do
      events = [
        %{
          action: %{lemma: "play", tense: :present},
          actor: %{text: "user", type: "pronoun"},
          object: %{text: "jazz", type: "noun"},
          confidence: 0.8
        }
      ]

      :ok = Writer.write_events(events, world)
      Process.sleep(200)

      assert_edge_exists(g("knowledge_graph", world), "ACTOR")
      assert_edge_exists(g("knowledge_graph", world), "ACTS_ON")
    end

    test "handles events without actor or object", %{world: world} do
      events = [%{action: %{lemma: "rain", tense: :present}, confidence: 0.7}]
      :ok = Writer.write_events(events, world)
      Process.sleep(200)

      assert {:ok, 1} = count_nodes(g("knowledge_graph", world), "Event")
    end
  end

  describe "write_belief/2" do
    test "creates belief node in the world's knowledge_graph", %{world: world} do
      belief = %{
        id: "test_belief_1",
        subject: :user,
        predicate: :knows,
        object: "Elixir is great",
        confidence: 0.9
      }

      :ok = Writer.write_belief(belief, world)
      Process.sleep(200)

      assert {:ok, 1} = count_nodes(g("knowledge_graph", world), "Belief")
    end

    test "creates user preference edge for :likes predicate in the world's user_graph", %{world: world} do
      belief = %{
        id: "test_pref_1",
        subject: :user,
        predicate: :likes,
        object: "jazz",
        confidence: 0.85
      }

      :ok = Writer.write_belief(belief, world)
      Process.sleep(300)

      assert_node_exists(g("user_graph", world), "Topic", "jazz")
      assert_edge_exists(g("user_graph", world), "LIKES")
    end
  end

  describe "write_user_preference/5" do
    test "creates user and topic nodes with preference edge", %{world: world} do
      :ok = Writer.write_user_preference("user_42", :wants, "pizza", 0.95, world)
      Process.sleep(200)

      assert_node_exists(g("user_graph", world), "User", "user_42")
      assert_node_exists(g("user_graph", world), "Topic", "pizza")
      assert_edge_exists(g("user_graph", world), "WANTS")
    end
  end

  describe "write_semantic_cluster/3" do
    test "raises without a world" do
      assert_raise ArgumentError, ~r/world/, fn ->
        Writer.write_semantic_cluster(%{representation: "x"}, ["ep_1"], nil)
      end
    end

    test "creates semantic fact node with evidence edges", %{world: world} do
      semantic = %{representation: "Action: query | Examples: weather in London"}
      episode_ids = ["ep_1", "ep_2", "ep_3"]

      :ok = Writer.write_semantic_cluster(semantic, episode_ids, world)
      Process.sleep(300)

      assert {:ok, 1} = count_nodes(g("semantic_graph", world), "SemanticFact")
      assert {:ok, 3} = count_nodes(g("semantic_graph", world), "Episode")
      assert {:ok, 3} = count_edges(g("semantic_graph", world), "EVIDENCE_FOR")
    end
  end

  describe "write_conversation/1" do
    test "creates conversation node in its world's conversation_graph", %{world: world} do
      :ok = Writer.write_conversation(%{id: "conv_test_1", world_id: world})
      Process.sleep(200)

      assert_node_exists(g("conversation_graph", world), "Conversation", "conv_test_1")
    end
  end

  describe "write_message/4" do
    test "creates message node linked to conversation", %{world: world} do
      :ok = Writer.write_conversation(%{id: "conv_msg_test", world_id: world})
      Process.sleep(200)

      message = %{id: "msg_test_1", role: "user", content: "Hello there!"}
      analysis = %{intent: "smalltalk.greetings.hello"}
      :ok = Writer.write_message("conv_msg_test", message, analysis, world)
      Process.sleep(300)

      assert_node_exists(g("conversation_graph", world), "Message", "msg_test_1")
      assert_edge_exists(g("conversation_graph", world), "CONTAINS")
      assert_edge_exists(g("conversation_graph", world), "HAS_TOPIC")
    end

    test "creates FOLLOWS edge between sequential messages", %{world: world} do
      :ok = Writer.write_conversation(%{id: "conv_seq_test", world_id: world})
      Process.sleep(200)

      :ok = Writer.write_message("conv_seq_test", %{id: "seq_1", role: "user", content: "Hi"}, %{intent: "smalltalk.greetings.hello"}, world)
      Process.sleep(200)
      :ok = Writer.write_message("conv_seq_test", %{id: "seq_2", role: "assistant", content: "Hello!"}, nil, world)
      Process.sleep(300)

      assert_edge_exists(g("conversation_graph", world), "FOLLOWS")
    end
  end

  describe "write_jtms_node/2" do
    test "creates JTMSNode in the world's epistemic_graph", %{world: world} do
      node = %{id: "jtms_test_1", datum: "test belief", node_type: :premise, label: :in}
      :ok = Writer.write_jtms_node(node, world)
      Process.sleep(200)

      assert_node_exists(g("epistemic_graph", world), "JTMSNode", "jtms_test_1")
    end
  end

  describe "write_justification/2" do
    test "creates justification with SUPPORTS edge", %{world: world} do
      node1 = %{id: "just_premise", datum: "A", node_type: :assumption, label: :in}
      node2 = %{id: "just_conclusion", datum: "B", node_type: :derived, label: :in}
      :ok = Writer.write_jtms_node(node1, world)
      :ok = Writer.write_jtms_node(node2, world)
      Process.sleep(200)

      just = %{id: "j_test_1", informant: "test_rule", conclusion_id: "just_conclusion", in_list: ["just_premise"], out_list: []}
      :ok = Writer.write_justification(just, world)
      Process.sleep(300)

      assert_node_exists(g("epistemic_graph", world), "Justification", "j_test_1")
      assert_edge_exists(g("epistemic_graph", world), "SUPPORTS")
      assert_edge_exists(g("epistemic_graph", world), "REQUIRES_IN")
    end

    test "links only to JTMS nodes of the same world", %{world: world} do
      other = other_world()
      :ok = Writer.write_jtms_node(%{id: "only_other", datum: "A", node_type: :premise, label: :in}, other)
      :ok = Writer.write_jtms_node(%{id: "here_conclusion", datum: "B", node_type: :derived, label: :in}, world)
      Process.sleep(200)

      just = %{id: "j_cross", informant: "r", conclusion_id: "here_conclusion", in_list: ["only_other"], out_list: []}
      :ok = Writer.write_justification(just, world)
      Process.sleep(300)

      assert {:ok, 1} = count_edges(g("epistemic_graph", world), "SUPPORTS")
      assert {:ok, 0} = count_edges(g("epistemic_graph", world), "REQUIRES_IN")
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

  describe "write_analysis/2" do
    test "writes entities, events, and POS tags from a full analysis model", %{world: world} do
      model = %{
        analyses: [
          %{
            entities: [%{entity_type: "location", value: "Sydney", confidence: 0.9}],
            events: [%{action: %{lemma: "visit", tense: :future}, actor: %{text: "I", type: "pronoun"}, object: %{text: "Sydney", type: "proper_noun"}, confidence: 0.8}],
            pos_tags: [{"I", "PRON"}, {"will", "AUX"}, {"visit", "VERB"}, {"Sydney", "PROPN"}]
          }
        ]
      }

      assert :ok = Writer.write_analysis(model, world)
      Process.sleep(400)

      assert_node_exists(g("knowledge_graph", world), "Location", "Sydney")
      assert {:ok, 1} = count_nodes(g("knowledge_graph", world), "Event")
    end
  end
end
