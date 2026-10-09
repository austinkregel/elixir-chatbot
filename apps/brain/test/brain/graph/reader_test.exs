defmodule Brain.Graph.ReaderTest do
  use Brain.Test.GraphCase, async: false
  @moduletag seed_graphs: true

  alias Brain.Graph.Reader

  # `GraphCase` seeds the test's own world, `:graph_world_id`. `other` is a
  # second world whose graphs exist and are empty: whatever the seeded world
  # holds must not be read through it.
  setup %{graph_world_id: world} do
    other = "test_graph_unseeded_#{System.unique_integer([:positive])}"
    Graph.ensure_world_graphs!(other)
    %{world: world, other: other}
  end

  describe "the world" do
    test "every world-scoped read raises without a world, or with one that cannot name a graph" do
      for opts <- [[], [world_id: nil], [world_id: ""], [world_id: "not a world"]] do
        for read <- [
              fn -> Reader.entity_context([%{entity_type: "Location", value: "Paris"}], opts) end,
              fn -> Reader.entity_context_multi_label("Paris", ["Location"], opts) end,
              fn -> Reader.relationship_path(%{value: "a"}, %{value: "b"}, opts) end,
              fn -> Reader.expand_query("q", [], opts) end,
              fn -> Reader.user_preferences("test_user_1", opts) end,
              fn -> Reader.evidence_chain("weather_pattern", opts) end,
              fn -> Reader.conversation_topics("test_conv_1", opts) end,
              fn -> Reader.topic_transitions(opts) end,
              fn -> Reader.recent_context("test_conv_1", opts) end,
              fn -> Reader.belief_justification_chain("recommend_jazz", opts) end,
              fn -> Reader.assumption_consequences("user_likes_jazz", opts) end
            ] do
          assert_raise ArgumentError, read
        end
      end
    end

    test "what one world's graphs hold is invisible from another world", %{other: other} do
      assert [%{node: nil, neighbors: []}] =
               Reader.entity_context([%{entity_type: "Location", value: "Paris"}], world_id: other)

      assert Reader.entity_context_multi_label("Paris", ["Location"], world_id: other) == %{}

      assert {:error, :not_found} =
               Reader.relationship_path(
                 %{entity_type: "Location", value: "Paris"},
                 %{entity_type: "Location", value: "France"},
                 world_id: other
               )

      assert Reader.user_preferences("test_user_1", world_id: other) == []
      assert Reader.evidence_chain("weather_pattern", world_id: other) == []
      assert Reader.conversation_topics("test_conv_1", world_id: other) == []
      assert Reader.topic_transitions(world_id: other) == []
      assert Reader.recent_context("test_conv_1", world_id: other) == []
      assert Reader.belief_justification_chain("recommend_jazz", world_id: other) == []
      assert Reader.assumption_consequences("user_likes_jazz", world_id: other) == []
    end
  end

  describe "entity_context/2" do
    test "returns neighbors for seeded entity", %{world: world} do
      result = Reader.entity_context([%{entity_type: "Location", value: "Paris"}], world_id: world)
      assert [%{entity: _, neighbors: neighbors, node: node}] = result
      assert node != nil
      neighbor_names = Enum.map(neighbors, &(&1.properties["name"]))
      assert "France" in neighbor_names
    end

    test "returns empty neighbors for unknown entity", %{world: world} do
      result = Reader.entity_context([%{entity_type: "Location", value: "Atlantis"}], world_id: world)
      assert [%{neighbors: [], node: nil}] = result
    end

    test "returns results for multiple entities", %{world: world} do
      entities = [
        %{entity_type: "Location", value: "Paris"},
        %{entity_type: "Location", value: "London"}
      ]

      result = Reader.entity_context(entities, world_id: world)
      assert length(result) == 2
      assert Enum.all?(result, fn r -> r.node != nil end)
    end
  end

  describe "relationship_path/3" do
    test "finds path between related entities", %{world: world} do
      result =
        Reader.relationship_path(
          %{entity_type: "Location", value: "Paris"},
          %{entity_type: "Location", value: "France"},
          world_id: world
        )

      assert {:ok, _path} = result
    end

    test "returns error for unrelated entities", %{world: world} do
      result =
        Reader.relationship_path(
          %{entity_type: "Location", value: "Atlantis"},
          %{entity_type: "Location", value: "Narnia"},
          world_id: world
        )

      assert {:error, :not_found} = result
    end
  end

  describe "expand_query/3" do
    test "expands query with related entity names", %{world: world} do
      entities = [%{entity_type: "Location", value: "Paris"}]
      {query, related} = Reader.expand_query("What about Paris?", entities, world_id: world)

      assert query == "What about Paris?"
      assert is_list(related)
      assert "France" in related
    end

    test "returns no graph neighbors for unknown entities", %{world: world} do
      entities = [%{entity_type: "Location", value: "Atlantis"}]
      {_query, related} = Reader.expand_query("Find Atlantis", entities, world_id: world)
      assert is_list(related)
      refute "Atlantis" in related,
             "Should not return the query entity itself as a related term"
    end
  end

  describe "user_preferences/2" do
    test "returns preference edges for seeded user", %{world: world} do
      prefs = Reader.user_preferences("test_user_1", world_id: world)
      assert Enum.any?(prefs, fn p -> p.topic == "jazz" end)
    end

    test "returns empty list for unknown user", %{world: world} do
      assert Reader.user_preferences("nonexistent_user", world_id: world) == []
    end
  end

  describe "evidence_chain/2" do
    test "returns episodes for seeded semantic fact", %{world: world} do
      episodes = Reader.evidence_chain("weather_pattern", world_id: world)
      assert length(episodes) == 2
    end

    test "returns empty for unknown fact", %{world: world} do
      assert Reader.evidence_chain("nonexistent_fact", world_id: world) == []
    end
  end

  describe "conversation_topics/2" do
    test "returns topics from seeded conversation", %{world: world} do
      topics = Reader.conversation_topics("test_conv_1", world_id: world)
      assert "weather.query" in topics
    end

    test "returns empty for unknown conversation", %{world: world} do
      assert Reader.conversation_topics("nonexistent_conv", world_id: world) == []
    end
  end

  describe "topic_transitions/1" do
    test "returns topic transition data", %{world: world} do
      assert [%{from: "smalltalk.greetings.hello", to: "weather.query", count: 2}] =
               Reader.topic_transitions(world_id: world, limit: 10)
    end
  end

  describe "recent_context/2" do
    test "returns messages from seeded conversation", %{world: world} do
      context = Reader.recent_context("test_conv_1", world_id: world, n: 5)
      assert length(context) == 2

      roles = Enum.map(context, fn c -> c.message["role"] end)
      assert "user" in roles
      assert "assistant" in roles
    end
  end

  describe "belief_justification_chain/2" do
    test "returns justification chain for derived belief", %{world: world} do
      chain = Reader.belief_justification_chain("recommend_jazz", world_id: world)
      assert length(chain) >= 1

      entry = hd(chain)
      assert entry.node != nil
      assert entry.justification != nil
    end

    test "returns empty for unknown node", %{world: world} do
      assert Reader.belief_justification_chain("nonexistent_node", world_id: world) == []
    end
  end

  describe "assumption_consequences/2" do
    test "finds nodes that depend on an assumption", %{world: world} do
      consequences = Reader.assumption_consequences("user_likes_jazz", world_id: world)
      names = Enum.map(consequences, &(&1.properties["name"]))
      assert "recommend_jazz" in names
    end

    test "returns empty for node with no dependents", %{world: world} do
      assert Reader.assumption_consequences("paris_capital", world_id: world) == []
    end
  end

  describe "tag_transitions/1" do
    test "returns FOLLOWED_BY edges for DET from the shared pos_graph" do
      transitions = Reader.tag_transitions("DET")
      assert length(transitions) >= 2

      noun_t = Enum.find(transitions, fn t -> t.to_tag == "NOUN" end)
      assert noun_t != nil
      assert noun_t.frequency == 0.65
    end

    test "returns empty for unused tag" do
      transitions = Reader.tag_transitions("SYM")
      assert transitions == []
    end
  end

  describe "ambiguous_tokens/1" do
    test "finds tokens with multiple POS tags in the shared pos_graph" do
      ambiguous = Reader.ambiguous_tokens(2)
      assert length(ambiguous) >= 1
      assert Enum.any?(ambiguous, fn a -> a.token == "run" end)
    end
  end
end
