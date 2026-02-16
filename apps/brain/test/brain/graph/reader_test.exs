defmodule Brain.Graph.ReaderTest do
  use Brain.Test.GraphCase, async: false
  @moduletag seed_graphs: true

  alias Brain.Graph.Reader

  describe "entity_context/2" do
    test "returns neighbors for seeded entity" do
      result = Reader.entity_context([%{entity_type: "Location", value: "Paris"}])
      assert [%{entity: _, neighbors: neighbors, node: node}] = result
      assert node != nil
      neighbor_names = Enum.map(neighbors, &(&1.properties["name"]))
      assert "France" in neighbor_names
    end

    test "returns empty neighbors for unknown entity" do
      result = Reader.entity_context([%{entity_type: "Location", value: "Atlantis"}])
      assert [%{neighbors: [], node: nil}] = result
    end

    test "returns results for multiple entities" do
      entities = [
        %{entity_type: "Location", value: "Paris"},
        %{entity_type: "Location", value: "London"}
      ]

      result = Reader.entity_context(entities)
      assert length(result) == 2
      assert Enum.all?(result, fn r -> r.node != nil end)
    end
  end

  describe "relationship_path/2" do
    test "finds path between related entities" do
      result = Reader.relationship_path(
        %{entity_type: "Location", value: "Paris"},
        %{entity_type: "Location", value: "France"}
      )

      assert {:ok, _path} = result
    end

    test "returns error for unrelated entities" do
      result = Reader.relationship_path(
        %{entity_type: "Location", value: "Atlantis"},
        %{entity_type: "Location", value: "Narnia"}
      )

      assert {:error, :not_found} = result
    end
  end

  describe "expand_query/2" do
    test "expands query with related entity names" do
      entities = [%{entity_type: "Location", value: "Paris"}]
      {query, related} = Reader.expand_query("What about Paris?", entities)

      assert query == "What about Paris?"
      assert is_list(related)
      assert "France" in related
    end

    test "returns empty list for unknown entities" do
      entities = [%{entity_type: "Location", value: "Atlantis"}]
      {_query, related} = Reader.expand_query("Find Atlantis", entities)
      assert related == []
    end
  end

  describe "user_preferences/1" do
    test "returns preference edges for seeded user" do
      prefs = Reader.user_preferences("test_user_1")
      assert length(prefs) >= 1
      assert Enum.any?(prefs, fn p -> p.topic == "jazz" end)
    end

    test "returns empty list for unknown user" do
      prefs = Reader.user_preferences("nonexistent_user")
      assert prefs == []
    end
  end

  describe "evidence_chain/1" do
    test "returns episodes for seeded semantic fact" do
      episodes = Reader.evidence_chain("weather_pattern")
      assert length(episodes) >= 2
    end

    test "returns empty for unknown fact" do
      episodes = Reader.evidence_chain("nonexistent_fact")
      assert episodes == []
    end
  end

  describe "conversation_topics/1" do
    test "returns topics from seeded conversation" do
      topics = Reader.conversation_topics("test_conv_1")
      assert "weather.query" in topics
    end

    test "returns empty for unknown conversation" do
      topics = Reader.conversation_topics("nonexistent_conv")
      assert topics == []
    end
  end

  describe "topic_transitions/1" do
    test "returns topic transition data" do
      transitions = Reader.topic_transitions(10)
      assert is_list(transitions)
      assert length(transitions) >= 1

      t = hd(transitions)
      assert Map.has_key?(t, :from)
      assert Map.has_key?(t, :to)
      assert Map.has_key?(t, :count)
    end
  end

  describe "recent_context/2" do
    test "returns messages from seeded conversation" do
      context = Reader.recent_context("test_conv_1", 5)
      assert length(context) >= 2

      roles = Enum.map(context, fn c -> c.message["role"] end)
      assert "user" in roles
      assert "assistant" in roles
    end
  end

  describe "belief_justification_chain/1" do
    test "returns justification chain for derived belief" do
      chain = Reader.belief_justification_chain("recommend_jazz")
      assert length(chain) >= 1

      entry = hd(chain)
      assert entry.node != nil
      assert entry.justification != nil
    end

    test "returns empty for unknown node" do
      chain = Reader.belief_justification_chain("nonexistent_node")
      assert chain == []
    end
  end

  describe "assumption_consequences/1" do
    test "finds nodes that depend on an assumption" do
      consequences = Reader.assumption_consequences("user_likes_jazz")
      assert length(consequences) >= 1
      names = Enum.map(consequences, &(&1.properties["name"]))
      assert "recommend_jazz" in names
    end

    test "returns empty for node with no dependents" do
      consequences = Reader.assumption_consequences("paris_capital")
      assert consequences == []
    end
  end

  describe "tag_transitions/1" do
    test "returns FOLLOWED_BY edges for DET" do
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
    test "finds tokens with multiple POS tags" do
      ambiguous = Reader.ambiguous_tokens(2)
      assert length(ambiguous) >= 1
      assert Enum.any?(ambiguous, fn a -> a.token == "run" end)
    end
  end
end
