defmodule Brain.Response.TemplateStoreTest do
  use ExUnit.Case

  alias Brain.Response.TemplateStore

  describe "filter_by_conditions/2" do
    test "filters templates that match condition" do
      templates = [
        %TemplateStore.Template{
          text: "Hello $person!",
          condition: "has_entity:person",
          embedding: nil,
          intent: "intro"
        },
        %TemplateStore.Template{
          text: "Hello stranger!",
          condition: "missing_entity:person",
          embedding: nil,
          intent: "intro"
        }
      ]

      context = %{
        entities: [%{entity_type: "person", value: "Austin"}]
      }

      result = TemplateStore.filter_by_conditions(templates, context)

      assert length(result) == 1
      assert List.first(result).text == "Hello $person!"
    end

    test "includes templates with nil condition (always match)" do
      templates = [
        %TemplateStore.Template{
          text: "Hello!",
          condition: nil,
          embedding: nil,
          intent: "smalltalk.greetings.hello"
        },
        %TemplateStore.Template{
          text: "Hello $person!",
          condition: "has_entity:person",
          embedding: nil,
          intent: "smalltalk.greetings.hello"
        }
      ]

      context = %{entities: []}

      result = TemplateStore.filter_by_conditions(templates, context)
      assert length(result) == 1
      assert List.first(result).text == "Hello!"
    end

    test "includes templates with empty string condition" do
      templates = [
        %TemplateStore.Template{
          text: "Hello!",
          condition: "",
          embedding: nil,
          intent: "smalltalk.greetings.hello"
        }
      ]

      context = %{entities: []}

      result = TemplateStore.filter_by_conditions(templates, context)
      assert length(result) == 1
    end

    test "filters by slot conditions" do
      templates = [
        %TemplateStore.Template{
          text: "Here's the weather for $location.",
          condition: "slot_filled:location",
          embedding: nil,
          intent: "weather.query"
        },
        %TemplateStore.Template{
          text: "What location?",
          condition: "slot_missing:location",
          embedding: nil,
          intent: "weather.query"
        }
      ]

      context = %{
        filled_slots: [],
        missing_slots: ["location"]
      }

      result = TemplateStore.filter_by_conditions(templates, context)

      assert length(result) == 1
      assert List.first(result).text == "What location?"
    end

    test "filters by confidence conditions" do
      templates = [
        %TemplateStore.Template{
          text: "I'm confident the answer is...",
          condition: "confidence:high",
          embedding: nil,
          intent: "answer"
        },
        %TemplateStore.Template{
          text: "I think the answer might be...",
          condition: "confidence:low",
          embedding: nil,
          intent: "answer"
        }
      ]

      context = %{confidence: 0.9}

      result = TemplateStore.filter_by_conditions(templates, context)

      assert length(result) == 1
      assert String.contains?(List.first(result).text, "confident")
    end

    test "filters by compound AND conditions" do
      templates = [
        %TemplateStore.Template{
          text: "Hello $person, here's your high-confidence answer.",
          condition: "has_entity:person AND confidence:high",
          embedding: nil,
          intent: "answer"
        }
      ]

      context_both = %{
        entities: [%{entity_type: "person", value: "Austin"}],
        confidence: 0.9
      }

      result_both = TemplateStore.filter_by_conditions(templates, context_both)
      assert length(result_both) == 1

      context_one = %{
        entities: [],
        confidence: 0.9
      }

      result_one = TemplateStore.filter_by_conditions(templates, context_one)
      assert result_one == []
    end

    test "filters by compound OR conditions" do
      templates = [
        %TemplateStore.Template{
          text: "I can help!",
          condition: "has_entity:person OR confidence:high",
          embedding: nil,
          intent: "help"
        }
      ]

      context_first = %{
        entities: [%{entity_type: "person", value: "Austin"}],
        confidence: 0.3
      }

      result_first = TemplateStore.filter_by_conditions(templates, context_first)
      assert length(result_first) == 1

      context_second = %{
        entities: [],
        confidence: 0.9
      }

      result_second = TemplateStore.filter_by_conditions(templates, context_second)
      assert length(result_second) == 1

      context_neither = %{
        entities: [],
        confidence: 0.3
      }

      result_neither = TemplateStore.filter_by_conditions(templates, context_neither)
      assert result_neither == []
    end
  end

  describe "rank_by_similarity/2" do
    test "ranks templates by embedding similarity" do
      templates = [
        %TemplateStore.Template{
          text: "Low similarity",
          condition: nil,
          embedding: [0.1, 0.0, 0.0],
          intent: "test"
        },
        %TemplateStore.Template{
          text: "High similarity",
          condition: nil,
          embedding: [0.9, 0.8, 0.7],
          intent: "test"
        },
        %TemplateStore.Template{
          text: "Medium similarity",
          condition: nil,
          embedding: [0.5, 0.4, 0.3],
          intent: "test"
        }
      ]

      query_embedding = [0.9, 0.8, 0.7]

      result = TemplateStore.rank_by_similarity(templates, query_embedding)
      assert List.first(result).text == "High similarity"
      assert List.last(result).text == "Low similarity"
    end

    test "handles templates without embeddings" do
      templates = [
        %TemplateStore.Template{
          text: "No embedding",
          condition: nil,
          embedding: nil,
          intent: "test"
        },
        %TemplateStore.Template{
          text: "Has embedding",
          condition: nil,
          embedding: [0.9, 0.8, 0.7],
          intent: "test"
        }
      ]

      query_embedding = [0.9, 0.8, 0.7]

      result = TemplateStore.rank_by_similarity(templates, query_embedding)
      assert List.first(result).text == "Has embedding"
    end

    test "returns empty list for empty input" do
      result = TemplateStore.rank_by_similarity([], [0.1, 0.2, 0.3])
      assert result == []
    end
  end

  describe "substitute_slots/2" do
    test "substitutes $slot format" do
      template = "Hello $person, welcome to $location!"

      entities = [
        %{entity_type: "person", value: "Austin"},
        %{entity_type: "location", value: "Texas"}
      ]

      result = TemplateStore.substitute_slots(template, entities)

      assert result == "Hello Austin, welcome to Texas!"
    end

    test "substitutes @slot format" do
      template = "Playing @song by @artist"

      entities = [
        %{entity_type: "song", value: "Bohemian Rhapsody"},
        %{entity_type: "music-artist", value: "Queen"}
      ]

      result = TemplateStore.substitute_slots(template, entities)

      assert result == "Playing Bohemian Rhapsody by Queen"
    end

    test "leaves unmatched placeholders" do
      template = "Hello $person!"

      entities = []

      result = TemplateStore.substitute_slots(template, entities)

      assert result == "Hello $person!"
    end

    test "handles multiple entities of same type" do
      template = "Weather in $location is sunny."

      entities = [
        %{entity_type: "location", value: "Austin"},
        %{entity_type: "location", value: "Dallas"}
      ]

      result = TemplateStore.substitute_slots(template, entities)
      assert result == "Weather in Austin is sunny." or result == "Weather in Dallas is sunny."
    end
  end

  describe "integration with TemplateStore GenServer" do
    test "ready? returns boolean" do
      result = TemplateStore.ready?()
      assert is_boolean(result)
    end

    test "list_intents returns list" do
      if TemplateStore.ready?() do
        result = TemplateStore.list_intents()
        assert is_list(result)
      end
    end

    test "stats returns map with expected keys" do
      if TemplateStore.ready?() do
        stats = TemplateStore.stats()

        assert is_map(stats)
        assert Map.has_key?(stats, :intent_count)
        assert Map.has_key?(stats, :template_count)
        assert Map.has_key?(stats, :ready)
      end
    end
  end
end
