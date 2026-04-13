defmodule Brain.Analysis.ContextResolverTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.ContextResolver
  alias Brain.Analysis.SlotResult

  describe "resolve/2" do
    test "fills slot from conversation history" do
      slot_result = %SlotResult{
        schema_name: "weather.query",
        filled_slots: %{},
        missing_required: ["location"],
        missing_optional: [],
        all_required_filled: false
      }

      history = [
        %{
          entities: %{"location" => "New York"},
          intent: "weather.query",
          timestamp: System.system_time(:millisecond)
        }
      ]

      resolved = ContextResolver.resolve(slot_result, conversation_history: history)

      assert SlotResult.get_slot_value(resolved, "location") == "New York"
      assert resolved.all_required_filled == true
    end

    test "fills slot from user profile" do
      slot_result = %SlotResult{
        schema_name: "weather.query",
        filled_slots: %{},
        missing_required: ["location"],
        missing_optional: [],
        all_required_filled: false
      }

      profile = %{
        "location" => "San Francisco",
        "timezone" => "America/Los_Angeles"
      }

      resolved = ContextResolver.resolve(slot_result, user_profile: profile)

      assert SlotResult.get_slot_value(resolved, "location") == "San Francisco"
    end

    test "prefers conversation history over user profile" do
      slot_result = %SlotResult{
        schema_name: "weather.query",
        filled_slots: %{},
        missing_required: ["location"],
        missing_optional: [],
        all_required_filled: false
      }

      history = [
        %{
          entities: %{"location" => "Boston"},
          intent: "weather.query",
          timestamp: System.system_time(:millisecond)
        }
      ]

      profile = %{"location" => "San Francisco"}

      resolved =
        ContextResolver.resolve(slot_result,
          conversation_history: history,
          user_profile: profile
        )

      # History should take precedence
      assert SlotResult.get_slot_value(resolved, "location") == "Boston"
    end

    test "respects history depth limit" do
      slot_result = %SlotResult{
        schema_name: "weather.query",
        filled_slots: %{},
        missing_required: ["location"],
        missing_optional: [],
        all_required_filled: false
      }

      # Create history with location only in the 6th entry (beyond default depth of 5)
      history =
        1..6
        |> Enum.map(fn i ->
          if i == 6 do
            %{entities: %{"location" => "Old Location"}, intent: "other", timestamp: 0}
          else
            %{entities: %{}, intent: "other", timestamp: i}
          end
        end)

      resolved =
        ContextResolver.resolve(slot_result,
          conversation_history: history,
          history_depth: 5
        )

      # Should not find the location since it's beyond depth
      assert SlotResult.get_slot_value(resolved, "location") == nil
    end

    test "returns unchanged result when no missing slots" do
      slot_result = %SlotResult{
        schema_name: "weather.query",
        filled_slots: %{
          "location" => %{value: "NYC", source: :explicit, confidence: 1.0}
        },
        missing_required: [],
        missing_optional: [],
        all_required_filled: true
      }

      resolved = ContextResolver.resolve(slot_result)

      assert resolved == slot_result
    end
  end

  describe "extract_context/3" do
    test "extracts context snapshot from entities" do
      entities = [
        %{entity_type: "location", value: "Paris", confidence: 0.9},
        %{entity_type: "date", value: "tomorrow", confidence: 0.8}
      ]

      context = ContextResolver.extract_context(entities, "weather.query")

      assert context.entities["location"] == "Paris"
      assert context.entities["date"] == "tomorrow"
      assert context.intent == "weather.query"
      assert is_integer(context.timestamp)
    end

    test "handles empty entities" do
      context = ContextResolver.extract_context([], "unknown")

      assert context.entities == %{}
      assert context.intent == "unknown"
    end
  end

  describe "generate_clarification_prompts/2" do
    test "generates prompts for missing slots" do
      slot_result = %SlotResult{
        schema_name: "weather.query",
        filled_slots: %{},
        missing_required: ["location"],
        missing_optional: [],
        all_required_filled: false
      }

      prompts = ContextResolver.generate_clarification_prompts(slot_result)

      assert length(prompts) == 1
      assert hd(prompts) =~ "location"
    end

    test "uses custom templates from schema" do
      slot_result = %SlotResult{
        schema_name: "weather.query",
        filled_slots: %{},
        missing_required: ["location"],
        missing_optional: [],
        all_required_filled: false
      }

      schema = %{
        "clarification_templates" => %{
          "location" => "Where would you like the weather for?"
        }
      }

      prompts = ContextResolver.generate_clarification_prompts(slot_result, schema)

      assert hd(prompts) == "Where would you like the weather for?"
    end

    test "returns empty list when no missing required slots" do
      slot_result = %SlotResult{
        schema_name: "news.query",
        filled_slots: %{},
        missing_required: [],
        missing_optional: ["topic"],
        all_required_filled: true
      }

      prompts = ContextResolver.generate_clarification_prompts(slot_result)

      assert prompts == []
    end
  end
end
