defmodule Brain.Response.ConditionEvaluatorTest do
  use ExUnit.Case, async: false

  alias Brain.Response.ConditionEvaluator

  describe "evaluate/2 with nil/empty conditions" do
    test "returns true for nil condition" do
      assert ConditionEvaluator.evaluate(nil, %{}) == true
    end

    test "returns true for empty string condition" do
      assert ConditionEvaluator.evaluate("", %{}) == true
    end
  end

  describe "evaluate/2 with entity conditions" do
    test "has_entity returns true when entity type is present" do
      context = %{
        entities: [
          %{entity_type: "person", value: "Austin"}
        ]
      }

      assert ConditionEvaluator.evaluate("has_entity:person", context) == true
    end

    test "has_entity returns false when entity type is not present" do
      context = %{
        entities: [
          %{entity_type: "location", value: "Austin"}
        ]
      }

      assert ConditionEvaluator.evaluate("has_entity:person", context) == false
    end

    test "missing_entity returns true when entity type is not present" do
      context = %{
        entities: [
          %{entity_type: "location", value: "Austin"}
        ]
      }

      assert ConditionEvaluator.evaluate("missing_entity:person", context) == true
    end

    test "missing_entity returns false when entity type is present" do
      context = %{
        entities: [
          %{entity_type: "person", value: "Austin"}
        ]
      }

      assert ConditionEvaluator.evaluate("missing_entity:person", context) == false
    end

    test "handles string keys in entity maps" do
      context = %{
        entities: [
          %{"entity_type" => "person", "value" => "Austin"}
        ]
      }

      assert ConditionEvaluator.evaluate("has_entity:person", context) == true
    end
  end

  describe "evaluate/2 with slot conditions" do
    test "slot_filled returns true when slot is in filled_slots" do
      context = %{
        filled_slots: ["address", "date"]
      }

      assert ConditionEvaluator.evaluate("slot_filled:address", context) == true
    end

    test "slot_filled returns false when slot is not filled" do
      context = %{
        filled_slots: ["date"]
      }

      assert ConditionEvaluator.evaluate("slot_filled:address", context) == false
    end

    test "slot_missing returns true when slot is in missing_slots" do
      context = %{
        missing_slots: ["location", "time"]
      }

      assert ConditionEvaluator.evaluate("slot_missing:location", context) == true
    end

    test "slot_missing returns false when slot is not missing" do
      context = %{
        missing_slots: ["time"]
      }

      assert ConditionEvaluator.evaluate("slot_missing:location", context) == false
    end
  end

  describe "evaluate/2 with confidence conditions" do
    test "confidence:high returns true when confidence >= 0.8" do
      context = %{confidence: 0.85}
      assert ConditionEvaluator.evaluate("confidence:high", context) == true
    end

    test "confidence:high returns false when confidence < 0.8" do
      context = %{confidence: 0.75}
      assert ConditionEvaluator.evaluate("confidence:high", context) == false
    end

    test "confidence:medium returns true when 0.5 <= confidence < 0.8" do
      context = %{confidence: 0.65}
      assert ConditionEvaluator.evaluate("confidence:medium", context) == true
    end

    test "confidence:low returns true when confidence < 0.5" do
      context = %{confidence: 0.3}
      assert ConditionEvaluator.evaluate("confidence:low", context) == true
    end

    test "defaults to 0.0 confidence when not provided" do
      context = %{}
      assert ConditionEvaluator.evaluate("confidence:low", context) == true
    end
  end

  describe "evaluate/2 with speech_act conditions" do
    test "speech_act returns true when sub_type matches (atom)" do
      context = %{
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      assert ConditionEvaluator.evaluate("speech_act:greeting", context) == true
    end

    test "speech_act returns true when sub_type matches (string)" do
      context = %{
        speech_act: %{category: :expressive, sub_type: "greeting"}
      }

      assert ConditionEvaluator.evaluate("speech_act:greeting", context) == true
    end

    test "speech_act returns false when sub_type does not match" do
      context = %{
        speech_act: %{category: :expressive, sub_type: :farewell}
      }

      assert ConditionEvaluator.evaluate("speech_act:greeting", context) == false
    end
  end

  describe "evaluate/2 with compound conditions" do
    test "AND returns true when both conditions are true" do
      context = %{
        entities: [%{entity_type: "person", value: "Austin"}],
        confidence: 0.9
      }

      assert ConditionEvaluator.evaluate("has_entity:person AND confidence:high", context) == true
    end

    test "AND returns false when first condition is false" do
      context = %{
        entities: [],
        confidence: 0.9
      }

      assert ConditionEvaluator.evaluate("has_entity:person AND confidence:high", context) == false
    end

    test "AND returns false when second condition is false" do
      context = %{
        entities: [%{entity_type: "person", value: "Austin"}],
        confidence: 0.5
      }

      assert ConditionEvaluator.evaluate("has_entity:person AND confidence:high", context) == false
    end

    test "OR returns true when first condition is true" do
      context = %{
        entities: [%{entity_type: "person", value: "Austin"}],
        confidence: 0.3
      }

      assert ConditionEvaluator.evaluate("has_entity:person OR confidence:high", context) == true
    end

    test "OR returns true when second condition is true" do
      context = %{
        entities: [],
        confidence: 0.9
      }

      assert ConditionEvaluator.evaluate("has_entity:person OR confidence:high", context) == true
    end

    test "OR returns false when both conditions are false" do
      context = %{
        entities: [],
        confidence: 0.3
      }

      assert ConditionEvaluator.evaluate("has_entity:person OR confidence:high", context) == false
    end
  end

  describe "evaluate/2 with chained conditions" do
    test "handles multiple ANDs" do
      context = %{
        entities: [%{entity_type: "person", value: "Austin"}],
        confidence: 0.9,
        speech_act: %{sub_type: :greeting}
      }

      condition = "has_entity:person AND confidence:high AND speech_act:greeting"
      assert ConditionEvaluator.evaluate(condition, context) == true
    end

    test "handles multiple ORs" do
      context = %{
        entities: [],
        confidence: 0.3,
        missing_slots: ["location"]
      }

      condition = "has_entity:person OR confidence:high OR slot_missing:location"
      assert ConditionEvaluator.evaluate(condition, context) == true
    end
  end

  describe "parse/1" do
    test "parses simple condition" do
      assert ConditionEvaluator.parse("has_entity:person") ==
               {:condition, "has_entity", "person"}
    end

    test "parses empty string as always true" do
      assert ConditionEvaluator.parse("") == {:always, true}
    end

    test "parses AND expression" do
      result = ConditionEvaluator.parse("has_entity:person AND confidence:high")

      assert result ==
               {:and, {:condition, "has_entity", "person"}, {:condition, "confidence", "high"}}
    end

    test "parses OR expression" do
      result = ConditionEvaluator.parse("has_entity:person OR confidence:high")

      assert result ==
               {:or, {:condition, "has_entity", "person"}, {:condition, "confidence", "high"}}
    end
  end

  describe "evaluate/2 edge cases" do
    test "handles unknown condition type gracefully" do
      context = %{}
      assert ConditionEvaluator.evaluate("unknown_type:value", context) == false
    end

    test "handles malformed condition gracefully" do
      context = %{}
      # Should not crash
      result = ConditionEvaluator.evaluate("malformed", context)
      assert is_boolean(result)
    end

    test "handles whitespace in condition" do
      context = %{
        entities: [%{entity_type: "person", value: "Austin"}]
      }

      assert ConditionEvaluator.evaluate("  has_entity:person  ", context) == true
    end
  end

  describe "evaluate/2 with enrichment conditions" do
    test "enriched returns true when field is present with atom key" do
      context = %{
        enriched_data: %{temperature: "72°F", conditions: "sunny"}
      }

      assert ConditionEvaluator.evaluate("enriched:temperature", context) == true
    end

    test "enriched returns true when field is present with string key" do
      context = %{
        enriched_data: %{"temperature" => "72°F", "conditions" => "sunny"}
      }

      assert ConditionEvaluator.evaluate("enriched:temperature", context) == true
    end

    test "enriched returns false when field is not present" do
      context = %{
        enriched_data: %{conditions: "sunny"}
      }

      assert ConditionEvaluator.evaluate("enriched:temperature", context) == false
    end

    test "enriched returns false when enriched_data is empty" do
      context = %{
        enriched_data: %{}
      }

      assert ConditionEvaluator.evaluate("enriched:temperature", context) == false
    end

    test "enriched returns false when enriched_data is not present" do
      context = %{}
      assert ConditionEvaluator.evaluate("enriched:temperature", context) == false
    end

    test "enrichment_failed returns true when status is :failed" do
      context = %{
        enrichment_status: :failed
      }

      assert ConditionEvaluator.evaluate("enrichment_failed", context) == true
    end

    test "enrichment_failed returns false when status is :success" do
      context = %{
        enrichment_status: :success
      }

      assert ConditionEvaluator.evaluate("enrichment_failed", context) == false
    end

    test "enrichment_failed returns false when status is not set" do
      context = %{}
      assert ConditionEvaluator.evaluate("enrichment_failed", context) == false
    end

    test "enrichment_success returns true when status is :success" do
      context = %{
        enrichment_status: :success
      }

      assert ConditionEvaluator.evaluate("enrichment_success", context) == true
    end

    test "enrichment_success returns false when status is :failed" do
      context = %{
        enrichment_status: :failed
      }

      assert ConditionEvaluator.evaluate("enrichment_success", context) == false
    end
  end

  describe "evaluate/2 with NOT conditions" do
    test "NOT negates a true condition" do
      context = %{
        entities: [%{entity_type: "person", value: "Austin"}]
      }

      assert ConditionEvaluator.evaluate("NOT has_entity:person", context) == false
    end

    test "NOT negates a false condition" do
      context = %{
        entities: []
      }

      assert ConditionEvaluator.evaluate("NOT has_entity:person", context) == true
    end

    test "NOT works with enrichment conditions" do
      context = %{
        enriched_data: %{temperature: "72°F"}
      }

      assert ConditionEvaluator.evaluate("NOT enriched:humidity", context) == true
    end

    test "AND with NOT works correctly" do
      context = %{
        filled_slots: ["location"],
        enriched_data: %{}
      }

      condition = "slot_filled:location AND NOT enriched:temperature"
      assert ConditionEvaluator.evaluate(condition, context) == true
    end
  end

  describe "evaluate/2 with compound enrichment conditions" do
    test "enriched AND slot_filled combination" do
      context = %{
        filled_slots: ["location"],
        enriched_data: %{temperature: "72°F", conditions: "sunny"}
      }

      condition = "slot_filled:location AND enriched:temperature"
      assert ConditionEvaluator.evaluate(condition, context) == true
    end

    test "enriched AND slot_filled fails when enrichment missing" do
      context = %{
        filled_slots: ["location"],
        enriched_data: %{}
      }

      condition = "slot_filled:location AND enriched:temperature"
      assert ConditionEvaluator.evaluate(condition, context) == false
    end

    test "complex weather template condition" do
      context = %{
        filled_slots: ["location"],
        enriched_data: %{temperature: "72°F", conditions: "sunny", humidity: "45%"},
        enrichment_status: :success
      }

      condition = "enriched:temperature AND enriched:humidity"
      assert ConditionEvaluator.evaluate(condition, context) == true
    end

    test "fallback template condition when enrichment fails" do
      context = %{
        filled_slots: ["location"],
        enriched_data: %{},
        enrichment_status: :failed
      }

      condition = "slot_filled:location AND enrichment_failed"
      assert ConditionEvaluator.evaluate(condition, context) == true
    end
  end

  describe "parse/1 with NOT" do
    test "parses NOT expression" do
      result = ConditionEvaluator.parse("NOT has_entity:person")
      assert result == {:not, {:condition, "has_entity", "person"}}
    end

    test "parses AND with NOT" do
      result = ConditionEvaluator.parse("slot_filled:location AND NOT enriched:temperature")

      assert result == {
               :and,
               {:condition, "slot_filled", "location"},
               {:not, {:condition, "enriched", "temperature"}}
             }
    end
  end
end
