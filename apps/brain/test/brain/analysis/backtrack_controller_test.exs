defmodule Brain.Analysis.BacktrackControllerTest do
  @moduledoc """
  Tests for the entity-mismatch contradiction check, which was a permanent
  `:ok` stub. Two guarded signals now fire: self-consistency (entities vs the
  intent domain's expected types) and cross-turn (same value, incompatible type
  across turns).
  """
  use ExUnit.Case, async: true

  alias Brain.Analysis.{BacktrackController, Interpretation}

  defp interp(intent, entities, activation \\ 0.8) do
    Interpretation.new(intent, "text", activation, :model)
    |> Interpretation.with_entities(entities)
  end

  describe "self-consistency entity mismatch" do
    test "flags when all qualifying entities are incompatible with the intent domain" do
      i = interp("weather.query", [%{entity_type: "person", value: "Alice", confidence: 0.9}])

      assert {:needs_backtrack, {:entity_mismatch, {:intent_domain, "weather.query", _}}} =
               BacktrackController.check_for_contradictions(i)
    end

    test "does not flag when an entity is compatible with the intent domain" do
      i = interp("weather.query", [%{entity_type: "location", value: "Paris", confidence: 0.9}])
      assert :ok = BacktrackController.check_for_contradictions(i)
    end

    test "guards: low-confidence and OOV proper-noun entities never force a backtrack" do
      low = interp("weather.query", [%{entity_type: "person", value: "Alice", confidence: 0.3}])
      assert :ok = BacktrackController.check_for_contradictions(low)

      propn =
        interp("weather.query", [
          %{entity_type: "person", value: "Zzy", confidence: 0.9, source: :pos_tagger_propn}
        ])

      assert :ok = BacktrackController.check_for_contradictions(propn)
    end

    test "no entities / unknown domain → no contradiction" do
      assert :ok = BacktrackController.check_for_contradictions(interp("weather.query", []))
      assert :ok = BacktrackController.check_for_contradictions(interp("mystery.thing", [
               %{entity_type: "person", value: "Alice", confidence: 0.9}
             ]))
    end
  end

  describe "cross-turn entity mismatch" do
    test "flags an entity whose type contradicts the same value in a recent turn" do
      current =
        interp("smalltalk.chitchat", [%{entity_type: "person", value: "Paris", confidence: 0.9}])

      history = [%{entities: [%{entity_type: "location", value: "Paris", confidence: 0.9}]}]

      assert {:needs_backtrack, {:entity_mismatch, {:cross_turn, "Paris", "person"}}} =
               BacktrackController.check_for_contradictions(current, conversation_history: history)
    end

    test "does not flag when the same value keeps a compatible type across turns" do
      current =
        interp("smalltalk.chitchat", [%{entity_type: "person", value: "Alice", confidence: 0.9}])

      history = [%{entities: [%{entity_type: "person", value: "Alice", confidence: 0.9}]}]
      assert :ok = BacktrackController.check_for_contradictions(current, conversation_history: history)
    end

    test "no history opt = no cross-turn contradiction (backward compatible /1)" do
      current =
        interp("smalltalk.chitchat", [%{entity_type: "person", value: "Paris", confidence: 0.9}])

      assert :ok = BacktrackController.check_for_contradictions(current)
    end
  end
end
