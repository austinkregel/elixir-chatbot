defmodule Brain.Analysis.SlotDetectorFreeTextTest do
  # Reads the capture model `mix train_free_text_capture` wrote to the models
  # directory, and the POS model capture tags an utterance with.
  use ExUnit.Case, async: false

  alias Brain.Analysis.{FreeTextCapture, SlotDetector, SlotResult}

  setup do
    %{model: FreeTextCapture.read!("en")}
  end

  defp detect(intent, entities, text, model),
    do: SlotDetector.detect(intent, entities, side_effects: false, text: text, free_text_capture: model)

  describe "a free-text slot" do
    test "is filled with the span the capture model captures, as captured, with its confidence", %{model: model} do
      result = detect("reminder.create", [], "Can you remind me to call John", model)

      assert %{value: "call John", source: :captured, confidence: confidence} = result.filled_slots["reminder_text"]
      assert confidence >= model.thresholds["reminder_text"]
      refute "reminder_text" in result.missing_optional
    end

    test "is captured for every intent that declares it: a check and a cancel name the reminder too", %{model: model} do
      for {intent, text} <- [{"reminder.check", "Did I set a reminder to call John"}, {"reminder.cancel", "cancel my reminder to call John"}] do
        assert SlotResult.get_slot_value(detect(intent, [], text, model), "reminder_text") == "call John", intent
      end
    end

    test "with no decidable span stays unfilled, among the missing slots, and its clarification asks for it", %{model: model} do
      result = detect("reminder.create", [], "Create a reminder", model)

      refute Map.has_key?(result.filled_slots, "reminder_text")
      assert "reminder_text" in result.missing_optional
      assert SlotDetector.get_clarification_prompt("reminder_text", "reminder.create") =~ ~r/\?$/
    end

    test "holds no token of an entity that fills another slot", %{model: model} do
      text = "Can you remind me to call John at 5pm"
      {at, length} = :binary.match(text, "5pm")
      time = %{entity_type: "time", value: "5pm", confidence: 0.9, start_pos: at, end_pos: at + length - 1}

      result = detect("reminder.create", [time], text, model)

      assert SlotResult.get_slot_value(result, "time") == "5pm"

      case SlotResult.get_slot_value(result, "reminder_text") do
        nil -> :ok
        value -> refute value =~ "5pm"
      end
    end

    test "is not captured without the utterance" do
      result = SlotDetector.detect("reminder.create", [], side_effects: false)
      refute Map.has_key?(result.filled_slots, "reminder_text")
    end

    test "an entity filling a slot with no position raises when the utterance is given: no span could be kept off it",
         %{model: model} do
      assert_raise ArgumentError, ~r/carries no :start_pos and :end_pos/, fn ->
        detect("reminder.create", [%{entity_type: "time", value: "5pm", confidence: 0.9}], "remind me to call John at 5pm", model)
      end
    end

    test "the utterance without a language or a model raises" do
      assert_raise ArgumentError, ~r/:text needs :language/, fn ->
        SlotDetector.detect("reminder.create", [], text: "remind me to call John")
      end
    end
  end
end
