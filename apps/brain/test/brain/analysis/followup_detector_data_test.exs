defmodule Brain.Analysis.FollowupDetectorDataTest do
  @moduledoc """
  Data-driven tests for FollowupDetector covering multi-turn conversation detection.
  """
  use ExUnit.Case, async: false

  alias Brain.Analysis.FollowupDetector

  # Helper to create a recent context - timestamp should be in milliseconds
  defp recent_context(attrs \\ %{}) do
    Map.merge(
      %{
        intent: "weather.query",
        missing_slots: ["location"],
        entities: [],
        slots: %{},
        timestamp: System.system_time(:millisecond)
      },
      attrs
    )
  end

  defp old_context(attrs \\ %{}) do
    # 10 minutes ago - beyond the 5 minute timeout (600_000 ms)
    old_timestamp = System.system_time(:millisecond) - 600_000
    Map.merge(
      %{
        intent: "weather.query",
        missing_slots: ["location"],
        entities: [],
        slots: %{},
        timestamp: old_timestamp
      },
      attrs
    )
  end

  # Test data - focus on return type correctness
  @followup_test_cases [
    # {input, context_fn, description}
    {"New York", nil, "nil context"},
    {"in Seattle", nil, "prepositional phrase without context"},
    {"New York", :old_context, "expired context"},
    {"in Seattle", :recent_context, "prepositional phrase with context"},
    {"at home", :recent_context, "prepositional 'at' with context"},
    {"Paris", :recent_context, "single capitalized word"},
    {"Los Angeles", :recent_context, "multi-word capitalized"},
    {"What's the weather?", :recent_context, "full question"},
    {"yes", :recent_context, "simple yes"},
    {"hello there", :recent_context, "greeting"},
  ]

  describe "is_followup?/2 - data driven" do
    for {input, context_type, description} <- @followup_test_cases do
      @input input
      @context_type context_type
      @description description

      test "returns boolean for: #{description}" do
        context = case @context_type do
          nil -> nil
          :old_context -> old_context()
          :recent_context -> recent_context()
        end

        result = FollowupDetector.is_followup?(@input, context)
        assert is_boolean(result)
      end
    end
  end

  # Test definite false cases
  describe "is_followup?/2 definite false cases" do
    test "nil context is not followup" do
      assert FollowupDetector.is_followup?("New York", nil) == false
    end

    test "expired context is not followup" do
      assert FollowupDetector.is_followup?("New York", old_context()) == false
    end
  end

  # Test get_carried_context/2
  describe "get_carried_context/2" do
    test "returns context map with required fields" do
      context = recent_context(%{intent: "weather.query", missing_slots: ["location"]})

      carried = FollowupDetector.get_carried_context("Seattle", context)

      assert is_map(carried)
      assert Map.has_key?(carried, :intent)
      assert Map.has_key?(carried, :carry_forward)
      assert Map.has_key?(carried, :new_input)
      assert carried.carry_forward == true
      assert carried.new_input == "Seattle"
    end
  end

  # Test merge_with_previous/2
  describe "merge_with_previous/2" do
    test "returns map with merged data" do
      context = %{
        intent: "weather.query",
        missing_slots: ["location"],
        previous_entities: [],
        previous_slots: %{}
      }
      new_entities = [%{type: "location", value: "Seattle"}]

      result = FollowupDetector.merge_with_previous(context, new_entities)

      assert is_map(result)
      # The function returns: :intent, :entities, :slots, :missing_slots, :all_required_filled
      assert Map.has_key?(result, :entities) or Map.has_key?(result, :missing_slots)
    end
  end

  # Edge cases
  describe "edge cases" do
    test "handles empty input" do
      result = FollowupDetector.is_followup?("", recent_context())
      assert is_boolean(result)
    end

    test "handles unicode input" do
      result = FollowupDetector.is_followup?("東京", recent_context())
      assert is_boolean(result)
    end

    test "handles long input" do
      long_input = String.duplicate("word ", 100)
      result = FollowupDetector.is_followup?(long_input, recent_context())
      assert is_boolean(result)
    end
  end
end
