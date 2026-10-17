defmodule Brain.Analysis.FollowupDetectorDataTest do
  @moduledoc """
  Data-driven tests for FollowupDetector covering multi-turn conversation detection.
  """
  use ExUnit.Case, async: false

  alias Brain.Analysis.FollowupDetector

  # A reply's entities fill the pending intent's slots through
  # `SlotDetector.detect/3`, which weighs an entity whose type is not the
  # slot's own by the type hierarchy, reading WordNet; and `is_followup?/2`
  # tags a message with the POS tagger, which reads WordNet through its
  # lexical features.
  setup_all do
    Brain.TestHelpers.start_pos_tagger!()
  end

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

  describe "a reply to a service's clarifying question" do
    @dentist %{event: "Dentist", date: "Friday", time: "3 pm"}
    @dentist_monday %{event: "Dentist", date: "Monday", time: "3 pm"}
    @standup %{event: "Standup", date: "", time: "5 pm"}

    defp ambiguous(candidates) do
      %{
        enrichment_status: :failed,
        enrichment_error: {:ambiguous, :calendar, candidates},
        enrichment_service: :local_scheduler,
        intent: "calendar.cancel",
        slots: %{time: "3 pm"}
      }
    end

    defp pending_context(candidates) do
      recent_context(FollowupDetector.awaiting_answer(ambiguous(candidates)))
    end

    defp pending(candidates),
      do: %{item_kind: :calendar, candidates: candidates, service: :local_scheduler, intent: "calendar.cancel"}

    defp reply(text, entities \\ []), do: %{text: text, entities: entities}

    test "an ambiguous cancel waits for one of its candidates, with the request it was asked with" do
      assert FollowupDetector.awaiting_answer(ambiguous([@dentist, @standup])) == %{
               intent: "calendar.cancel",
               slots: %{"time" => "3 pm"},
               missing_slots: [],
               pending_clarification: pending([@dentist, @standup])
             }
    end

    test "a cancel that identified no event waits for one of every event, which the reply lists" do
      enrichment = %{ambiguous([]) | enrichment_error: {:event_not_identified, [@dentist, @standup]}, slots: %{}}

      assert %{pending_clarification: pending} = FollowupDetector.awaiting_answer(enrichment)
      assert pending == pending([@dentist, @standup])

      {:ok, question} =
        {:event_not_identified, [@dentist, @standup]}
        |> Brain.Response.DiscoursePlanner.enrichment_error_primitive(%{intent: "calendar.cancel"})
        |> Map.fetch!(:content)
        |> Brain.Response.Synthesizer.synthesize_service_reply()

      assert question =~ "Dentist on Friday at 3 pm or Standup at 5 pm"

      assert FollowupDetector.resolve_pending(reply("the standup"), pending) == {:resolved, @standup}
      assert FollowupDetector.complete_pending(%{slots: %{}, pending_clarification: pending}, @standup) ==
               %{"event" => "Standup", "time" => "5 pm"}
    end

    test "every reply after a clarification is read against it first, until the context expires" do
      assert FollowupDetector.is_followup?("what's the weather in paris", pending_context([@dentist, @standup]))
      refute FollowupDetector.is_followup?("the dentist", old_context(%{pending_clarification: pending([@dentist])}))
    end

    test "a question a slot answers waits for that slot" do
      enrichment = %{
        enrichment_status: :failed,
        enrichment_error: :location_not_found,
        enrichment_service: :weather,
        intent: "weather.query",
        slots: %{location: "Atlantis", date: "today"}
      }

      assert FollowupDetector.awaiting_answer(enrichment) == %{
               intent: "weather.query",
               slots: %{"date" => "today"},
               missing_slots: ["location"]
             }
    end

    test "a reply that asked nothing waits for nothing" do
      assert FollowupDetector.awaiting_answer(%{
               enrichment_status: :failed,
               enrichment_error: :rate_limited,
               enrichment_service: :weather,
               intent: "weather.query",
               slots: %{}
             }) == %{}

      assert FollowupDetector.awaiting_answer(%{enrichment_status: :success}) == %{}
    end

    test "naming a candidate's field resolves to it, and completes the request with its slots" do
      context = pending_context([@dentist, @standup])

      assert FollowupDetector.resolve_pending(reply("the dentist one"), context.pending_clarification) ==
               {:resolved, @dentist}

      assert FollowupDetector.is_followup?("the dentist one", context)

      assert FollowupDetector.complete_pending(context, @dentist) ==
               %{"event" => "Dentist", "date" => "Friday", "time" => "3 pm"}
    end

    test "the candidate naming the most fields wins" do
      assert FollowupDetector.resolve_pending(reply("dentist on monday"), pending([@dentist, @dentist_monday])) ==
               {:resolved, @dentist_monday}
    end

    test "naming some of the candidates narrows them" do
      assert FollowupDetector.resolve_pending(reply("the dentist"), pending([@dentist, @dentist_monday, @standup])) ==
               {:narrowed, [@dentist, @dentist_monday]}
    end

    test "a reply naming none of them, or all of them, is unrelated" do
      context = pending_context([@dentist, @dentist_monday])

      assert FollowupDetector.resolve_pending(reply("what's the weather in paris"), context.pending_clarification) ==
               :unrelated

      assert FollowupDetector.resolve_pending(reply("the dentist"), context.pending_clarification) == :unrelated
    end

    test "part of a value names its candidate, and a reply that names several equally asks again among them" do
      appointment = %{event: "Dentist appointment", date: "Friday", time: "3 pm"}
      checkup = %{event: "Dentist checkup", date: "Monday", time: "9 am"}

      assert FollowupDetector.resolve_pending(reply("the dentist"), pending([appointment, @standup])) ==
               {:resolved, appointment}

      assert FollowupDetector.resolve_pending(reply("dentist"), pending([appointment, checkup, @standup])) ==
               {:narrowed, [appointment, checkup]}

      assert FollowupDetector.resolve_pending(reply("the checkup"), pending([appointment, checkup, @standup])) ==
               {:resolved, checkup}
    end

    test "a word every candidate holds tells none apart" do
      at_three = %{event: "Standup", date: "", time: "3 pm"}

      assert FollowupDetector.resolve_pending(reply("the 3 pm one"), pending([at_three, @standup])) ==
               {:resolved, at_three}

      assert FollowupDetector.resolve_pending(reply("the pm one"), pending([at_three, @standup])) == :unrelated
    end

    test "a date or time entity of the reply names the candidate with that date or time, case and spacing aside" do
      tomorrow = %{event: "Standup", date: "tomorrow", time: "9 am"}

      assert FollowupDetector.resolve_pending(
               reply("the 3pm one", [%{entity_type: "time", value: "3pm"}]),
               pending([@dentist, @standup])
             ) == {:resolved, @dentist}

      assert FollowupDetector.resolve_pending(
               reply("the one tomorrow", [%{entity_type: "date", value: "Tomorrow"}]),
               pending([@dentist, tomorrow])
             ) == {:resolved, tomorrow}

      # Without the entity, "3pm" is no word of "3 pm".
      assert FollowupDetector.resolve_pending(reply("the 3pm one"), pending([@dentist, @standup])) == :unrelated
    end

    test "a reply naming a candidate by position names none: no lexicon records ordinals" do
      for text <- ["the first one", "the second", "the last one"] do
        assert FollowupDetector.resolve_pending(reply(text), pending([@dentist, @standup])) == :unrelated
      end
    end

    test "a single candidate is resolved by naming it" do
      assert FollowupDetector.resolve_pending(reply("dentist"), pending([@dentist])) == {:resolved, @dentist}
      assert FollowupDetector.resolve_pending(reply("yes"), pending([@dentist])) == :unrelated
    end

    test "candidates from a service that is not a scheduling driver raise" do
      assert_raise ArgumentError, ~r/no registered service/, fn ->
        FollowupDetector.resolve_pending(reply("dentist"), %{pending([@dentist]) | service: :nonexistent})
      end

      assert_raise ArgumentError, ~r/is not a scheduling driver/, fn ->
        FollowupDetector.resolve_pending(reply("dentist"), %{pending([@dentist]) | service: :weather})
      end
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
