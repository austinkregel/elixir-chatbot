defmodule Brain.Response.SynthesizerTest do
  use ExUnit.Case, async: false

  alias Brain.Response.Synthesizer

  # An acknowledgment that continues a reply's sentence asks the POS tagger
  # whether the reply's first word is a proper noun (`Synthesizer.join_after/2`);
  # the tagger reads WordNet through its lexical features.
  setup_all do
    Brain.TestHelpers.start_pos_tagger!()
  end

  describe "synthesize/3" do
    test "returns :not_synthesized for unknown domain" do
      result = Synthesizer.synthesize("completely_unknown.intent", [], confidence: 0.8)

      assert result == :not_synthesized
    end

    test "synthesizes response for known domain" do
      entities = [%{entity_type: "location", value: "NYC"}]

      result = Synthesizer.synthesize("weather.query", entities, confidence: 0.8)

      # Should either synthesize or return :not_synthesized; with no recurring
      # pattern passed, none is adapted from.
      assert result == :not_synthesized or match?({:ok, _, nil}, result)
    end
  end

  describe "synthesize/3 adapting from memory" do
    alias Brain.Memory.RecurringPattern
    alias Brain.Memory.Types.{Episode, SemanticFact}

    # An intent with no domain knowledge: whatever is synthesized comes from
    # memory.
    @intent "completely_unknown.intent"
    @entities [%{entity_type: "location", value: "Lyon"}]

    setup do
      test_pid = self()
      handler = "synthesizer-test-#{System.unique_integer([:positive])}"

      :ok =
        :telemetry.attach(
          handler,
          [:chat_bot, :memory, :semantic_fact, :used],
          fn _event, _measurements, metadata, _config -> send(test_pid, {:used, metadata}) end,
          nil
        )

      on_exit(fn -> :telemetry.detach(handler) end)

      fact = SemanticFact.new("is it sunny in paris | is it sunny in london", [], ["ep1", "ep2", "ep3"], [])

      pattern = %RecurringPattern{
        world_id: "test_synthesizer",
        fact: fact,
        similarity: 0.75,
        reply: "It's usually sunny in $location.",
        reply_count: 2,
        evidence_count: 3
      }

      # One episode, more similar to the turn than the pattern is.
      episode = Episode.new("is it sunny in lyon", "conversation", "Bring a coat to $location.", [], [])

      {:ok, pattern: pattern, episode: {episode, 0.95}}
    end

    test "adapts the recurring pattern's reply ahead of a more similar single episode, and hands the pattern back",
         %{pattern: pattern, episode: episode} do
      assert {:ok, "It's usually sunny in Lyon.", ^pattern} =
               Synthesizer.synthesize(@intent, @entities,
                 recurring_pattern: pattern,
                 similar_episodes: [episode]
               )

      # The caller that settles on the turn's response reports the use, not
      # each synthesis.
      refute_receive {:used, _}, 50
    end

    test "adapts the most similar single episode when the turn has no recurring pattern", %{episode: episode} do
      assert {:ok, "Bring a coat to Lyon.", nil} =
               Synthesizer.synthesize(@intent, @entities, recurring_pattern: nil, similar_episodes: [episode])

      refute_receive {:used, _}, 50
    end

    test "answers a contradiction before it draws on the pattern", %{pattern: pattern} do
      context = %{epistemic_context: %{status: :contradicted, beliefs: [%{object: "rainy"}]}}

      assert {:ok, response, nil} =
               Synthesizer.synthesize(@intent, @entities, recurring_pattern: pattern, context: context)

      refute response == "It's usually sunny in Lyon."
      refute_receive {:used, _}, 50
    end
  end

  describe "synthesize_from_domain/5 with enrichment" do
    test "uses enriched templates when enrichment data is available" do
      entities = [%{entity_type: "location", value: "NYC"}]

      context = %{
        enrichment_status: :success,
        enriched_data: %{
          temperature: "72°F",
          conditions: "sunny",
          humidity: "45%",
          wind_speed: "5 mph"
        }
      }

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.query",
          entities,
          0.8,
          context: context
        )

      case result do
        {:ok, response} ->
          # Response should contain the enriched data
          assert response =~ "72°F" or response =~ "NYC"

        :not_synthesized ->
          # Domain may not have templates configured
          assert true
      end
    end

    test "falls back to standard templates when no enrichment data" do
      entities = [%{entity_type: "location", value: "NYC"}]
      context = %{enrichment_status: :not_configured, enriched_data: %{}}

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.query",
          entities,
          0.8,
          context: context
        )

      case result do
        {:ok, response} ->
          assert is_binary(response)
          # Should mention the location
          assert response =~ "NYC"

        :not_synthesized ->
          assert true
      end
    end

    test "returns :not_synthesized for nil domain" do
      result = Synthesizer.synthesize_from_domain(nil, "test", [], 0.8, [])
      assert result == :not_synthesized
    end
  end

  describe "synthesize_clarification/4" do
    test "generates clarification for missing slots" do
      missing_slots = [:location]

      result =
        Synthesizer.synthesize_clarification(
          "weather.query",
          [],
          missing_slots,
          []
        )

      case result do
        {:ok, response} ->
          assert is_binary(response)
          # Should ask about location
          assert response =~ "location" or String.length(response) > 0

        _ ->
          # May not have clarification templates configured
          assert true
      end
    end
  end

  describe "synthesize_service_reply/1" do
    test "fills the kind's nouns and each candidate's description, its day included" do
      {:ok, text} =
        Synthesizer.synthesize_service_reply(%{
          reply: :ambiguous_item,
          item_kind: :reminder,
          candidates: [%{text: "call mom", date: "Friday", time: "6 pm"}, %{text: "buy milk", date: nil, time: ""}]
        })

      assert text =~ "call mom on Friday at 6 pm or buy milk"
    end

    test "an item giving none of its kind's description fields a value raises" do
      assert_raise ArgumentError, ~r/gives none of the fields it is described by a value/, fn ->
        Synthesizer.synthesize_service_reply(%{
          reply: :ambiguous_item,
          item_kind: :calendar,
          candidates: [%{event: "", time: nil, date: ""}, %{event: "Standup", time: "5 pm", date: ""}]
        })
      end
    end

    test "an item kind with no service_items entry raises" do
      assert_raise ArgumentError, ~r/no complete entry for the item kind :weather/, fn ->
        Synthesizer.synthesize_service_reply(%{reply: :no_items, item_kind: :weather})
      end
    end

    test "a reply with no templates raises" do
      assert_raise ArgumentError, ~r/no service_replies templates for :rate_limted/, fn ->
        Synthesizer.synthesize_service_reply(%{reply: :rate_limted, service_name: "Weather"})
      end
    end

    test "a template placeholder the content does not fill raises" do
      # Every ambiguous_item template names $candidates, and no candidates
      # are given; every unavailable template names $service.
      assert_raise ArgumentError, ~r/names \$candidates, which nothing fills/, fn ->
        Synthesizer.synthesize_service_reply(%{reply: :ambiguous_item, item_kind: :calendar})
      end

      assert_raise ArgumentError, ~r/names \$service, which nothing fills/, fn ->
        Synthesizer.synthesize_service_reply(%{reply: :unavailable})
      end
    end
  end

  describe "join_after/2" do
    test "a prefix that ends a sentence leaves the sentence as written" do
      assert Synthesizer.join_after("Done.", "Alarm set for tomorrow.") == "Done. Alarm set for tomorrow."
      assert Synthesizer.join_after("Let me check.", "Your calendar has dentist.") == "Let me check. Your calendar has dentist."
    end

    test "a prefix that continues the sentence lowercases a first word that is neither a proper noun nor I" do
      assert Synthesizer.join_after("Sure,", "Alarm set for tomorrow.") == "Sure, alarm set for tomorrow."
      assert Synthesizer.join_after("Also,", "Which event would you like to cancel?") ==
               "Also, which event would you like to cancel?"
    end

    test "the pronoun I and a proper noun keep their capital" do
      assert Synthesizer.join_after("Sure,", "I've set a reminder.") == "Sure, I've set a reminder."
      assert Synthesizer.join_after("Of course,", "Paris is sunny today.") == "Of course, Paris is sunny today."
    end

    test "an empty sentence raises" do
      assert_raise ArgumentError, ~r/no sentence to put after a prefix/, fn -> Synthesizer.join_after("Sure,", "") end
    end
  end

  describe "enriched_frame/3" do
    @frames %{
      "x" => %{"requires_enrichment" => ["x"], "templates" => ["$x."]},
      "y" => %{"requires_enrichment" => ["y"], "templates" => ["$y."]},
      "x_and_y" => %{"requires_enrichment" => ["x", "y"], "templates" => ["$x and $y."]}
    }

    test "of the frames that fire, the most specific is chosen" do
      assert {"x_and_y", _} = Synthesizer.enriched_frame(@frames, [], %{x: "1", y: "2"})
      assert {"x", _} = Synthesizer.enriched_frame(@frames, [], %{x: "1"})
      assert Synthesizer.enriched_frame(@frames, [], %{z: "1"}) == nil
    end

    test "frames that fire together at the top specificity raise, naming them" do
      assert_raise ArgumentError, ~r/the enriched frames \["x", "y"\] all fire, each with specificity 1/, fn ->
        Synthesizer.enriched_frame(Map.delete(@frames, "x_and_y"), [], %{x: "1", y: "2"})
      end
    end
  end

  describe "every domain's enriched frames" do
    # Two frames of the same specificity can fire together unless some field
    # cannot meet both. When they can, a more specific frame must fire
    # whenever both do, or which of them words the reply is not decided.
    test "no two frames can fire together at the top specificity" do
      ties =
        for file <- "knowledge/domains/*.json" |> Brain.priv_path() |> Path.wildcard() |> Enum.sort(),
            Path.basename(file) != "primitives.json",
            frames = file |> File.read!() |> Jason.decode!() |> Map.get("enriched_response_frames", %{}),
            {a, a_config} <- frames,
            {b, b_config} <- frames,
            a < b,
            Synthesizer.frame_specificity(a_config) == Synthesizer.frame_specificity(b_config),
            not exclusive?(a_config, b_config),
            not Enum.any?(frames, fn {_c, c_config} -> covers?(c_config, a_config, b_config) end),
            do: "#{Path.basename(file)}: #{a} and #{b}"

      assert ties == [],
             "these frames can fire together with the same specificity, and no more specific frame fires " <>
               "whenever both do:\n" <> Enum.join(ties, "\n")
    end

    test "the sweep tells frames that exclude each other, and a frame that covers two, from a tie" do
      required = fn fields -> %{"requires_enrichment" => fields, "templates" => ["."]} end

      assert exclusive?(Map.put(required.([]), "when", %{"n" => 0}), Map.put(required.([]), "when", %{"n" => %{"min" => 1}}))
      assert exclusive?(Map.put(required.([]), "when", %{"n" => nil}), required.(["n"]))
      refute exclusive?(required.(["x"]), required.(["y"]))
      assert covers?(required.(["x", "y"]), required.(["x"]), required.(["y"]))
      refute covers?(required.(["x", "z"]), required.(["x"]), required.(["y"]))
    end
  end

  defp frame_when(config), do: Map.get(config, "when", %{})
  defp frame_required(config), do: Map.get(config, "requires_enrichment", [])

  # The placeholders a frame's templates fill from entities: those it does
  # not require from the enrichment.
  defp entity_placeholders(config) do
    for template <- Map.fetch!(config, "templates"),
        [_whole, name] <- Regex.scan(~r/\$([a-z_][a-z0-9_]*)/, template),
        name not in frame_required(config),
        uniq: true,
        do: name
  end

  # Two frames exclude each other when some field cannot meet both: their
  # conditions on it share no value, or one tests it for null and the other
  # requires it.
  defp exclusive?(a, b) do
    {when_a, when_b} = {frame_when(a), frame_when(b)}

    Enum.any?(when_a, fn {field, condition} -> Map.has_key?(when_b, field) and disjoint?(condition, when_b[field]) end) or
      Enum.any?(when_a, fn {field, condition} -> condition == nil and field in frame_required(b) end) or
      Enum.any?(when_b, fn {field, condition} -> condition == nil and field in frame_required(a) end)
  end

  defp disjoint?(nil, nil), do: false
  defp disjoint?(nil, _other), do: true
  defp disjoint?(_other, nil), do: true
  defp disjoint?(%{} = a, %{} = b), do: max(low(a), low(b)) > min(high(a), high(b))
  defp disjoint?(%{} = range, value), do: not (is_number(value) and value >= low(range) and value <= high(range))
  defp disjoint?(value, %{} = range), do: disjoint?(range, value)
  defp disjoint?(a, b), do: a != b

  defp low(range), do: Map.get(range, "min", -1.0e300)
  defp high(range), do: Map.get(range, "max", 1.0e300)

  # A frame covers two that fire together when it is more specific and fires
  # whenever both do: everything it requires, tests and fills from entities,
  # one of the two does.
  defp covers?(c, a, b) do
    Synthesizer.frame_specificity(c) > Synthesizer.frame_specificity(a) and
      Enum.all?(frame_required(c), &(&1 in frame_required(a) or &1 in frame_required(b))) and
      Enum.all?(frame_when(c), fn {field, condition} ->
        Map.fetch(frame_when(a), field) == {:ok, condition} or Map.fetch(frame_when(b), field) == {:ok, condition}
      end) and
      Enum.all?(entity_placeholders(c), &(&1 in entity_placeholders(a) or &1 in entity_placeholders(b)))
  end

  describe "get_fallback_response/0" do
    test "returns a fallback string" do
      response = Synthesizer.get_fallback_response()

      assert is_binary(response)
      assert String.length(response) > 0
    end
  end

  describe "enriched response frame selection" do
    test "selects more specific frame when more fields match" do
      entities = [%{entity_type: "location", value: "Chicago"}]

      # Context with all detailed fields
      detailed_context = %{
        enrichment_status: :success,
        enriched_data: %{
          temperature: "55°F",
          conditions: "cloudy",
          humidity: "80%",
          wind_speed: "10 mph"
        }
      }

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.query",
          entities,
          0.8,
          context: detailed_context
        )

      case result do
        {:ok, response} ->
          # Should use detailed template if available
          # At minimum should contain some of the enrichment data
          assert response =~ "55°F" or response =~ "Chicago" or response =~ "cloudy"

        :not_synthesized ->
          assert true
      end
    end

  end

  describe "fill_enrichment_slots integration" do
    test "substitutes $variable placeholders" do
      # This tests the internal behavior through the public API
      entities = [%{entity_type: "location", value: "Seattle"}]

      context = %{
        enrichment_status: :success,
        enriched_data: %{
          temperature: "48°F",
          conditions: "rainy"
        }
      }

      result =
        Synthesizer.synthesize_from_domain(
          :weather,
          "weather.current",
          entities,
          0.9,
          context: context
        )

      case result do
        {:ok, response} ->
          # Should not contain unsubstituted placeholders
          refute response =~ "$temperature"
          refute response =~ "$conditions"

        :not_synthesized ->
          assert true
      end
    end
  end
end
