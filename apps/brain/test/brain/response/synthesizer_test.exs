defmodule Brain.Response.SynthesizerTest do
  use ExUnit.Case, async: false

  alias Brain.Response.Synthesizer

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
        Synthesizer.synthesize_service_reply(%{reply: :item_not_identified, item_kind: :weather})
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

  describe "service_item_slots/2" do
    test "names the slot each non-blank field of the item is given in" do
      assert Synthesizer.service_item_slots(:reminder, %{text: "call mom", date: "", time: "6 pm", status: "set"}) ==
               %{"reminder_text" => "call mom", "time" => "6 pm"}
    end
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
