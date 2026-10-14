defmodule Brain.ML.EntityExtractorPassTwoTest do
  @moduledoc """
  What pass 2 of the analysis does with extraction's results once the chunk's
  profile and intent are known: entities take only the type fields a second
  disambiguation gives them, matched by offsets (`put_types/2`), and an
  unresolved span is weighed again only when it has a hypothesis to weigh and
  the intent has a domain to say something with (`resolve_with_intent/3`).

  The span reliabilities and slot type preferences these read are exercised
  in their own tests; deciding a span with them is in `EntityExtractorTest`.
  """
  use ExUnit.Case, async: true

  alias Brain.ML.EntityExtractor

  defp entity(type, s, e, extra \\ %{}) do
    Map.merge(%{entity_type: type, value: "v#{s}", match: "v#{s}", start_pos: s, end_pos: e, confidence: 0.8, source: :gazetteer}, extra)
  end

  describe "put_types/2" do
    test "writes back only the type fields, matched by offsets" do
      pass1 = [
        entity("location", 0, 4, %{types: [%{entity_type: "location"}], graph_known: true, disambiguation_source: :context_analysis}),
        entity("number", 10, 10)
      ]

      retyped = [
        entity("number", 10, 10, %{confidence: 0.1}),
        entity("room", 0, 4, %{value: "other", confidence: 0.2, disambiguation_source: :type_inferrer, disambiguation_confidence: 0.7})
      ]

      [room, number] = EntityExtractor.put_types(pass1, retyped)

      assert room.entity_type == "room"
      assert room.disambiguation_source == :type_inferrer
      assert room.disambiguation_confidence == 0.7
      # A type field the second disambiguation does not give is removed.
      refute Map.has_key?(room, :types)
      # Everything else is pass 1's.
      assert room.value == "v0"
      assert room.confidence == 0.8
      assert room.graph_known

      assert number == hd(tl(pass1))
    end

    test "an entity anaphora resolution supplied has no position and is left as it is" do
      referent = %{entity_type: "device", value: "lamp", confidence: 0.75, source: :anaphora_resolution}
      assert EntityExtractor.put_types([referent], []) == [referent]
    end

    test "a positioned entity with no retyped entity at its offsets raises" do
      assert_raise ArgumentError, ~r/0 retyped entities at 0\.\.4/, fn ->
        EntityExtractor.put_types([entity("location", 0, 4)], [entity("location", 0, 5)])
      end
    end
  end

  describe "refine_entity_types/3" do
    alias Brain.Analysis.SlotTypePreferences

    # A hierarchy of its own, so the preferences below are the test's data and
    # not entity_types.json's.
    defmodule Hierarchy do
      def all_types, do: ~w(device room change_value final_value amount start_year)
      def parents_of(_type), do: []
    end

    defp span(type, tag, word, numeric), do: %{type: type, preceding_tag: tag, preceding_word: word, numeric: numeric}

    # Lowering the hue gives the change after "by" twice as often as the final
    # value after "to"; a music request names an amount after "top" and a
    # start year after "from" equally often.
    defp preferences do
      hue =
        for i <- 1..6 do
          number = if rem(i, 3) == 0, do: span("final_value", "ADP", "to", true), else: span("change_value", "ADP", "by", true)
          %{key: {:hue, i}, intent: "smarthome.device.hue.down", spans: [span("device", "DET", "the", false), number], ordinary: ["VERB"], gazetteer: [:entity]}
        end

      music =
        for i <- 1..6 do
          %{
            key: {:music, i},
            intent: "music.play",
            spans: [span("amount", "ADJ", "top", true), span("start_year", "ADP", "from", true)],
            ordinary: ["VERB", "NOUN"],
            gazetteer: [:entity]
          }
        end

      SlotTypePreferences.fit(hue ++ music,
        language: "en",
        canonical: %{},
        hierarchy: Hierarchy,
        listed: [],
        reading_types: ["device"],
        tag_vocabulary: Brain.ML.POSTagger.valid_tags() ++ [SlotTypePreferences.sentence_start()],
        smoothing: %{
          global: 1,
          domain: 1,
          intent: 1,
          hierarchy: 0.0,
          tag_global: 1,
          tag_type: 1,
          word_global: 1,
          word_type: 1,
          numeric_global: 1,
          numeric_type: 1,
          ordinary_global: 1,
          ordinary_domain: 1,
          ordinary_intent: 1
        }
      )
    end

    defp number_at(text, value) do
      {start_pos, _} = :binary.match(text, value)
      entity("number", start_pos, start_pos + String.length(value) - 1, %{value: value, match: value, source: :system})
    end

    defp refine(entities, intent, text), do: EntityExtractor.refine_entity_types(entities, intent, text: text, preferences: preferences())

    test "a number takes the slot the word before it says: a final value after \"to\", a change after \"by\"" do
      text = "lower the hue by 10 to 30"
      refined = refine([number_at(text, "10"), number_at(text, "30")], "smarthome.device.hue.down", text)

      assert Enum.map(refined, &{&1.value, &1.entity_type}) == [{"10", "change_value"}, {"30", "final_value"}]
    end

    test "the order of the numbers decides nothing" do
      text = "lower the hue to 30 by 10"
      refined = refine([number_at(text, "30"), number_at(text, "10")], "smarthome.device.hue.down", text)

      assert Enum.map(refined, &{&1.value, &1.entity_type}) == [{"30", "final_value"}, {"10", "change_value"}]
    end

    test "a year is told from an amount in a music request, whatever slot types the schema lists" do
      text = "play top 10 songs from 2014"
      refined = refine([number_at(text, "10"), number_at(text, "2014")], "music.play", text)

      assert Enum.map(refined, &{&1.value, &1.entity_type}) == [{"10", "amount"}, {"2014", "start_year"}]
    end

    test "every other entity is left as it is, in its place" do
      text = "lower the hue by 10"
      room = entity("room", 0, 4)
      [first, second] = refine([room, number_at(text, "10")], "smarthome.device.hue.down", text)

      assert first == room
      assert second.entity_type == "change_value"
    end

    test "a referent anaphora resolution supplied is weighed by the intent and its form alone" do
      model = preferences()
      referent = %{entity_type: "number", value: "40", confidence: 0.42, source: :anaphora_resolution}

      [refined] = EntityExtractor.refine_entity_types([referent], "music.play", text: "play that many", preferences: model)

      # music.play's numbers are amounts and start years equally often: a tie
      # decides nothing, and the referent keeps its type.
      assert refined == referent

      # Lowering the hue gives a change twice as often as a final value.
      posteriors =
        SlotTypePreferences.label_posteriors(model, "smarthome.device.hue.down", ~w(change_value device final_value room), %{numeric: true})

      assert posteriors["change_value"] > posteriors["final_value"]

      [refined] = EntityExtractor.refine_entity_types([referent], "smarthome.device.hue.down", text: "set it", preferences: model)
      assert refined.entity_type == "change_value"
    end

    test "a number stays a number when no slot of the intent's schema is a gold label, or the intent has no schema" do
      text = "say 40"
      number = number_at(text, "40")

      # smalltalk.agent.age's one slot, age, is no label of these preferences.
      assert refine([number], "smalltalk.agent.age", text) == [number]
      assert refine([number], "no_such.intent", text) == [number]
    end

    test "any other number without a position, or one that does not sit on tokens of the text, raises" do
      positionless = %{entity_type: "number", value: "40", confidence: 0.9, source: :gazetteer}

      assert_raise ArgumentError, ~r/a number with no position in the text/, fn ->
        refine([positionless], "smarthome.device.hue.down", "lower the hue by 40")
      end

      assert_raise ArgumentError, ~r/does not start and end at tokens/, fn ->
        refine([entity("number", 1, 2)], "smarthome.device.hue.down", "lower the hue by 40")
      end
    end

    test "the text the positions index is required" do
      assert_raise ArgumentError, ~r/needs :text/, fn ->
        EntityExtractor.refine_entity_types([], "music.play", [])
      end
    end
  end

  describe "resolve_with_intent/3" do
    defp unresolved(reason, s, e),
      do: %{entity_type: nil, value: "u#{s}", match: "u#{s}", start_pos: s, end_pos: e, confidence: 0.0, resolved: false, unresolved_reason: reason, readings: []}

    test "a span with nothing to weigh, or an intent with nothing to say, stays unresolved and says why" do
      spans = [unresolved(:no_measured_reading, 0, 3), unresolved(:below_confidence_floor, 5, 6), unresolved(:insufficient_evidence, 8, 9)]

      no_intent = EntityExtractor.resolve_with_intent(spans, nil, [])
      assert no_intent.entities == []
      assert no_intent.unresolved_spans == spans

      assert Enum.map(no_intent.outcomes, &{&1.start_pos, &1.outcome, &1.reason}) == [
               {0, :not_redecided, :no_measured_reading},
               {5, :not_redecided, :below_confidence_floor},
               {8, :not_redecided, :no_intent}
             ]

      # "question" is no prefix IntentDomains declares.
      undomained = EntityExtractor.resolve_with_intent(spans, "question.factual", [])
      assert List.last(undomained.outcomes).reason == :intent_without_domain
      assert undomained.unresolved_spans == spans
    end

    test "anything but an unresolved span raises" do
      assert_raise ArgumentError, ~r/not an unresolved span/, fn ->
        EntityExtractor.resolve_with_intent([entity("room", 0, 4)], "smarthome.lights.on", [])
      end
    end
  end
end
