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
        entity("room", 0, 4, %{value: "other", confidence: 0.2, entity: "room", disambiguation_source: :type_inferrer, disambiguation_confidence: 0.7})
      ]

      [room, number] = EntityExtractor.put_types(pass1, retyped)

      assert room.entity_type == "room"
      assert room.entity == "room"
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
