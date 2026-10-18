defmodule Brain.ML.NerSpanEvaluationTest do
  use ExUnit.Case, async: true

  alias Brain.ML.NerSpanEvaluation

  defmodule Hierarchy do
    def is_a?("final_value", "number"), do: true
    def is_a?(_child, _parent), do: false
  end

  @opts [canonical: %{"music_artist" => "artist"}, hierarchy: Hierarchy]

  defp gold(type, s, e), do: %{type: type, start_pos: s, end_pos: e}
  defp entity(type, s, e, source), do: %{entity_type: type, start_pos: s, end_pos: e, source: source, resolved: true}

  defp unresolved(type, s, e, reason),
    do: %{entity_type: type, start_pos: s, end_pos: e, resolved: false, unresolved_reason: reason}

  test "matches by extent and mapped type, exactly and up to the hierarchy" do
    results = [
      # exact through the mapping
      %{gold: [gold("artist", 5, 9)], entities: [entity("music_artist", 5, 9, :gazetteer)], unresolved_spans: []},
      # agrees: the gold refines number
      %{gold: [gold("final_value", 7, 7)], entities: [entity("number", 7, 7, :system)], unresolved_spans: []},
      # right type, wrong extent; and a gold span nothing predicted
      %{
        gold: [gold("artist", 0, 9), gold("final_value", 12, 13)],
        entities: [entity("artist", 5, 9, :entity_model)],
        unresolved_spans: []
      }
    ]

    score = NerSpanEvaluation.score(results, @opts)

    assert score.gold == 4
    assert score.predictions == 3
    assert score.exact.tp == 1
    assert score.agree.tp == 2
    assert_in_delta score.exact.precision, 1 / 3, 1.0e-12
    assert_in_delta score.agree.recall, 2 / 4, 1.0e-12

    assert score.by_source == %{
             gazetteer: %{predictions: 1, exact: 1, agree: 1},
             system: %{predictions: 1, exact: 0, agree: 1},
             entity_model: %{predictions: 1, exact: 0, agree: 0}
           }
  end

  test "an unresolved span is an abstention: it costs recall, not precision" do
    results = [
      %{
        gold: [gold("artist", 5, 9), gold("final_value", 11, 11)],
        entities: [entity("final_value", 11, 11, :entity_model)],
        unresolved_spans: [unresolved("person", 5, 9, :insufficient_evidence)]
      }
    ]

    score = NerSpanEvaluation.score(results, @opts)

    assert score.predictions == 1
    assert score.abstentions == 1
    assert score.abstentions_by_reason == %{insufficient_evidence: 1}
    assert score.exact.precision == 1.0
    assert score.exact.recall == 0.5
  end

  test "abstentions are counted by reason across rows, and a typeless one is still an abstention" do
    results = [
      %{gold: [gold("artist", 0, 4)], entities: [], unresolved_spans: [unresolved(nil, 0, 4, :insufficient_evidence)]},
      %{
        gold: [],
        entities: [],
        unresolved_spans: [unresolved("number", 2, 2, :below_confidence_floor), unresolved(nil, 4, 6, :no_measured_reading)]
      }
    ]

    score = NerSpanEvaluation.score(results, @opts)

    assert score.predictions == 0
    assert score.abstentions == 3
    assert score.abstentions_by_reason == %{insufficient_evidence: 1, below_confidence_floor: 1, no_measured_reading: 1}
    assert score.exact.recall == 0.0
  end

  test "an unresolved span that names no reason raises" do
    results = [%{gold: [], entities: [], unresolved_spans: [%{entity_type: nil, start_pos: 0, end_pos: 1, resolved: false}]}]
    assert_raise ArgumentError, ~r/names no reason/, fn -> NerSpanEvaluation.score(results, @opts) end
  end

  test "a decided entity with no type raises" do
    results = [%{gold: [], entities: [%{entity_type: nil, start_pos: 0, end_pos: 1, resolved: true}], unresolved_spans: []}]
    assert_raise ArgumentError, ~r/no type/, fn -> NerSpanEvaluation.score(results, @opts) end
  end
end
