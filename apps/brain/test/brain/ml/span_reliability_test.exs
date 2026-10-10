defmodule Brain.ML.SpanReliabilityTest do
  use ExUnit.Case, async: true

  alias Brain.Lattice
  alias Brain.ML.SpanReliability

  # A hierarchy of its own, so these test the learner and the decision and not
  # entity_types.json: final_value < number, music_artist < artist < person,
  # given_name < person.
  defmodule Hierarchy do
    @parents %{
      "final_value" => ["number"],
      "change_value" => ["number"],
      "music_artist" => ["artist"],
      "artist" => ["person"],
      "given_name" => ["person"]
    }

    def parents_of(type), do: Map.get(@parents, type, [])

    def is_a?(child, parent), do: parent in ancestors(child)

    defp ancestors(type), do: Enum.flat_map(parents_of(type), &[&1 | ancestors(&1)])
  end

  @canonical %{"music_artist" => "artist"}
  @opts [language: "en", canonical: @canonical, hierarchy: Hierarchy]

  defp reading(source, type, s, e, extra \\ %{}),
    do: Map.merge(%{source: source, entity_type: type, start_pos: s, end_pos: e, confidence: 0.9}, extra)

  defp gold(type, s, e), do: %{type: type, start_pos: s, end_pos: e}

  # Expected tallies are worked out by hand in the tests below.
  defp examples, do: [
    # "set to 5": the number parser and the model on the gold final_value.
    %{key: 1, gold: [gold("final_value", 7, 7)],
      readings: [reading(:system, "number", 7, 7), reading(:entity_model, "final_value", 7, 7)]},
    # "play Adele": the gazetteer's music_artist maps to the gold artist.
    %{key: 2, gold: [gold("artist", 5, 9)],
      readings: [reading(:gazetteer, "music_artist", 5, 9), reading(:pos_tagger_propn, "person", 5, 9),
                 reading(:entity_model, "artist", 5, 9)]},
    # "text Mike": the gold refines the hint's person; the gazetteer's emoji
    # cannot be confirmed by any gold label; the model's artist is wrong.
    %{key: 3, gold: [gold("given_name", 5, 8)],
      readings: [reading(:pos_tagger_propn, "person", 5, 8), reading(:gazetteer, "emoji", 5, 8),
                 reading(:entity_model, "artist", 5, 8)]},
    # A gazetteer reading where the gold has nothing, and a model reading
    # whose extent overlaps a gold span without matching it.
    %{key: 4, gold: [gold("final_value", 10, 11)],
      readings: [reading(:gazetteer, "music_artist", 0, 3), reading(:entity_model, "final_value", 9, 11)]}
  ]

  describe "fit/2" do
    setup do
      {:ok, model: SpanReliability.fit(examples(), Keyword.put(@opts, :prior_strength, 1))}
    end

    test "a reading the gold type refines agrees at its own level", %{model: model} do
      assert %{n: 1, exact: 0, refined: 1, agree: 1} = model.pairs[{:system, "number"}]
      assert %{n: 2, exact: 0, refined: 2, agree: 2} = model.pairs[{:pos_tagger_propn, "person"}]
    end

    test "runtime types are compared through the type mappings", %{model: model} do
      assert %{n: 2, exact: 1, no_entity: 1, agree: 1} = model.pairs[{:gazetteer, "music_artist"}]
    end

    test "misses are recorded by kind", %{model: model} do
      assert %{n: 2, exact: 1, wrong_type: %{"given_name" => 1}, agree: 1} = model.pairs[{:entity_model, "artist"}]
      assert %{n: 2, exact: 1, boundary: 1, agree: 1} = model.pairs[{:entity_model, "final_value"}]
    end

    test "a type no gold label is or refines is unmeasurable, with its count", %{model: model} do
      refute Map.has_key?(model.pairs, {:gazetteer, "emoji"})
      assert model.unmeasurable == %{{:gazetteer, "emoji"} => %{n: 1}}
      assert model.gold_types == ["artist", "final_value", "given_name"]
    end

    test "smoothing pulls a pair toward its source and a source toward the overall rate", %{model: model} do
      # 9 measured readings, 6 agree.
      assert model.overall == %{n: 9, agree: 6, rate: 6 / 9}
      # The model: 4 readings, 2 agree -> (2 + 1 * 2/3) / (4 + 1) = 8/15.
      assert_in_delta model.sources[:entity_model].reliability, 8 / 15, 1.0e-12
      # Its artist pair: 1 of 2 -> (1 + 8/15) / (2 + 1) = 23/45.
      assert_in_delta model.pairs[{:entity_model, "artist"}].reliability, 23 / 45, 1.0e-12
      # The number parser: 1 of 1 -> source (1 + 2/3) / 2 = 5/6, pair (1 + 5/6) / 2.
      assert_in_delta model.pairs[{:system, "number"}].reliability, (1 + 5 / 6) / 2, 1.0e-12
    end

    test "a larger prior strength pulls further toward the source" do
      strong = SpanReliability.fit(examples(), Keyword.put(@opts, :prior_strength, 100))
      prior = strong.sources[:entity_model].reliability
      p = strong.pairs[{:entity_model, "artist"}].reliability
      assert abs(p - prior) < abs(23 / 45 - 8 / 15)
    end

    test "a rate of 0 or 1 over every reading cannot be learned from" do
      all_right = [%{key: 1, gold: [gold("artist", 0, 3)], readings: [reading(:gazetteer, "music_artist", 0, 3)]}]

      assert_raise ArgumentError, ~r/1 of 1 measured readings agree/, fn ->
        SpanReliability.fit(all_right, Keyword.put(@opts, :prior_strength, 1))
      end
    end

    test "a non-positive prior strength and a reading without a source raise" do
      assert_raise ArgumentError, ~r/positive number/, fn -> SpanReliability.fit(examples(), Keyword.put(@opts, :prior_strength, 0)) end

      no_source = [%{key: 1, gold: [], readings: [%{entity_type: "artist", start_pos: 0, end_pos: 1}]}]
      assert_raise ArgumentError, ~r/needs a :source/, fn -> SpanReliability.fit(no_source, Keyword.put(@opts, :prior_strength, 1)) end
    end
  end

  describe "decide/3" do
    setup do
      model = SpanReliability.fit(examples(), Keyword.put(@opts, :prior_strength, 1))
      {:ok, model: %{model | min_score: 0.0, min_margin: 0.0}}
    end

    defp decide(readings, model, text), do: SpanReliability.decide(readings, model, text: text, canonical: @canonical, hierarchy: Hierarchy)

    defp logit(p), do: :math.log(p / (1 - p))

    test "combines a span's readings in log-odds of their learned reliabilities, not their own confidences", %{model: model} do
      readings = [reading(:system, "number", 7, 7, %{confidence: 0.99}), reading(:entity_model, "final_value", 7, 7, %{confidence: 0.01})]
      [span] = decide(readings, model, "set to 5")

      p_number = model.pairs[{:system, "number"}].reliability
      p_final = model.pairs[{:entity_model, "final_value"}].reliability

      # final_value refines number, so the number reading supports it too.
      assert %{resolved: true, entity_type: "final_value", source: :entity_model} = span
      assert_in_delta span.decision.score, logit(p_number) + logit(p_final), 1.0e-6
      assert_in_delta span.decision.margin, logit(p_final), 1.0e-6
      assert_in_delta span.confidence, 1 / (1 + :math.exp(-span.decision.score)), 1.0e-6
      assert %Lattice{candidates: [_, _]} = span.candidates
      assert Enum.map(span.readings, & &1.source) == [:system, :entity_model]
    end

    test "keeps every reading, measured or not, and merges the keys of the winning hypothesis's readings", %{model: model} do
      readings = [
        reading(:gazetteer, "music_artist", 5, 9, %{metadata: %{ha_entity_id: "media_player.x"}}),
        reading(:entity_model, "artist", 5, 9, %{model_only: true}),
        reading(:pos_tagger_propn, "person", 5, 9),
        reading(:gazetteer, "emoji", 5, 9)
      ]

      [span] = decide(readings, model, "play Adele")

      # The gazetteer's music_artist (14/27) is trusted more than the model's
      # artist (23/45); both read the gold label artist.
      assert %{resolved: true, entity_type: "music_artist", source: :gazetteer} = span
      assert span.metadata == %{ha_entity_id: "media_player.x"}
      assert span.model_only
      assert length(span.readings) == 4
      assert %{measured: false, reliability: nil, evidence: 0} = Enum.find(span.readings, &(&1.entity_type == "emoji"))
    end

    test "abstains when the lead is below min_margin, typed at the readings' common ancestor", %{model: model} do
      readings = [reading(:system, "number", 7, 7), reading(:entity_model, "final_value", 7, 7)]
      [span] = decide(readings, %{model | min_margin: 1.0}, "set to 5")

      assert %{resolved: false, unresolved_reason: :insufficient_evidence, entity_type: "number",
               value: "5", match: "5", start_pos: 7, end_pos: 7} = span
      assert span.confidence == 0.0
      assert length(span.readings) == 2
      assert %Lattice{best: %{label: {7, 7, "final_value"}}} = span.candidates
    end

    test "abstains when the best score is below min_score", %{model: model} do
      [span] = decide([reading(:entity_model, "artist", 0, 3)], %{model | min_score: 5.0}, "Mike")
      assert %{resolved: false, unresolved_reason: :insufficient_evidence, entity_type: "artist"} = span
    end

    test "a span read only by unmeasurable sources stays unresolved, typed as they read it", %{model: model} do
      [span] = decide([reading(:gazetteer, "emoji", 0, 7), reading(:location_hint, "location", 0, 7)], model, "umbrella")
      assert %{resolved: false, unresolved_reason: :no_measured_reading, entity_type: nil} = span

      [span] = decide([reading(:gazetteer, "emoji", 0, 7)], model, "umbrella")
      assert %{resolved: false, unresolved_reason: :no_measured_reading, entity_type: "emoji"} = span
    end

    test "equal evidence for different readings never decides" do
      model = %SpanReliability{
        language: "en", prior_strength: 1, gold_types: ["artist", "final_value"], fitted_rows: 1,
        overall: %{n: 2, agree: 1, rate: 0.5}, sources: %{}, unmeasurable: %{},
        pairs: %{{:a, "artist"} => %{n: 5, reliability: 0.8}, {:b, "final_value"} => %{n: 5, reliability: 0.8}},
        min_score: -10.0, min_margin: 0.0
      }

      [span] = decide([reading(:a, "artist", 0, 3), reading(:b, "final_value", 0, 3)], model, "Rush")
      # artist and final_value share no ancestor.
      assert %{resolved: false, unresolved_reason: :insufficient_evidence, entity_type: nil} = span
    end

    test "overlapping readings of different extents are one span; separate ones are not", %{model: model} do
      readings = [reading(:entity_model, "artist", 0, 9), reading(:gazetteer, "music_artist", 5, 9), reading(:system, "number", 12, 12)]
      spans = decide(readings, model, "Lady Gaga, 5")
      assert [%{start_pos: s1}, %{start_pos: 12}] = spans
      assert s1 in [0, 5]
      assert length(hd(spans).readings) == 2
    end

    test "a model without thresholds cannot decide" do
      model = SpanReliability.fit(examples(), Keyword.put(@opts, :prior_strength, 1))
      assert_raise ArgumentError, ~r/no decision thresholds/, fn -> decide([], model, "") end
    end

    test "withhold/4 turns a decision back into an unresolved span with its readings", %{model: model} do
      [span] = decide([reading(:system, "number", 7, 7), reading(:entity_model, "final_value", 7, 7)], model, "set to 5")
      withheld = SpanReliability.withhold(span, :below_confidence_floor, "set to 5", Hierarchy)

      assert %{resolved: false, unresolved_reason: :below_confidence_floor, entity_type: "number"} = withheld
      assert withheld.readings == span.readings
    end
  end

  describe "common_type/2" do
    test "the most specific type every reading is or descends from" do
      assert SpanReliability.common_type(["music_artist", "given_name"], Hierarchy) == "person"
      assert SpanReliability.common_type(["music_artist", "artist"], Hierarchy) == "artist"
      assert SpanReliability.common_type(["final_value", "artist"], Hierarchy) == nil
    end
  end

  describe "select/2" do
    # A trustworthy source and an untrustworthy one disagreeing on most rows.
    defp synthetic do
      conflicts =
        for i <- 1..40 do
          %{key: {:conflict, i}, gold: [gold("final_value", 0, 1)],
            readings: [reading(:good, "final_value", 0, 1), reading(:bad, "artist", 0, 1)]}
        end

      bad_right =
        for i <- 1..10 do
          %{key: {:artist, i}, gold: [gold("artist", 0, 1)], readings: [reading(:bad, "artist", 0, 1)]}
        end

      conflicts ++ bad_right
    end

    test "chooses a prior strength and thresholds by cross-validation and decides for the reliable source" do
      {model, report} = SpanReliability.select(synthetic(), @opts)

      assert Map.keys(report) |> Enum.sort() == Enum.sort(Map.keys(model.cross_validation))
      assert model.prior_strength in Map.keys(report)
      assert is_number(model.min_score)
      assert is_number(model.min_margin) or model.min_margin == :infinity
      assert report[model.prior_strength].f1 == report |> Map.values() |> Enum.map(& &1.f1) |> Enum.max()

      [span] =
        SpanReliability.decide([reading(:good, "final_value", 0, 1), reading(:bad, "artist", 0, 1)], model,
          text: "50", canonical: @canonical, hierarchy: Hierarchy)

      assert %{resolved: true, entity_type: "final_value", source: :good} = span
    end

    test "the prior strength is not the caller's to pass" do
      assert_raise ArgumentError, ~r/chooses the prior strength/, fn ->
        SpanReliability.select(synthetic(), Keyword.put(@opts, :prior_strength, 1))
      end
    end
  end
end
