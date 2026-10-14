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

  describe "with the NER type mappings and gold standard" do
    # Needs the type hierarchy running.
    test "a lock status is compared as itself, which no gold label is or refines, so its readings are unmeasured" do
      canonical = Brain.ML.EntityExtractor.load_type_mappings()
      hierarchy = Brain.Analysis.TypeHierarchy

      # The gold's lock is the lockable thing ("door"); a lock status is an
      # action or a state ("lock", "open"), and the gold labels none.
      refute Map.has_key?(canonical, "locks_status")

      gold_labels =
        for row <- Brain.ML.EvaluationStore.load_gold_standard("ner", :all), %{"type" => type} <- row["expected"], uniq: true, do: type

      refute Enum.any?(gold_labels, &(&1 == "locks_status" or hierarchy.is_a?(&1, "locks_status")))

      # "lock the front door, then the den in the hall": the gazetteer reads
      # "den" as a room the gold does not label.
      example = %{
        key: 1,
        gold: [gold("lock", 9, 18), gold("room", 41, 44)],
        readings: [reading(:gazetteer, "locks_status", 0, 3), reading(:gazetteer, "lock", 9, 18), reading(:gazetteer, "room", 30, 32)]
      }

      model = SpanReliability.fit([example], language: "en", canonical: canonical, hierarchy: hierarchy, prior_strength: 1)

      assert model.unmeasurable == %{{:gazetteer, "locks_status"} => %{n: 1}}
      assert %{exact: 1} = model.pairs[{:gazetteer, "lock"}]
    end

    # Needs the type hierarchy and the temporal vocabulary loaded. The gold
    # standard labels the lengths of timer and snooze rows duration, and an
    # offset ("in 2 hours") date_time, which holds no length. The length
    # source alone is read, so no span has competing hypotheses, and a fold of
    # its readings can agree throughout, which select/2 cannot fit; the model
    # is fitted on the whole training split and decides a span whose
    # reliability is at least even odds.
    test "the temporal resolver's lengths are measured on the training split and decide the held-out duration spans" do
      alias Brain.ML.{EntityExtractor, EvaluationStore, NerGold}

      canonical = EntityExtractor.load_type_mappings()
      hierarchy = Brain.Analysis.TypeHierarchy
      examples = for row <- EvaluationStore.load_gold_standard("ner", :train), do: %{key: row["text"], readings: EntityExtractor.length_readings(row["text"]), gold: NerGold.spans!(row)}

      model =
        examples
        |> SpanReliability.fit(language: "en", canonical: canonical, hierarchy: hierarchy, prior_strength: 1)
        |> Map.merge(%{min_score: 0.0, min_margin: 0.0})

      assert %{exact: exact, reliability: reliability} = model.pairs[{:temporal_resolver, "duration"}]
      assert exact > 0
      assert reliability > 0.5

      held = for row <- EvaluationStore.load_gold_standard("ner", :held_out), do: {row["text"], NerGold.spans!(row)}
      assert Enum.any?(held, fn {_text, gold} -> Enum.any?(gold, &(&1.type == "duration")) end), "no held-out row labels a duration"

      for {text, gold} <- held do
        decided = text |> EntityExtractor.length_readings() |> SpanReliability.decide(model, text: text, canonical: canonical, hierarchy: hierarchy) |> Enum.filter(& &1.resolved)
        durations = for span <- gold, span.type == "duration", do: {span.start_pos, span.end_pos}

        assert Enum.map(decided, &{&1.start_pos, &1.end_pos}) == durations, "#{inspect(text)}: decided #{inspect(Enum.map(decided, & &1.value))}"
        assert Enum.all?(decided, &(&1.entity_type == "duration")), inspect(text)
      end
    end
  end

  describe "fit_extent/2" do
    setup do
      {:ok, extent: SpanReliability.fit_extent(examples(), prior_strength: 1)}
    end

    test "a reading at a gold span's extent agrees whatever its type; a partial overlap is an extent error", %{extent: extent} do
      # The model's artist is the wrong type for "Mike" (given_name) but at
      # its extent, and right for "Adele".
      assert %{n: 2, agree: 2, boundary: 0, no_entity: 0} = extent.pairs[{:entity_model, "artist"}]
      # 9..11 overlaps the gold 10..11 without matching it.
      assert %{n: 2, agree: 1, boundary: 1, no_entity: 0} = extent.pairs[{:entity_model, "final_value"}]
      assert %{n: 2, agree: 1, boundary: 0, no_entity: 1} = extent.pairs[{:gazetteer, "music_artist"}]
    end

    test "a type no gold label confirms still has an extent the gold standard judges", %{extent: extent} do
      assert %{n: 1, agree: 1} = extent.pairs[{:gazetteer, "emoji"}]
    end

    test "smoothing pulls a pair toward its source, and a source toward the overall rate", %{extent: extent} do
      # 10 readings, 8 at a gold extent.
      assert extent.overall == %{n: 10, agree: 8, rate: 0.8}
      # The model: 4 readings, 3 at a gold extent -> (3 + 0.8) / (4 + 1).
      assert_in_delta extent.sources[:entity_model].reliability, 0.76, 1.0e-12
      # Its final_value: 1 of 2 -> (1 + 0.76) / (2 + 1).
      assert_in_delta extent.pairs[{:entity_model, "final_value"}].reliability, 1.76 / 3, 1.0e-12
      assert extent.fitted_rows == 4
      assert extent.min_score == nil and extent.min_margin == nil
    end

    test "a reading is keyed by its source and type alone, whatever else it carries" do
      rows = [
        %{key: 1, gold: [gold("song", 10, 19)], readings: [reading(:entity_model, "song", 10, 19, %{confidence: 1.0})]},
        %{key: 2, gold: [gold("song", 10, 19)], readings: [reading(:entity_model, "song", 16, 19, %{confidence: 1.0})]}
      ]

      extent = SpanReliability.fit_extent(rows, prior_strength: 1)

      assert Map.keys(extent.pairs) == [{:entity_model, "song"}]
      assert %{n: 2, agree: 1, boundary: 1} = extent.pairs[{:entity_model, "song"}]
    end

    test "a rate of 0 or 1 over every reading's extent cannot be learned from" do
      all_right = [%{key: 1, gold: [gold("artist", 0, 3)], readings: [reading(:gazetteer, "music_artist", 0, 3)]}]

      assert_raise ArgumentError, ~r/1 of 1 readings have a gold span's extent/, fn ->
        SpanReliability.fit_extent(all_right, prior_strength: 1)
      end
    end
  end

  describe "redecide/4" do
    setup do
      model = SpanReliability.fit(examples(), Keyword.put(@opts, :prior_strength, 1))
      # Extent thresholds every span with one extent passes, shared by every span.
      extent =
        examples()
        |> SpanReliability.fit_extent(prior_strength: 1)
        |> Map.merge(%{min_score: -10.0, min_margin: 0.0, granularity: :global, groups: %{}})
      # Thresholds the readings of "Mike" alone (the model's artist, 23/45) do
      # not reach: logit(23/45) is about 0.044. The redecision thresholds are
      # the same here, so what decides a span again is the evidence.
      {:ok, model: %{model | min_score: 1.0, min_margin: 0.0, redecision: %{min_score: 1.0, min_margin: 0.0, extent: extent}}}
    end

    defp undecided(model) do
      readings = [reading(:entity_model, "artist", 0, 3, %{model_only: true})]
      [span] = SpanReliability.decide(readings, model, text: "Mike", canonical: @canonical, hierarchy: Hierarchy)
      span
    end

    test "an unresolved span keeps its readings as the sources gave them", %{model: model} do
      span = undecided(model)
      assert %{resolved: false, unresolved_reason: :insufficient_evidence} = span
      assert [%{source: :entity_model, entity_type: "artist", model_only: true}] = span.raw_readings
    end

    test "evidence that lifts the hypothesis over the thresholds decides it, with the readings' keys", %{model: model} do
      span = undecided(model)
      p = model.pairs[{:entity_model, "artist"}].reliability

      assert {:decided, entity} = SpanReliability.redecide(span, %{"artist" => 2.0}, model, Hierarchy)

      assert %{resolved: true, entity_type: "artist", source: :entity_model, model_only: true} = entity

      assert_in_delta entity.decision.score, :math.log(p / (1 - p)) + 2.0, 1.0e-6
      assert entity.decision.margin == :infinity
      assert entity.decision.evidence == %{"artist" => 2.0}
      # The extent, decided first by the model's artist extent reliability alone.
      q = model.redecision.extent.pairs[{:entity_model, "artist"}].reliability
      assert_in_delta entity.decision.extent.score, :math.log(q / (1 - q)), 1.0e-6
      assert entity.decision.extent.margin == :infinity
      assert_in_delta entity.confidence, 1 / (1 + :math.exp(-entity.decision.score)), 1.0e-6
      assert entity.readings == span.readings
    end

    test "evidence that does not reach the thresholds leaves the span undecided", %{model: model} do
      span = undecided(model)

      assert {:undecided, %Lattice{best: %{label: {0, 3, "artist"}, score: score}}} =
               SpanReliability.redecide(span, %{"artist" => -1.0}, model, Hierarchy)

      assert score < model.min_score
    end

    test "evidence can change which hypothesis leads", %{model: model} do
      # number and final_value at one extent; final_value refines number.
      readings = [reading(:system, "number", 7, 7), reading(:entity_model, "final_value", 7, 7)]
      model = %{model | min_margin: 5.0, redecision: %{model.redecision | min_score: 1.0, min_margin: 5.0}}
      [span] = SpanReliability.decide(readings, model, text: "set to 5", canonical: @canonical, hierarchy: Hierarchy)
      assert span.unresolved_reason == :insufficient_evidence

      assert {:decided, entity} = SpanReliability.redecide(span, %{"number" => 6.0, "final_value" => -1.0}, model, Hierarchy)

      assert entity.entity_type == "number"
      assert entity.decision.margin >= 5.0
    end

    test "decides with the redecision thresholds, not the first decision's", %{model: model} do
      span = undecided(model)
      # Evidence of 2.0 decides "Mike" at a redecision min_score of 1.0 ...
      assert {:decided, _} = SpanReliability.redecide(span, %{"artist" => 2.0}, model, Hierarchy)
      # ... but not at 3.0, whatever the first decision's thresholds are.
      strict = %{model | min_score: -10.0, redecision: %{model.redecision | min_score: 3.0, min_margin: 0.0}}
      assert {:undecided, _} = SpanReliability.redecide(span, %{"artist" => 2.0}, strict, Hierarchy)
      # Thresholds that decide nothing again.
      never = %{model | redecision: %{model.redecision | min_score: :infinity, min_margin: :infinity}}
      assert {:undecided, _} = SpanReliability.redecide(span, %{"artist" => 50.0}, never, Hierarchy)
    end

    test "a model without redecision thresholds cannot decide again", %{model: model} do
      assert_raise ArgumentError, ~r/no redecision thresholds/, fn ->
        SpanReliability.redecide(undecided(model), %{"artist" => 2.0}, %{model | redecision: nil}, Hierarchy)
      end
    end

    test "a redecision without extent reliabilities, or without their thresholds, cannot decide again", %{model: model} do
      no_extent = %{model | redecision: Map.delete(model.redecision, :extent)}
      unchosen = %{model | redecision: %{model.redecision | extent: %{model.redecision.extent | min_score: nil}}}

      for broken <- [no_extent, unchosen] do
        assert_raise ArgumentError, ~r/extent reliabilities with thresholds/, fn ->
          SpanReliability.redecide(undecided(model), %{"artist" => 2.0}, broken, Hierarchy)
        end
      end
    end

    test "extent thresholds that name no granularity and groups are refused", %{model: model} do
      unchosen = %{model | redecision: %{model.redecision | extent: Map.drop(model.redecision.extent, [:granularity, :groups])}}

      assert_raise ArgumentError, ~r/name no granularity and groups.*mix train_span_reliability/, fn ->
        SpanReliability.redecide(undecided(model), %{"artist" => 2.0}, unchosen, Hierarchy)
      end
    end

    test "a span's extent is decided under its group's thresholds: its pair's, else its source's, else the shared ones", %{model: model} do
      span = undecided(model)
      extent = model.redecision.extent

      for {groups, decided?} <- [
            {%{{:entity_model, "artist"} => %{min_score: 50.0, min_margin: 0.0}}, false},
            {%{:entity_model => %{min_score: 50.0, min_margin: 0.0}}, false},
            {%{{:entity_model, "artist"} => %{min_score: -10.0, min_margin: 0.0}, :entity_model => %{min_score: 50.0, min_margin: 0.0}}, true},
            {%{{:gazetteer, "artist"} => %{min_score: 50.0, min_margin: 0.0}, :gazetteer => %{min_score: 50.0, min_margin: 0.0}}, true}
          ] do
        grouped = %{model | redecision: %{model.redecision | extent: %{extent | granularity: {:pair, 1}, groups: groups}}}
        result = SpanReliability.redecide(span, %{"artist" => 2.0}, grouped, Hierarchy)
        assert match?({:decided, _}, result) == decided?, "groups #{inspect(groups)} gave #{inspect(elem(result, 0))}"
      end
    end

    test "a span whose extent is not decided is never decided again, whatever the evidence", %{model: model} do
      span = undecided(model)
      undecidable = %{model | redecision: %{model.redecision | extent: %{model.redecision.extent | min_score: 50.0}}}

      assert {:extent_undecided, %Lattice{best: %{label: {0, 3}}}} =
               SpanReliability.redecide(span, %{"artist" => 1000.0}, undecidable, Hierarchy)

      # Extents the readings trust equally are never decided either.
      equal = %{
        model.redecision.extent
        | pairs: %{{:a, "artist"} => %{n: 5, reliability: 0.8}, {:b, "artist"} => %{n: 5, reliability: 0.8}}
      }

      model = %{model | pairs: Map.merge(model.pairs, %{{:a, "artist"} => %{n: 5, reliability: 0.5}, {:b, "artist"} => %{n: 5, reliability: 0.5}})}
      [span] = SpanReliability.decide([reading(:a, "artist", 0, 9), reading(:b, "artist", 5, 9)], model, text: "Lady Gaga", canonical: @canonical, hierarchy: Hierarchy)
      assert span.unresolved_reason == :insufficient_evidence

      assert {:extent_undecided, %Lattice{}} =
               SpanReliability.redecide(span, %{"artist" => 1000.0}, %{model | redecision: %{model.redecision | extent: equal}}, Hierarchy)
    end

    test "only the hypotheses at the decided extent are weighed", %{model: model} do
      # The model's artist over 0..3 and its final_value over 2..3: one span
      # whose two hypotheses the type reliabilities tie (23/45 each).
      readings = [reading(:entity_model, "artist", 0, 3), reading(:entity_model, "final_value", 2, 3)]
      [span] = SpanReliability.decide(readings, model, text: "Poker Face", canonical: @canonical, hierarchy: Hierarchy)
      assert span.unresolved_reason == :insufficient_evidence

      extent = %{
        model.redecision.extent
        | pairs: %{
            {:entity_model, "artist"} => %{n: 10, reliability: 0.9},
            {:entity_model, "final_value"} => %{n: 10, reliability: 0.6}
          }
      }

      model = %{model | redecision: %{model.redecision | extent: extent}}

      # 0..3 is the extent: logit(0.9) against logit(0.6). Evidence for the
      # final_value at 2..3 cannot decide it, and evidence against the artist
      # leaves the span undecided.
      assert {:undecided, %Lattice{best: %{label: {0, 3, "artist"}}, candidates: candidates}} =
               SpanReliability.redecide(span, %{"artist" => -2.0, "final_value" => 50.0}, model, Hierarchy)

      assert Enum.all?(candidates, &match?(%{label: {0, 3, _}}, &1))

      assert {:decided, entity} = SpanReliability.redecide(span, %{"artist" => 2.0, "final_value" => 50.0}, model, Hierarchy)
      assert %{entity_type: "artist", start_pos: 0, end_pos: 3} = entity
      assert_in_delta entity.decision.extent.score, :math.log(0.9 / 0.1), 1.0e-6
      assert_in_delta entity.decision.extent.margin, :math.log(0.9 / 0.1) - :math.log(0.6 / 0.4), 1.0e-6
    end

    test "an extent no measured reading gives a type at leaves nothing to weigh", %{model: model} do
      # The emoji reading is unmeasured for its type but its extent is trusted.
      readings = [reading(:entity_model, "artist", 0, 3), reading(:gazetteer, "emoji", 0, 9)]
      [span] = SpanReliability.decide(readings, model, text: "Mean Kitty", canonical: @canonical, hierarchy: Hierarchy)
      assert span.unresolved_reason == :insufficient_evidence

      extent = %{
        model.redecision.extent
        | pairs: %{{:entity_model, "artist"} => %{n: 10, reliability: 0.3}, {:gazetteer, "emoji"} => %{n: 10, reliability: 0.9}}
      }

      assert {:undecided, %Lattice{best: nil}} =
               SpanReliability.redecide(span, %{"artist" => 50.0}, %{model | redecision: %{model.redecision | extent: extent}}, Hierarchy)
    end

    test "every hypothesis label needs evidence", %{model: model} do
      assert_raise ArgumentError, ~r/no evidence for \["artist"\]/, fn ->
        SpanReliability.redecide(undecided(model), %{"person" => 3.0}, model, Hierarchy)
      end
    end

    test "only a span left for insufficient evidence can be decided again", %{model: model} do
      [unmeasured] = SpanReliability.decide([reading(:gazetteer, "emoji", 0, 7)], model, text: "umbrella", canonical: @canonical, hierarchy: Hierarchy)
      assert_raise ArgumentError, ~r/only a span decide\/3 left unresolved/, fn -> SpanReliability.redecide(unmeasured, %{}, model, Hierarchy) end

      [decided] = SpanReliability.decide([reading(:system, "number", 7, 7)], %{model | min_score: -5.0}, text: "set to 5", canonical: @canonical, hierarchy: Hierarchy)
      withheld = SpanReliability.withhold(decided, :below_confidence_floor, "set to 5", Hierarchy)
      assert_raise ArgumentError, ~r/only a span decide\/3 left unresolved/, fn -> SpanReliability.redecide(withheld, %{"number" => 1.0}, model, Hierarchy) end
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

    test "select_redecision/3 decides extents first and keeps evidence that decides right, leaving evidence that does not" do
      # Two sources the data trusts about equally read every tied row
      # differently, and the first decision's min_score is out of their reach,
      # so no row is decided by its readings. The evidence names the gold
      # label strongly on 40 tied rows, and the wrong one weakly on 10.
      tied =
        for i <- 1..50 do
          type = if rem(i, 2) == 0, do: "artist", else: "final_value"
          %{key: {:row, i}, gold: [gold(type, 0, 3)], readings: [reading(:a, "artist", 0, 3), reading(:b, "final_value", 0, 3)]}
        end

      # On 20 more rows the first source reads only the end of a longer gold
      # artist ("Poker Face" read as "Face"), alone. The evidence names artist
      # as strongly as it names the right label on the tied rows, and the
      # reading's type reliability is the one it has there, so the type alone
      # cannot tell these apart; what the second source's reading of the same
      # extent adds on the tied rows can.
      partial =
        for i <- 1..20 do
          %{key: {:partial, i}, gold: [gold("artist", 0, 7)], readings: [reading(:a, "artist", 4, 7)]}
        end

      examples = tied ++ partial

      evidence = fn _training ->
        fn
          %{key: {:row, i}, gold: [%{type: right}]}, labels ->
            {to_right, to_other} = if i <= 40, do: {3.0, -3.0}, else: {-0.5, 0.5}
            Map.new(labels, &{&1, if(&1 == right, do: to_right, else: to_other)})

          %{key: {:partial, _i}}, labels ->
            Map.new(labels, &{&1, if(&1 == "artist", do: 3.0, else: -3.0)})
        end
      end

      model = examples |> SpanReliability.fit(Keyword.put(@opts, :prior_strength, 1)) |> Map.merge(%{min_score: 5.0, min_margin: 0.0})
      model = SpanReliability.select_redecision(model, examples, canonical: @canonical, hierarchy: Hierarchy, evidence: evidence)

      %{min_score: min_score, min_margin: min_margin, extent: extent, cross_validation: cv} = model.redecision
      assert cv.none.f1 == 0.0
      assert cv.first_thresholds.decided == 0
      assert cv.points == 70
      # Weighing every hypothesis whatever its extent decides the partial
      # readings wrong along with the right ones ...
      assert cv.without_extent.decided > cv.without_extent.correct
      # ... and deciding the extent first leaves them, and only them, out.
      assert %{decided: 40, correct: 40} = cv.chosen
      assert cv.chosen.f1 > cv.without_extent.f1
      assert cv.chosen.f1 == cv.by_extent_prior_strength[extent.prior_strength].f1
      assert cv.chosen.f1 == cv.by_extent_prior_strength |> Map.values() |> Enum.map(& &1.f1) |> Enum.max()
      assert min_score > 0.5 or min_margin > 1.0
      assert extent.fitted_rows == 70
      assert {extent.min_score, extent.min_margin} == {cv.chosen.extent_min_score, cv.chosen.extent_min_margin}

      [span] = SpanReliability.decide(hd(tied).readings, model, text: "Rush", canonical: @canonical, hierarchy: Hierarchy)
      assert {:decided, %{entity_type: "artist"}} = SpanReliability.redecide(span, %{"artist" => 3.0, "final_value" => -3.0}, model, Hierarchy)
      assert {:undecided, _} = SpanReliability.redecide(span, %{"artist" => 0.5, "final_value" => -0.5}, model, Hierarchy)

      [span] = SpanReliability.decide(hd(partial).readings, model, text: "Poker Face", canonical: @canonical, hierarchy: Hierarchy)
      assert {:extent_undecided, _} = SpanReliability.redecide(span, %{"artist" => 3.0}, model, Hierarchy)
    end

    test "select_redecision/3 finds the evidence that decides right among spans that all share one extent score" do
      # A reliable source decides 20 rows by itself. A second source reads one
      # extent on 20 more rows, the right type on 4 of them and no entity on
      # 16, so every one of its spans has the same extent score and is left
      # undecided. The evidence is for its type on the 4 and against it on the
      # 16: with every type hypothesis decided, deciding those extents decides
      # 16 wrong, but with the evidence weighed only the 4 right ones pass.
      reliable = for i <- 1..20, do: %{key: {:reliable, i}, gold: [gold("final_value", 0, 3)], readings: [reading(:reliable, "final_value", 0, 3)], said: :other}
      named = for i <- 1..4, do: %{key: {:named, i}, gold: [gold("artist", 0, 3)], readings: [reading(:word, "artist", 0, 3)], said: :about_artists}
      word = for i <- 1..16, do: %{key: {:word, i}, gold: [], readings: [reading(:word, "artist", 0, 3)], said: :other}
      examples = reliable ++ named ++ word

      {model, _report} = SpanReliability.select(examples, @opts)

      evidence = fn said, labels -> Map.new(labels, &{&1, if(said == :about_artists, do: 3.0, else: -3.0)}) end

      model =
        SpanReliability.select_redecision(model, examples,
          canonical: @canonical,
          hierarchy: Hierarchy,
          evidence: fn _training -> fn example, labels -> evidence.(example.said, labels) end end
        )

      assert %{decided: 24, correct: 24} = model.redecision.cross_validation.chosen

      [span] = SpanReliability.decide(hd(named).readings, model, text: "Rush", canonical: @canonical, hierarchy: Hierarchy)
      assert %{resolved: false, unresolved_reason: :insufficient_evidence} = span
      assert {:decided, %{entity_type: "artist"}} = SpanReliability.redecide(span, evidence.(:about_artists, ["artist"]), model, Hierarchy)
      refute match?({:decided, _}, SpanReliability.redecide(span, evidence.(:other, ["artist"]), model, Hierarchy))
    end

    # Nothing is decided by its readings (the first decision's min_score is out
    # of reach), so every row is weighed again with its evidence.
    defp redecided(examples, evidence) do
      examples
      |> SpanReliability.fit(Keyword.put(@opts, :prior_strength, 1))
      |> Map.merge(%{min_score: 50.0, min_margin: 0.0})
      |> SpanReliability.select_redecision(examples, canonical: @canonical, hierarchy: Hierarchy, evidence: fn _training -> evidence end)
    end

    test "select_redecision/3 gives each source extent thresholds of its own where that decides held-out folds better" do
      # The second source places its extents well but types them wrong, and
      # the evidence for its type is strong enough that its summed score is
      # above the first source's: deciding its spans is always wrong, and
      # neither their type scores nor their extent scores (both above the
      # first source's) leave them out without the first source's spans. The
      # first source's extents are right more often than not. Shared
      # thresholds cannot take the first source's spans without the second's.
      right = for i <- 1..15, do: %{key: {:right, i}, gold: [gold("artist", 0, 3)], readings: [reading(:near, "artist", 0, 3)], strength: 3.0}
      off = for i <- 1..5, do: %{key: {:off, i}, gold: [gold("artist", 0, 5)], readings: [reading(:near, "artist", 0, 3)], strength: 3.0}
      mistyped = for i <- 1..20, do: %{key: {:mistyped, i}, gold: [gold("final_value", 0, 3)], readings: [reading(:typo, "artist", 0, 3)], strength: 15.0}

      model = redecided(right ++ off ++ mistyped, fn example, labels -> Map.new(labels, &{&1, example.strength}) end)
      %{extent: extent, cross_validation: %{granularity: granularity}} = model.redecision

      assert {level, _n} = granularity.chosen
      assert level in [:source, :pair]
      assert granularity.by_granularity[granularity.chosen].f1 > granularity.by_granularity[:global].f1
      assert extent.granularity == granularity.chosen

      decide_again = fn example ->
        [span] = SpanReliability.decide(example.readings, model, text: "Rush hour", canonical: @canonical, hierarchy: Hierarchy)
        SpanReliability.redecide(span, %{"artist" => example.strength}, model, Hierarchy)
      end

      assert {:decided, %{entity_type: "artist", source: :near}} = decide_again.(hd(right))
      refute match?({:decided, _}, decide_again.(hd(mistyped)))
    end

    test "select_redecision/3 keeps shared extent thresholds where its own decide held-out folds no better" do
      # Two sources alike: each right where the evidence is for its type and
      # wrong where it is against.
      examples =
        for source <- [:one, :other], i <- 1..20 do
          gold = if rem(i, 4) == 0, do: [], else: [gold("artist", 0, 3)]
          %{key: {source, i}, gold: gold, readings: [reading(source, "artist", 0, 3)], strength: if(gold == [], do: -3.0, else: 3.0)}
        end

      model = redecided(examples, fn example, labels -> Map.new(labels, &{&1, example.strength}) end)
      %{extent: extent, cross_validation: %{granularity: granularity}} = model.redecision

      assert granularity.chosen == :global
      assert extent.granularity == :global and extent.groups == %{}
      assert Enum.all?(granularity.by_granularity, fn {_g, r} -> r.f1 <= granularity.by_granularity[:global].f1 end)
    end

    test "a model whose extent reliabilities are keyed by more than source and type is refused" do
      extent = examples() |> SpanReliability.fit_extent(prior_strength: 1) |> Map.merge(%{min_score: -10.0, min_margin: 0.0})
      keyed_by_cue = %{extent | pairs: Map.new(extent.pairs, fn {{source, type}, p} -> {{source, {:begin, :outside}, type}, p} end)}

      model =
        examples()
        |> SpanReliability.fit(Keyword.put(@opts, :prior_strength, 1))
        |> Map.merge(%{min_score: 1.0, min_margin: 0.0, redecision: %{min_score: 1.0, min_margin: 0.0, extent: keyed_by_cue}})

      [span] = SpanReliability.decide([reading(:entity_model, "artist", 0, 3)], model, text: "Mike", canonical: @canonical, hierarchy: Hierarchy)

      assert_raise ArgumentError, ~r/extent reliabilities are keyed .* not \{source, entity_type\}.*mix train_span_reliability/, fn ->
        SpanReliability.redecide(span, %{"artist" => 3.0}, model, Hierarchy)
      end
    end

    test "the prior strength is not the caller's to pass" do
      assert_raise ArgumentError, ~r/chooses the prior strength/, fn ->
        SpanReliability.select(synthetic(), Keyword.put(@opts, :prior_strength, 1))
      end
    end
  end
end
