defmodule Brain.Analysis.ComprehensionAssessor.DimensionEvaluatorsTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.ComprehensionAssessor.DimensionEvaluators
  alias Brain.Analysis.{ChunkAnalysis, SpeechActResult, DiscourseResult, SlotResult}
  alias Brain.Analysis.Types.Event

  # Helper to build a minimal ChunkAnalysis with overrides
  defp build_analysis(overrides \\ %{}) do
    base = %ChunkAnalysis{
      chunk_index: 0,
      text: Map.get(overrides, :text, "This is a test sentence for analysis."),
      discourse: Map.get(overrides, :discourse, %DiscourseResult{}),
      speech_act: Map.get(overrides, :speech_act, %SpeechActResult{}),
      intent: Map.get(overrides, :intent, nil),
      entities: Map.get(overrides, :entities, []),
      slots: Map.get(overrides, :slots, nil),
      missing_context: Map.get(overrides, :missing_context, []),
      events: Map.get(overrides, :events, []),
      sentiment: Map.get(overrides, :sentiment, nil),
      fact_verification: Map.get(overrides, :fact_verification, nil),
      related_beliefs: Map.get(overrides, :related_beliefs, []),
      epistemic_status: Map.get(overrides, :epistemic_status, :unchecked)
    }

    base
  end

  defp make_event(opts \\ []) do
    %Event{
      id: "evt_test_#{:rand.uniform(10000)}",
      action: Keyword.get(opts, :action, %{verb: "want", lemma: "want", tense: :present}),
      actor: Keyword.get(opts, :actor, nil),
      object: Keyword.get(opts, :object, nil),
      modifiers: Keyword.get(opts, :modifiers, []),
      confidence: Keyword.get(opts, :confidence, 0.8),
      source_tokens: []
    }
  end

  describe "referential_clarity/1" do
    test "scores high with multiple high-confidence entities" do
      analysis =
        build_analysis(%{
          entities: [
            %{value: "France", entity_type: "location", confidence: 0.9},
            %{value: "Europe", entity_type: "location", confidence: 0.8}
          ]
        })

      {score, evidence} = DimensionEvaluators.referential_clarity(analysis)
      assert score == 1.0
      assert evidence.entity_count == 2
      assert evidence.has_high_confidence_entity == true
    end

    test "scores medium with one entity" do
      analysis =
        build_analysis(%{
          entities: [%{value: "coffee", entity_type: "object", confidence: 0.4}]
        })

      {score, _evidence} = DimensionEvaluators.referential_clarity(analysis)
      assert score == 0.6
    end

    test "scores low with no entities and no event objects" do
      analysis = build_analysis(%{entities: [], events: []})
      {score, evidence} = DimensionEvaluators.referential_clarity(analysis)
      assert score == 0.1
      assert evidence.entity_count == 0
    end

    test "scores medium with event objects but no entities" do
      analysis =
        build_analysis(%{
          entities: [],
          events: [
            make_event(object: %{text: "coffee", type: "noun", token_index: 2})
          ]
        })

      {score, evidence} = DimensionEvaluators.referential_clarity(analysis)
      assert score == 0.4
      assert evidence.objects_identified == 1
    end
  end

  describe "actor_identification/1" do
    test "scores high with known addressee and event actors" do
      analysis =
        build_analysis(%{
          discourse: %DiscourseResult{addressee: :bot, confidence: 0.9},
          events: [
            make_event(actor: %{text: "I", type: "pronoun", token_index: 0})
          ]
        })

      {score, evidence} = DimensionEvaluators.actor_identification(analysis)
      assert score == 1.0
      assert evidence.addressee_confidence == 0.9
      assert evidence.actors_identified == 1
    end

    test "scores low with unknown addressee and no actors" do
      analysis =
        build_analysis(%{
          discourse: %DiscourseResult{addressee: :unknown, confidence: 0.1},
          events: [make_event()]
        })

      {score, _evidence} = DimensionEvaluators.actor_identification(analysis)
      assert score == 0.1
    end

    test "scores medium with only addressee" do
      analysis =
        build_analysis(%{
          discourse: %DiscourseResult{addressee: :bot, confidence: 0.6},
          events: []
        })

      {score, _evidence} = DimensionEvaluators.actor_identification(analysis)
      assert score == 0.7
    end
  end

  describe "propositional_content/1" do
    test "scores high with assertive speech act and complete events" do
      analysis =
        build_analysis(%{
          speech_act: %SpeechActResult{category: :assertive, confidence: 0.9},
          events: [
            make_event(
              action: %{verb: "is", lemma: "be", tense: :present},
              object: %{text: "country", type: "noun", token_index: 3}
            )
          ]
        })

      {score, evidence} = DimensionEvaluators.propositional_content(analysis)
      assert score == 1.0
      assert evidence.is_assertive == true
      assert evidence.complete_events == 1
    end

    test "scores low with no assertive and no events" do
      analysis =
        build_analysis(%{
          speech_act: %SpeechActResult{category: :expressive, confidence: 0.5},
          events: []
        })

      {score, evidence} = DimensionEvaluators.propositional_content(analysis)
      assert score == 0.1
      assert evidence.is_assertive == false
    end

    test "scores medium with assertive but only action, no object" do
      analysis =
        build_analysis(%{
          speech_act: %SpeechActResult{category: :assertive, confidence: 0.7},
          events: [
            make_event(action: %{verb: "exists", lemma: "exist", tense: :present})
          ]
        })

      {score, evidence} = DimensionEvaluators.propositional_content(analysis)
      assert score == 0.7
      assert evidence.has_action == true
    end
  end

  describe "temporal_grounding/1" do
    test "scores high with temporal entity and explicit tense" do
      analysis =
        build_analysis(%{
          entities: [%{value: "2024", entity_type: "date", confidence: 0.9}],
          events: [
            make_event(action: %{verb: "happened", lemma: "happen", tense: :past})
          ]
        })

      {score, evidence} = DimensionEvaluators.temporal_grounding(analysis)
      assert score == 1.0
      assert evidence.has_explicit_tense == true
      assert evidence.temporal_entities == 1
    end

    test "scores low with no temporal info" do
      analysis = build_analysis(%{entities: [], events: []})
      {score, evidence} = DimensionEvaluators.temporal_grounding(analysis)
      assert score == 0.1
      assert evidence.has_explicit_tense == false
      assert evidence.temporal_entities == 0
    end

    test "scores medium with temporal modifiers" do
      analysis =
        build_analysis(%{
          events: [
            make_event(
              action: %{verb: "run", lemma: "run", tense: :unknown},
              modifiers: [%{type: :temporal, text: "yesterday", token_index: 0}]
            )
          ]
        })

      {score, evidence} = DimensionEvaluators.temporal_grounding(analysis)
      assert score == 0.7
      assert evidence.temporal_modifiers == 1
    end
  end

  describe "contextual_sufficiency/1" do
    test "scores high when all required slots filled and no missing context" do
      analysis =
        build_analysis(%{
          slots: %SlotResult{all_required_filled: true, missing_required: []},
          missing_context: []
        })

      {score, evidence} = DimensionEvaluators.contextual_sufficiency(analysis)
      assert score == 1.0
      assert evidence.missing_context_count == 0
      assert evidence.all_required_filled == true
    end

    test "scores low with many missing required slots" do
      analysis =
        build_analysis(%{
          slots: %SlotResult{
            all_required_filled: false,
            missing_required: ["location", "time", "subject", "duration"]
          },
          missing_context: [:topic]
        })

      {score, evidence} = DimensionEvaluators.contextual_sufficiency(analysis)
      assert score <= 0.3
      assert evidence.missing_required_slots == 4
    end

    test "scores medium with one missing item" do
      analysis =
        build_analysis(%{
          slots: %SlotResult{all_required_filled: false, missing_required: ["location"]},
          missing_context: []
        })

      {score, _evidence} = DimensionEvaluators.contextual_sufficiency(analysis)
      assert score == 0.7
    end
  end

  describe "epistemic_grounding/1" do
    test "scores high with verified fact and related beliefs" do
      analysis =
        build_analysis(%{
          fact_verification: {:verified, 0.9},
          related_beliefs: [%{id: "b1", predicate: :is_a}]
        })

      {score, evidence} = DimensionEvaluators.epistemic_grounding(analysis)
      assert score == 1.0
      assert evidence.related_belief_count == 1
      assert evidence.verification_score == 0.9
    end

    test "scores low with no verification and no beliefs" do
      analysis = build_analysis(%{fact_verification: nil, related_beliefs: []})
      {score, evidence} = DimensionEvaluators.epistemic_grounding(analysis)
      assert score == 0.1
      assert evidence.related_belief_count == 0
    end

    test "handles contradicted fact" do
      analysis =
        build_analysis(%{
          fact_verification: {:contradicted, [%{claim: "X"}]},
          related_beliefs: []
        })

      {score, evidence} = DimensionEvaluators.epistemic_grounding(analysis)
      assert score == 0.5
      assert evidence.verification_score == 0.3
    end
  end

  describe "structural_coherence/1" do
    test "scores high with good speech act confidence and normal text" do
      analysis =
        build_analysis(%{
          text: "France is a country in Western Europe with several overseas regions.",
          speech_act: %SpeechActResult{category: :assertive, confidence: 0.85}
        })

      {score, evidence} = DimensionEvaluators.structural_coherence(analysis)
      assert score >= 0.7
      assert evidence.speech_act_confidence == 0.85
    end

    test "scores very low for garbled text with mostly non-alpha characters" do
      analysis =
        build_analysis(%{
          text: "<<<>>>{{{}}}[[[]]]==!!@@##$$%%^^&&**(())__++--//\\\\|||",
          speech_act: %SpeechActResult{category: :unknown, confidence: 0.1}
        })

      {score, evidence} = DimensionEvaluators.structural_coherence(analysis)
      assert score < 0.2
      assert evidence.alpha_ratio < 0.3
    end

    test "scores low for very short text" do
      analysis =
        build_analysis(%{
          text: "hi",
          speech_act: %SpeechActResult{category: :expressive, confidence: 0.3}
        })

      {score, _evidence} = DimensionEvaluators.structural_coherence(analysis)
      assert score <= 0.3
    end
  end

  describe "illocutionary_clarity/1" do
    test "scores high with known category, high confidence, and intent" do
      analysis =
        build_analysis(%{
          speech_act: %SpeechActResult{category: :assertive, confidence: 0.9},
          intent: "weather.query"
        })

      {score, evidence} = DimensionEvaluators.illocutionary_clarity(analysis)
      assert score == 1.0
      assert evidence.category_known == true
      assert evidence.has_intent == true
    end

    test "scores low with unknown category and no intent" do
      analysis =
        build_analysis(%{
          speech_act: %SpeechActResult{category: :unknown, confidence: 0.1},
          intent: nil
        })

      {score, evidence} = DimensionEvaluators.illocutionary_clarity(analysis)
      assert score == 0.1
      assert evidence.category_known == false
    end

    test "scores medium with category known but low confidence" do
      analysis =
        build_analysis(%{
          speech_act: %SpeechActResult{category: :directive, confidence: 0.3},
          intent: nil
        })

      {score, _evidence} = DimensionEvaluators.illocutionary_clarity(analysis)
      assert score == 0.6
    end
  end

  describe "evaluate_all/1" do
    test "returns all 8 dimensions" do
      analysis = build_analysis()
      result = DimensionEvaluators.evaluate_all(analysis)

      assert map_size(result) == 8

      expected_dims = [
        :referential_clarity,
        :actor_identification,
        :propositional_content,
        :temporal_grounding,
        :contextual_sufficiency,
        :epistemic_grounding,
        :structural_coherence,
        :illocutionary_clarity
      ]

      for dim <- expected_dims do
        assert Map.has_key?(result, dim), "Missing dimension: #{dim}"
        {score, evidence} = result[dim]
        assert is_float(score) or is_integer(score)
        assert score >= 0.0 and score <= 1.0
        assert is_map(evidence)
      end
    end

    test "well-formed assertive text scores highly across most dimensions" do
      analysis =
        build_analysis(%{
          text: "France is a country in Western Europe with several overseas regions.",
          speech_act: %SpeechActResult{category: :assertive, confidence: 0.9},
          intent: "knowledge.statement",
          discourse: %DiscourseResult{addressee: :bot, confidence: 0.8},
          entities: [
            %{value: "France", entity_type: "location", confidence: 0.95},
            %{value: "Europe", entity_type: "location", confidence: 0.9}
          ],
          events: [
            make_event(
              action: %{verb: "is", lemma: "be", tense: :present},
              actor: %{text: "France", type: "noun", token_index: 0},
              object: %{text: "country", type: "noun", token_index: 3}
            )
          ],
          fact_verification: {:verified, 0.8},
          related_beliefs: [%{id: "b1"}]
        })

      result = DimensionEvaluators.evaluate_all(analysis)

      {ref_score, _} = result[:referential_clarity]
      {prop_score, _} = result[:propositional_content]
      {struct_score, _} = result[:structural_coherence]
      {ill_score, _} = result[:illocutionary_clarity]

      assert ref_score >= 0.8, "referential_clarity should be high: #{ref_score}"
      assert prop_score >= 0.8, "propositional_content should be high: #{prop_score}"
      assert struct_score >= 0.7, "structural_coherence should be high: #{struct_score}"
      assert ill_score >= 0.8, "illocutionary_clarity should be high: #{ill_score}"

      # temporal_grounding should be low (no temporal info in the sentence)
      {temp_score, _} = result[:temporal_grounding]
      assert temp_score <= 0.5, "temporal_grounding should be low: #{temp_score}"
    end
  end
end
