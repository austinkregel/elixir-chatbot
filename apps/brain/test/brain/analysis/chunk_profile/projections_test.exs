defmodule Brain.Analysis.ChunkProfile.ProjectionsTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.ChunkProfile.Projections
  alias Brain.Analysis.SpeechActResult

  describe "project_modality/1" do
    test "returns :interrogative for questions" do
      analysis = %{
        speech_act: %SpeechActResult{
          is_question: true,
          is_imperative: false,
          category: :directive
        }
      }

      assert Projections.project_modality(analysis) == :interrogative
    end

    test "returns :imperative for imperatives" do
      analysis = %{
        speech_act: %SpeechActResult{
          is_question: false,
          is_imperative: true,
          category: :directive
        }
      }

      assert Projections.project_modality(analysis) == :imperative
    end

    test "returns :exclamatory for expressives" do
      analysis = %{
        speech_act: %SpeechActResult{
          is_question: false,
          is_imperative: false,
          category: :expressive
        }
      }

      assert Projections.project_modality(analysis) == :exclamatory
    end

    test "returns :declarative by default" do
      analysis = %{
        speech_act: %SpeechActResult{
          is_question: false,
          is_imperative: false,
          category: :assertive
        }
      }

      assert Projections.project_modality(analysis) == :declarative
    end
  end

  describe "project_polarity/1" do
    test "detects negative polarity from negation tokens" do
      analysis = %{
        text: "I don't want that",
        pos_tags: [{"I", :PRON}, {"don't", :PART}, {"want", :VERB}, {"that", :DET}]
      }

      assert Projections.project_polarity(analysis) == :negative
    end

    test "returns affirmative when no negation" do
      analysis = %{
        text: "I want coffee",
        pos_tags: [{"I", :PRON}, {"want", :VERB}, {"coffee", :NOUN}]
      }

      assert Projections.project_polarity(analysis) == :affirmative
    end
  end

  describe "project_response_posture/1" do
    test "direct for high confidence" do
      signals = %{
        confidence: 0.9,
        certainty: :committed,
        slot_completeness: 1.0,
        novelty_score: 0.1
      }

      assert Projections.project_response_posture(signals) == :direct
    end

    test "clarify for low confidence" do
      signals = %{
        confidence: 0.2,
        certainty: :hedged,
        slot_completeness: 0.5,
        novelty_score: 0.8
      }

      assert Projections.project_response_posture(signals) == :clarify
    end
  end

  describe "project_temporal_framing/1" do
    test "timeless for atemporal" do
      assert Projections.project_temporal_framing(%{
               tense: :atemporal,
               aspect: :simple,
               polarity: :affirmative
             }) == :timeless
    end

    test "completed_past for past simple" do
      assert Projections.project_temporal_framing(%{
               tense: :past,
               aspect: :simple,
               polarity: :affirmative
             }) == :completed_past
    end

    test "hypothetical_future for future" do
      assert Projections.project_temporal_framing(%{
               tense: :future,
               aspect: :simple,
               polarity: :affirmative
             }) == :hypothetical_future
    end

    test "negated_past for negative past" do
      assert Projections.project_temporal_framing(%{
               tense: :past,
               aspect: :simple,
               polarity: :negative
             }) == :negated_past
    end
  end

  describe "project_engagement_level/1" do
    test "urgent_demand for bot-addressed imperatives with high urgency" do
      signals = %{
        addressee: :bot,
        target: :agent,
        modality: :imperative,
        urgency: :critical
      }

      assert Projections.project_engagement_level(signals) == :urgent_demand
    end

    test "active_request for bot-addressed questions" do
      signals = %{
        addressee: :bot,
        target: :agent,
        modality: :interrogative,
        urgency: :low
      }

      assert Projections.project_engagement_level(signals) == :active_request
    end

    test "passive_observation for user-addressed" do
      signals = %{
        addressee: :user,
        target: :self,
        modality: :declarative,
        urgency: :low
      }

      assert Projections.project_engagement_level(signals) == :passive_observation
    end
  end

  describe "safe_to_atom/1" do
    test "converts strings to atoms" do
      assert Projections.safe_to_atom("weather") == :weather
    end

    test "handles existing atoms" do
      assert is_atom(Projections.safe_to_atom("test_atom_12345"))
    end

    test "passes through atoms unchanged" do
      assert Projections.safe_to_atom(:already_atom) == :already_atom
    end

    test "returns :unknown for non-string non-atom input" do
      assert Projections.safe_to_atom(42) == :unknown
    end
  end
end
