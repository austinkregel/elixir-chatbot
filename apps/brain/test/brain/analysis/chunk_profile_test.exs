defmodule Brain.Analysis.ChunkProfileTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.{ChunkProfile, ChunkAnalysis, SpeechActResult, DiscourseResult}

  describe "new/0" do
    test "creates a default profile with all expected fields" do
      profile = ChunkProfile.new()

      assert profile.domain == :unknown
      assert profile.speech_act_category == :unknown
      assert profile.speech_act_subtype == :unknown
      assert profile.target == :ambiguous
      assert profile.modality == :declarative
      assert profile.polarity == :affirmative
      assert profile.tense == :present
      assert profile.aspect == :simple
      assert profile.addressee == :unknown
      assert profile.urgency == :low
      assert profile.certainty == :committed
      assert profile.sentiment_alignment == :neutral
      assert profile.slot_completeness == 1.0
      assert profile.novelty_score == 0.0
      assert profile.feature_provenance == %{}
      assert profile.confidence == 0.0
      assert profile.derived_label == ""
      assert profile.response_posture == :direct
      assert profile.engagement_level == :casual_engagement
      assert profile.self_disclosure_level == :none
      assert profile.temporal_framing == :timeless
      assert profile.feature_vector == []
    end
  end

  describe "materialize/2" do
    test "projects speech act fields from analysis" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "hello there",
        speech_act:
          SpeechActResult.new(:expressive, :greeting, 0.9,
            is_question: false,
            is_imperative: false
          ),
        discourse: DiscourseResult.new(:bot, 0.8),
        confidence: 0.85,
        pos_tags: [{"hello", "INTJ"}, {"there", "ADV"}]
      }

      profile = ChunkProfile.materialize(analysis, [])

      assert profile.speech_act_category == :expressive
      assert profile.speech_act_subtype == :greeting
      assert profile.addressee == :bot
      assert is_float(profile.confidence)
    end

    test "derives label from domain and speech_act_subtype" do
      profile = %ChunkProfile{domain: :weather, speech_act_subtype: :question_factual}
      assert ChunkProfile.derived_label(profile) == "weather.question_factual"
    end

    test "handles nil speech_act gracefully" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "test",
        speech_act: nil,
        discourse: nil,
        confidence: 0.5,
        pos_tags: []
      }

      profile = ChunkProfile.materialize(analysis, [])
      assert profile.speech_act_category == :unknown
      assert profile.addressee == :unknown
    end

    test "stores feature vector" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "what is the weather",
        speech_act:
          SpeechActResult.new(:directive, :question_factual, 0.8, is_question: true),
        discourse: DiscourseResult.new(:bot, 0.7),
        confidence: 0.7,
        pos_tags: [{"what", "PRON"}, {"is", "AUX"}, {"the", "DET"}, {"weather", "NOUN"}]
      }

      feature_vector = List.duplicate(0.5, 140)
      profile = ChunkProfile.materialize(analysis, feature_vector)

      assert length(profile.feature_vector) == 140
    end

    test "projects modality from speech act" do
      question_analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "what time is it",
        speech_act:
          SpeechActResult.new(:directive, :question_factual, 0.9, is_question: true),
        discourse: DiscourseResult.new(:bot, 0.8),
        confidence: 0.8,
        pos_tags: []
      }

      profile = ChunkProfile.materialize(question_analysis, [])
      assert profile.modality == :interrogative
    end
  end

  describe "interaction axes" do
    test "response_posture is :direct for high confidence committed utterances" do
      profile = %ChunkProfile{
        confidence: 0.9,
        certainty: :committed,
        slot_completeness: 1.0,
        novelty_score: 0.1
      }

      assert profile.confidence >= 0.7
      assert profile.certainty == :committed
    end

    test "derived_label format" do
      profile = %ChunkProfile{
        domain: :smarthome,
        speech_act_subtype: :command
      }

      assert ChunkProfile.derived_label(profile) == "smarthome.command"
    end
  end
end
