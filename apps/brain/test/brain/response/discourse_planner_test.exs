defmodule Brain.Response.DiscoursePlannerTest do
  use ExUnit.Case, async: false

  alias Brain.Response.DiscoursePlanner
  alias Brain.Analysis.{InternalModel, ChunkAnalysis, SpeechActResult}

  defp build_model(text, opts \\ []) do
    speech_act = Keyword.get(opts, :speech_act, SpeechActResult.new(:directive, :question, 0.8, is_question: true))
    sentiment = Keyword.get(opts, :sentiment, %{label: :neutral, confidence: 0.5})
    confidence = Keyword.get(opts, :confidence, 0.8)
    response_strategy = Keyword.get(opts, :response_strategy, :can_respond)

    analysis = %ChunkAnalysis{
      chunk_index: 0,
      text: text,
      speech_act: speech_act,
      intent: Keyword.get(opts, :intent, "general.query"),
      entities: Keyword.get(opts, :entities, []),
      sentiment: sentiment,
      confidence: confidence,
      response_strategy: response_strategy,
      epistemic_status: Keyword.get(opts, :epistemic_status, :unchecked),
      related_beliefs: Keyword.get(opts, :related_beliefs, []),
      slots: Keyword.get(opts, :slots, nil)
    }

    %InternalModel{
      raw_input: text,
      analyses: [analysis],
      overall_strategy: response_strategy
    }
  end

  describe "plan/1" do
    test "plans a greeting response" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9)
      )

      primitives = DiscoursePlanner.plan(model)

      assert is_list(primitives)
      assert length(primitives) >= 1

      types = Enum.map(primitives, & &1.type)
      assert :acknowledgment in types
    end

    test "plans a factual question response" do
      model = build_model("What's the weather?",
        speech_act: SpeechActResult.new(:directive, :question, 0.8, is_question: true),
        intent: "weather.query"
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :framing in types or :content in types
    end

    test "plans with attunement for negative sentiment" do
      model = build_model("I had a terrible day",
        speech_act: SpeechActResult.new(:assertive, :statement, 0.7),
        sentiment: %{label: :negative, confidence: 0.8}
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :attunement in types
    end

    test "inserts hedging for low confidence" do
      model = build_model("What's that?",
        speech_act: SpeechActResult.new(:directive, :question, 0.3, is_question: true),
        confidence: 0.2
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :hedging in types
    end

    test "handles contradiction" do
      model = build_model("Actually my name is Bob",
        speech_act: SpeechActResult.new(:assertive, :statement, 0.8),
        epistemic_status: :contradicted
      )

      primitives = DiscoursePlanner.plan(model)
      types = Enum.map(primitives, & &1.type)

      assert :contradiction_response in types
    end

    test "seeds content from analysis" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9)
      )

      primitives = DiscoursePlanner.plan(model)
      social_ack = Enum.find(primitives, &(&1.type == :acknowledgment and &1.variant == :social))

      if social_ack do
        assert Map.has_key?(social_ack.content, :speech_act_sub_type)
      end
    end

    test "returns fallback for nil input" do
      primitives = DiscoursePlanner.plan(nil)
      assert is_list(primitives)
      assert length(primitives) >= 1
    end
  end
end
