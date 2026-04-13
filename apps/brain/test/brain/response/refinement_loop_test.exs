defmodule Brain.Response.RefinementLoopTest do
  use ExUnit.Case, async: false

  alias Brain.Response.{RefinementLoop, Primitive}
  alias Brain.Response.ResponseEvaluator.Score
  alias Brain.Analysis.{InternalModel, ChunkAnalysis, SpeechActResult}

  defp build_model(text, opts) do
    speech_act = Keyword.fetch!(opts, :speech_act)
    sentiment = Keyword.get(opts, :sentiment, %{label: :neutral, confidence: 0.5})

    analysis = %ChunkAnalysis{
      chunk_index: 0,
      text: text,
      speech_act: speech_act,
      intent: Keyword.get(opts, :intent, "general.query"),
      entities: Keyword.get(opts, :entities, []),
      sentiment: sentiment,
      confidence: Keyword.get(opts, :confidence, 0.7),
      response_strategy: Keyword.get(opts, :response_strategy, :can_respond),
      epistemic_status: Keyword.get(opts, :epistemic_status, :unchecked)
    }

    %InternalModel{
      raw_input: text,
      analyses: [analysis],
      overall_strategy: :can_respond
    }
  end

  describe "generate/2" do
    test "greeting produces social acknowledgment with evaluator convergence" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9)
      )

      assert {:ok, response, metadata} = RefinementLoop.generate(model)

      assert metadata.method == :synthesis_pipeline
      assert %Score{} = metadata.score
      assert metadata.score.speech_act_alignment >= 0.8
      assert metadata.score.overall > 0.0

      primitive_types = Enum.map(metadata.primitives, & &1.type)
      assert :acknowledgment in primitive_types

      social = Enum.find(metadata.primitives, &(&1.type == :acknowledgment and &1.variant == :social))
      assert social != nil
      assert social.rendered == response or String.contains?(response, social.rendered)
    end

    test "question plans informative framing and evaluator scores speech act alignment" do
      model = build_model("What's the weather?",
        speech_act: SpeechActResult.new(:directive, :question, 0.8, is_question: true),
        intent: "weather.query"
      )

      assert {:ok, _response, metadata} = RefinementLoop.generate(model)

      assert %Score{} = metadata.score
      assert metadata.score.speech_act_alignment >= 0.7

      primitive_types = Enum.map(metadata.primitives, & &1.type)
      assert :framing in primitive_types or :content in primitive_types
    end

    test "negative sentiment sharing includes attunement primitive" do
      model = build_model("I had a terrible day",
        speech_act: SpeechActResult.new(:assertive, :statement, 0.7),
        sentiment: %{label: :negative, confidence: 0.8}
      )

      assert {:ok, _response, metadata} = RefinementLoop.generate(model)

      primitive_types = Enum.map(metadata.primitives, & &1.type)
      assert :attunement in primitive_types

      attunement = Enum.find(metadata.primitives, &(&1.type == :attunement))
      assert attunement.rendered != nil
      assert attunement.rendered != ""

      assert metadata.score.speech_act_alignment >= 0.5
    end

    test "low confidence input produces hedging and appropriate confidence score" do
      model = build_model("What is quantum decoherence?",
        speech_act: SpeechActResult.new(:directive, :question, 0.3, is_question: true),
        confidence: 0.2
      )

      assert {:ok, _response, metadata} = RefinementLoop.generate(model)

      primitive_types = Enum.map(metadata.primitives, & &1.type)
      assert :hedging in primitive_types

      hedging = Enum.find(metadata.primitives, &(&1.type == :hedging))
      assert hedging.content[:confidence_level] <= 0.4

      assert metadata.score.confidence_alignment >= 0.7
    end

    test "empty analyses returns error tuple" do
      model = %InternalModel{raw_input: "", analyses: []}

      assert {:error, :empty_analysis} = RefinementLoop.generate(model)
    end

    test "max_iterations is respected" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9)
      )

      assert {:ok, _response, metadata} = RefinementLoop.generate(model, max_iterations: 1)
      assert metadata.iterations <= 1
    end

    test "converged responses stop iterating early" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9),
        confidence: 0.9
      )

      assert {:ok, _response, metadata} = RefinementLoop.generate(model, max_iterations: 5)

      assert metadata.score.converged == true or metadata.iterations < 5
      assert metadata.score.overall >= 0.7
    end
  end

  describe "single_pass/2" do
    test "runs exactly one iteration and returns scored primitives" do
      model = build_model("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9)
      )

      assert {:ok, response, metadata} = RefinementLoop.single_pass(model)

      assert is_binary(response), "Response should be a string"
      assert response != "", "Response should not be empty"
      assert metadata.iterations == 1
      assert %Score{} = metadata.score
      assert is_list(metadata.primitives)
      assert length(metadata.primitives) >= 1

      for p <- metadata.primitives do
        assert p.source == :ouro, "Each primitive should have source :ouro"
        assert Primitive.rendered?(p), "Each primitive should be rendered"
      end
    end
  end
end
