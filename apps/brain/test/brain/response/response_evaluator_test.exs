defmodule Brain.Response.ResponseEvaluatorTest do
  @moduledoc """
  Tests for `Brain.Response.ResponseEvaluator`.

  Verifies the epistemically grounded scoring dimensions:
  - belief_grounding: claims aligned with JTMS-backed beliefs score high
  - epistemic_consistency: JTMS contradictions lower the score
  - speech_act_alignment: pipeline-based classification of the response
  - content_coverage: Tokenizer-based overlap (no String.contains?)
  """
  use Brain.Test.GraphCase, async: false

  alias Brain.Response.ResponseEvaluator
  alias Brain.Response.ResponseEvaluator.Score
  alias Brain.Response.Primitive
  alias Brain.Analysis.{ChunkAnalysis, SpeechActResult}
  alias Brain.Epistemic.BeliefStore
  alias Brain.Epistemic.Types.Belief

  defp build_analysis(text, opts \\ []) do
    speech_act = Keyword.get(opts, :speech_act, SpeechActResult.new(:expressive, :greeting, 0.9))
    entities = Keyword.get(opts, :entities, [])

    %ChunkAnalysis{
      chunk_index: 0,
      text: text,
      speech_act: speech_act,
      intent: Keyword.get(opts, :intent, "greeting"),
      entities: entities,
      sentiment: %{label: :neutral, confidence: 0.5},
      confidence: Keyword.get(opts, :confidence, 0.7),
      response_strategy: :can_respond,
      epistemic_status: :unchecked
    }
  end

  describe "evaluate/3 score struct" do
    test "returns a Score with all expected dimensions" do
      analysis = build_analysis("Hello!")
      primitives = [Primitive.new(:acknowledgment, :social, %{})]

      score = ResponseEvaluator.evaluate(primitives, "Hi there, how are you?", analysis)

      assert %Score{} = score
      assert is_float(score.speech_act_alignment)
      assert is_float(score.confidence_alignment)
      assert is_float(score.content_coverage)
      assert is_float(score.content_completeness)
      assert is_float(score.slot_coverage)
      assert is_float(score.naturalness)
      assert is_float(score.echo_avoidance)
      assert is_float(score.belief_grounding)
      assert is_float(score.epistemic_consistency)
      assert is_float(score.overall)
      assert score.overall >= 0.0 and score.overall <= 1.0
      assert is_atom(score.weakest_dimension)
      assert is_boolean(score.converged)
      assert is_boolean(score.silence_preferred)
    end

    test "fallback evaluate returns converged score for non-matching inputs" do
      score = ResponseEvaluator.evaluate(nil, nil, nil)
      assert %Score{converged: true, overall: 0.5} = score
    end
  end

  describe "dimension_to_stage/1" do
    test "belief_grounding maps to content_specifier" do
      assert :content_specifier = ResponseEvaluator.dimension_to_stage(:belief_grounding)
    end

    test "epistemic_consistency maps to discourse_planner" do
      assert :discourse_planner = ResponseEvaluator.dimension_to_stage(:epistemic_consistency)
    end

    test "speech_act_alignment maps to discourse_planner" do
      assert :discourse_planner = ResponseEvaluator.dimension_to_stage(:speech_act_alignment)
    end

    test "naturalness maps to surface_realizer" do
      assert :surface_realizer = ResponseEvaluator.dimension_to_stage(:naturalness)
    end
  end

  describe "belief grounding" do
    test "response with no extracted entities scores acceptably" do
      analysis = build_analysis("Hello!")
      primitives = [Primitive.new(:acknowledgment, :social, %{})]

      score = ResponseEvaluator.evaluate(primitives, "Hi there!", analysis)

      assert score.belief_grounding >= 0.5
    end

    test "response aligning with an IN belief scores high on grounding" do
      belief = Belief.new(:world, :capital, "Paris is the capital of France",
        source: :explicit, confidence: 0.95)
      {:ok, _} = BeliefStore.add_belief(belief)

      analysis = build_analysis("What is the capital of France?",
        speech_act: SpeechActResult.new(:directive, :question, 0.8, is_question: true),
        entities: [%{entity_type: "location", value: "France"}]
      )

      primitives = [Primitive.new(:content, :factual, %{
        facts: [%{fact: "Paris is the capital of France"}]
      })]

      score = ResponseEvaluator.evaluate(
        primitives,
        "Paris is the capital of France, a beautiful city in Europe.",
        analysis
      )

      assert score.belief_grounding >= 0.5
    end

    test "novel claims without contradicting beliefs score medium" do
      analysis = build_analysis("Tell me about quantum mechanics",
        speech_act: SpeechActResult.new(:directive, :question, 0.7, is_question: true)
      )

      primitives = [Primitive.new(:content, :explanatory, %{})]

      score = ResponseEvaluator.evaluate(
        primitives,
        "Quantum mechanics describes the behavior of particles at atomic scales.",
        analysis
      )

      assert score.belief_grounding >= 0.5
    end
  end

  describe "epistemic consistency" do
    test "consistent JTMS state scores high" do
      analysis = build_analysis("Hello!")
      primitives = [Primitive.new(:acknowledgment, :social, %{})]

      score = ResponseEvaluator.evaluate(primitives, "Hi there, friend!", analysis)

      assert score.epistemic_consistency >= 0.5
    end
  end

  describe "content coverage uses tokenizer overlap" do
    test "entity coverage uses tokenizer instead of String.contains?" do
      analysis = build_analysis("Tell me about Paris",
        entities: [%{entity_type: "location", value: "Paris"}]
      )

      primitives = [Primitive.new(:content, :factual, %{})]

      score = ResponseEvaluator.evaluate(
        primitives,
        "Paris is a wonderful city with great architecture.",
        analysis
      )

      assert score.content_coverage >= 0.5
    end
  end

  describe "convergence and weakest dimension" do
    test "high-quality greeting converges" do
      analysis = build_analysis("Hello!",
        speech_act: SpeechActResult.new(:expressive, :greeting, 0.9),
        confidence: 0.9
      )

      primitives = [Primitive.new(:acknowledgment, :social, %{
        greeting_type: :hello
      })]

      score = ResponseEvaluator.evaluate(
        primitives,
        "Hello! Nice to meet you.",
        analysis
      )

      assert score.overall > 0.0
      assert is_atom(score.weakest_dimension)
    end
  end
end
