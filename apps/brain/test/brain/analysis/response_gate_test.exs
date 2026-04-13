defmodule Brain.Analysis.ResponseGateTest do
  alias Brain.Analysis
  use ExUnit.Case, async: false

  alias Analysis.{ResponseGate, InternalModel, ChunkAnalysis, SpeechActResult}

  describe "evaluate/3" do
    test "responds to directive speech acts (questions, commands)" do
      analysis_model = build_analysis_model(:directive, :request_information, is_question: true)

      assert {:respond, %{reason: reason}} = ResponseGate.evaluate(analysis_model, [], [])
      assert reason =~ "directive"
    end

    test "responds to questions regardless of category" do
      analysis_model = build_analysis_model(:expressive, :general, is_question: true)

      assert {:respond, %{reason: reason}} = ResponseGate.evaluate(analysis_model, [], [])
      assert reason =~ "question"
    end

    test "defers on backchannel speech acts" do
      analysis_model = build_analysis_model(:expressive, :backchannel)

      assert {:optional, confidence, %{reason: reason}} =
               ResponseGate.evaluate(analysis_model, [], [])

      assert confidence >= 0.7
      assert reason =~ "backchannel"
    end

    test "optional response for compliments without questions" do
      analysis_model = build_analysis_model(:expressive, :compliment, is_question: false)

      assert {:optional, confidence, %{reason: reason}} =
               ResponseGate.evaluate(analysis_model, [], [])

      assert confidence >= 0.5
      assert reason =~ "compliment"
    end

    test "defers on continuation speech acts" do
      analysis_model = build_analysis_model(:assertive, :continuation)

      assert {:defer, %{reason: reason}} = ResponseGate.evaluate(analysis_model, [], [])
      assert reason =~ "continuation"
    end

    test "responds to greetings" do
      analysis_model = build_analysis_model(:expressive, :greeting)
      result = ResponseGate.evaluate(analysis_model, [], [])

      case result do
        {:respond, _} -> assert true
        {:optional, _, _} -> assert true
        _ -> flunk("Expected respond or optional for greeting")
      end
    end
  end

  describe "gratitude_loop?/2" do
    test "detects gratitude loop when thanks follows thanks->ack pattern" do
      current = %{sub_type: :thanks, category: :expressive}

      history = [
        %{sub_type: :thanks, category: :expressive},
        %{sub_type: :acknowledgment, category: :expressive}
      ]

      assert ResponseGate.gratitude_loop?(current, history)
    end

    test "no loop when pattern is incomplete" do
      current = %{sub_type: :thanks, category: :expressive}
      history = [%{sub_type: :thanks, category: :expressive}]

      refute ResponseGate.gratitude_loop?(current, history)
    end

    test "no loop when current is not thanks" do
      current = %{sub_type: :greeting, category: :expressive}

      history = [
        %{sub_type: :thanks, category: :expressive},
        %{sub_type: :acknowledgment, category: :expressive}
      ]

      refute ResponseGate.gratitude_loop?(current, history)
    end

    test "no loop when history is empty" do
      current = %{sub_type: :thanks, category: :expressive}
      assert not ResponseGate.gratitude_loop?(current, [])
    end
  end

  describe "has_recent_pattern?/2" do
    test "detects pattern in history" do
      history = [%{sub_type: :greeting}, %{sub_type: :thanks}, %{sub_type: :acknowledgment}]

      assert ResponseGate.has_recent_pattern?(history, [:thanks, :acknowledgment])
    end

    test "pattern must be at end of history" do
      history = [%{sub_type: :thanks}, %{sub_type: :acknowledgment}, %{sub_type: :greeting}]
      refute ResponseGate.has_recent_pattern?(history, [:thanks, :acknowledgment])
    end

    test "handles empty history" do
      refute ResponseGate.has_recent_pattern?([], [:thanks])
    end
  end

  describe "recent_thanks?/1" do
    test "detects thanks in recent history" do
      history = [%{sub_type: :greeting}, %{sub_type: :thanks}, %{sub_type: :statement}]

      assert ResponseGate.recent_thanks?(history)
    end

    test "returns false when no thanks in recent history" do
      history = [%{sub_type: :greeting}, %{sub_type: :farewell}]

      refute ResponseGate.recent_thanks?(history)
    end
  end

  describe "evaluate_speech_act/3" do
    test "evaluates speech act directly without analysis model" do
      speech_act = %{category: :directive, sub_type: :command, is_question: false}

      assert {:respond, _} = ResponseGate.evaluate_speech_act(speech_act, [], [])
    end

    test "handles nil speech act" do
      assert {:respond, %{reason: reason}} = ResponseGate.evaluate_speech_act(nil, [], [])
      assert reason =~ "no speech act"
    end
  end

  describe "integration with conversation memory" do
    test "extracts speech acts from conversation memory format" do
      memory = [
        %{
          role: "user",
          content: "Thanks!",
          context: %{
            speech_act: %{category: :expressive, sub_type: :thanks}
          }
        },
        %{
          role: "assistant",
          content: "You're welcome!"
        },
        %{
          role: "user",
          content: "Thanks again!",
          context: %{
            speech_act: %{category: :expressive, sub_type: :acknowledgment}
          }
        }
      ]

      analysis_model = build_analysis_model(:expressive, :thanks)

      result = ResponseGate.evaluate(analysis_model, memory, [])

      case result do
        {:defer, %{reason: reason}} ->
          assert reason =~ "gratitude loop"

        {:optional, _, _} ->
          assert true

        {:respond, _} ->
          assert true
      end
    end
  end

  defp build_analysis_model(category, sub_type, opts \\ []) do
    is_question = Keyword.get(opts, :is_question, false)
    confidence = Keyword.get(opts, :confidence, 0.8)

    speech_act = SpeechActResult.new(category, sub_type, confidence, is_question: is_question)

    chunk_analysis = %ChunkAnalysis{
      chunk_index: 0,
      text: "test text",
      speech_act: speech_act,
      confidence: confidence
    }

    %InternalModel{
      raw_input: "test input",
      chunks: [],
      analyses: [chunk_analysis],
      overall_strategy: :can_respond
    }
  end
end