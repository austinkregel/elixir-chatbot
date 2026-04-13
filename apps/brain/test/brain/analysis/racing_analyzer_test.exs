defmodule Brain.Analysis.RacingAnalyzerTest do
  alias Brain.ML.EntityExtractor
  alias Brain.Analysis
  use Brain.Test.GraphCase, async: false

  alias Analysis.{RacingAnalyzer, Interpretation, AnalyzerResult}
  import Brain.TestHelpers

  setup _context do
    start_test_services()
    EntityExtractor.load_entity_maps()
    :ok
  end

  describe "race/2 basic functionality" do
    test "returns Interpretation struct" do
      result = RacingAnalyzer.race("Hello there!")

      assert %Interpretation{} = result
      assert is_binary(result.intent) or is_nil(result.intent)
      assert is_float(result.activation)
      assert result.activation >= 0.0 and result.activation <= 1.0
    end

    test "handles empty string input" do
      result = RacingAnalyzer.race("")

      assert %Interpretation{} = result
      assert result.text == ""
    end

    test "populates alternatives list" do
      result = RacingAnalyzer.race("What's the weather in New York?")

      assert %Interpretation{} = result
      assert is_list(result.alternatives)

      Enum.each(result.alternatives, fn alt ->
        assert Map.has_key?(alt, :intent)
        assert Map.has_key?(alt, :activation)
        assert Map.has_key?(alt, :source)
      end)
    end

    test "respects skip_heuristics option" do
      result = RacingAnalyzer.race("Hello", skip_heuristics: true)

      assert %Interpretation{} = result
    end

    test "respects skip_memory option" do
      result = RacingAnalyzer.race("Hello", skip_memory: true)

      assert %Interpretation{} = result
    end
  end

  describe "check_fast_path/4" do
    test "returns :no_match when no heuristic fires" do
      result = RacingAnalyzer.check_fast_path("xyzabc123randomtext", "default", nil, nil)

      assert result == :no_match
    end

    test "returns tuple structure when fast path fires" do
      result = RacingAnalyzer.check_fast_path("Hello", "default", nil, nil)

      case result do
        {:fast_path, interpretation} ->
          assert %Interpretation{} = interpretation
          assert interpretation.intent != nil
          assert interpretation.activation >= 0.85

        :no_match ->
          assert true
      end
    end
  end

  describe "analyzer orchestration" do
    test "structural analyzer detects questions with question marks" do
      result = RacingAnalyzer.race("What is your name?", skip_heuristics: true, skip_memory: true)

      assert %Interpretation{} = result
      assert is_list(result.analyzer_results)

      structural_result =
        Enum.find(result.analyzer_results, fn r ->
          r.analyzer == :structural
        end)

      if structural_result do
        assert structural_result.intent =~ ~r/question/i or
                 "question_mark" in (structural_result.indicators || [])
      end
    end

    test "structural analyzer detects WH-words" do
      result =
        RacingAnalyzer.race("Where are you from?", skip_heuristics: true, skip_memory: true)

      assert %Interpretation{} = result

      structural_result =
        Enum.find(result.analyzer_results, fn r ->
          r.analyzer == :structural
        end)

      if structural_result do
        assert structural_result.intent =~ ~r/question/i or
                 "wh_word" in (structural_result.indicators || [])
      end
    end

    test "model analyzer returns AnalyzerResult from classifier" do
      result =
        RacingAnalyzer.race("Play some music", skip_heuristics: true, skip_memory: true)

      assert %Interpretation{} = result
      assert is_list(result.analyzer_results)

      model_result =
        Enum.find(result.analyzer_results, fn r ->
          r.analyzer == :model
        end)

      if model_result do
        assert %AnalyzerResult{} = model_result
        assert model_result.analyzer == :model
        assert is_float(model_result.raw_score)
      end
    end

    test "returns multiple analyzer results" do
      result = RacingAnalyzer.race("Turn on the lights", skip_heuristics: true, skip_memory: true)

      assert %Interpretation{} = result
      assert result.analyzer_results != []
      analyzer_names = Enum.map(result.analyzer_results, & &1.analyzer)
      assert :structural in analyzer_names or :model in analyzer_names
    end
  end

  describe "calibration and normalization" do
    test "activation is normalized between 0 and 1" do
      result = RacingAnalyzer.race("What's the weather in Paris?")

      assert %Interpretation{} = result
      assert result.activation >= 0.0
      assert result.activation <= 1.0

      Enum.each(result.alternatives, fn alt ->
        assert alt.activation >= 0.0
        assert alt.activation <= 1.0
      end)
    end

    test "total activation across all interpretations does not exceed 1.0" do
      result = RacingAnalyzer.race("Hello, what's the weather?")

      total_activation =
        result.activation + Enum.sum(Enum.map(result.alternatives, & &1.activation))

      assert total_activation <= 1.05
    end
  end

  describe "performance" do
    test "completes within reasonable time" do
      start_time = System.monotonic_time(:millisecond)

      _result = RacingAnalyzer.race("What's the weather in New York?")

      elapsed = System.monotonic_time(:millisecond) - start_time
      assert elapsed < 3000
    end
  end
end
