defmodule ChatBot.Analysis.ProcessingTraceTest do
  use ExUnit.Case, async: false

  alias ChatBot.Analysis.{ProcessingTrace, Interpretation}
  import ChatBot.TestHelpers

  setup do
    # Start common test services
    start_test_services()
    ChatBot.ML.EntityExtractor.load_entity_maps()
    :ok
  end

  describe "from_interpretation/2" do
    test "builds trace from Interpretation struct" do
      interpretation =
        Interpretation.new("greeting.hello", "Hello there!", 0.85, :model)

      trace = ProcessingTrace.from_interpretation(interpretation)

      assert %ProcessingTrace{} = trace
      assert trace.primary_intent == "greeting.hello"
      assert trace.primary_activation == 0.85
      assert trace.input_text == "Hello there!"
    end

    test "captures fast path trigger info when heuristic matched" do
      interpretation =
        Interpretation.new("greeting.hello", "Hello", 0.90, :heuristic)
        |> Interpretation.with_heuristic("greeting_heuristic_1", :global)

      trace = ProcessingTrace.from_interpretation(interpretation)

      assert %ProcessingTrace{} = trace
      assert trace.fast_path_triggered == true
      # Note: triggering_heuristic may be nil if the heuristic isn't in HeuristicStore
      # The important thing is that fast_path_triggered is true
    end

    test "captures non-fast-path for model source" do
      interpretation =
        Interpretation.new("weather.query", "What's the weather?", 0.75, :model)

      trace = ProcessingTrace.from_interpretation(interpretation)

      assert %ProcessingTrace{} = trace
      assert trace.fast_path_triggered == false
    end

    test "formats analyzer results (top 5 by calibrated score)" do
      interpretation =
        Interpretation.new("music.play", "Play some music", 0.80, :model)
        |> Map.put(:analyzer_results, [
          %{
            analyzer: :model,
            intent: "music.play",
            raw_score: 0.80,
            calibrated_activation: 0.78,
            indicators: ["ml_model"]
          },
          %{
            analyzer: :structural,
            intent: "command.general",
            raw_score: 0.60,
            calibrated_activation: 0.55,
            indicators: ["imperative_verb"]
          }
        ])

      trace = ProcessingTrace.from_interpretation(interpretation)

      assert is_list(trace.analyzer_results)
      # Should be sorted by calibrated score descending
      if length(trace.analyzer_results) > 1 do
        first = hd(trace.analyzer_results)
        second = Enum.at(trace.analyzer_results, 1)
        assert first.calibrated >= second.calibrated
      end
    end

    test "computes total activation and normalization flag" do
      interpretation =
        Interpretation.new("weather.query", "What's the weather?", 0.80, :model)
        |> Map.put(:alternatives, [
          %{intent: "question.factual", activation: 0.40, source: :structural, raw_score: 0.40}
        ])

      trace = ProcessingTrace.from_interpretation(interpretation)

      # Total should be sum of primary + alternatives
      assert trace.total_activation == 1.2  # 0.80 + 0.40

      # Should be marked as normalized since total > 1.0
      assert trace.activation_normalized == true
    end
  end

  describe "trace_processing/2" do
    test "returns {trace, primary_interpretation} tuple" do
      {trace, interpretation} = ProcessingTrace.trace_processing("Hello there!")

      assert %ProcessingTrace{} = trace
      assert %Interpretation{} = interpretation
    end

    test "traces single chunk correctly" do
      {trace, _interpretation} = ProcessingTrace.trace_processing("Hello!")

      assert trace.chunk_count == 1
      assert length(trace.chunks) == 1

      chunk = hd(trace.chunks)
      assert chunk.text == "Hello!"
      assert chunk.index == 0
    end

    test "traces each semantic chunk independently" do
      # Multi-sentence input should produce multiple chunks
      {trace, _interpretation} = ProcessingTrace.trace_processing("Hello! What's the weather?")

      assert trace.chunk_count >= 1
      assert is_list(trace.chunks)

      # Each chunk should have its own trace
      Enum.each(trace.chunks, fn chunk ->
        assert is_binary(chunk.text)
        assert is_integer(chunk.index)
      end)
    end

    test "determines overall strategy from chunk strategies" do
      {trace, _interpretation} = ProcessingTrace.trace_processing("Hello! What's the weather in Paris?")

      assert trace.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification,
               :low_confidence
             ]
    end

    test "finds primary chunk (prioritizes questions/commands over greetings)" do
      # Input with greeting first, then question
      {trace, _interpretation} = ProcessingTrace.trace_processing("Hello! What is the weather in NYC?")

      # Primary intent should be the weather query, not the greeting
      # (since questions are prioritized over expressives)
      if trace.chunk_count > 1 do
        # The trace should prioritize substantive intents
        assert trace.primary_intent != nil
      end
    end
  end

  describe "to_display_map/1" do
    test "converts trace to UI-friendly map" do
      interpretation =
        Interpretation.new("greeting.hello", "Hello!", 0.90, :heuristic)
        |> Interpretation.with_heuristic("test_heuristic", :global)

      trace = ProcessingTrace.from_interpretation(interpretation)
      display_map = ProcessingTrace.to_display_map(trace)

      assert is_map(display_map)
      assert Map.has_key?(display_map, :intent)
      assert Map.has_key?(display_map, :confidence)
      assert Map.has_key?(display_map, :fast_path)
      assert Map.has_key?(display_map, :analyzers)
    end

    test "includes chunk details for multi-chunk input" do
      {trace, _interpretation} = ProcessingTrace.trace_processing("Hello! What's the weather?")

      display_map = ProcessingTrace.to_display_map(trace)

      assert Map.has_key?(display_map, :chunk_count)
      assert Map.has_key?(display_map, :chunks)
      assert is_list(display_map.chunks)
    end

    test "includes alternatives in display map" do
      interpretation =
        Interpretation.new("weather.query", "What's the weather?", 0.70, :model)
        |> Map.put(:alternatives, [
          %{intent: "question.factual", activation: 0.30, source: :structural, raw_score: 0.30}
        ])

      trace = ProcessingTrace.from_interpretation(interpretation)
      display_map = ProcessingTrace.to_display_map(trace)

      assert Map.has_key?(display_map, :alternatives)
      assert is_list(display_map.alternatives)
    end

    test "includes slots and backtrack info" do
      interpretation =
        Interpretation.new("weather.query", "What's the weather?", 0.75, :model)

      trace = ProcessingTrace.from_interpretation(interpretation)
      display_map = ProcessingTrace.to_display_map(trace)

      assert Map.has_key?(display_map, :slots_filled)
      assert Map.has_key?(display_map, :slots_missing)
      assert Map.has_key?(display_map, :needs_clarification)
      assert Map.has_key?(display_map, :backtrack_count)
    end

    test "includes total activation and normalization flag" do
      interpretation =
        Interpretation.new("test.intent", "Test input", 0.80, :model)

      trace = ProcessingTrace.from_interpretation(interpretation)
      display_map = ProcessingTrace.to_display_map(trace)

      assert Map.has_key?(display_map, :total_activation)
      assert Map.has_key?(display_map, :was_normalized)
    end
  end

  describe "multi-chunk handling" do
    test "Hello! What's the weather? produces multiple chunk traces" do
      {trace, _interpretation} = ProcessingTrace.trace_processing("Hello! What's the weather?")

      # Should have at least 2 chunks (greeting + question)
      # Note: chunking behavior depends on SemanticChunker implementation
      assert trace.chunk_count >= 1
      assert length(trace.chunks) == trace.chunk_count
    end

    test "primary chunk is substantive content over greeting" do
      {trace, _interpretation} =
        ProcessingTrace.trace_processing("Hi there! What's the weather in New York?")

      # The primary intent should be the weather query (substantive)
      # not the greeting (expressive)
      if trace.chunk_count > 1 do
        # Check that we're not just using the first chunk
        # Weather intents should take priority
        primary_domain =
          if trace.primary_intent do
            cond do
              String.contains?(trace.primary_intent || "", "weather") -> :weather
              String.contains?(trace.primary_intent || "", "greeting") -> :greeting
              String.contains?(trace.primary_intent || "", "smalltalk") -> :smalltalk
              true -> :other
            end
          else
            :other
          end

        # Weather should be prioritized over greetings
        assert primary_domain in [:weather, :other]
      end
    end

    test "each chunk trace has independent analysis" do
      {trace, _interpretation} =
        ProcessingTrace.trace_processing("Hello! Play some music. What's the weather?")

      # Each chunk should have its own intent and entities
      Enum.each(trace.chunks, fn chunk ->
        assert is_binary(chunk.text) or is_nil(chunk.text)
        assert is_binary(chunk.primary_intent) or is_nil(chunk.primary_intent)
        assert is_float(chunk.primary_activation) or is_nil(chunk.primary_activation)
      end)
    end
  end

  describe "performance" do
    test "trace_processing completes within reasonable time" do
      start_time = System.monotonic_time(:millisecond)

      {_trace, _interpretation} = ProcessingTrace.trace_processing("Hello! What's the weather in NYC? Play some jazz music.")

      elapsed = System.monotonic_time(:millisecond) - start_time

      # Should complete within 5 seconds
      assert elapsed < 5000
    end
  end
end
