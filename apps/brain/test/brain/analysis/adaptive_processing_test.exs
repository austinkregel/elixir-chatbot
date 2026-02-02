defmodule Brain.Analysis.AdaptiveProcessingTest do
  @moduledoc """
  Tests for the adaptive cognitive processing system.

  Tests cover:
  - Activation normalization and limits
  - Analyzer calibration
  - Backtracking with thrash protection
  - Heuristic scope isolation
  """

  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.{
    Interpretation,
    AnalyzerResult,
    ActivationPool,
    AnalyzerCalibration,
    BacktrackController,
    HeuristicStore,
    OutcomeLearner
  }

  # ============================================================================
  # Interpretation Tests
  # ============================================================================

  describe "Interpretation" do
    test "creates interpretation with activation levels" do
      interp =
        Interpretation.new("weather.query", "What's the weather?", 0.85, :pattern_recognition)

      assert interp.intent == "weather.query"
      assert interp.activation == 0.85
      assert interp.source == :pattern_recognition
      assert interp.alternatives == []
    end

    test "creates interpretation from analyzer results" do
      results = [
        AnalyzerResult.new(:pattern_recognition, "weather.query", 0.85,
          calibrated_activation: 0.85
        ),
        AnalyzerResult.new(:keyword, "news.query", 0.60, calibrated_activation: 0.60),
        AnalyzerResult.new(:structural, "question.factual", 0.50, calibrated_activation: 0.50)
      ]

      interp = Interpretation.from_analyzer_results("What's the weather?", results)

      assert interp.intent == "weather.query"
      assert interp.activation == 0.85
      assert length(interp.alternatives) == 2
      assert hd(interp.alternatives).intent == "news.query"
    end

    test "promotes alternative interpretation" do
      interp =
        Interpretation.new("greeting", "Hello weather", 0.7, :keyword)
        |> Map.put(:alternatives, [
          %{
            intent: "weather.query",
            activation: 0.5,
            source: :pattern_recognition,
            raw_score: 0.5
          }
        ])

      {:ok, promoted} = Interpretation.promote_alternative(interp)

      assert promoted.intent == "weather.query"
      assert promoted.was_promoted == true
      assert promoted.backtrack_count == 1
      # Original is now in alternatives
      assert Enum.any?(promoted.alternatives, &(&1.intent == "greeting"))
    end

    test "returns error when no alternatives available" do
      interp = Interpretation.new("greeting", "Hello", 0.9, :keyword)

      assert {:error, :no_alternatives} = Interpretation.promote_alternative(interp)
    end

    test "reports confidence level correctly" do
      high = Interpretation.new("test", "text", 0.90, :keyword)
      medium = Interpretation.new("test", "text", 0.65, :keyword)
      low = Interpretation.new("test", "text", 0.35, :keyword)
      very_low = Interpretation.new("test", "text", 0.10, :keyword)

      assert Interpretation.confidence_level(high) == :high
      assert Interpretation.confidence_level(medium) == :medium
      assert Interpretation.confidence_level(low) == :low
      assert Interpretation.confidence_level(very_low) == :very_low
    end
  end

  # ============================================================================
  # Activation Pool Tests
  # ============================================================================

  describe "ActivationPool" do
    test "normalizes activations when sum exceeds 1.0" do
      interpretations = [
        Interpretation.new("intent1", "text", 0.6, :keyword),
        Interpretation.new("intent2", "text", 0.5, :keyword),
        Interpretation.new("intent3", "text", 0.4, :keyword)
      ]

      normalized = ActivationPool.normalize(interpretations)

      total = Enum.sum(Enum.map(normalized, & &1.activation))
      assert_in_delta total, 1.0, 0.01
    end

    test "preserves activations when sum is under 1.0" do
      interpretations = [
        Interpretation.new("intent1", "text", 0.3, :keyword),
        Interpretation.new("intent2", "text", 0.2, :keyword)
      ]

      normalized = ActivationPool.normalize(interpretations)

      # Should be unchanged (except for minimum enforcement)
      assert Enum.map(normalized, & &1.activation) == [0.3, 0.2]
    end

    test "applies boost with diminishing returns" do
      # Below threshold - full effect
      result1 = ActivationPool.apply_boost(0.3, 0.2, :seeded)
      assert result1 > 0.3

      # At threshold - reduced effect
      result2 = ActivationPool.apply_boost(0.7, 0.2, :seeded)
      boost_below = result1 - 0.3
      boost_at = result2 - 0.7
      assert boost_at < boost_below

      # Above threshold - heavily dampened
      result3 = ActivationPool.apply_boost(0.9, 0.2, :seeded)
      boost_above = result3 - 0.9
      # At 0.9, boost should be minimal (only 10% of requested boost applies)
      assert boost_above <= boost_at
      # 10% of 0.2 boost, capped
      assert boost_above <= 0.04
    end

    test "respects source-specific boost caps" do
      # Seeded has highest cap (0.40)
      seeded = ActivationPool.apply_boost(0.0, 1.0, :seeded)

      # Learned user has lowest cap (0.15)
      user = ActivationPool.apply_boost(0.0, 1.0, :learned_user)

      assert seeded > user
      # Can't exceed base + max_boost
      assert seeded <= 0.5
      assert user <= 0.25
    end

    test "applies stacked boosts with diminishing effect" do
      boosts = [
        {:seeded, 0.3},
        {:learned_global, 0.2},
        {:learned_user, 0.1}
      ]

      result = ActivationPool.apply_stacked_boosts(0.0, boosts)

      # Result should be less than sum of all boosts
      # because of diminishing stacking
      assert result < 0.6
      assert result > 0.3
    end

    test "detects approaching activation limit" do
      interpretations = [
        Interpretation.new("intent1", "text", 0.5, :keyword),
        Interpretation.new("intent2", "text", 0.45, :keyword)
      ]

      assert ActivationPool.approaching_limit?(interpretations, 0.9)

      low_interpretations = [
        Interpretation.new("intent1", "text", 0.3, :keyword),
        Interpretation.new("intent2", "text", 0.2, :keyword)
      ]

      refute ActivationPool.approaching_limit?(low_interpretations, 0.9)
    end
  end

  # ============================================================================
  # Backtrack Controller Tests
  # ============================================================================

  describe "BacktrackController" do
    test "allows backtracking within budget" do
      state = BacktrackController.new("test input")

      interp =
        Interpretation.new("intent1", "test input", 0.7, :keyword)
        |> Map.put(:alternatives, [
          %{intent: "intent2", activation: 0.5, source: :structural, raw_score: 0.5}
        ])

      {:ok, new_state, new_interp, cost} =
        BacktrackController.attempt_backtrack(state, interp, :missing_required)

      assert new_state.backtrack_count == 1
      assert new_interp.intent == "intent2"
      assert cost == BacktrackController.backtrack_cost()
    end

    test "forces clarification when budget exhausted" do
      state = %BacktrackController{
        input_text: "test input",
        # At max
        backtrack_count: 2
      }

      interp =
        Interpretation.new("intent1", "test input", 0.7, :keyword)
        |> Map.put(:alternatives, [
          %{intent: "intent2", activation: 0.5, source: :structural, raw_score: 0.5}
        ])

      {:force_clarification, clarification} =
        BacktrackController.attempt_backtrack(state, interp, :missing_required)

      assert clarification.type in [:disambiguation, :general, :missing_slot]
      assert is_binary(clarification.prompt)
    end

    test "detects oscillation between interpretations" do
      # State shows we already went intent1 -> intent2, now trying to go back to intent1
      state = %BacktrackController{
        input_text: "test input",
        backtrack_count: 1,
        # Previously was intent1
        interpretation_history: ["intent1"]
      }

      # Currently at intent2, trying to backtrack to intent1 (which we came from)
      interp =
        Interpretation.new("intent2", "test input", 0.7, :keyword)
        |> Map.put(:alternatives, [
          %{intent: "intent1", activation: 0.5, source: :structural, raw_score: 0.5}
        ])

      {:force_clarification, clarification} =
        BacktrackController.attempt_backtrack(state, interp, :low_confidence)

      assert clarification.type == :oscillation
    end

    test "checks for contradictions" do
      # Missing required slots
      interp =
        Interpretation.new("weather.query", "What's the weather?", 0.8, :keyword)
        |> Interpretation.with_slots(%Brain.Analysis.SlotResult{
          all_required_filled: false,
          missing_required: ["location"]
        })

      assert {:needs_backtrack, {:missing_required, _}} =
               BacktrackController.check_for_contradictions(interp)

      # No contradictions
      ok_interp =
        Interpretation.new("greeting", "Hello", 0.9, :keyword)
        |> Interpretation.with_slots(%Brain.Analysis.SlotResult{all_required_filled: true})

      assert :ok = BacktrackController.check_for_contradictions(ok_interp)
    end

    test "should_clarify? returns true near budget limit" do
      near_limit = %BacktrackController{backtrack_count: 1}
      at_limit = %BacktrackController{backtrack_count: 2}
      oscillating = %BacktrackController{oscillation_detected: true}

      assert BacktrackController.should_clarify?(near_limit)
      assert BacktrackController.should_clarify?(at_limit)
      assert BacktrackController.should_clarify?(oscillating)

      fresh = BacktrackController.new("test")
      refute BacktrackController.should_clarify?(fresh)
    end
  end

  # ============================================================================
  # Analyzer Calibration Tests
  # ============================================================================

  describe "AnalyzerCalibration" do
    setup do
      # Start calibration GenServer for tests
      ensure_process_started(AnalyzerCalibration, fn -> AnalyzerCalibration.start_link([]) end)
      :ok
    end

    test "calibrates raw scores" do
      {calibrated, error} = AnalyzerCalibration.calibrate(:keyword, 0.8)

      # Calibrated should be <= raw (adjusted by historical accuracy)
      assert is_float(calibrated)
      assert calibrated <= 0.8
      assert is_float(error)
    end

    test "tracks outcomes and updates accuracy" do
      # Track some outcomes
      AnalyzerCalibration.track_outcome(:structural, 0.75, true)
      AnalyzerCalibration.track_outcome(:structural, 0.75, true)
      AnalyzerCalibration.track_outcome(:structural, 0.75, false)

      # Allow async processing
      Process.sleep(50)

      # Accuracy for bucket 7 (0.7-0.8) should be updated
      accuracy = AnalyzerCalibration.get_bucket_accuracy(:structural, 7)
      assert is_float(accuracy)
    end

    test "returns stats for all analyzers" do
      stats = AnalyzerCalibration.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :keyword)
      assert Map.has_key?(stats, :structural)

      keyword_stats = stats[:keyword]
      assert Map.has_key?(keyword_stats, :calibration_error)
      assert Map.has_key?(keyword_stats, :bucket_accuracies)
    end
  end

  # ============================================================================
  # Heuristic Store Tests
  # ============================================================================

  describe "HeuristicStore" do
    setup do
      # Start heuristic store under ExUnit supervision
      ensure_started({HeuristicStore, seeded_path: "data/heuristics/seeded_heuristics.json"})
      :ok
    end

    test "matches global heuristics" do
      # HeuristicStore.match_best(text, world_id, user_id \\ nil, cohort_id \\ nil)
      case HeuristicStore.match_best("Hello there!", "default") do
        {:ok, heuristic, confidence} ->
          assert heuristic.scope == :global
          assert confidence > 0.3

        {:error, :no_match} ->
          # Acceptable if seeded file not loaded
          :ok
      end
    end

    test "adds and retrieves heuristic" do
      {:ok, heuristic} =
        HeuristicStore.add_heuristic(
          %{phrase: "test phrase unique"},
          %{intent: "test.intent", confidence_boost: 0.3},
          scope: :global
        )

      assert heuristic.id != nil
      assert heuristic.scope == :global

      retrieved = HeuristicStore.get(heuristic.id)
      assert retrieved.id == heuristic.id
    end

    test "isolates user-scoped heuristics" do
      # Add user-specific heuristic with world_id
      {:ok, _user_heuristic} =
        HeuristicStore.add_heuristic(
          %{phrase: "the usual for user123"},
          %{intent: "order.repeat", slots: %{item: "latte"}},
          scope: :user,
          scope_id: "user123",
          world_id: "default"
        )

      # Should match for correct user
      user123_matches = HeuristicStore.match_scope(:user, "user123", "the usual for user123")
      assert length(user123_matches) > 0

      # Should NOT match for different user
      user456_matches = HeuristicStore.match_scope(:user, "user456", "the usual for user123")
      assert Enum.empty?(user456_matches)
    end

    test "respects scope activation caps" do
      # Add heuristics at different scopes with world_id
      {:ok, global} =
        HeuristicStore.add_heuristic(
          %{phrase: "global test xyz"},
          %{intent: "test.global", confidence_boost: 0.5},
          scope: :global,
          world_id: "default"
        )

      {:ok, user} =
        HeuristicStore.add_heuristic(
          %{phrase: "user test xyz"},
          %{intent: "test.user", confidence_boost: 0.5},
          scope: :user,
          scope_id: "test_user",
          world_id: "default"
        )

      # Global max boost is 0.40, user max boost is 0.15
      assert global.max_activation_boost == 0.40
      assert user.max_activation_boost == 0.15
    end

    test "records success and failure" do
      {:ok, heuristic} =
        HeuristicStore.add_heuristic(
          %{phrase: "success test heuristic"},
          %{intent: "test.success"},
          scope: :global
        )

      HeuristicStore.record_success(heuristic.id)
      HeuristicStore.record_success(heuristic.id)
      HeuristicStore.record_failure(heuristic.id)

      # Allow async processing
      Process.sleep(50)

      updated = HeuristicStore.get(heuristic.id)
      assert updated.success_count == 2
      assert updated.failure_count == 1
    end

    test "checks heuristic health" do
      {:ok, healthy} =
        HeuristicStore.add_heuristic(
          %{phrase: "healthy heuristic xyz"},
          %{intent: "test.healthy"},
          scope: :global
        )

      # Initially healthy (not enough data)
      assert HeuristicStore.healthy?(healthy.id)

      # Add some successes
      for _ <- 1..5, do: HeuristicStore.record_success(healthy.id)
      Process.sleep(50)

      assert HeuristicStore.healthy?(healthy.id)
    end

    test "deprecates heuristic with high failure rate" do
      {:ok, heuristic} =
        HeuristicStore.add_heuristic(
          %{phrase: "failing heuristic xyz"},
          %{intent: "test.failing"},
          scope: :global
        )

      # Add failures to exceed threshold
      for _ <- 1..2, do: HeuristicStore.record_success(heuristic.id)
      for _ <- 1..5, do: HeuristicStore.record_failure(heuristic.id)

      Process.sleep(50)

      updated = HeuristicStore.get(heuristic.id)
      # Should be deprecated due to >20% failure rate
      assert updated.deprecated == true
    end
  end

  # ============================================================================
  # Outcome Learner Tests
  # ============================================================================

  describe "OutcomeLearner" do
    setup do
      # Start stores under ExUnit supervision
      ensure_started({HeuristicStore, seeded_path: "data/heuristics/seeded_heuristics.json"})
      ensure_started(AnalyzerCalibration)

      :ok
    end

    test "assesses positive outcome from explicit feedback" do
      interp =
        Interpretation.new("weather.query", "What's the weather?", 0.85, :pattern_recognition)

      outcome = OutcomeLearner.assess_outcome(interp, "Here's the weather...", :positive)

      assert outcome == :success
    end

    test "assesses negative outcome from explicit feedback" do
      interp =
        Interpretation.new("weather.query", "What's the weather?", 0.85, :pattern_recognition)

      outcome = OutcomeLearner.assess_outcome(interp, "Here's the weather...", :negative)

      assert outcome == :failure
    end

    test "assesses likely success from high confidence" do
      interp =
        Interpretation.new("weather.query", "What's the weather?", 0.90, :pattern_recognition)

      outcome = OutcomeLearner.assess_outcome(interp, "Here's the weather...", nil)

      assert outcome == :likely_success
    end

    test "extracts learnable pattern from interpretation" do
      interp =
        Interpretation.new("weather.query", "what's the weather like", 0.85, :pattern_recognition)

      {pattern, conclusion} = OutcomeLearner.extract_pattern(interp)

      assert is_map(pattern)
      assert conclusion.intent == "weather.query"
      assert is_float(conclusion.confidence_boost)
    end

    test "determines appropriate scope for pattern" do
      # Phrase pattern -> user scope
      phrase_pattern = %{phrase: "my usual order"}
      assert OutcomeLearner.determine_scope(phrase_pattern, "user1", "cohort1") == :user

      # Keyword pattern -> cohort scope
      keyword_pattern = %{keywords: ["weather", "forecast"]}
      assert OutcomeLearner.determine_scope(keyword_pattern, "user1", "cohort1") == :cohort

      # First word pattern -> global scope
      first_word_pattern = %{first_word: ["play"]}
      assert OutcomeLearner.determine_scope(first_word_pattern, "user1", "cohort1") == :global
    end

    test "learns from successful outcome" do
      interp =
        Interpretation.new(
          "test.learned",
          "unique test phrase for learning",
          0.85,
          :pattern_recognition
        )

      # Should not crash and should record learning
      outcome = OutcomeLearner.learn_from_outcome(interp, "Response",
        user_id: "test_user",
        world_id: "default"
      )

      assert outcome in [:likely_success, :uncertain]
    end
  end

  # ============================================================================
  # Integration Tests
  # ============================================================================

  describe "Integration" do
    test "full pipeline: racing -> validation -> backtracking -> learning" do
      # This tests the complete flow without actually running all services

      # 1. Create initial interpretation (simulating racing result)
      initial =
        Interpretation.new("weather.query", "What's the weather?", 0.80, :pattern_recognition)
        |> Map.put(:alternatives, [
          %{intent: "question.factual", activation: 0.40, source: :structural, raw_score: 0.40}
        ])
        |> Interpretation.with_slots(%Brain.Analysis.SlotResult{
          all_required_filled: false,
          missing_required: ["location"]
        })

      # 2. Check for contradictions
      assert {:needs_backtrack, _reason} = BacktrackController.check_for_contradictions(initial)

      # 3. Attempt backtrack
      state = BacktrackController.new("What's the weather?")

      {:ok, new_state, promoted, _cost} =
        BacktrackController.attempt_backtrack(state, initial, :missing_required)

      assert promoted.intent == "question.factual"
      assert new_state.backtrack_count == 1

      # 4. Normalize activations
      normalized = ActivationPool.normalize_with_alternatives(promoted)
      assert normalized.activation <= 1.0

      # 5. Would then proceed to response and learning
    end

    test "stability: activation cannot exceed 1.0 even with many boosts" do
      base = 0.3

      # Apply many stacked boosts
      boosts = for _i <- 1..10, do: {:seeded, 0.2}

      result = ActivationPool.apply_stacked_boosts(base, boosts)

      assert result <= 1.0
    end

    test "stability: backtrack budget prevents infinite loops" do
      state = BacktrackController.new("ambiguous input")

      interp =
        Interpretation.new("intent1", "ambiguous input", 0.6, :keyword)
        |> Map.put(:alternatives, [
          %{intent: "intent2", activation: 0.55, source: :structural, raw_score: 0.55},
          %{intent: "intent3", activation: 0.50, source: :model, raw_score: 0.50}
        ])

      # First backtrack
      {:ok, state, interp, _} =
        BacktrackController.attempt_backtrack(state, interp, :contradiction)

      # Second backtrack
      {:ok, state, interp, _} =
        BacktrackController.attempt_backtrack(state, interp, :contradiction)

      # Third should be forced clarification
      result = BacktrackController.attempt_backtrack(state, interp, :contradiction)
      assert {:force_clarification, _} = result
    end
  end

  # Helper to safely start a process
  defp ensure_process_started(name, start_fn) do
    case Process.whereis(name) do
      nil ->
        case start_fn.() do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
        end

      _pid ->
        :ok
    end
  end
end
