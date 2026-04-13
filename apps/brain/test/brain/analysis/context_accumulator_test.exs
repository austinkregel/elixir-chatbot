defmodule Brain.Analysis.ContextAccumulatorTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.ContextAccumulator

  describe "add_signal/4" do
    test "adds signals to the accumulator" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.9)
        |> ContextAccumulator.add_signal(:speech_act, :directive, 0.8)

      assert length(acc.signals) == 2
    end

    test "clamps confidence to valid range" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:test, :val, 1.5)
        |> ContextAccumulator.add_signal(:test2, :val, -0.5)

      [{_, _, c2}, {_, _, c1}] = acc.signals
      assert c1 <= 1.0
      assert c2 >= 0.0
    end
  end

  describe "log-odds combination" do
    test "two agreeing high-confidence signals produce higher combined confidence" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :yes, 0.9)
        |> ContextAccumulator.add_signal(:b, :yes, 0.85)
        |> ContextAccumulator.accumulate()

      assert acc.combined_confidence > 0.9
      assert acc.combined_confidence > 0.85
    end

    test "two agreeing low-confidence signals produce lower combined confidence" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :no, 0.1)
        |> ContextAccumulator.add_signal(:b, :no, 0.15)
        |> ContextAccumulator.accumulate()

      assert acc.combined_confidence < 0.15
      assert acc.combined_confidence < 0.1
    end

    test "combination is order-independent (commutative)" do
      acc1 =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :x, 0.9)
        |> ContextAccumulator.add_signal(:b, :y, 0.3)
        |> ContextAccumulator.accumulate()

      acc2 =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:b, :y, 0.3)
        |> ContextAccumulator.add_signal(:a, :x, 0.9)
        |> ContextAccumulator.accumulate()

      assert_in_delta acc1.combined_confidence, acc2.combined_confidence, 1.0e-9
    end
  end

  describe "conflict detection" do
    test "two disagreeing high-confidence signals produce high conflict K" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :yes, 0.95)
        |> ContextAccumulator.add_signal(:b, :no, 0.05)
        |> ContextAccumulator.accumulate()

      assert acc.conflict_measure > 0.3
    end

    test "two agreeing signals produce low conflict K" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :yes, 0.9)
        |> ContextAccumulator.add_signal(:b, :yes, 0.85)
        |> ContextAccumulator.accumulate()

      assert acc.conflict_measure < 0.3
    end

    test "single signal has zero conflict" do
      assert ContextAccumulator.compute_conflict([{:a, :yes, 0.9}]) == 0.0
    end

    test "high conflict reduces effective confidence" do
      agreeing =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :yes, 0.9)
        |> ContextAccumulator.add_signal(:b, :yes, 0.85)
        |> ContextAccumulator.accumulate()

      disagreeing =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :yes, 0.9)
        |> ContextAccumulator.add_signal(:b, :no, 0.1)
        |> ContextAccumulator.accumulate()

      eff_agree = ContextAccumulator.effective_confidence(agreeing)
      eff_disagree = ContextAccumulator.effective_confidence(disagreeing)

      assert eff_agree > eff_disagree
    end
  end

  describe "entropy gating" do
    test "uninformative signal has near-zero effect on combination" do
      single =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :yes, 0.9)
        |> ContextAccumulator.accumulate()

      with_noise =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:a, :yes, 0.9)
        |> ContextAccumulator.add_signal(:noise, :maybe, 0.5)
        |> ContextAccumulator.accumulate()

      assert_in_delta single.combined_confidence,
                      with_noise.combined_confidence,
                      0.05
    end

    test "high-certainty signals get high relevance weight" do
      weights =
        ContextAccumulator.compute_relevance_weights([
          {:certain, :yes, 0.99},
          {:uncertain, :maybe, 0.5}
        ])

      {_, _, _, certain_w} = Enum.find(weights, fn {s, _, _, _} -> s == :certain end)
      {_, _, _, uncertain_w} = Enum.find(weights, fn {s, _, _, _} -> s == :uncertain end)

      assert certain_w > uncertain_w
    end
  end

  describe "should_hedge?/1" do
    test "returns true when conflict is high" do
      acc = %ContextAccumulator{conflict_measure: 0.5, entity_familiarity: 0.8, combined_confidence: 0.8}
      assert ContextAccumulator.should_hedge?(acc)
    end

    test "returns true when entity familiarity is low" do
      acc = %ContextAccumulator{conflict_measure: 0.0, entity_familiarity: 0.1, combined_confidence: 0.8}
      assert ContextAccumulator.should_hedge?(acc)
    end

    test "returns true when effective confidence is low" do
      acc = %ContextAccumulator{conflict_measure: 0.0, entity_familiarity: 0.8, combined_confidence: 0.3}
      assert ContextAccumulator.should_hedge?(acc)
    end

    test "returns false when all indicators are strong" do
      acc = %ContextAccumulator{conflict_measure: 0.1, entity_familiarity: 0.8, combined_confidence: 0.85}
      refute ContextAccumulator.should_hedge?(acc)
    end
  end

  describe "dominant_strategy/1" do
    test "returns the source of the dominant signal" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:speech_act, :directive, 0.95)
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.6)
        |> ContextAccumulator.accumulate()

      assert ContextAccumulator.dominant_strategy(acc) == :speech_act
    end

    test "returns :unknown when no signals" do
      assert ContextAccumulator.dominant_strategy(%ContextAccumulator{}) == :unknown
    end
  end

  describe "interlocutor_adaptation/2" do
    test "returns adaptation when present" do
      acc = %ContextAccumulator{interlocutor_adaptations: %{fahrenheit: "true"}}
      assert ContextAccumulator.interlocutor_adaptation(acc, :fahrenheit) == "true"
    end

    test "returns nil when absent" do
      acc = %ContextAccumulator{interlocutor_adaptations: %{}}
      assert ContextAccumulator.interlocutor_adaptation(acc, :celsius) == nil
    end
  end

  describe "memory_context/1" do
    test "returns structured memory context" do
      acc = %ContextAccumulator{
        relevant_episodes: [{:episode, 0.9}],
        relevant_semantics: [{:fact, 0.8}],
        conversation_topics: ["weather"]
      }

      ctx = ContextAccumulator.memory_context(acc)
      assert ctx.episodes == [{:episode, 0.9}]
      assert ctx.semantics == [{:fact, 0.8}]
      assert ctx.conversation_topics == ["weather"]
    end
  end

  describe "intent signal integration" do
    test "low intent confidence drags down effective confidence" do
      without_intent =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.9)
        |> ContextAccumulator.add_signal(:speech_act, :directive, 0.85)
        |> ContextAccumulator.add_signal(:sentiment, :positive, 0.8)
        |> ContextAccumulator.accumulate()

      with_low_intent =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.9)
        |> ContextAccumulator.add_signal(:speech_act, :directive, 0.85)
        |> ContextAccumulator.add_signal(:sentiment, :positive, 0.8)
        |> ContextAccumulator.add_signal(:intent, "account.earning.check", 0.278)
        |> ContextAccumulator.accumulate()

      eff_without = ContextAccumulator.effective_confidence(without_intent)
      eff_with = ContextAccumulator.effective_confidence(with_low_intent)

      assert eff_with < eff_without
    end

    test "high intent confidence preserves effective confidence" do
      with_high_intent =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.9)
        |> ContextAccumulator.add_signal(:speech_act, :directive, 0.85)
        |> ContextAccumulator.add_signal(:intent, "social.greeting", 0.95)
        |> ContextAccumulator.accumulate()

      eff = ContextAccumulator.effective_confidence(with_high_intent)
      assert eff > 0.8
    end

    test "low intent confidence triggers should_hedge?" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.5)
        |> ContextAccumulator.add_signal(:intent, "unknown", 0.2)
        |> ContextAccumulator.accumulate()

      assert ContextAccumulator.should_hedge?(acc)
    end
  end

  describe "accumulate/1 with empty signals" do
    test "returns unchanged accumulator" do
      acc = ContextAccumulator.accumulate(%ContextAccumulator{})
      assert acc.combined_confidence == 0.5
      assert acc.conflict_measure == 0.0
      assert acc.dominant_signal == nil
    end
  end

  describe "entity familiarity extraction" do
    test "extracts entity_familiarity signal when present" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:entity_familiarity, true, 0.9)
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.8)
        |> ContextAccumulator.accumulate()

      assert acc.entity_familiarity == 0.9
    end

    test "defaults to 0.5 when entity_familiarity signal absent" do
      acc =
        %ContextAccumulator{}
        |> ContextAccumulator.add_signal(:discourse, :bot, 0.8)
        |> ContextAccumulator.accumulate()

      assert acc.entity_familiarity == 0.5
    end
  end
end
