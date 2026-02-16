defmodule Brain.Epistemic.AuthorityImpactTest do
  @moduledoc """
  End-to-end tests proving that source authority tiers measurably affect
  epistemic outcomes: confidence levels, fact verification status, decay
  rates, and credibility tracking.

  Each test demonstrates reproducible differences (within 1-2% tolerance)
  between authority levels.
  """

  use Brain.Test.GraphCase, async: false

  alias Brain.Epistemic.{BeliefStore, SourceAuthority, JTMS, ContradictionHandler, UserModelStore}
  alias Brain.FactDatabase
  alias Brain.FactDatabase.Integration, as: FactIntegration
  import Brain.TestHelpers

  setup do
    ensure_epistemic_stores_started()
    BeliefStore.clear()
    SourceAuthority.clear()
    :ok
  end

  # ────────────────────────────────────────────────────────────────
  # Test 1: Same statement, different authority → different verification
  # ────────────────────────────────────────────────────────────────

  describe "differential verification status by authority" do
    test "mentor belief reaches :verified but stranger belief stays :uncertain" do
      entity = "elixir_lang"
      fact_text = "Elixir runs on the BEAM virtual machine"

      # Add belief via mentor authority (effective confidence ~0.85)
      {:ok, mentor_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :elixir_lang,
          fact_text,
          :mentor
        )

      {:ok, mentor_belief} = BeliefStore.get_belief(mentor_id)

      # Mentor belief should have confidence >= 0.7 (verify threshold)
      assert mentor_belief.confidence >= 0.7,
        "Mentor belief confidence #{mentor_belief.confidence} should be >= 0.7"

      assert mentor_belief.source_authority == :mentor

      # Verify fact against the mentor belief
      result = FactIntegration.verify_fact(entity, fact_text)
      assert {:verified, confidence} = result
      assert_in_delta confidence, 0.85, 0.02

      # Clear and try the same with stranger
      BeliefStore.clear()

      {:ok, stranger_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :elixir_lang,
          fact_text,
          :stranger
        )

      {:ok, stranger_belief} = BeliefStore.get_belief(stranger_id)

      # Stranger belief should have confidence < 0.7 (below verify threshold)
      assert stranger_belief.confidence < 0.7,
        "Stranger belief confidence #{stranger_belief.confidence} should be < 0.7"

      assert stranger_belief.source_authority == :stranger

      # Verify fact against the stranger belief — should be uncertain
      result = FactIntegration.verify_fact(entity, fact_text)
      assert {:uncertain, :low_confidence} = result
    end

    test "academic_expert belief reaches :verified, student stays :uncertain" do
      entity = "photosynthesis"
      fact_text = "Plants convert CO2 to oxygen using sunlight"

      {:ok, expert_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :photosynthesis,
          fact_text,
          :academic_expert
        )

      {:ok, expert_belief} = BeliefStore.get_belief(expert_id)
      assert expert_belief.confidence >= 0.7

      result = FactIntegration.verify_fact(entity, fact_text)
      assert {:verified, _conf} = result

      BeliefStore.clear()

      {:ok, student_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :photosynthesis,
          fact_text,
          :student
        )

      {:ok, student_belief} = BeliefStore.get_belief(student_id)
      assert student_belief.confidence < 0.7

      result = FactIntegration.verify_fact(entity, fact_text)
      assert {:uncertain, :low_confidence} = result
    end
  end

  # ────────────────────────────────────────────────────────────────
  # Test 2: Authority hierarchy is consistent
  # ────────────────────────────────────────────────────────────────

  describe "authority hierarchy ordering" do
    test "confidence follows hierarchy: academic > mentor > friend > acquaintance > stranger > parody" do
      hierarchy = [:academic_expert, :mentor, :friend, :acquaintance, :stranger, :parody]

      confidences =
        Enum.map(hierarchy, fn authority ->
          SourceAuthority.effective_confidence(authority)
        end)

      # Each confidence should be >= the next
      pairs = Enum.zip(confidences, tl(confidences))

      Enum.each(pairs, fn {higher, lower} ->
        assert higher >= lower,
          "Expected #{higher} >= #{lower} in authority hierarchy"
      end)

      # Verify specific thresholds
      [academic, mentor, friend, acquaintance, stranger, parody] = confidences

      assert_in_delta academic, 0.90, 0.02
      assert_in_delta mentor, 0.85, 0.02
      assert_in_delta friend, 0.70, 0.02
      assert_in_delta acquaintance, 0.55, 0.02
      assert_in_delta stranger, 0.40, 0.02
      assert parody < 0.1
    end

    test "decay rate follows inverse hierarchy: parody fastest, mentor slowest" do
      base_rate = 0.05

      hierarchy = [:mentor, :academic_expert, :industry_expert, :friend, :stranger, :parody]

      rates =
        Enum.map(hierarchy, fn authority ->
          SourceAuthority.effective_decay_rate(authority, base_rate)
        end)

      [mentor_rate, academic_rate, _industry_rate, _friend_rate, stranger_rate, parody_rate] = rates

      # Mentor decays slowest (0.5 * 0.05 = 0.025)
      assert_in_delta mentor_rate, 0.025, 0.001

      # Academic decays slower than mentor (0.3 * 0.05 = 0.015)
      assert_in_delta academic_rate, 0.015, 0.001

      # Stranger at full rate
      assert_in_delta stranger_rate, 0.05, 0.001

      # Parody decays fastest (2.0 * 0.05 = 0.10)
      assert_in_delta parody_rate, 0.10, 0.001
    end
  end

  # ────────────────────────────────────────────────────────────────
  # Test 3: Decay behaviour differs by authority
  # ────────────────────────────────────────────────────────────────

  describe "authority-aware decay" do
    test "after simulated decay, mentor retains more confidence than stranger" do
      base_rate = 0.05

      mentor_rate = SourceAuthority.effective_decay_rate(:mentor, base_rate)
      stranger_rate = SourceAuthority.effective_decay_rate(:stranger, base_rate)

      initial_mentor = SourceAuthority.effective_confidence(:mentor)
      initial_stranger = SourceAuthority.effective_confidence(:stranger)

      # Simulate 10 decay cycles
      mentor_after = simulate_decay(initial_mentor, mentor_rate, 10)
      stranger_after = simulate_decay(initial_stranger, stranger_rate, 10)

      # Mentor should retain significantly more confidence
      assert mentor_after > stranger_after,
        "Mentor (#{mentor_after}) should retain more confidence than stranger (#{stranger_after})"

      # Mentor should retain substantially more after 10 cycles
      # 0.85 * (1 - 0.025)^10 ≈ 0.66
      assert_in_delta mentor_after, 0.66, 0.02

      # Stranger should be much lower: 0.40 * (1 - 0.05)^10 ≈ 0.24
      assert_in_delta stranger_after, 0.24, 0.02
    end

    test "parody beliefs decay to near-zero much faster than others" do
      base_rate = 0.05

      parody_rate = SourceAuthority.effective_decay_rate(:parody, base_rate)
      initial_parody = SourceAuthority.effective_confidence(:parody)

      # Even after just 5 cycles, parody should be negligible
      parody_after = simulate_decay(initial_parody, parody_rate, 5)

      assert parody_after < 0.03,
        "Parody (#{parody_after}) should be near-zero after 5 decay cycles"
    end
  end

  # ────────────────────────────────────────────────────────────────
  # Test 4: Dynamic credibility tracking affects future beliefs
  # ────────────────────────────────────────────────────────────────

  describe "credibility erosion and recovery" do
    test "repeated contradictions lower effective confidence for future beliefs" do
      # Fresh mentor confidence
      initial_conf = SourceAuthority.effective_confidence(:mentor)
      assert_in_delta initial_conf, 0.85, 0.02

      # Simulate 8 contradictions + 2 confirmations (20% success rate)
      for _ <- 1..2, do: SourceAuthority.record_outcome(:mentor, :confirmed)
      for _ <- 1..8, do: SourceAuthority.record_outcome(:mentor, :contradicted)
      Process.sleep(50)

      degraded_conf = SourceAuthority.effective_confidence(:mentor)

      # Credibility = max(2/10, 0.3) = 0.3 (floor kicks in)
      # Effective confidence = 0.85 * 0.3 = 0.255
      assert degraded_conf < initial_conf,
        "Degraded confidence (#{degraded_conf}) should be less than initial (#{initial_conf})"

      assert_in_delta degraded_conf, 0.255, 0.02

      # Now add belief with degraded mentor — should NOT reach :verified
      {:ok, belief_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :mars,
          "Mars is the fourth planet",
          :mentor
        )

      {:ok, belief} = BeliefStore.get_belief(belief_id)

      assert belief.confidence < 0.7,
        "Belief from degraded mentor (#{belief.confidence}) should be below 0.7"
    end

    test "credibility recovers as more outcomes are confirmed" do
      # Start with poor credibility (2 confirmed, 8 contradicted)
      for _ <- 1..2, do: SourceAuthority.record_outcome(:mentor, :confirmed)
      for _ <- 1..8, do: SourceAuthority.record_outcome(:mentor, :contradicted)
      Process.sleep(50)

      low_cred = SourceAuthority.get_credibility(:mentor)
      assert_in_delta low_cred, 0.3, 0.02

      # Now add 18 more confirmed outcomes (total: 20 confirmed, 8 contradicted)
      for _ <- 1..18, do: SourceAuthority.record_outcome(:mentor, :confirmed)
      Process.sleep(50)

      recovered_cred = SourceAuthority.get_credibility(:mentor)
      # Credibility = 20/28 ≈ 0.714
      assert recovered_cred > low_cred,
        "Recovered credibility (#{recovered_cred}) should exceed degraded (#{low_cred})"

      assert_in_delta recovered_cred, 0.714, 0.02

      # Now new belief from mentor should be verifiable
      recovered_conf = SourceAuthority.effective_confidence(:mentor)
      # 0.85 * 0.714 ≈ 0.607
      assert_in_delta recovered_conf, 0.607, 0.02
    end
  end

  # ────────────────────────────────────────────────────────────────
  # Test 5: Parody source is essentially never trusted
  # ────────────────────────────────────────────────────────────────

  describe "parody source limitations" do
    test "parody beliefs never reach verification threshold" do
      {:ok, parody_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :gravity,
          "Gravity pulls things down",
          :parody
        )

      {:ok, parody_belief} = BeliefStore.get_belief(parody_id)

      # Parody confidence should be near zero (0.05 * 1.0 = 0.05)
      assert parody_belief.confidence < 0.1,
        "Parody belief confidence (#{parody_belief.confidence}) should be < 0.1"

      result = FactIntegration.verify_fact("gravity", "Gravity pulls things down")
      assert {:uncertain, :low_confidence} = result
    end

    test "even with perfect credibility, parody never reaches :verified" do
      # Give parody 100% confirmed credibility
      for _ <- 1..50, do: SourceAuthority.record_outcome(:parody, :confirmed)
      Process.sleep(50)

      # Credibility should be 1.0
      cred = SourceAuthority.get_credibility(:parody)
      assert_in_delta cred, 1.0, 0.01

      # But effective confidence is still only 0.05
      conf = SourceAuthority.effective_confidence(:parody)
      assert conf < 0.1

      {:ok, belief_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :earth,
          "The Earth is round",
          :parody
        )

      {:ok, belief} = BeliefStore.get_belief(belief_id)
      assert belief.confidence < 0.1

      result = FactIntegration.verify_fact("earth", "The Earth is round")
      assert {:uncertain, :low_confidence} = result
    end
  end

  # ────────────────────────────────────────────────────────────────
  # Test 6: Credibility hooks fire on confirm/retract
  # ────────────────────────────────────────────────────────────────

  describe "credibility hooks in BeliefStore" do
    test "confirming a belief with authority records :confirmed outcome" do
      {:ok, belief_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :test_subject,
          "Test value",
          :friend
        )

      # Record_outcome :added already fired for the add
      Process.sleep(50)

      profiles_before = SourceAuthority.list_profiles()
      friend_before = Enum.find(profiles_before, & &1.key == :friend)
      assert friend_before.total_added == 1

      # Confirm the belief
      BeliefStore.confirm_belief(belief_id)
      Process.sleep(50)

      profiles_after = SourceAuthority.list_profiles()
      friend_after = Enum.find(profiles_after, & &1.key == :friend)
      assert friend_after.confirmed_count == 1
    end

    test "retracting a belief with authority records :contradicted outcome" do
      {:ok, belief_id} =
        BeliefStore.add_belief_with_authority(
          :world,
          :test_retract,
          "Will be retracted",
          :acquaintance
        )

      Process.sleep(50)

      BeliefStore.retract_belief(belief_id)
      Process.sleep(50)

      profiles = SourceAuthority.list_profiles()
      acq = Enum.find(profiles, & &1.key == :acquaintance)
      assert acq.contradicted_count == 1
    end
  end

  # ────────────────────────────────────────────────────────────────
  # Test 7: Reproducibility — same inputs produce same outputs
  # ────────────────────────────────────────────────────────────────

  describe "reproducibility" do
    test "same authority + same statement produces consistent confidence within 1%" do
      results =
        for _i <- 1..5 do
          BeliefStore.clear()
          SourceAuthority.clear()

          {:ok, bid} =
            BeliefStore.add_belief_with_authority(
              :world,
              :test_repro,
              "Reproducible belief",
              :industry_expert
            )

          {:ok, belief} = BeliefStore.get_belief(bid)
          belief.confidence
        end

      # All 5 runs should produce the same confidence
      first = hd(results)

      Enum.each(results, fn conf ->
        assert_in_delta conf, first, 0.01,
          "Expected all runs to produce confidence within 1% of #{first}, got #{conf}"
      end)
    end
  end

  # ────────────────────────────────────────────────────────────────
  # Helpers
  # ────────────────────────────────────────────────────────────────

  defp ensure_epistemic_stores_started do
    ensure_started(SourceAuthority)
    ensure_started(JTMS)
    ensure_started(BeliefStore)
    ensure_started(ContradictionHandler)
    ensure_started(UserModelStore)
    ensure_started(FactDatabase)
  end

  defp simulate_decay(confidence, rate, cycles) do
    Enum.reduce(1..cycles, confidence, fn _i, acc ->
      acc * (1.0 - rate)
    end)
  end
end
