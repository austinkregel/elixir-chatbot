defmodule Brain.Epistemic.SourceAuthorityTest do
  @moduledoc """
  Unit tests for the SourceAuthority GenServer.

  Verifies profile loading, effective confidence/decay calculations,
  credibility tracking, and credibility floor enforcement.
  """

  use Brain.Test.GraphCase, async: false

  alias Brain.Epistemic.SourceAuthority
  import Brain.TestHelpers

  setup _context do
    ensure_started(SourceAuthority)
    SourceAuthority.clear()
    :ok
  end

  describe "profile loading" do
    test "all expected profiles are loaded" do
      profiles = SourceAuthority.list_profiles()
      keys = Enum.map(profiles, & &1.key)

      assert :mentor in keys
      assert :academic_expert in keys
      assert :industry_expert in keys
      assert :postgraduate in keys
      assert :student in keys
      assert :hobbyist in keys
      assert :friend in keys
      assert :acquaintance in keys
      assert :stranger in keys
      assert :parody in keys
    end

    test "get_profile returns correct fields for mentor" do
      profile = SourceAuthority.get_profile(:mentor)

      assert profile != nil
      assert profile.label == "Mentor"
      assert profile.initial_confidence == 0.85
      assert profile.decay_rate_multiplier == 0.5
      assert profile.jtms_node_type == :assumption
      assert profile.credibility_floor == 0.3
      assert profile.category == "personal"
    end

    test "get_profile returns nil for unknown authority" do
      assert SourceAuthority.get_profile(:nonexistent) == nil
    end
  end

  describe "effective_confidence" do
    test "returns initial_confidence when credibility is fresh (1.0)" do
      conf = SourceAuthority.effective_confidence(:mentor)
      assert_in_delta conf, 0.85, 0.02
    end

    test "academic_expert has higher confidence than mentor" do
      academic = SourceAuthority.effective_confidence(:academic_expert)
      mentor = SourceAuthority.effective_confidence(:mentor)

      assert academic >= mentor
    end

    test "stranger has low confidence" do
      conf = SourceAuthority.effective_confidence(:stranger)
      assert_in_delta conf, 0.40, 0.02
    end

    test "parody has near-zero confidence" do
      conf = SourceAuthority.effective_confidence(:parody)
      assert conf < 0.1
    end

    test "returns 0.5 for unknown authority" do
      conf = SourceAuthority.effective_confidence(:nonexistent)
      assert_in_delta conf, 0.5, 0.01
    end
  end

  describe "effective_decay_rate" do
    test "mentor decays at half base rate" do
      rate = SourceAuthority.effective_decay_rate(:mentor, 0.05)
      assert_in_delta rate, 0.025, 0.001
    end

    test "stranger decays at full base rate" do
      rate = SourceAuthority.effective_decay_rate(:stranger, 0.05)
      assert_in_delta rate, 0.05, 0.001
    end

    test "parody decays at double base rate" do
      rate = SourceAuthority.effective_decay_rate(:parody, 0.05)
      assert_in_delta rate, 0.10, 0.001
    end

    test "returns base rate for unknown authority" do
      rate = SourceAuthority.effective_decay_rate(:nonexistent, 0.05)
      assert_in_delta rate, 0.05, 0.001
    end
  end

  describe "credibility tracking" do
    test "credibility starts at 1.0 with no outcomes" do
      cred = SourceAuthority.get_credibility(:mentor)
      assert_in_delta cred, 1.0, 0.01
    end

    test "credibility after 8 confirmed + 2 contradicted is 0.8" do
      for _ <- 1..8, do: SourceAuthority.record_outcome(:mentor, :confirmed)
      for _ <- 1..2, do: SourceAuthority.record_outcome(:mentor, :contradicted)

      # Need a small delay for async cast processing
      Process.sleep(50)

      cred = SourceAuthority.get_credibility(:mentor)
      assert_in_delta cred, 0.8, 0.02
    end

    test "effective confidence reflects credibility" do
      for _ <- 1..8, do: SourceAuthority.record_outcome(:mentor, :confirmed)
      for _ <- 1..2, do: SourceAuthority.record_outcome(:mentor, :contradicted)
      Process.sleep(50)

      conf = SourceAuthority.effective_confidence(:mentor)
      # 0.85 * 0.8 = 0.68
      assert_in_delta conf, 0.68, 0.02
    end

    test "added outcomes are tracked separately" do
      SourceAuthority.record_outcome(:mentor, :added)
      SourceAuthority.record_outcome(:mentor, :added)
      Process.sleep(50)

      profiles = SourceAuthority.list_profiles()
      mentor = Enum.find(profiles, & &1.key == :mentor)

      assert mentor.total_added == 2
      # Added doesn't affect credibility
      assert_in_delta mentor.credibility, 1.0, 0.01
    end
  end

  describe "credibility floor" do
    test "mentor credibility never drops below 0.3" do
      # All contradictions
      for _ <- 1..20, do: SourceAuthority.record_outcome(:mentor, :contradicted)
      Process.sleep(50)

      cred = SourceAuthority.get_credibility(:mentor)
      assert cred >= 0.3
      assert_in_delta cred, 0.3, 0.01
    end

    test "stranger credibility never drops below 0.1" do
      for _ <- 1..20, do: SourceAuthority.record_outcome(:stranger, :contradicted)
      Process.sleep(50)

      cred = SourceAuthority.get_credibility(:stranger)
      assert cred >= 0.1
      assert_in_delta cred, 0.1, 0.01
    end

    test "parody credibility can reach 0.0" do
      for _ <- 1..10, do: SourceAuthority.record_outcome(:parody, :contradicted)
      Process.sleep(50)

      cred = SourceAuthority.get_credibility(:parody)
      assert_in_delta cred, 0.0, 0.01
    end
  end

  describe "ready?" do
    test "returns true when started" do
      assert SourceAuthority.ready?() == true
    end
  end
end
