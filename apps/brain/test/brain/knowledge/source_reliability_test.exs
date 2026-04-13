defmodule Brain.Knowledge.SourceReliabilityTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Knowledge.SourceReliability

  setup_all do
    Brain.TestHelpers.require_services!([Brain.Knowledge.SourceReliability])
    :ok
  end

  setup _context do
    ensure_pubsub_started()
    ensure_started(SourceReliability)

    :ok
  end

  describe "lookup/1" do
    test "looks up known reliable source" do
      {:ok, profile} = SourceReliability.lookup("wikipedia.org")

      assert profile.domain == "wikipedia.org"
      assert profile.trust_tier == :verified
      # Reliability score is calculated from base accuracy (0.85) * 0.7 + feedback * 0.3
      # Without feedback, this gives 0.85 * 0.7 + 0.5 * 0.3 = 0.745
      assert profile.reliability_score >= 0.7
    end

    test "returns neutral for unknown domain" do
      {:ok, profile} = SourceReliability.lookup("unknown-site-xyz-123.com")

      assert profile.trust_tier == :neutral
      assert profile.reliability_score == 0.5
    end

    test "extracts domain from full URL" do
      {:ok, profile} = SourceReliability.lookup("https://en.wikipedia.org/wiki/France")

      assert profile.domain == "en.wikipedia.org"
    end

    test "identifies blocked domains" do
      {:ok, profile} = SourceReliability.lookup("theonion.com")

      assert profile.trust_tier == :blocked
    end
  end

  describe "record_feedback/3" do
    test "admin feedback adjusts score over time" do
      # Record multiple rejections for a domain
      for _ <- 1..5 do
        SourceReliability.record_feedback("test-feedback-domain.com", :rejected)
      end

      # Allow async updates to process
      Process.sleep(100)

      {:ok, updated} = SourceReliability.lookup("test-feedback-domain.com")

      # Score should be low after rejections
      # Base 0.5 * 0.7 + (0.0 feedback) * 0.3 = 0.35
      assert updated.reliability_score <= 0.4
    end

    test "approvals increase reliability" do
      # Record multiple approvals
      for _ <- 1..5 do
        SourceReliability.record_feedback("test-approval-domain.com", :approved)
      end

      Process.sleep(100)

      {:ok, profile} = SourceReliability.lookup("test-approval-domain.com")

      # Score should be higher than neutral baseline
      assert profile.reliability_score > 0.5
    end
  end

  describe "get_stats/0" do
    test "returns statistics about source index" do
      stats = SourceReliability.get_stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :total_sources)
      assert Map.has_key?(stats, :blocked_domains)
      assert Map.has_key?(stats, :by_trust_tier)
    end
  end

  describe "blocked?/1" do
    test "returns true for blocked domains" do
      assert SourceReliability.blocked?("theonion.com") == true
    end

    test "returns false for normal domains" do
      assert SourceReliability.blocked?("wikipedia.org") == false
    end
  end

  describe "ready?/0" do
    test "returns true when service is running" do
      assert SourceReliability.ready?() == true
    end
  end
end
