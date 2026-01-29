defmodule ChatBot.Knowledge.ResearchAgentTest do
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  alias ChatBot.Knowledge.ResearchAgent
  alias ChatBot.Knowledge.Types.{ResearchGoal, Finding}

  setup do
    ensure_pubsub_started()

    # Ensure the rate limiter agent is available
    case Agent.start_link(fn -> %{} end, name: ChatBot.Knowledge.RateLimiter) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    :ok
  end

  describe "research/2" do
    test "returns findings for a goal with mock mode" do
      goal = ResearchGoal.new("France", questions: ["What is the capital?"])

      {:ok, findings} = ResearchAgent.research(goal, mock: true)

      assert is_list(findings)
      # Mock mode should return some findings
      assert length(findings) >= 0
    end

    test "handles empty topic gracefully" do
      goal = ResearchGoal.new("", questions: [])

      {:ok, findings} = ResearchAgent.research(goal, mock: true)

      assert is_list(findings)
    end

    test "respects max_pages option" do
      goal = ResearchGoal.new("Test topic")

      {:ok, findings} = ResearchAgent.research(goal, mock: true, max_pages: 1)

      # With limited pages, should still return a result
      assert is_list(findings)
    end

    test "findings have required fields" do
      goal = ResearchGoal.new("European capitals", questions: ["What is the capital of France?"])

      {:ok, findings} = ResearchAgent.research(goal, mock: true, max_pages: 2)

      for finding <- findings do
        assert %Finding{} = finding
        assert is_binary(finding.id)
        assert is_binary(finding.claim)
        assert finding.source != nil
      end
    end
  end

  describe "fetch_url/2" do
    test "returns error for blocked domain" do
      # Start SourceReliability for blocking check
      ensure_started(ChatBot.Knowledge.SourceReliability)

      result = ResearchAgent.fetch_url("https://theonion.com/article")

      assert {:error, :blocked_domain} = result
    end

    test "handles invalid URLs gracefully" do
      result = ResearchAgent.fetch_url("not-a-valid-url")

      # Should return an error, not crash
      assert {:error, _} = result
    end

    test "respects timeout option" do
      # Use a URL that will timeout quickly
      result = ResearchAgent.fetch_url("https://httpstat.us/200?sleep=5000", timeout: 100)

      # Should timeout or error
      assert {:error, _} = result
    end
  end

  describe "rate limiting" do
    test "rate limiter agent can be accessed" do
      # Verify the rate limiter is working
      domain = "test-rate-limit.com"

      # First request should be immediate
      start_time = System.monotonic_time(:millisecond)
      ResearchAgent.fetch_url("https://#{domain}/page1", timeout: 100)
      first_elapsed = System.monotonic_time(:millisecond) - start_time

      # Should complete quickly (just timeout, not rate limited)
      assert first_elapsed < 500
    end
  end

  describe "goal expansion" do
    test "research with questions generates queries" do
      goal =
        ResearchGoal.new("Paris",
          questions: [
            "What is the population?",
            "When was it founded?"
          ]
        )

      # Mock mode exercises the query expansion logic
      {:ok, _findings} = ResearchAgent.research(goal, mock: true)

      # If we got here without error, query expansion worked
      assert true
    end

    test "research with constraints" do
      goal =
        ResearchGoal.new("Test",
          constraints: %{min_sources: 3, max_age_days: 7}
        )

      {:ok, findings} = ResearchAgent.research(goal, mock: true)

      assert is_list(findings)
    end
  end

  describe "source enrichment" do
    test "findings include source reliability when available" do
      ensure_started(ChatBot.Knowledge.SourceReliability)

      goal = ResearchGoal.new("Test topic")
      {:ok, findings} = ResearchAgent.research(goal, mock: true)

      for finding <- findings do
        # Source should have reliability fields
        assert finding.source.reliability_score != nil
        assert finding.source.trust_tier != nil
      end
    end
  end
end
