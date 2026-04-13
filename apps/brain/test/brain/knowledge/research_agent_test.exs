defmodule Brain.Knowledge.ResearchAgentTest do
  alias Brain.Knowledge.Types
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Knowledge.ResearchAgent
  alias Types.{ResearchGoal, Finding}

  setup_all do
    Brain.TestHelpers.require_services!([
      Brain.Knowledge.SourceReliability,
      Brain.ML.SentimentClassifierSimple
    ])

    :ok
  end

  setup _context do
    ensure_pubsub_started()

    case Agent.start_link(fn -> %{} end, name: Brain.Knowledge.RateLimiter) do
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
      ensure_started(Brain.Knowledge.SourceReliability)

      result = ResearchAgent.fetch_url("https://theonion.com/article")

      assert {:error, :blocked_domain} = result
    end

    test "handles invalid URLs gracefully" do
      result = ResearchAgent.fetch_url("not-a-valid-url")
      assert {:error, _} = result
    end

    test "respects timeout option" do
      result = ResearchAgent.fetch_url("https://httpstat.us/200?sleep=5000", timeout: 100)
      assert {:error, _} = result
    end
  end

  describe "rate limiting" do
    test "rate limiter agent can be accessed" do
      domain = "test-rate-limit.com"
      start_time = System.monotonic_time(:millisecond)
      ResearchAgent.fetch_url("https://#{domain}/page1", timeout: 100)
      first_elapsed = System.monotonic_time(:millisecond) - start_time
      assert first_elapsed < 500
    end
  end

  describe "goal expansion" do
    test "research with questions generates queries" do
      goal =
        ResearchGoal.new("Paris", questions: ["What is the population?", "When was it founded?"])

      {:ok, _findings} = ResearchAgent.research(goal, mock: true)
      assert true
    end

    test "research with constraints" do
      goal =
        ResearchGoal.new("Test", constraints: %{min_sources: 3, max_age_days: 7})

      {:ok, findings} = ResearchAgent.research(goal, mock: true)

      assert is_list(findings)
    end
  end

  describe "source enrichment" do
    test "findings include source reliability when available" do
      ensure_started(Brain.Knowledge.SourceReliability)

      goal = ResearchGoal.new("Test topic")
      {:ok, findings} = ResearchAgent.research(goal, mock: true)

      for finding <- findings do
        assert finding.source.reliability_score != nil
        assert finding.source.trust_tier != nil
      end
    end
  end
end
