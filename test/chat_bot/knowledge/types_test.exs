defmodule ChatBot.Knowledge.TypesTest do
  use ExUnit.Case, async: true

  alias ChatBot.Knowledge.Types.{
    SourceInfo,
    Finding,
    ResearchGoal,
    ReviewCandidate,
    LearningSession,
    SourceProfile
  }

  describe "SourceInfo" do
    test "creates from URL with domain extraction" do
      source = SourceInfo.new("https://www.wikipedia.org/wiki/France")

      assert source.url == "https://www.wikipedia.org/wiki/France"
      assert source.domain == "wikipedia.org"
      assert source.reliability_score == 0.5
      assert source.trust_tier == :neutral
    end

    test "extracts domain correctly from various URL formats" do
      assert SourceInfo.extract_domain("https://en.wikipedia.org/wiki/Page") == "en.wikipedia.org"
      assert SourceInfo.extract_domain("http://www.example.com/path") == "example.com"
      assert SourceInfo.extract_domain("https://subdomain.site.co.uk/page") == "subdomain.site.co.uk"
    end

    test "accepts custom reliability options" do
      source =
        SourceInfo.new("https://reuters.com/article",
          reliability_score: 0.92,
          bias_rating: :center,
          trust_tier: :verified
        )

      assert source.reliability_score == 0.92
      assert source.bias_rating == :center
      assert source.trust_tier == :verified
    end
  end

  describe "Finding" do
    test "creates with generated ID" do
      source = SourceInfo.new("https://example.com")
      finding = Finding.new("Paris is the capital of France", "France", source)

      assert is_binary(finding.id)
      assert String.length(finding.id) > 0
      assert finding.claim == "Paris is the capital of France"
      assert finding.entity == "France"
      assert finding.source == source
    end

    test "accepts optional entity type and confidence" do
      source = SourceInfo.new("https://example.com")

      finding =
        Finding.new("Paris is the capital of France", "France", source,
          entity_type: "location",
          confidence: 0.85
        )

      assert finding.entity_type == "location"
      assert finding.confidence == 0.85
    end
  end

  describe "ResearchGoal" do
    test "creates with topic" do
      goal = ResearchGoal.new("European capitals")

      assert is_binary(goal.id)
      assert goal.topic == "European capitals"
      assert goal.status == :pending
      assert goal.priority == :normal
    end

    test "accepts questions and constraints" do
      goal =
        ResearchGoal.new("France",
          questions: ["What is the capital?", "What is the population?"],
          constraints: %{min_sources: 2},
          priority: :high
        )

      assert length(goal.questions) == 2
      assert goal.constraints.min_sources == 2
      assert goal.priority == :high
    end

    test "updates status" do
      goal = ResearchGoal.new("Test topic")
      assert goal.status == :pending

      updated = ResearchGoal.update_status(goal, :in_progress)
      assert updated.status == :in_progress

      completed = ResearchGoal.update_status(updated, :completed)
      assert completed.status == :completed
    end
  end

  describe "ReviewCandidate" do
    test "creates from finding" do
      source = SourceInfo.new("https://example.com", reliability_score: 0.8)
      finding = Finding.new("Test claim", "Test Entity", source, confidence: 0.75)

      candidate = ReviewCandidate.new(finding)

      assert is_binary(candidate.id)
      assert candidate.finding == finding
      assert candidate.status == :pending
      assert candidate.aggregate_confidence == 0.75
    end

    test "approve updates status and timestamp" do
      source = SourceInfo.new("https://example.com")
      finding = Finding.new("Test", "Entity", source)
      candidate = ReviewCandidate.new(finding)

      approved = ReviewCandidate.approve(candidate, "Looks good")

      assert approved.status == :approved
      assert approved.reviewer_notes == "Looks good"
      assert approved.reviewed_at != nil
    end

    test "reject updates status and timestamp" do
      source = SourceInfo.new("https://example.com")
      finding = Finding.new("Test", "Entity", source)
      candidate = ReviewCandidate.new(finding)

      rejected = ReviewCandidate.reject(candidate, "Incorrect")

      assert rejected.status == :rejected
      assert rejected.reviewer_notes == "Incorrect"
    end
  end

  describe "LearningSession" do
    test "creates with default values" do
      session = LearningSession.new()

      assert is_binary(session.id)
      assert session.status == :active
      assert session.findings_count == 0
    end

    test "adds goals and records findings" do
      session = LearningSession.new(topic: "Test topic")
      goal = ResearchGoal.new("Subtopic")

      session = LearningSession.add_goal(session, goal)
      assert length(session.goals) == 1

      session = LearningSession.record_findings(session, 5)
      assert session.findings_count == 5

      session = LearningSession.record_approval(session)
      assert session.approved_count == 1
    end

    test "completes session" do
      session = LearningSession.new()
      completed = LearningSession.complete(session)

      assert completed.status == :completed
      assert completed.completed_at != nil
    end
  end

  describe "SourceProfile" do
    test "creates with domain" do
      profile = SourceProfile.new("example.com")

      assert profile.domain == "example.com"
      assert profile.factual_accuracy == 0.5
      assert profile.trust_tier == :neutral
    end

    test "records decisions and calculates reliability" do
      profile = SourceProfile.new("example.com", factual_accuracy: 0.7)

      # Record several approvals
      profile = SourceProfile.record_decision(profile, :approved)
      profile = SourceProfile.record_decision(profile, :approved)
      profile = SourceProfile.record_decision(profile, :approved)

      reliability = SourceProfile.calculate_reliability(profile)
      # Base 0.7 * 0.7 + feedback adjustment * 0.3
      # With all approvals, feedback should be ~1.0
      assert reliability > 0.7
    end

    test "rejections lower reliability" do
      profile = SourceProfile.new("sketchy.com", factual_accuracy: 0.5)

      # Record several rejections
      profile = SourceProfile.record_decision(profile, :rejected)
      profile = SourceProfile.record_decision(profile, :rejected)
      profile = SourceProfile.record_decision(profile, :rejected)

      reliability = SourceProfile.calculate_reliability(profile)
      # With all rejections, feedback should be ~0.0
      assert reliability < 0.5
    end
  end
end
