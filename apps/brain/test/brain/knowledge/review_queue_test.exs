defmodule Brain.Knowledge.ReviewQueueTest do
  alias Brain.Knowledge.Types
  alias Brain.Knowledge
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Knowledge.{ReviewQueue, SourceReliability}
  alias Types.{Finding, SourceInfo, ReviewCandidate}

  setup _context do
    ensure_pubsub_started()
    ensure_started(SourceReliability)
    ensure_started(ReviewQueue)
    ReviewQueue.clear()

    :ok
  end

  describe "add/1" do
    test "adds candidate to queue" do
      candidate = build_test_candidate("France", "Paris is the capital")

      {:ok, id} = ReviewQueue.add(candidate)

      assert is_binary(id)
      assert id == candidate.id
    end

    test "increments pending count" do
      initial_stats = ReviewQueue.stats()

      candidate = build_test_candidate("Test", "Test claim")
      ReviewQueue.add(candidate)

      final_stats = ReviewQueue.stats()
      assert final_stats.pending == initial_stats.pending + 1
    end
  end

  describe "get_pending/1" do
    test "returns pending candidates" do
      candidate1 = build_test_candidate("Entity1", "Claim 1")
      candidate2 = build_test_candidate("Entity2", "Claim 2")

      ReviewQueue.add(candidate1)
      ReviewQueue.add(candidate2)

      pending = ReviewQueue.get_pending()

      assert length(pending) == 2
    end

    test "respects limit option" do
      for i <- 1..10 do
        ReviewQueue.add(build_test_candidate("Entity#{i}", "Claim #{i}"))
      end

      pending = ReviewQueue.get_pending(limit: 5)

      assert length(pending) == 5
    end

    test "sorts by confidence by default" do
      low_conf = build_test_candidate("Low", "Low confidence", confidence: 0.3)
      high_conf = build_test_candidate("High", "High confidence", confidence: 0.9)

      ReviewQueue.add(low_conf)
      ReviewQueue.add(high_conf)

      pending = ReviewQueue.get_pending()
      assert hd(pending).aggregate_confidence == 0.9
    end
  end

  describe "approve/2" do
    test "updates status to approved" do
      candidate = build_test_candidate("France", "Paris is the capital")
      ReviewQueue.add(candidate)

      {:ok, approved} = ReviewQueue.approve(candidate.id, "Verified correct")

      assert approved.status == :approved
      assert approved.reviewer_notes == "Verified correct"
      assert approved.reviewed_at != nil
    end

    test "removes from pending list" do
      candidate = build_test_candidate("Test", "Test claim")
      ReviewQueue.add(candidate)

      initial_pending = length(ReviewQueue.get_pending())

      ReviewQueue.approve(candidate.id)

      final_pending = length(ReviewQueue.get_pending())
      assert final_pending == initial_pending - 1
    end

    test "returns error for non-existent ID" do
      {:error, :not_found} = ReviewQueue.approve("nonexistent-id")
    end
  end

  describe "reject/2" do
    test "updates status to rejected" do
      candidate = build_test_candidate("Test", "Test claim")
      ReviewQueue.add(candidate)

      {:ok, rejected} = ReviewQueue.reject(candidate.id, "Factually incorrect")

      assert rejected.status == :rejected
      assert rejected.reviewer_notes == "Factually incorrect"
    end

    test "updates daily rejection count" do
      candidate = build_test_candidate("Test", "Test claim")
      ReviewQueue.add(candidate)

      initial_stats = ReviewQueue.stats()

      ReviewQueue.reject(candidate.id)

      final_stats = ReviewQueue.stats()
      assert final_stats.rejected_today == initial_stats.rejected_today + 1
    end
  end

  describe "bulk_approve/1" do
    test "approves multiple candidates" do
      candidates =
        for i <- 1..5 do
          build_test_candidate("Entity#{i}", "Claim #{i}")
        end

      ids =
        Enum.map(candidates, fn c ->
          ReviewQueue.add(c)
          c.id
        end)

      {:ok, count} = ReviewQueue.bulk_approve(ids)

      assert count == 5
      assert ReviewQueue.get_pending() == []
    end
  end

  describe "bulk_reject/1" do
    test "rejects multiple candidates" do
      candidates =
        for i <- 1..3 do
          build_test_candidate("Entity#{i}", "Claim #{i}")
        end

      ids =
        Enum.map(candidates, fn c ->
          ReviewQueue.add(c)
          c.id
        end)

      {:ok, count} = ReviewQueue.bulk_reject(ids)

      assert count == 3
    end
  end

  describe "stats/0" do
    test "returns queue statistics" do
      stats = ReviewQueue.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :pending)
      assert Map.has_key?(stats, :approved)
      assert Map.has_key?(stats, :rejected)
      assert Map.has_key?(stats, :approved_today)
      assert Map.has_key?(stats, :rejected_today)
    end
  end

  describe "add_contradiction/2" do
    test "adds contradiction for review" do
      new_fact = %{entity: "France", claim: "Lyon is the capital"}
      existing_belief = %{object: "Paris is the capital", confidence: 0.9}

      {:ok, id} = ReviewQueue.add_contradiction(new_fact, existing_belief)

      assert is_binary(id)

      pending = ReviewQueue.get_pending()
      contradiction = Enum.find(pending, &String.contains?(&1.finding.claim, "CONTRADICTION"))
      assert contradiction != nil
    end
  end

  describe "persistence" do
    test "persists across restarts" do
      candidate = build_test_candidate("Persist", "Persistence test claim")
      ReviewQueue.add(candidate)
      ReviewQueue.persist()
      GenServer.stop(ReviewQueue)
      Process.sleep(100)

      ensure_started(ReviewQueue)

      pending = ReviewQueue.get_pending()
      found = Enum.find(pending, &(&1.id == candidate.id))
      assert found != nil
    end
  end

  defp build_test_candidate(entity, claim, opts \\ []) do
    confidence = Keyword.get(opts, :confidence, 0.7)
    domain = Keyword.get(opts, :domain, "test-source.com")

    source =
      SourceInfo.new("https://#{domain}/article", reliability_score: 0.8, trust_tier: :verified)

    finding = Finding.new(claim, entity, source, entity_type: "location", confidence: confidence)

    ReviewCandidate.new(finding, aggregate_confidence: confidence)
  end
end
