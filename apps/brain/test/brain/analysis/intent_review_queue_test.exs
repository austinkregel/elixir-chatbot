defmodule Brain.Analysis.IntentReviewQueueTest do
  alias Brain.Analysis
  use Brain.Test.GraphCase, async: false

  alias Analysis.{IntentReviewQueue, Types.IntentReviewCandidate}

  setup _context do
    IntentReviewQueue.clear()
    :ok
  end

  describe "add/1" do
    test "adds a candidate to the queue" do
      candidate = IntentReviewCandidate.new("test utterance", "weather.query", 0.45)

      assert {:ok, id} = IntentReviewQueue.add(candidate)
      assert is_binary(id)
      assert id == candidate.id
    end

    test "increments pending count" do
      candidate = IntentReviewCandidate.new("test", "weather.query", 0.5)
      IntentReviewQueue.add(candidate)

      stats = IntentReviewQueue.stats()
      assert stats.pending == 1
    end
  end

  describe "get_pending/1" do
    test "returns pending candidates" do
      candidate1 = IntentReviewCandidate.new("test 1", "weather.query", 0.3)
      candidate2 = IntentReviewCandidate.new("test 2", "music.play", 0.4)

      IntentReviewQueue.add(candidate1)
      IntentReviewQueue.add(candidate2)

      pending = IntentReviewQueue.get_pending()
      assert length(pending) == 2
    end

    test "respects limit" do
      for i <- 1..10 do
        candidate = IntentReviewCandidate.new("test #{i}", "weather.query", 0.5)
        IntentReviewQueue.add(candidate)
      end

      pending = IntentReviewQueue.get_pending(limit: 5)
      assert length(pending) == 5
    end
  end

  describe "approve/2" do
    test "marks candidate as approved" do
      candidate = IntentReviewCandidate.new("test", "weather.query", 0.5)
      IntentReviewQueue.add(candidate)

      assert {:ok, approved} =
               IntentReviewQueue.approve(candidate.id, "test notes", :variation, "weather.query")

      assert approved.status == :approved
      assert approved.reviewer_notes == "test notes"
      assert approved.promotion_action == :variation
    end

    test "updates stats" do
      candidate = IntentReviewCandidate.new("test", "weather.query", 0.5)
      IntentReviewQueue.add(candidate)

      IntentReviewQueue.approve(candidate.id)

      stats = IntentReviewQueue.stats()
      assert stats.pending == 0
      assert stats.approved == 1
    end
  end

  describe "update_annotation/2" do
    test "updates candidate annotation" do
      candidate = IntentReviewCandidate.new("test", "weather.query", 0.5)
      IntentReviewQueue.add(candidate)

      updates = %{
        tags: [:priority, :domain_guess],
        notes: "Test note",
        domain_guess: "weather"
      }

      assert {:ok, updated} = IntentReviewQueue.update_annotation(candidate.id, updates)
      assert :priority in updated.annotation[:tags]
      assert updated.annotation[:notes] == "Test note"
      assert updated.annotation[:domain_guess] == "weather"
    end
  end

  describe "persistence" do
    test "persists and loads candidates" do
      candidate = IntentReviewCandidate.new("test", "weather.query", 0.5)
      IntentReviewQueue.add(candidate)
      IntentReviewQueue.persist()
      IntentReviewQueue.clear()
      stats = IntentReviewQueue.stats()
      assert is_map(stats)
    end
  end
end
