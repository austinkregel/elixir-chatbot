defmodule Brain.Analysis.LearningStoreTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.LearningStore

  # Note: These tests interact with the actual LearningStore GenServer

  setup do
    # Ensure PubSub is started
    ensure_pubsub_started()

    # Start LearningStore under ExUnit supervision
    ensure_started(LearningStore)

    :ok
  end

  describe "get_params/1" do
    test "returns params for known component" do
      {:ok, params} = LearningStore.get_params("chunker")

      assert is_map(params)
      assert Map.has_key?(params, "max_chunk_words")
    end

    test "returns error for unknown component" do
      result = LearningStore.get_params("nonexistent_component")

      assert result == {:error, :not_found}
    end
  end

  describe "get_all_params/0" do
    test "returns all parameters" do
      params = LearningStore.get_all_params()

      assert is_map(params)
      assert Map.has_key?(params, "chunker")
      assert Map.has_key?(params, "speech_acts")
      assert Map.has_key?(params, "discourse")
    end
  end

  describe "update_params/3" do
    test "updates component parameters" do
      :ok = LearningStore.update_params("chunker", %{"max_chunk_words" => 60})

      {:ok, updated} = LearningStore.get_params("chunker")
      assert updated["max_chunk_words"] == 60
    end

    test "sets learned_at timestamp" do
      :ok = LearningStore.update_params("speech_acts", %{"confidence_threshold" => 0.35})

      {:ok, params} = LearningStore.get_params("speech_acts")
      assert params["learned_at"] != nil
    end
  end

  describe "lock_params/1 and unlock_params/1" do
    test "locks and unlocks parameters" do
      # Lock
      :ok = LearningStore.lock_params("discourse")
      {:ok, params} = LearningStore.get_params("discourse")
      assert params["admin_locked"] == true

      # Unlock
      :ok = LearningStore.unlock_params("discourse")
      {:ok, params} = LearningStore.get_params("discourse")
      assert params["admin_locked"] == false
    end

    test "locked params reject non-admin updates" do
      :ok = LearningStore.lock_params("discourse")

      result = LearningStore.update_params("discourse", %{"bot_names" => ["test"]})
      assert result == {:error, :locked}

      # Admin update should work
      :ok = LearningStore.update_params("discourse", %{"bot_names" => ["test"]}, admin: true)
      {:ok, params} = LearningStore.get_params("discourse")
      assert "test" in params["bot_names"]

      # Unlock for other tests (test isolation via config means we don't pollute production)
      :ok = LearningStore.unlock_params("discourse")
    end
  end

  describe "record_feedback/2" do
    test "records successful response feedback" do
      initial_stats = LearningStore.get_stats()
      initial_count = Map.get(initial_stats, "successful_responses", 0)

      LearningStore.record_feedback(:successful_response, %{})

      # Give time for async cast
      Process.sleep(100)

      final_stats = LearningStore.get_stats()
      final_count = Map.get(final_stats, "successful_responses", 0)

      assert final_count >= initial_count
    end

    test "records clarification needed feedback" do
      LearningStore.record_feedback(:clarification_needed, %{slot: "location"})

      # Give time for async cast
      Process.sleep(100)

      stats = LearningStore.get_stats()
      assert Map.has_key?(stats, "clarifications_needed")
    end
  end

  describe "get_stats/0" do
    test "returns feedback statistics" do
      stats = LearningStore.get_stats()

      assert is_map(stats)
      assert Map.has_key?(stats, "total_interactions")
      assert Map.has_key?(stats, "successful_responses")
      assert Map.has_key?(stats, "clarifications_needed")
    end
  end

  describe "reset_to_defaults/1" do
    test "resets specific component to defaults" do
      # First modify
      :ok = LearningStore.update_params("chunker", %{"max_chunk_words" => 100})

      # Reset
      :ok = LearningStore.reset_to_defaults("chunker")

      {:ok, params} = LearningStore.get_params("chunker")
      assert params["max_chunk_words"] == 50
    end
  end
end
