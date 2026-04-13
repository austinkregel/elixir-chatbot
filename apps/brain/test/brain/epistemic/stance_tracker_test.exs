defmodule Brain.Epistemic.StanceTrackerTest do
  use ExUnit.Case, async: false

  alias Brain.Epistemic.StanceTracker

  setup do
    {:ok, pid} = StanceTracker.start_link(name: :"tracker_#{:rand.uniform(100000)}")
    %{tracker: pid}
  end

  describe "ready?/1" do
    test "reports ready after start", %{tracker: tracker} do
      assert StanceTracker.ready?(tracker)
    end
  end

  describe "record_stance/5" do
    test "records stance observations", %{tracker: tracker} do
      StanceTracker.record_stance("conv1", "climate", 0.5, :system, tracker)
      StanceTracker.record_stance("conv1", "climate", 0.7, :user, tracker)

      # Give the async casts time to process
      Process.sleep(50)

      {:ok, stances} = StanceTracker.conversation_stances("conv1", tracker)
      assert Map.has_key?(stances, "climate")
      assert length(stances["climate"]) == 2
    end

    test "clamps positions to [-1, 1]", %{tracker: tracker} do
      StanceTracker.record_stance("conv1", "topic", 5.0, :system, tracker)
      Process.sleep(50)

      {:ok, stances} = StanceTracker.conversation_stances("conv1", tracker)
      obs = hd(stances["topic"])
      assert obs.position == 1.0
    end
  end

  describe "check_drift/3" do
    test "reports no drift with insufficient data", %{tracker: tracker} do
      StanceTracker.record_stance("conv1", "topic", 0.5, :system, tracker)
      Process.sleep(50)

      {:ok, result} = StanceTracker.check_drift("conv1", "topic", tracker)
      assert result == :no_drift
    end

    test "detects drift when system stance shifts significantly", %{tracker: tracker} do
      StanceTracker.record_stance("conv1", "topic", 0.2, :system, tracker)
      StanceTracker.record_stance("conv1", "topic", 0.8, :user, tracker)
      StanceTracker.record_stance("conv1", "topic", 0.6, :system, tracker)
      Process.sleep(50)

      {:ok, drift_info} = StanceTracker.check_drift("conv1", "topic", tracker)

      assert is_map(drift_info)
      assert drift_info.absolute_drift > 0.3
      assert drift_info.exceeds_threshold == true
      assert drift_info.initial_position == 0.2
      assert drift_info.current_position == 0.6
    end

    test "reports no drift for small changes", %{tracker: tracker} do
      StanceTracker.record_stance("conv1", "topic", 0.5, :system, tracker)
      StanceTracker.record_stance("conv1", "topic", 0.55, :system, tracker)
      Process.sleep(50)

      {:ok, drift_info} = StanceTracker.check_drift("conv1", "topic", tracker)

      if is_map(drift_info) do
        assert drift_info.exceeds_threshold == false
      else
        assert drift_info == :no_drift
      end
    end
  end

  describe "stats/1" do
    test "tracks observation counts", %{tracker: tracker} do
      StanceTracker.record_stance("conv1", "topic", 0.5, :system, tracker)
      StanceTracker.record_stance("conv1", "topic", 0.6, :user, tracker)
      StanceTracker.record_stance("conv2", "other", 0.3, :system, tracker)
      Process.sleep(50)

      {:ok, stats} = StanceTracker.stats(tracker)

      assert stats.total_observations == 3
      assert stats.conversations_tracked == 2
    end
  end
end
