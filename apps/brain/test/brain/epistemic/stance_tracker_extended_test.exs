defmodule Brain.Epistemic.StanceTrackerExtendedTest do
  use ExUnit.Case, async: false

  alias Brain.Epistemic.StanceTracker

  setup do
    name = :"stance_ext_test_#{:rand.uniform(100_000)}"
    {:ok, _pid} = StanceTracker.start_link(name: name)
    {:ok, tracker: name}
  end

  describe "multi-topic tracking" do
    test "tracks multiple topics independently", %{tracker: tracker} do
      StanceTracker.record_stance("conv1", "climate", 0.5, :user, tracker)
      StanceTracker.record_stance("conv1", "climate", 0.3, :system, tracker)
      StanceTracker.record_stance("conv1", "taxes", -0.3, :user, tracker)
      StanceTracker.record_stance("conv1", "taxes", -0.1, :system, tracker)
      StanceTracker.record_stance("conv1", "climate", 0.6, :system, tracker)
      StanceTracker.record_stance("conv1", "taxes", -0.5, :system, tracker)

      Process.sleep(50)

      {:ok, climate_result} = StanceTracker.check_drift("conv1", "climate", tracker)
      {:ok, tax_result} = StanceTracker.check_drift("conv1", "taxes", tracker)

      assert is_map(climate_result)
      assert is_map(tax_result)
      assert Map.has_key?(climate_result, :absolute_drift)
      assert Map.has_key?(tax_result, :absolute_drift)
    end

    test "different conversations are independent", %{tracker: tracker} do
      StanceTracker.record_stance("conv_a", "topic1", 0.9, :user, tracker)
      StanceTracker.record_stance("conv_a", "topic1", 0.1, :system, tracker)
      StanceTracker.record_stance("conv_b", "topic1", -0.9, :user, tracker)
      StanceTracker.record_stance("conv_b", "topic1", -0.2, :system, tracker)
      StanceTracker.record_stance("conv_a", "topic1", 0.8, :system, tracker)
      StanceTracker.record_stance("conv_b", "topic1", -0.5, :system, tracker)

      Process.sleep(50)

      {:ok, drift_a} = StanceTracker.check_drift("conv_a", "topic1", tracker)
      {:ok, drift_b} = StanceTracker.check_drift("conv_b", "topic1", tracker)

      assert is_map(drift_a)
      assert is_map(drift_b)
      assert drift_a.absolute_drift != drift_b.absolute_drift
    end
  end

  describe "position clamping" do
    test "extreme positive value clamped to 1.0", %{tracker: tracker} do
      StanceTracker.record_stance("conv", "topic", 5.0, :user, tracker)
      StanceTracker.record_stance("conv", "topic", 5.0, :system, tracker)
      Process.sleep(50)

      {:ok, stances} = StanceTracker.conversation_stances("conv", tracker)
      observations = Map.get(stances, "topic", [])

      for obs <- observations do
        assert obs.position >= -1.0 and obs.position <= 1.0
      end
    end

    test "extreme negative value clamped to -1.0", %{tracker: tracker} do
      StanceTracker.record_stance("conv", "topic", -10.0, :user, tracker)
      StanceTracker.record_stance("conv", "topic", -10.0, :system, tracker)
      Process.sleep(50)

      {:ok, stances} = StanceTracker.conversation_stances("conv", tracker)
      observations = Map.get(stances, "topic", [])

      for obs <- observations do
        assert obs.position >= -1.0 and obs.position <= 1.0
      end
    end
  end

  describe "drift detection details" do
    test "no drift when only one observation", %{tracker: tracker} do
      StanceTracker.record_stance("conv", "topic", 0.5, :system, tracker)
      Process.sleep(50)

      {:ok, drift_result} = StanceTracker.check_drift("conv", "topic", tracker)
      assert drift_result == :no_drift
    end

    test "drift between two system observations", %{tracker: tracker} do
      StanceTracker.record_stance("conv", "topic", 0.2, :system, tracker)
      StanceTracker.record_stance("conv", "topic", 0.8, :user, tracker)
      StanceTracker.record_stance("conv", "topic", 0.7, :system, tracker)
      Process.sleep(50)

      {:ok, drift_info} = StanceTracker.check_drift("conv", "topic", tracker)
      assert is_map(drift_info)
      assert drift_info.absolute_drift > 0.4
    end

    test "check_drift for unknown conversation returns no_drift", %{tracker: tracker} do
      {:ok, drift_result} = StanceTracker.check_drift("unknown_conv", "unknown_topic", tracker)
      assert drift_result == :no_drift
    end
  end

  describe "stats" do
    test "stats reflect recorded data", %{tracker: tracker} do
      StanceTracker.record_stance("c1", "t1", 0.5, :user, tracker)
      StanceTracker.record_stance("c2", "t2", -0.5, :user, tracker)
      Process.sleep(50)

      {:ok, stats} = StanceTracker.stats(tracker)
      assert is_map(stats)
      assert stats.total_observations >= 2
    end
  end
end
