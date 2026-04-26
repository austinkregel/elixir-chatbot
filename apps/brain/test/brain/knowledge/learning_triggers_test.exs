defmodule Brain.Knowledge.LearningTriggersTest do
  use ExUnit.Case, async: false

  alias Brain.Knowledge.LearningTriggers

  setup do
    name = :"learning_triggers_test_#{:rand.uniform(100_000)}"
    {:ok, pid} = LearningTriggers.start_link(name: name)
    on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)
    {:ok, name: name, pid: pid}
  end

  describe "ready?/1" do
    test "returns true when started", %{name: name} do
      assert LearningTriggers.ready?(name)
    end

    test "returns false for unregistered name" do
      refute LearningTriggers.ready?(:nonexistent_triggers)
    end
  end

  describe "stats/1" do
    test "returns initial stats", %{name: name} do
      stats = LearningTriggers.stats(name)

      assert is_map(stats)
      assert stats.sessions_total_today == 0
      assert stats.max_sessions_per_day_per_domain == 2
      assert stats.max_sessions_per_day_total == 5
      assert stats.triggered_sessions == 0
      assert stats.novel_input_domains == []
      assert stats.novel_input_counts == %{}
    end
  end

  describe "novel input handling" do
    test "accumulates novel inputs by domain", %{name: name, pid: pid} do
      send(pid, {:novel_input, "what is quantum computing", 0.8, :science})
      Process.sleep(20)

      stats = LearningTriggers.stats(name)
      assert :science in stats.novel_input_domains
      assert stats.novel_input_counts[:science] == 1
    end

    test "accumulates multiple inputs in same domain", %{name: name, pid: pid} do
      send(pid, {:novel_input, "quantum computing basics", 0.8, :science})
      send(pid, {:novel_input, "quantum entanglement", 0.7, :science})
      Process.sleep(20)

      stats = LearningTriggers.stats(name)
      assert stats.novel_input_counts[:science] == 2
    end

    test "tracks separate domains independently", %{name: name, pid: pid} do
      send(pid, {:novel_input, "what is photosynthesis", 0.8, :biology})
      send(pid, {:novel_input, "explain gravity", 0.7, :physics})
      Process.sleep(20)

      stats = LearningTriggers.stats(name)
      assert :biology in stats.novel_input_domains
      assert :physics in stats.novel_input_domains
    end

    test "handles enriched message format with entities", %{name: name, pid: pid} do
      entities = [%{value: "quantum", confidence: 0.9, type: "concept"}]
      send(pid, {:novel_input, "quantum computing", 0.8, :science, entities})
      Process.sleep(20)

      stats = LearningTriggers.stats(name)
      assert stats.novel_input_counts[:science] == 1
    end
  end

  describe "cleanup" do
    test "handles cleanup message", %{pid: pid} do
      send(pid, :cleanup)
      Process.sleep(20)
      assert Process.alive?(pid)
    end
  end

  describe "unknown messages" do
    test "ignores unknown messages", %{pid: pid} do
      send(pid, :some_random_message)
      Process.sleep(10)
      assert Process.alive?(pid)
    end
  end
end
