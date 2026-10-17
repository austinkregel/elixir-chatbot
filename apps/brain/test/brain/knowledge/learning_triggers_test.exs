defmodule Brain.Knowledge.LearningTriggersTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Knowledge.LearningTriggers

  # The auto-learning world is stored in Atlas, so every instance reads and
  # writes inside this test's sandbox transaction.
  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

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

  describe "auto_learning_world/1" do
    test "starts as the configured auto-learning world and stores it", %{name: name} do
      configured = Application.fetch_env!(:brain, :auto_learning_world_id)

      assert LearningTriggers.auto_learning_world(name) == configured

      assert Atlas.Settings.fetch(LearningTriggers.auto_learning_world_setting()) ==
               {:ok, configured}
    end
  end

  describe "a restart" do
    setup do
      ensure_started(World.Manager)
      setup_world_sandbox()
    end

    test "keeps the auto-learning world that was set", %{name: name, pid: pid} do
      {:ok, world} = create_test_world("auto_learning_restart")
      assert world.id != Application.fetch_env!(:brain, :auto_learning_world_id)

      assert :ok = LearningTriggers.set_auto_learning_world(world.id, name)
      assert Atlas.Settings.fetch(LearningTriggers.auto_learning_world_setting()) == {:ok, world.id}

      GenServer.stop(pid)
      {:ok, restarted} = LearningTriggers.start_link(name: name)
      on_exit(fn -> if Process.alive?(restarted), do: GenServer.stop(restarted) end)

      assert LearningTriggers.auto_learning_world(name) == world.id
    end
  end

  describe "set_auto_learning_world/2" do
    setup do
      ensure_started(World.Manager)
      setup_world_sandbox()
    end

    test "switches to a loaded world", %{name: name} do
      {:ok, world} = create_test_world("auto_learning")

      assert :ok = LearningTriggers.set_auto_learning_world(world.id, name)
      assert LearningTriggers.auto_learning_world(name) == world.id
    end

    test "refuses a world that is not loaded and keeps the current one", %{name: name} do
      before = LearningTriggers.auto_learning_world(name)
      unloaded = "test_unloaded_auto_learning_#{System.unique_integer([:positive])}"

      assert {:error, {:world_not_loaded, ^unloaded, :not_found}} =
               LearningTriggers.set_auto_learning_world(unloaded, name)

      assert LearningTriggers.auto_learning_world(name) == before
    end
  end

  describe "auto-learning world configuration" do
    setup do
      previous = Application.fetch_env(:brain, :auto_learning_world_id)

      on_exit(fn ->
        case previous do
          {:ok, world_id} -> Application.put_env(:brain, :auto_learning_world_id, world_id)
          :error -> Application.delete_env(:brain, :auto_learning_world_id)
        end
      end)

      Process.flag(:trap_exit, true)
      :ok
    end

    @tag :capture_log
    test "starting with neither a stored nor a configured auto-learning world raises" do
      delete_stored_auto_learning_world!()
      Application.delete_env(:brain, :auto_learning_world_id)
      name = :"learning_triggers_unconfigured_#{System.unique_integer([:positive])}"

      assert {:error, {%ArgumentError{message: message}, _stacktrace}} =
               LearningTriggers.start_link(name: name)

      assert message =~ "config :brain, :auto_learning_world_id is not set"
    end

    test "a stored auto-learning world is used without the application setting" do
      stored = "test_stored_auto_learning_#{System.unique_integer([:positive])}"
      Atlas.Settings.put!(LearningTriggers.auto_learning_world_setting(), stored)
      Application.delete_env(:brain, :auto_learning_world_id)
      name = :"learning_triggers_stored_#{System.unique_integer([:positive])}"

      {:ok, pid} = LearningTriggers.start_link(name: name)
      on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)

      assert LearningTriggers.auto_learning_world(name) == stored
    end

    @tag :capture_log
    test "a triggered session raises naming the auto-learning world when it is not loaded" do
      ensure_started(World.Manager)
      unloaded = "test_unloaded_auto_learning_#{System.unique_integer([:positive])}"
      assert {:error, :not_found} = World.Manager.get(unloaded)

      Atlas.Settings.put!(LearningTriggers.auto_learning_world_setting(), unloaded)
      name = :"learning_triggers_unloaded_#{System.unique_integer([:positive])}"
      {:ok, pid} = LearningTriggers.start_link(name: name)
      assert LearningTriggers.auto_learning_world(name) == unloaded

      # Three novel inputs in one domain reach the novelty threshold and
      # trigger a session in the auto-learning world.
      entities = [%{value: "quantum", confidence: 0.9, type: "concept"}]
      send(pid, {:novel_input, "what is quantum computing", 0.8, :science, entities})
      send(pid, {:novel_input, "quantum entanglement", 0.7, :science, entities})
      send(pid, {:novel_input, "quantum error correction", 0.9, :science, entities})

      assert_receive {:EXIT, ^pid, {%RuntimeError{message: message}, _stacktrace}}, 1_000
      assert message =~ "the auto-learning world #{inspect(unloaded)} is not loaded"
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

  # The instance this module's setup starts stores the configured world; a test
  # about starting with nothing stored removes that row first.
  defp delete_stored_auto_learning_world! do
    Atlas.Schemas.Setting
    |> Atlas.Repo.get!(LearningTriggers.auto_learning_world_setting())
    |> Atlas.Repo.delete!()
  end
end
