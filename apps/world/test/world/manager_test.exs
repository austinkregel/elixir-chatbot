defmodule World.ManagerTest do
  use World.AtlasCase, async: false
  import Brain.TestHelpers

  alias Brain.Epistemic.BeliefStore
  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.Services.{Dispatcher, LocalScheduler}
  alias World.Manager

  setup do
    start_world_test_services()
    setup_world_sandbox()

    eventually(
      fn -> Manager.ready?() end,
      fn ready -> ready end,
      50,
      20
    )

    :ok
  end

  describe "ready?/0" do
    test "is true once the manager has started" do
      assert Manager.ready?() == true
    end
  end

  describe "list_worlds/1" do
    test "leaves out worlds marked as test worlds unless asked for them" do
      {:ok, world} = create_test_world("manager_list")

      refute world.id in Enum.map(Manager.list_worlds(), & &1.id)
      assert world.id in Enum.map(Manager.list_worlds(include_test: true), & &1.id)
    end

    test "lists a world without the test mark" do
      # Created without the mark, so it is listed; it is destroyed when the
      # test ends and, being ephemeral, has nothing on disk.
      world_id = "test_manager_listed_#{System.unique_integer([:positive])}"
      {:ok, world} = Manager.create(world_id, id: world_id, mode: :ephemeral)
      on_exit(fn -> Manager.destroy(world.id) end)

      assert world.id in Enum.map(Manager.list_worlds(), & &1.id)
    end
  end

  describe "a world's graphs" do
    test "are created when the world is created" do
      {:ok, world} = create_test_world("manager_graphs_created")

      for name <- Atlas.Graph.world_graphs(world.id) do
        assert Atlas.Graph.graph_exists?(name), "no graph #{name} for world #{world.id}"
      end
    end

    test "are created when a persisted world is loaded" do
      {:ok, world} = create_test_world("manager_graphs_loaded", mode: :persistent)
      names = Atlas.Graph.world_graphs(world.id)

      # Unloaded, with its graphs gone: its persisted data stays on disk.
      :ok = Manager.destroy(world.id)

      for name <- names do
        Atlas.Repo.query!("SELECT ag_catalog.drop_graph($1, true)", [name])
      end

      refute Enum.any?(names, &Atlas.Graph.graph_exists?/1)

      assert {:ok, _world} = Manager.load_world(world.id)
      assert Enum.all?(names, &Atlas.Graph.graph_exists?/1)
    end
  end

  describe "a world's scheduling driver" do
    test "is the local scheduler when the world is created" do
      {:ok, world} = create_test_world("manager_scheduling_created")

      assert Dispatcher.scheduling_driver(world.id) == {:ok, LocalScheduler}
      assert Atlas.WorldServiceDrivers.fetch(world.id, "scheduling") == {:ok, "local_scheduler"}
    end

    test "is the local scheduler for a world created again under an id whose driver was changed" do
      {:ok, world} = create_test_world("manager_scheduling_recreated")
      Atlas.WorldServiceDrivers.put!(world.id, "scheduling", "test_other_driver")
      :ok = Manager.destroy(world.id)

      {:ok, again} = create_test_world("manager_scheduling_recreated", id: world.id)

      assert Dispatcher.scheduling_driver(again.id) == {:ok, LocalScheduler}
    end

    test "is kept when the world is destroyed and deleted when it is purged" do
      {:ok, world} = create_test_world("manager_scheduling_kept")
      :ok = Manager.destroy(world.id)
      assert Dispatcher.scheduling_driver(world.id) == {:ok, LocalScheduler}

      {:ok, purged} = create_test_world("manager_scheduling_purged")
      assert :ok = Manager.purge(purged.id)
      assert Dispatcher.scheduling_driver(purged.id) == :none
    end
  end

  # The turn the scheduler is asked in.
  @turn_time ~U[2026-10-12 09:00:00Z]

  # A world with something in each of the places a world keeps data in Atlas
  # through the running processes: an episode in the memory store, a belief, a
  # scheduler item, and a node in its knowledge graph. Its scheduling driver
  # was stored when it was created.
  defp populate!(world_id) do
    {:ok, _id} = MemoryStore.add_episode("the weather in paris is sunny", "weather", "", [], world_id: world_id)
    {:ok, _id} = BeliefStore.add_belief(world_id, :user, :likes, "tea")
    {:ok, _} = LocalScheduler.enrich("timer.set", %{duration: "5 minutes", turn_time: @turn_time}, %{}, world_id)
    {:ok, _} = Atlas.Graph.add_node(Atlas.Graph.world_graph("knowledge_graph", world_id), "Thing", %{name: "tea"})
    :ok = Brain.AtlasIntegration.drain()
  end

  defp atlas_rows(world_id) do
    import Ecto.Query

    Map.new(Atlas.Worlds.world_tables(), fn {schema, column} ->
      {schema, schema |> where([row], field(row, ^column) == ^world_id) |> Atlas.Repo.aggregate(:count)}
    end)
  end

  describe "purge/1" do
    test "drops the world's graphs and deletes what it holds in Atlas, and leaves another world's" do
      {:ok, world} = create_test_world("manager_purged", mode: :persistent)
      {:ok, other} = create_test_world("manager_purge_kept")
      populate!(world.id)
      populate!(other.id)

      assert atlas_rows(world.id)[Atlas.Schemas.Episode] == 1
      assert atlas_rows(world.id)[Atlas.Schemas.Belief] == 1
      assert atlas_rows(world.id)[Atlas.Schemas.SchedulerItem] == 1
      assert atlas_rows(world.id)[Atlas.Schemas.WorldServiceDriver] == 1

      assert :ok = Manager.purge(world.id)

      assert Manager.get(world.id) == {:error, :not_found}
      refute File.exists?(World.Persistence.world_path(world.id))
      refute Enum.any?(Atlas.Graph.world_graphs(world.id), &Atlas.Graph.graph_exists?/1)
      assert Enum.all?(atlas_rows(world.id), fn {_schema, count} -> count == 0 end)

      # Nothing the running processes held of it is left either.
      assert {:ok, []} = BeliefStore.query_beliefs(world.id)
      assert {:ok, %{timer_status: "none"}} = LocalScheduler.enrich("timer.check", %{turn_time: @turn_time}, %{}, world.id)
      assert {:ok, []} = MemoryStore.all_episodes(world_id: world.id)

      assert Enum.all?(Atlas.Graph.world_graphs(other.id), &Atlas.Graph.graph_exists?/1)
      assert {:ok, 1} = Atlas.Graph.count_nodes(Atlas.Graph.world_graph("knowledge_graph", other.id))
      assert atlas_rows(other.id)[Atlas.Schemas.Episode] == 1
      assert atlas_rows(other.id)[Atlas.Schemas.Belief] == 1
      assert {:ok, [_belief]} = BeliefStore.query_beliefs(other.id)
      assert {:ok, %{timer_status: "running"}} = LocalScheduler.enrich("timer.check", %{turn_time: @turn_time}, %{}, other.id)
      assert atlas_rows(other.id)[Atlas.Schemas.WorldServiceDriver] == 1
    end

    test "is refused for a world the manager does not have" do
      assert Manager.purge("test_manager_never_created") == {:error, :not_found}
    end

    test "purges Atlas before the disk; a world whose files cannot be deleted is an error, stays listed, " <>
           "and can be purged again" do
      {:ok, world} = create_test_world("manager_purge_disk_fails", mode: :persistent)
      populate!(world.id)
      path = World.Persistence.world_path(world.id)
      assert File.dir?(path)

      # A directory that cannot be written: its files cannot be deleted.
      File.chmod!(path, 0o500)
      on_exit(fn -> File.chmod(path, 0o755) end)

      assert {:error, {:disk_delete_failed, :eacces}} = Manager.purge(world.id)

      # Atlas went first; the world and its files are still there.
      assert Enum.all?(atlas_rows(world.id), fn {_schema, count} -> count == 0 end)
      refute Enum.any?(Atlas.Graph.world_graphs(world.id), &Atlas.Graph.graph_exists?/1)
      assert {:ok, _world} = Manager.get(world.id)
      assert File.dir?(path)

      File.chmod!(path, 0o755)

      assert :ok = Manager.purge(world.id)
      assert Manager.get(world.id) == {:error, :not_found}
      refute File.exists?(path)
    end

    test "raises while queued Atlas writes have not finished, and leaves the world to be purged again" do
      {:ok, world} = create_test_world("manager_purge_drain_fails")
      populate!(world.id)

      # A queued Atlas write that does not finish.
      {:ok, blocked} = Task.Supervisor.start_child(Brain.AtlasTaskSupervisor, fn -> Process.sleep(:infinity) end)
      on_exit(fn -> Process.exit(blocked, :kill) end)

      assert {:error, {:drain_timeout, 1}} = Brain.AtlasIntegration.drain(50)

      assert_raise RuntimeError, ~r/queued Atlas writes did not finish/, fn -> Manager.purge(world.id) end

      assert {:ok, _world} = Manager.get(world.id)
      assert Enum.all?(Atlas.Graph.world_graphs(world.id), &Atlas.Graph.graph_exists?/1)

      Process.exit(blocked, :kill)
      eventually(fn -> Task.Supervisor.children(Brain.AtlasTaskSupervisor) end, &(&1 == []), 50, 20)

      assert :ok = Manager.purge(world.id)
      assert Manager.get(world.id) == {:error, :not_found}
      assert Enum.all?(atlas_rows(world.id), fn {_schema, count} -> count == 0 end)
    end
  end

  describe "an ephemeral world's knowledge" do
    test "is held in memory, never on disk, and goes with the world" do
      {:ok, world} = create_test_world("manager_ephemeral_knowledge")

      assert :ok = Brain.KnowledgeStore.add_to_world(world.id, "people", "zed", %{"name" => "Zed"})
      assert %{"zed" => %{"name" => "Zed"}} = Brain.KnowledgeStore.get_world_knowledge(world.id, "people")
      refute File.exists?(World.Persistence.world_path(world.id))

      assert :ok = Manager.destroy(world.id)
      assert Brain.KnowledgeStore.get_world_knowledge(world.id) == %{}
      refute File.exists?(World.Persistence.world_path(world.id))
    end

    test "a persistent world's is its knowledge.json" do
      {:ok, world} = create_test_world("manager_persistent_knowledge", mode: :persistent)

      assert :ok = Brain.KnowledgeStore.add_to_world(world.id, "people", "zed", %{"name" => "Zed"})
      assert File.exists?(Path.join(World.Persistence.world_path(world.id), "knowledge.json"))
    end
  end

  describe "destroy/1" do
    test "keeps the world's graphs and what it holds in Atlas" do
      {:ok, world} = create_test_world("manager_destroyed_keeps")
      populate!(world.id)
      before = atlas_rows(world.id)

      assert :ok = Manager.destroy(world.id)

      assert Enum.all?(Atlas.Graph.world_graphs(world.id), &Atlas.Graph.graph_exists?/1)
      assert {:ok, 1} = Atlas.Graph.count_nodes(Atlas.Graph.world_graph("knowledge_graph", world.id))
      assert atlas_rows(world.id) == before
    end
  end

  describe "while unscoped graphs exist" do
    # The unscoped graph is created inside this test's sandbox transaction, so
    # it is gone when the test ends.
    test "creating or loading a world raises in the caller, and the manager and its worlds keep running" do
      {:ok, world} = create_test_world("manager_unscoped_kept")
      assert {:ok, embedding} = World.Embedder.embed(world.id, "hello there")
      manager = Process.whereis(Manager)

      Atlas.Repo.query!("SELECT ag_catalog.create_graph('knowledge_graph')", [])

      assert_raise ArgumentError, ~r/unscoped graphs \["knowledge_graph"\].*mix atlas.move_graphs_to_world/, fn ->
        Manager.create("test_manager_unscoped_refused", id: "test_manager_unscoped_refused")
      end

      assert_raise ArgumentError, ~r/unscoped graphs \["knowledge_graph"\]/, fn -> Manager.load_world(world.id) end
      assert_raise ArgumentError, ~r/unscoped graphs \["knowledge_graph"\]/, fn -> Manager.reload_persisted_worlds() end

      assert Process.whereis(Manager) == manager
      assert Manager.get("test_manager_unscoped_refused") == {:error, :not_found}
      assert {:ok, %{id: id}} = Manager.get(world.id)
      assert id == world.id
      assert World.Embedder.embed(world.id, "hello there") == {:ok, embedding}
    end
  end

  describe "get_candidates/2" do
    test "is empty for a new world" do
      {:ok, world} = create_test_world("manager_candidates")

      assert Manager.get_candidates(world.id) == []
    end
  end

  describe "test_world_id?/1" do
    test "is true only for ids that start with test_" do
      assert Manager.test_world_id?("test_anything")
      assert Manager.test_world_id?("test_")
      refute Manager.test_world_id?("default")
      refute Manager.test_world_id?("realworld")
      refute Manager.test_world_id?("my_test_world")
      refute Manager.test_world_id?("testworld")
    end
  end

  describe "cleanup_orphaned_worlds/1" do
    # The cleanup runs against a directory of this test's own: with
    # `:test_world_sandbox` off, `World.Persistence.base_path/0` returns
    # `:training_worlds_path`, which points there for the duration.
    setup do
      base_path =
        Path.join(System.tmp_dir!(), "manager_cleanup_test_#{System.unique_integer([:positive])}")

      File.mkdir_p!(base_path)

      original_sandbox = Application.fetch_env(:world, :test_world_sandbox)
      original_path = Application.fetch_env(:world, :training_worlds_path)

      Application.put_env(:world, :test_world_sandbox, false)
      Application.put_env(:world, :training_worlds_path, base_path)

      on_exit(fn ->
        restore_env(:test_world_sandbox, original_sandbox)
        restore_env(:training_worlds_path, original_path)
        File.rm_rf!(base_path)
      end)

      assert World.Persistence.base_path() == base_path

      u = System.unique_integer([:positive])

      dirs = %{
        real_no_config: "realworld_#{u}",
        real_with_config: "realconfigured_#{u}",
        test_no_config: "test_orphan_#{u}",
        test_with_config: "test_configured_#{u}"
      }

      for {kind, id} <- dirs do
        dir = Path.join(base_path, id)
        File.mkdir_p!(dir)

        if kind in [:real_with_config, :test_with_config] do
          File.write!(Path.join(dir, "config.json"), "{}")
        end

        # Two days old: past any cutoff these tests use.
        File.touch!(dir, System.os_time(:second) - 48 * 3600)
      end

      %{base_path: base_path, dirs: dirs}
    end

    test "deletes old test worlds and never a real world", %{base_path: base_path, dirs: dirs} do
      assert {:ok, 2} = Manager.cleanup_orphaned_worlds(dry_run: false, max_age_hours: 0)

      assert File.dir?(Path.join(base_path, dirs.real_no_config))
      assert File.dir?(Path.join(base_path, dirs.real_with_config))
      refute File.exists?(Path.join(base_path, dirs.test_no_config))
      refute File.exists?(Path.join(base_path, dirs.test_with_config))
    end

    test "a dry run names only the test worlds and deletes nothing", %{base_path: base_path, dirs: dirs} do
      assert {:ok, {:would_delete, would_delete}} =
               Manager.cleanup_orphaned_worlds(dry_run: true, max_age_hours: 0)

      assert Enum.sort(would_delete) == Enum.sort([dirs.test_no_config, dirs.test_with_config])

      for {_kind, id} <- dirs do
        assert File.dir?(Path.join(base_path, id))
      end
    end

    test "leaves a test world that is loaded", %{base_path: base_path, dirs: dirs} do
      {:ok, world} = create_test_world("manager_cleanup_loaded")
      loaded_dir = Path.join(base_path, world.id)
      File.mkdir_p!(loaded_dir)
      File.touch!(loaded_dir, System.os_time(:second) - 48 * 3600)

      assert {:ok, 2} = Manager.cleanup_orphaned_worlds(dry_run: false, max_age_hours: 0)

      assert File.dir?(loaded_dir)
      assert File.dir?(Path.join(base_path, dirs.real_no_config))
      refute File.exists?(Path.join(base_path, dirs.test_no_config))
    end
  end

  defp restore_env(key, {:ok, value}), do: Application.put_env(:world, key, value)
  defp restore_env(key, :error), do: Application.delete_env(:world, key)
end
