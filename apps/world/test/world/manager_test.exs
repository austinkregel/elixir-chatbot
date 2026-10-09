defmodule World.ManagerTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

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
    setup tags do
      # The graphs are created inside this test's sandbox transaction.
      owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
      on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
      :ok
    end

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
