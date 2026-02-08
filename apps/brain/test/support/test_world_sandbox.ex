defmodule Brain.TestWorldSandbox do
  @moduledoc "Sandbox module for managing test world lifecycle.\n\nProvides automatic cleanup of test worlds to ensure complete isolation\nbetween tests and prevent pollution of production world data.\n\n## Usage\n\nIn your test module:\n\n    defmodule MyTest do\n      use ExUnit.Case\n      import Brain.TestHelpers\n\n      setup do\n        setup_world_sandbox()\n      end\n\n      test \"creates a test world\" do\n        {:ok, world} = create_test_world(\"my_test\")\n        assert world.metadata.test == true\n        # World is automatically cleaned up after test\n      end\n    end\n\n## How It Works\n\n1. `setup_world_sandbox/0` registers an `on_exit` callback that cleans up\n   all worlds created during the test.\n\n2. `create_test_world/2` creates a world with `:test` metadata and tracks\n   it for cleanup.\n\n3. When the test completes (pass or fail), all tracked worlds are destroyed.\n\n4. Test worlds are stored in a temp directory (configured via `:test_world_sandbox`)\n   to ensure complete isolation from production worlds.\n"

  alias World.Manager, as: WorldManager

  alias ExUnit.Callbacks
  @ets_table :test_world_sandbox

  @doc "Sets up the test world sandbox for the current test.\n\nCall this in your test's `setup` block. Returns an `on_exit` callback\nthat will clean up all worlds created during the test.\n\n## Example\n\n    setup do\n      setup_world_sandbox()\n    end\n"
  def setup_world_sandbox do
    ensure_ets_table()
    test_pid = self()
    :ets.insert(@ets_table, {test_pid, []})

    Callbacks.on_exit(fn ->
      cleanup_worlds_for_pid(test_pid)
    end)

    :ok
  end

  @doc "Creates a test world with automatic cleanup tracking.\n\nThe world is created with `:test` metadata and will be automatically\ndestroyed when the test completes.\n\n## Options\n\nAll options are passed to `WorldManager.create/2`, plus:\n  - `:mode` - Defaults to `:ephemeral` for tests (can override to `:persistent`)\n\n## Examples\n\n    {:ok, world} = create_test_world(\"my_feature\")\n    {:ok, world} = create_test_world(\"persistent_test\", mode: :persistent)\n"
  def create_test_world(name, opts \\ []) do
    test_pid = self()
    opts = Keyword.put_new(opts, :mode, :ephemeral)
    metadata = Keyword.get(opts, :metadata, %{})
    metadata = Map.put(metadata, :test, true)
    metadata = Map.put(metadata, :test_pid, inspect(test_pid))
    opts = Keyword.put(opts, :metadata, metadata)

    case WorldManager.create(name, opts) do
      {:ok, world} ->
        track_world(test_pid, world.id)
        {:ok, world}

      error ->
        error
    end
  end

  @doc "Manually cleans up all worlds created by the current test.\n\nThis is called automatically by the `on_exit` callback, but can be\ncalled manually if needed.\n"
  def cleanup do
    cleanup_worlds_for_pid(self())
  end

  @doc "Cleans up all test worlds across all tests.\n\nUse this for global cleanup at the end of a test suite.\n"
  def cleanup_all do
    ensure_ets_table()

    try do
      :ets.tab2list(@ets_table)
      |> Enum.each(fn {pid, world_ids} ->
        Enum.each(world_ids, &destroy_world_safely/1)
        :ets.delete(@ets_table, pid)
      end)
    rescue
      _ -> :ok
    end

    cleanup_test_directory()
    cleanup_orphaned_test_worlds()
  end

  defp ensure_ets_table do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table])
    end
  rescue
    ArgumentError ->
      :ok
  end

  defp track_world(test_pid, world_id) do
    ensure_ets_table()

    case :ets.lookup(@ets_table, test_pid) do
      [{^test_pid, world_ids}] ->
        :ets.insert(@ets_table, {test_pid, [world_id | world_ids]})

      [] ->
        :ets.insert(@ets_table, {test_pid, [world_id]})
    end
  rescue
    _ -> :ok
  end

  defp cleanup_worlds_for_pid(test_pid) do
    ensure_ets_table()

    try do
      case :ets.lookup(@ets_table, test_pid) do
        [{^test_pid, world_ids}] ->
          Enum.each(world_ids, &destroy_world_safely/1)
          :ets.delete(@ets_table, test_pid)

        [] ->
          :ok
      end
    rescue
      _ -> :ok
    end
  end

  defp destroy_world_safely(world_id) do
    try do
      WorldManager.destroy(world_id)
    rescue
      _ -> :ok
    catch
      :exit, _ -> :ok
    end
  end

  defp cleanup_test_directory do
    test_path = Path.join(System.tmp_dir!(), "chat_bot_test_worlds")

    if File.exists?(test_path) do
      File.rm_rf(test_path)
    end
  rescue
    _ -> :ok
  end

  @doc "Cleans up orphaned test worlds that were created in priv/training_worlds/.\n\nThis handles edge cases where tests:\n- Created worlds directly via WorldManager instead of create_test_world\n- Failed before cleanup could run\n- Used hardcoded paths that bypassed the test sandbox\n\nA world is considered a test world if:\n- Its ID starts with \"test_world_\"\n- Its config.json contains \"test\": true in metadata\n- It has no config.json (incomplete world from crashed test)\n"
  def cleanup_orphaned_test_worlds do
    prod_path = "priv/training_worlds"

    if File.dir?(prod_path) do
      prod_path
      |> File.ls!()
      |> Enum.filter(&is_test_world?(&1, prod_path))
      |> Enum.each(fn world_id ->
        world_path = Path.join(prod_path, world_id)
        File.rm_rf(world_path)
      end)
    end
  rescue
    _ -> :ok
  end

  defp is_test_world?(world_id, base_path) do
    if String.starts_with?(world_id, "test_world_") do
      true
    else
      config_path = Path.join([base_path, world_id, "config.json"])

      if File.exists?(config_path) do
        try do
          config = File.read!(config_path) |> Jason.decode!()
          metadata = config["metadata"] || %{}
          metadata["test"] == true
        rescue
          _ -> false
        end
      else
        world_path = Path.join(base_path, world_id)

        has_only_knowledge =
          File.dir?(world_path) and
            File.ls!(world_path) == ["knowledge.json"]

        has_only_knowledge
      end
    end
  end
end