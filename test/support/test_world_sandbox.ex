defmodule ChatBot.TestWorldSandbox do
  @moduledoc """
  Sandbox module for managing test world lifecycle.

  Provides automatic cleanup of test worlds to ensure complete isolation
  between tests and prevent pollution of production world data.

  ## Usage

  In your test module:

      defmodule MyTest do
        use ExUnit.Case
        import ChatBot.TestWorldSandbox

        setup do
          setup_world_sandbox()
        end

        test "creates a test world" do
          {:ok, world} = create_test_world("my_test")
          assert world.metadata.test == true
          # World is automatically cleaned up after test
        end
      end

  ## How It Works

  1. `setup_world_sandbox/0` registers an `on_exit` callback that cleans up
     all worlds created during the test.

  2. `create_test_world/2` creates a world with `:test` metadata and tracks
     it for cleanup.

  3. When the test completes (pass or fail), all tracked worlds are destroyed.

  4. Test worlds are stored in a temp directory (configured via `:test_world_sandbox`)
     to ensure complete isolation from production worlds.
  """

  alias ChatBot.Learning.WorldManager

  @ets_table :test_world_sandbox

  # ============================================================================
  # Setup Functions
  # ============================================================================

  @doc """
  Sets up the test world sandbox for the current test.

  Call this in your test's `setup` block. Returns an `on_exit` callback
  that will clean up all worlds created during the test.

  ## Example

      setup do
        setup_world_sandbox()
      end
  """
  def setup_world_sandbox do
    ensure_ets_table()
    test_pid = self()

    # Register this test process
    :ets.insert(@ets_table, {test_pid, []})

    # Return on_exit callback for ExUnit
    ExUnit.Callbacks.on_exit(fn ->
      cleanup_worlds_for_pid(test_pid)
    end)

    :ok
  end

  # ============================================================================
  # World Creation
  # ============================================================================

  @doc """
  Creates a test world with automatic cleanup tracking.

  The world is created with `:test` metadata and will be automatically
  destroyed when the test completes.

  ## Options

  All options are passed to `WorldManager.create/2`, plus:
    - `:mode` - Defaults to `:ephemeral` for tests (can override to `:persistent`)

  ## Examples

      {:ok, world} = create_test_world("my_feature")
      {:ok, world} = create_test_world("persistent_test", mode: :persistent)
  """
  def create_test_world(name, opts \\ []) do
    test_pid = self()

    # Default to ephemeral mode for tests, but allow override
    opts = Keyword.put_new(opts, :mode, :ephemeral)

    # Add test metadata
    metadata = Keyword.get(opts, :metadata, %{})
    metadata = Map.put(metadata, :test, true)
    metadata = Map.put(metadata, :test_pid, inspect(test_pid))
    opts = Keyword.put(opts, :metadata, metadata)

    # Create the world
    case WorldManager.create(name, opts) do
      {:ok, world} ->
        # Track for cleanup
        track_world(test_pid, world.id)
        {:ok, world}

      error ->
        error
    end
  end

  @doc """
  Manually cleans up all worlds created by the current test.

  This is called automatically by the `on_exit` callback, but can be
  called manually if needed.
  """
  def cleanup do
    cleanup_worlds_for_pid(self())
  end

  @doc """
  Cleans up all test worlds across all tests.

  Use this for global cleanup at the end of a test suite.
  """
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

    # Clean up the test worlds temp directory
    cleanup_test_directory()

    # Also clean up any orphaned test worlds in the production directory
    # This handles worlds created by tests that bypassed the sandbox
    cleanup_orphaned_test_worlds()
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp ensure_ets_table do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table])
    end
  rescue
    ArgumentError ->
      # Table already exists
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

  @doc """
  Cleans up orphaned test worlds that were created in priv/training_worlds/.

  This handles edge cases where tests:
  - Created worlds directly via WorldManager instead of create_test_world
  - Failed before cleanup could run
  - Used hardcoded paths that bypassed the test sandbox

  A world is considered a test world if:
  - Its ID starts with "test_world_"
  - Its config.json contains "test": true in metadata
  - It has no config.json (incomplete world from crashed test)
  """
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
    # Check for explicit test world naming convention
    if String.starts_with?(world_id, "test_world_") do
      true
    else
      # Check for test metadata in config.json
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
        # World has no config - might be from crashed test
        # Only delete if it looks like a test ID (short random string with only knowledge.json)
        world_path = Path.join(base_path, world_id)

        has_only_knowledge =
          File.dir?(world_path) and
            File.ls!(world_path) == ["knowledge.json"]

        has_only_knowledge
      end
    end
  end
end
