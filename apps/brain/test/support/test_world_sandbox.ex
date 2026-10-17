defmodule Brain.TestWorldSandbox do
  @moduledoc "Sandbox module for managing test world lifecycle.\n\nProvides automatic cleanup of test worlds to ensure complete isolation\nbetween tests and prevent pollution of real world data.\n\n## Usage\n\nIn your test module:\n\n    defmodule MyTest do\n      use ExUnit.Case\n      import Brain.TestHelpers\n\n      setup do\n        setup_world_sandbox()\n      end\n\n      test \"creates a test world\" do\n        {:ok, world} = create_test_world(\"my_test\")\n        assert world.metadata.test == true\n        # World is automatically cleaned up after test\n      end\n    end\n\n## How It Works\n\n1. `setup_world_sandbox/0` checks that worlds are written to the test world\n   directory (`:test_world_sandbox`), not the real one.\n\n2. `create_test_world/2` creates a world whose id starts with `test_`, with\n   `:test` metadata, and registers an `on_exit` callback that removes it.\n\n3. When the test completes (pass or fail), each world it created is destroyed\n   and its persisted data deleted.\n\n4. Cleanup only ever touches worlds whose id starts with `test_`. Every other\n   world -- `default`, the one the suite runs in, among them -- is a real\n   world and is never deleted.\n"

  # World.Manager is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, [World.Manager, World.Persistence]}

  alias World.Manager, as: WorldManager

  alias ExUnit.Callbacks

  @test_world_prefix "test_"

  @doc "Sets up the test world sandbox for the current test.\n\nCall this in your test's `setup` block. Raises unless worlds are written to\nthe test world directory, so a test world can never land among real ones.\n\n## Example\n\n    setup do\n      setup_world_sandbox()\n    end\n"
  def setup_world_sandbox do
    unless Application.get_env(:world, :test_world_sandbox) == true do
      raise "Brain.TestWorldSandbox: :test_world_sandbox is not enabled for :world; " <>
              "test worlds would be written to #{World.Persistence.base_path()}"
    end

    :ok
  end

  @doc "Creates a test world with automatic cleanup.\n\nThe world's id is `test_<name>_<unique integer>`, it carries `:test` metadata,\nand an `on_exit` callback destroys it and deletes its persisted data when the\ntest completes.\n\n## Options\n\nAll options are passed to `WorldManager.create/2`, plus:\n  - `:mode` - Defaults to `:ephemeral` for tests (can override to `:persistent`)\n  - `:id` - Must start with `test_`; defaults to a unique `test_` id\n\n## Examples\n\n    {:ok, world} = create_test_world(\"my_feature\")\n    {:ok, world} = create_test_world(\"persistent_test\", mode: :persistent)\n"
  def create_test_world(name, opts \\ []) do
    test_pid = self()

    id =
      case Keyword.fetch(opts, :id) do
        {:ok, id} ->
          unless test_world?(id) do
            raise ArgumentError,
                  "create_test_world: id #{inspect(id)} must start with #{inspect(@test_world_prefix)}"
          end

          id

        :error ->
          "#{@test_world_prefix}#{name}_#{System.unique_integer([:positive])}"
      end

    opts = Keyword.put(opts, :id, id)
    opts = Keyword.put_new(opts, :mode, :ephemeral)
    metadata = Keyword.get(opts, :metadata, %{})
    metadata = Map.put(metadata, :test, true)
    metadata = Map.put(metadata, :test_pid, inspect(test_pid))
    opts = Keyword.put(opts, :metadata, metadata)

    case WorldManager.create(name, opts) do
      {:ok, world} ->
        Callbacks.on_exit({__MODULE__, world.id}, fn -> remove_test_world!(world.id) end)
        {:ok, world}

      error ->
        error
    end
  end

  @doc "Deletes everything persisted under a test world's directory.\n\nFor a test that writes into a world directory without creating the world\nthrough `create_test_world/2`. Raises for an id that does not start with\n`test_`.\n"
  def delete_test_world_data!(world_id) do
    refuse_unless_test_world!(world_id)
    :ok = World.Persistence.delete(world_id)
  end

  # Destroys the world's runtime state -- a test may already have done that
  # itself -- and deletes whatever it persisted. Refuses any id that is not a
  # test world's.
  defp remove_test_world!(world_id) do
    refuse_unless_test_world!(world_id)

    case WorldManager.destroy(world_id) do
      :ok -> :ok
      {:error, :not_found} -> :ok
    end

    delete_test_world_data!(world_id)
  end

  defp refuse_unless_test_world!(world_id) do
    unless test_world?(world_id) do
      raise ArgumentError,
            "Brain.TestWorldSandbox: refusing to remove #{inspect(world_id)}, which is not a test world"
    end
  end

  defp test_world?(world_id) when is_binary(world_id) do
    String.starts_with?(world_id, @test_world_prefix)
  end
end
