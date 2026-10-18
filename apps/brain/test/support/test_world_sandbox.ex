defmodule Brain.TestWorldSandbox do
  @moduledoc "Sandbox module for managing test world lifecycle.\n\nProvides automatic cleanup of test worlds to ensure complete isolation\nbetween tests and prevent pollution of real world data.\n\n## Usage\n\nIn your test module:\n\n    defmodule MyTest do\n      use ExUnit.Case\n      import Brain.TestHelpers\n\n      setup do\n        setup_world_sandbox()\n      end\n\n      test \"creates a test world\" do\n        {:ok, world} = create_test_world(\"my_test\")\n        assert world.metadata.test == true\n        # World is automatically cleaned up after test\n      end\n    end\n\n## How It Works\n\n1. `setup_world_sandbox/0` checks that worlds are written to the test world\n   directory (`config :brain, :test_world_sandbox`), not the real one.\n\n2. `create_test_world/2` creates a world in brain (`Brain.Worlds.create!/2`)\n   whose id starts with `test_`, with `:test` metadata, stores the local\n   scheduler as its scheduling driver, as a world created to be managed is,\n   and registers an `on_exit` callback that removes it.\n\n3. When the test completes (pass or fail), each world it created is stopped\n   (`Brain.Worlds.stop!/1`), its gazetteer overlay dropped, and its persisted\n   data deleted.\n\n4. Cleanup only ever touches worlds whose id starts with `test_`. Every other\n   world -- `default`, the one the suite runs in, among them -- is a real\n   world and is never deleted.\n\nAn app that manages worlds builds its own test worlds from\n`test_world_options/2` and removes them with `delete_test_world_data!/1`.\n"

  alias Brain.ML.Gazetteer
  alias Brain.Services.{Dispatcher, LocalScheduler}
  alias ExUnit.Callbacks

  @test_world_prefix "test_"
  @brain_options [:id, :mode, :metadata]

  @doc "Sets up the test world sandbox for the current test.\n\nCall this in your test's `setup` block. Raises unless worlds are written to\nthe test world directory, so a test world can never land among real ones.\n\n## Example\n\n    setup do\n      setup_world_sandbox()\n    end\n"
  def setup_world_sandbox do
    unless Application.get_env(:brain, :test_world_sandbox) == true do
      raise "Brain.TestWorldSandbox: :test_world_sandbox is not enabled for :brain; " <>
              "test worlds would be written to #{Brain.Worlds.base_path()}"
    end

    :ok
  end

  @doc """
  The options a test world is created with: `opts` with `:id` set to a unique
  `test_<name>_<unique integer>` id (or the given one, which must start with
  `test_`), `:mode` defaulting to `:ephemeral`, and `:metadata` carrying
  `test: true` and the creating test's pid.
  """
  def test_world_options(name, opts) when is_binary(name) and is_list(opts) do
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

    metadata =
      opts
      |> Keyword.get(:metadata, %{})
      |> Map.put(:test, true)
      |> Map.put(:test_pid, inspect(self()))

    opts
    |> Keyword.put(:id, id)
    |> Keyword.put_new(:mode, :ephemeral)
    |> Keyword.put(:metadata, metadata)
  end

  @doc "Creates a test world in brain with automatic cleanup.\n\nThe world's id is `test_<name>_<unique integer>`, it carries `:test` metadata,\nand an `on_exit` callback stops it and deletes its persisted data when the\ntest completes. Returns `{:ok, %{id: id, name: name, mode: mode, metadata: metadata}}`.\n\n## Options\n\n  - `:mode` - Defaults to `:ephemeral` for tests (can override to `:persistent`)\n  - `:id` - Must start with `test_`; defaults to a unique `test_` id\n  - `:metadata` - Additional metadata\n\nAny other option raises: a world's name, configuration and base are kept by\nwhatever manages worlds, not by brain.\n\n## Examples\n\n    {:ok, world} = create_test_world(\"my_feature\")\n    {:ok, world} = create_test_world(\"persistent_test\", mode: :persistent)\n"
  def create_test_world(name, opts \\ []) do
    case Keyword.keys(opts) -- @brain_options do
      [] ->
        :ok

      unknown ->
        raise ArgumentError,
              "create_test_world: brain creates a world from #{inspect(@brain_options)}, got #{inspect(unknown)}"
    end

    opts = test_world_options(name, opts)
    id = Keyword.fetch!(opts, :id)
    mode = Keyword.fetch!(opts, :mode)

    :ok = Brain.Worlds.create!(id, mode)
    Callbacks.on_exit({__MODULE__, id}, fn -> remove_test_world!(id) end)
    :ok = Dispatcher.assign_scheduling_driver(id, LocalScheduler)

    {:ok, %{id: id, name: name, mode: mode, metadata: Keyword.fetch!(opts, :metadata)}}
  end

  @doc "Deletes everything persisted under a test world's directory.\n\nFor a test that writes into a world directory without creating the world\nthrough `create_test_world/2`. Raises for an id that does not start with\n`test_`.\n"
  def delete_test_world_data!(world_id) do
    refuse_unless_test_world!(world_id)
    File.rm_rf!(Brain.Worlds.world_path(world_id))
    :ok
  end

  @doc "Raises `ArgumentError` for an id that is not a test world's (one starting with `test_`)."
  def refuse_unless_test_world!(world_id) do
    unless test_world?(world_id) do
      raise ArgumentError,
            "Brain.TestWorldSandbox: refusing to remove #{inspect(world_id)}, which is not a test world"
    end

    :ok
  end

  # Stops the world's tree -- a test may already have done that itself --
  # drops its gazetteer overlay, and deletes whatever it persisted. Refuses
  # any id that is not a test world's.
  defp remove_test_world!(world_id) do
    refuse_unless_test_world!(world_id)
    :ok = Brain.Worlds.stop!(world_id)
    Gazetteer.destroy_world_overlay(world_id)
    delete_test_world_data!(world_id)
  end

  defp test_world?(world_id) when is_binary(world_id) do
    String.starts_with?(world_id, @test_world_prefix)
  end
end
