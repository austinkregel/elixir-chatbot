defmodule Mix.Tasks.CleanupWorldsTest do
  @moduledoc """
  `mix cleanup_worlds` only ever deletes a test world's directory -- one whose
  id starts with `test_` -- and never a real world's, whatever its age or
  contents.

  The task runs against a directory of this test's own: with
  `:test_world_sandbox` off, `Brain.Worlds.base_path/0` returns
  `:training_worlds_path`, which points there for the duration. The directory
  exists, so the task returns instead of halting the VM.
  """
  use ExUnit.Case, async: false

  alias Mix.Tasks.CleanupWorlds

  setup do
    base_path =
      Path.join(System.tmp_dir!(), "cleanup_worlds_task_test_#{System.unique_integer([:positive])}")

    File.mkdir_p!(base_path)

    original_sandbox = Application.fetch_env(:brain, :test_world_sandbox)
    original_path = Application.fetch_env(:brain, :training_worlds_path)

    Application.put_env(:brain, :test_world_sandbox, false)
    Application.put_env(:brain, :training_worlds_path, base_path)

    previous_shell = Mix.shell()
    Mix.shell(Mix.Shell.Process)

    on_exit(fn ->
      Mix.shell(previous_shell)
      restore_env(:test_world_sandbox, original_sandbox)
      restore_env(:training_worlds_path, original_path)
      File.rm_rf!(base_path)
    end)

    assert Brain.Worlds.base_path() == base_path

    u = System.unique_integer([:positive])
    real_id = "realworld_#{u}"
    test_id = "test_orphan_#{u}"

    for id <- [real_id, test_id] do
      dir = Path.join(base_path, id)
      File.mkdir_p!(dir)
      # Two days old and without config.json: orphaned by every rule the task
      # has, were it a test world.
      File.touch!(dir, System.os_time(:second) - 48 * 3600)
    end

    %{base_path: base_path, real_id: real_id, test_id: test_id}
  end

  test "--delete removes the old test world and keeps the real one", ctx do
    CleanupWorlds.run(["--delete", "--max-age", "0"])

    assert File.dir?(Path.join(ctx.base_path, ctx.real_id))
    refute File.exists?(Path.join(ctx.base_path, ctx.test_id))

    real_line = "  #{ctx.real_id} (real world, never deleted)"
    deleted_line = "  Deleted: #{ctx.test_id}"
    assert_received {:mix_shell, :info, [^real_line]}
    assert_received {:mix_shell, :info, [^deleted_line]}
    assert_received {:mix_shell, :info, ["Deleted 1 directories"]}
  end

  test "a dry run deletes nothing", ctx do
    CleanupWorlds.run(["--max-age", "0"])

    assert File.dir?(Path.join(ctx.base_path, ctx.real_id))
    assert File.dir?(Path.join(ctx.base_path, ctx.test_id))
    assert_received {:mix_shell, :info, ["Would delete 1 directories"]}
  end

  defp restore_env(key, {:ok, value}), do: Application.put_env(:brain, key, value)
  defp restore_env(key, :error), do: Application.delete_env(:brain, key)
end
