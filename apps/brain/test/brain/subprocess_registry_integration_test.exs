defmodule Brain.SubprocessRegistryIntegrationTest do
  use ExUnit.Case, async: false

  import Brain.TestHelpers

  # Note: The SubprocessRegistry module is a GenServer that wraps an Elixir Registry.
  # The application.ex only starts the Registry, not the GenServer. This test suite
  # tests the Registry functionality directly since the GenServer can't coexist with
  # the Registry using the same name.
  #
  # Elixir's Registry only allows the calling process to register itself, so we spawn
  # subprocesses that register themselves.

  @registry_name Brain.SubprocessRegistry

  setup do
    # Start PubSub first
    ensure_pubsub_started()

    # The SubprocessRegistry is an Elixir Registry (not the GenServer)
    ensure_started({Registry, keys: :unique, name: @registry_name})

    :ok
  end

  # Helper to spawn a process that registers itself with the registry
  defp spawn_and_register(type, subprocess_id) do
    test_pid = self()
    key = {type, subprocess_id}

    pid =
      spawn(fn ->
        case Registry.register(@registry_name, key, nil) do
          {:ok, _} ->
            send(test_pid, {:registered, self()})
            # Keep process alive
            Process.sleep(60_000)

          {:error, {:already_registered, _}} ->
            send(test_pid, {:registration_failed, :already_registered})
            Process.sleep(60_000)
        end
      end)

    receive do
      {:registered, ^pid} -> {:ok, pid}
      {:registration_failed, reason} -> {:error, reason}
    after
      1000 -> {:error, :timeout}
    end
  end

  defp unregister_subprocess(type, subprocess_id) do
    key = {type, subprocess_id}

    case Registry.lookup(@registry_name, key) do
      [{pid, _}] ->
        # Kill the process which auto-unregisters
        Process.exit(pid, :kill)
        :ok

      [] ->
        :ok
    end
  end

  defp get_subprocess(type, subprocess_id) do
    key = {type, subprocess_id}

    case Registry.lookup(@registry_name, key) do
      [{pid, _}] -> {:ok, pid}
      [] -> {:error, :not_found}
    end
  end

  defp list_subprocesses do
    Registry.select(@registry_name, [{{:"$1", :"$2", :"$3"}, [], [{{:"$1", :"$2", :"$3"}}]}])
    |> Enum.map(fn {{type, subprocess_id}, pid, _value} ->
      %{type: type, subprocess_id: subprocess_id, pid: pid}
    end)
  end

  defp list_subprocesses_by_type(type) do
    list_subprocesses()
    |> Enum.filter(fn sp -> sp.type == type end)
  end

  describe "register_subprocess" do
    test "successfully registers a subprocess" do
      {:ok, pid} = spawn_and_register(:test, "subprocess_1")

      on_exit(fn ->
        if Process.alive?(pid), do: Process.exit(pid, :kill)
      end)

      # Verify it's registered
      assert {:ok, ^pid} = get_subprocess(:test, "subprocess_1")
    end

    test "returns error for duplicate registration" do
      {:ok, pid1} = spawn_and_register(:test, "dup_subprocess")

      on_exit(fn ->
        if Process.alive?(pid1), do: Process.exit(pid1, :kill)
      end)

      # Second registration with same type and id should fail
      result = spawn_and_register(:test, "dup_subprocess")

      assert {:error, :already_registered} = result
    end
  end

  describe "unregister_subprocess" do
    test "removes subprocess from registry" do
      {:ok, pid} = spawn_and_register(:test, "to_unregister")

      # Verify it's registered
      assert {:ok, ^pid} = get_subprocess(:test, "to_unregister")

      # Unregister (kills the process)
      unregister_subprocess(:test, "to_unregister")

      # Give it a moment to clean up
      Process.sleep(20)

      # Verify it's gone
      assert {:error, :not_found} = get_subprocess(:test, "to_unregister")
    end
  end

  describe "get_subprocess" do
    test "returns pid for registered subprocess" do
      {:ok, pid} = spawn_and_register(:test, "get_test")

      on_exit(fn ->
        if Process.alive?(pid), do: Process.exit(pid, :kill)
      end)

      result = get_subprocess(:test, "get_test")

      assert {:ok, ^pid} = result
    end

    test "returns error for non-existent subprocess" do
      result = get_subprocess(:test, "nonexistent_12345")

      assert {:error, :not_found} = result
    end
  end

  describe "list_subprocesses" do
    test "list_subprocesses returns all registered" do
      {:ok, pid1} = spawn_and_register(:type_a, "list_test_1")
      {:ok, pid2} = spawn_and_register(:type_b, "list_test_2")

      on_exit(fn ->
        if Process.alive?(pid1), do: Process.exit(pid1, :kill)
        if Process.alive?(pid2), do: Process.exit(pid2, :kill)
      end)

      subprocesses = list_subprocesses()

      assert is_list(subprocesses)
      assert length(subprocesses) >= 2

      ids = Enum.map(subprocesses, & &1.subprocess_id)
      assert "list_test_1" in ids
      assert "list_test_2" in ids
    end

    test "list_subprocesses_by_type filters by type" do
      {:ok, pid1} = spawn_and_register(:filter_type_a, "filter_1")
      {:ok, pid2} = spawn_and_register(:filter_type_a, "filter_2")
      {:ok, pid3} = spawn_and_register(:filter_type_b, "filter_3")

      on_exit(fn ->
        if Process.alive?(pid1), do: Process.exit(pid1, :kill)
        if Process.alive?(pid2), do: Process.exit(pid2, :kill)
        if Process.alive?(pid3), do: Process.exit(pid3, :kill)
      end)

      type_a_list = list_subprocesses_by_type(:filter_type_a)
      type_b_list = list_subprocesses_by_type(:filter_type_b)

      assert length(type_a_list) == 2
      assert length(type_b_list) == 1

      # All type_a entries should have correct type
      Enum.each(type_a_list, fn sp ->
        assert sp.type == :filter_type_a
      end)

      Enum.each(type_b_list, fn sp ->
        assert sp.type == :filter_type_b
      end)
    end
  end

  describe "subprocess info structure" do
    test "returns correct info structure" do
      {:ok, pid} = spawn_and_register(:info_type, "info_test")

      on_exit(fn ->
        if Process.alive?(pid), do: Process.exit(pid, :kill)
      end)

      subprocesses = list_subprocesses_by_type(:info_type)
      sp = hd(subprocesses)

      # Check structure
      assert Map.has_key?(sp, :type)
      assert Map.has_key?(sp, :subprocess_id)
      assert Map.has_key?(sp, :pid)

      assert sp.type == :info_type
      assert sp.subprocess_id == "info_test"
      assert sp.pid == pid
    end
  end
end
