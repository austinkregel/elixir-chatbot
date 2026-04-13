defmodule World.ManagerTest do
  use ExUnit.Case, async: false

  alias World.Manager

  describe "ready?/0" do
    test "returns boolean" do
      if Process.whereis(World.Manager) do
        result = Manager.ready?()
        assert is_boolean(result)
      else
        refute Manager.ready?()
      end
    end
  end

  describe "list_worlds/0" do
    test "returns a list" do
      if Process.whereis(World.Manager) do
        result = Manager.list_worlds()
        assert is_list(result)
      end
    end
  end

  describe "world operations" do
    setup do
      if Process.whereis(World.Manager) do
        :ok
      else
        name = :"manager_test_#{:rand.uniform(100_000)}"

        case Manager.start_link(name: name) do
          {:ok, pid} ->
            on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)
            {:ok, name: name}

          {:error, {:already_started, _}} ->
            {:ok, name: World.Manager}
        end
      end
    end

    test "get_candidates returns a list" do
      if Process.whereis(World.Manager) do
        result = Manager.get_candidates("default")
        assert is_list(result)
      end
    end
  end
end
