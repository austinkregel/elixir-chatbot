defmodule World.PersistenceTest do
  use ExUnit.Case, async: false

  alias World.Persistence
  alias World.TrainingWorld

  # Every world here is written to a directory of this test's own. With
  # `:test_world_sandbox` on, `Persistence.base_path/0` returns the shared test
  # world directory -- where the suite's "default" world lives -- and ignores
  # `:training_worlds_path`, so the sandbox is turned off for the duration and
  # `:training_worlds_path` points at the private directory instead.
  setup do
    sandbox_dir = Path.join(System.tmp_dir!(), "persistence_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(sandbox_dir)

    original_sandbox = Application.fetch_env(:world, :test_world_sandbox)
    original_path = Application.fetch_env(:world, :training_worlds_path)

    Application.put_env(:world, :test_world_sandbox, false)
    Application.put_env(:world, :training_worlds_path, sandbox_dir)

    World.TypeInferrer.init()

    on_exit(fn ->
      File.rm_rf!(sandbox_dir)
      restore_env(:test_world_sandbox, original_sandbox)
      restore_env(:training_worlds_path, original_path)
    end)

    {:ok, sandbox_dir: sandbox_dir}
  end

  defp restore_env(key, {:ok, value}), do: Application.put_env(:world, key, value)
  defp restore_env(key, :error), do: Application.delete_env(:world, key)

  describe "base_path/0" do
    test "returns temp dir when test_world_sandbox enabled" do
      Application.put_env(:world, :test_world_sandbox, true)

      path = Persistence.base_path()
      assert String.contains?(path, "chat_bot_test_worlds")
    end

    test "returns training_worlds_path when test_world_sandbox is off", %{sandbox_dir: sandbox_dir} do
      assert Persistence.base_path() == sandbox_dir
    end
  end

  describe "world_path/1" do
    test "joins base path with world id" do
      path = Persistence.world_path("test_my_world")
      assert String.ends_with?(path, "test_my_world")
    end
  end

  describe "save/2 and load/1" do
    test "saves and loads a persistent world", %{sandbox_dir: sandbox_dir} do
      world = %TrainingWorld{
        id: "test_save",
        name: "Save Test World",
        mode: :persistent,
        base_world: "default",
        created_at: DateTime.utc_now(),
        config: %{},
        metadata: %{}
      }

      data = %{
        world: world,
        metrics: nil,
        candidates: [],
        overlay: [],
        events: []
      }

      assert :ok = Persistence.save("test_save", data)
      assert File.dir?(Path.join(sandbox_dir, "test_save"))

      assert {:ok, loaded} = Persistence.load("test_save")
      assert loaded.world.id == "test_save"
      assert loaded.world.name == "Save Test World"
      assert loaded.world.mode == :persistent
    end

    test "returns error for ephemeral world" do
      world = %TrainingWorld{
        id: "test_ephemeral",
        name: "Ephemeral",
        mode: :ephemeral,
        created_at: DateTime.utc_now()
      }

      data = %{world: world}
      assert {:error, :ephemeral_world} = Persistence.save("test_ephemeral", data)
    end

    test "returns error for nil world in data" do
      assert {:error, :ephemeral_world} = Persistence.save("test_nil_world", %{world: nil})
    end
  end

  describe "load/1" do
    test "returns error for nonexistent world" do
      assert {:error, :not_found} =
               Persistence.load("test_nonexistent_world_#{System.unique_integer([:positive])}")
    end
  end

  describe "delete/1" do
    test "deletes existing world data" do
      world = %TrainingWorld{
        id: "test_delete",
        name: "Delete Me",
        mode: :persistent,
        created_at: DateTime.utc_now(),
        config: %{},
        metadata: %{}
      }

      data = %{world: world, metrics: nil, candidates: [], overlay: [], events: []}
      Persistence.save("test_delete", data)

      assert :ok = Persistence.delete("test_delete")
      assert {:error, :not_found} = Persistence.load("test_delete")
    end

    test "returns ok for nonexistent world" do
      assert :ok = Persistence.delete("test_never_existed_#{System.unique_integer([:positive])}")
    end
  end

  describe "list_persisted_worlds/0" do
    test "returns empty list when no worlds persisted" do
      assert Persistence.list_persisted_worlds() == []
    end

    test "lists saved worlds" do
      world = %TrainingWorld{
        id: "test_listed_world",
        name: "Listed",
        mode: :persistent,
        created_at: DateTime.utc_now(),
        config: %{},
        metadata: %{}
      }

      data = %{world: world, metrics: nil, candidates: [], overlay: [], events: []}
      Persistence.save("test_listed_world", data)

      worlds = Persistence.list_persisted_worlds()
      ids = Enum.map(worlds, & &1.id)
      assert "test_listed_world" in ids
    end
  end
end
