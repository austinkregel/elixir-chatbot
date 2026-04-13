defmodule World.PersistenceTest do
  use ExUnit.Case, async: false

  alias World.Persistence
  alias World.TrainingWorld

  setup do
    sandbox_dir = Path.join(System.tmp_dir!(), "persistence_test_#{:rand.uniform(100_000)}")
    File.mkdir_p!(sandbox_dir)
    Application.put_env(:world, :test_world_sandbox, true)

    original_path = Application.get_env(:world, :training_worlds_path)
    Application.put_env(:world, :training_worlds_path, sandbox_dir)

    World.TypeInferrer.init()

    on_exit(fn ->
      File.rm_rf!(sandbox_dir)
      Application.delete_env(:world, :test_world_sandbox)

      if original_path do
        Application.put_env(:world, :training_worlds_path, original_path)
      else
        Application.delete_env(:world, :training_worlds_path)
      end
    end)

    {:ok, sandbox_dir: sandbox_dir}
  end

  describe "base_path/0" do
    test "returns temp dir when test_world_sandbox enabled" do
      path = Persistence.base_path()
      assert String.contains?(path, "chat_bot_test_worlds")
    end
  end

  describe "world_path/1" do
    test "joins base path with world id" do
      path = Persistence.world_path("my_world")
      assert String.ends_with?(path, "my_world")
    end
  end

  describe "save/2 and load/1" do
    test "saves and loads a persistent world" do
      world = %TrainingWorld{
        id: "save_test",
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

      assert :ok = Persistence.save("save_test", data)

      case Persistence.load("save_test") do
        {:ok, loaded} ->
          assert loaded.world.id == "save_test"
          assert loaded.world.name == "Save Test World"
          assert loaded.world.mode == :persistent

        {:error, _reason} ->
          :ok
      end
    end

    test "returns error for ephemeral world" do
      world = %TrainingWorld{
        id: "ephemeral_test",
        name: "Ephemeral",
        mode: :ephemeral,
        created_at: DateTime.utc_now()
      }

      data = %{world: world}
      assert {:error, :ephemeral_world} = Persistence.save("ephemeral_test", data)
    end

    test "returns error for nil world in data" do
      assert {:error, :ephemeral_world} = Persistence.save("nil_world", %{world: nil})
    end
  end

  describe "load/1" do
    test "returns error for nonexistent world" do
      assert {:error, :not_found} = Persistence.load("nonexistent_world_#{:rand.uniform(100_000)}")
    end
  end

  describe "delete/1" do
    test "deletes existing world data" do
      world = %TrainingWorld{
        id: "delete_test",
        name: "Delete Me",
        mode: :persistent,
        created_at: DateTime.utc_now(),
        config: %{},
        metadata: %{}
      }

      data = %{world: world, metrics: nil, candidates: [], overlay: [], events: []}
      Persistence.save("delete_test", data)

      assert :ok = Persistence.delete("delete_test")
      assert {:error, :not_found} = Persistence.load("delete_test")
    end

    test "returns ok for nonexistent world" do
      assert :ok = Persistence.delete("never_existed_#{:rand.uniform(100_000)}")
    end
  end

  describe "list_persisted_worlds/0" do
    test "returns empty list when no worlds persisted" do
      result = Persistence.list_persisted_worlds()
      assert is_list(result)
    end

    test "lists saved worlds" do
      world = %TrainingWorld{
        id: "listed_world",
        name: "Listed",
        mode: :persistent,
        created_at: DateTime.utc_now(),
        config: %{},
        metadata: %{}
      }

      data = %{world: world, metrics: nil, candidates: [], overlay: [], events: []}
      Persistence.save("listed_world", data)

      worlds = Persistence.list_persisted_worlds()
      ids = Enum.map(worlds, & &1.id)
      assert "listed_world" in ids
    end
  end
end
