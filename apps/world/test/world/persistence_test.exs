defmodule World.PersistenceTest do
  use ExUnit.Case, async: false

  alias World.Persistence
  alias World.TrainingWorld

  # Every world here is written to a directory of this test's own. With
  # `:test_world_sandbox` on, `Brain.Worlds.base_path/0` returns the shared test
  # world directory -- where the suite's "default" world lives -- and ignores
  # `:training_worlds_path`, so the sandbox is turned off for the duration and
  # `:training_worlds_path` points at the private directory instead.
  setup do
    sandbox_dir = Path.join(System.tmp_dir!(), "persistence_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(sandbox_dir)

    original_sandbox = Application.fetch_env(:brain, :test_world_sandbox)
    original_path = Application.fetch_env(:brain, :training_worlds_path)

    Application.put_env(:brain, :test_world_sandbox, false)
    Application.put_env(:brain, :training_worlds_path, sandbox_dir)

    Brain.Worlds.TypeInferrer.init()

    on_exit(fn ->
      File.rm_rf!(sandbox_dir)
      restore_env(:test_world_sandbox, original_sandbox)
      restore_env(:training_worlds_path, original_path)
    end)

    {:ok, sandbox_dir: sandbox_dir}
  end

  defp restore_env(key, {:ok, value}), do: Application.put_env(:brain, key, value)
  defp restore_env(key, :error), do: Application.delete_env(:brain, key)

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

  describe "a world's semantic facts" do
    alias Brain.Memory.Types.SemanticFact

    defp persistent_world(id) do
      %TrainingWorld{id: id, name: id, mode: :persistent, created_at: DateTime.utc_now(), config: %{}, metadata: %{}}
    end

    test "are saved and loaded with their action apart from their text" do
      fact = SemanticFact.new("is it raining | will it rain today", [0.1], ["ep1"], [], action: "weather.query")
      data = %{world: persistent_world("test_semantics"), metrics: nil, candidates: [], overlay: [], events: [], semantics: [fact]}

      assert :ok = Persistence.save("test_semantics", data)
      assert {:ok, %{semantics: [loaded]}} = Persistence.load("test_semantics")

      assert loaded.representation == "is it raining | will it rain today"
      assert loaded.action == "weather.query"
    end

    test "saved before facts had an action are split into the action and the text", %{sandbox_dir: sandbox_dir} do
      data = %{world: persistent_world("test_legacy_semantics"), metrics: nil, candidates: [], overlay: [], events: []}
      assert :ok = Persistence.save("test_legacy_semantics", data)

      legacy = [
        %{"id" => "a", "representation" => "Action: weather.query | Examples: is it raining", "evidence_ids" => ["ep1"]},
        %{"id" => "b", "representation" => "a fact of its own", "evidence_ids" => []}
      ]

      File.write!(Path.join([sandbox_dir, "test_legacy_semantics", "semantics.json"]), Jason.encode!(legacy))

      assert {:ok, %{semantics: [a, b]}} = Persistence.load("test_legacy_semantics")
      assert {a.action, a.representation} == {"weather.query", "is it raining"}
      assert {b.action, b.representation} == {nil, "a fact of its own"}
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
