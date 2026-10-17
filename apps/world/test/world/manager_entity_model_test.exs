defmodule World.ManagerEntityModelTest do
  @moduledoc """
  A world holds its own copy of the trained entity model (`Brain.ML.EntityModel`)
  from the moment `World.Manager` creates it, and loses it when the world is
  destroyed.

  A persistent world's copy is a file in its models directory, byte for byte the
  trained model; an ephemeral world has no directory, so its copy lives only in
  memory. Either way a destroyed world's loaded model is gone, so extraction in
  it raises rather than running on a model the world no longer holds.
  """
  use World.AtlasCase, async: false
  import Brain.TestHelpers

  alias Brain.ML.EntityModel
  alias World.Manager, as: WorldManager

  setup do
    start_world_test_services()
    setup_world_sandbox()

    eventually(
      fn -> WorldManager.ready?() end,
      fn ready -> ready end,
      50,
      20
    )

    :ok
  end

  describe "a persistent world" do
    test "is created with a byte-identical copy of the trained model in its models directory" do
      {:ok, world} = create_test_world("entity_model_persistent", mode: :persistent)
      on_exit(fn -> WorldManager.purge(world.id) end)

      path = EntityModel.path(world.id)

      assert File.regular?(path)
      assert File.read!(path) == File.read!(EntityModel.trained_path())
      assert %{tag_vocabulary: _} = EntityModel.loaded!(world.id)
    end

    test "destroying it drops its loaded model" do
      {:ok, world} = create_test_world("entity_model_persistent_destroy", mode: :persistent)
      on_exit(fn -> File.rm_rf!(World.Persistence.world_path(world.id)) end)

      assert %{tag_vocabulary: _} = EntityModel.loaded!(world.id)

      :ok = WorldManager.destroy(world.id)

      # The file stays with the persisted world; with it gone, nothing of the
      # model is left in memory to answer for the world.
      File.rm!(EntityModel.path(world.id))

      assert_raise RuntimeError, ~r/has no entity model/, fn -> EntityModel.loaded!(world.id) end
    end
  end

  describe "an ephemeral world" do
    test "is created with the trained model loaded and no file" do
      {:ok, world} = create_test_world("entity_model_ephemeral")

      refute File.exists?(EntityModel.path(world.id))
      assert EntityModel.loaded!(world.id) == EntityModel.read!(EntityModel.trained_path(), :trained)
    end

    test "destroying it drops its loaded model" do
      {:ok, world} = create_test_world("entity_model_ephemeral_destroy")
      assert %{tag_vocabulary: _} = EntityModel.loaded!(world.id)

      :ok = WorldManager.destroy(world.id)

      assert_raise RuntimeError, ~r/has no entity model/, fn -> EntityModel.loaded!(world.id) end
    end
  end
end
