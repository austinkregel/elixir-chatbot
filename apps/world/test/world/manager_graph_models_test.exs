defmodule World.ManagerGraphModelsTest do
  @moduledoc """
  A world holds its own copies of the trained Poincare embeddings
  (`Brain.ML.Poincare.Embeddings`) and KG triple scorer
  (`Brain.ML.KnowledgeGraph.TripleScorer`) from the moment `World.Manager`
  creates it, and loses them when the world is destroyed.

  A persistent world's copies are files in its models directory, byte for byte
  the trained ones; an ephemeral world has no directory, so its copies live only
  in memory. Either way a destroyed world's loaded copies are gone, so using
  them raises rather than answering from models the world no longer holds.
  """
  use World.AtlasCase, async: false
  import Brain.TestHelpers, except: [create_test_world: 1, create_test_world: 2, start_world_test_services: 0]
  import World.TestWorlds

  alias Brain.ML.KnowledgeGraph.TripleScorer
  alias Brain.ML.Poincare.Embeddings
  alias World.Manager, as: WorldManager

  @models [
    {Embeddings, ~r/has no Poincare embeddings/},
    {TripleScorer, ~r/has no triple scorer/}
  ]

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
    test "is created with byte-identical copies of the trained models in its models directory" do
      {:ok, world} = create_test_world("graph_models_persistent", mode: :persistent)
      on_exit(fn -> WorldManager.purge(world.id) end)

      for {model, _missing} <- @models do
        path = model.path(world.id)

        assert File.regular?(path), "#{inspect(model)}: no copy at #{path}"
        assert File.read!(path) == File.read!(model.trained_path())
      end

      assert %{embeddings: %Nx.Tensor{}} = Embeddings.loaded!(world.id)
      assert %{model: %Axon{}} = TripleScorer.loaded!(world.id)
    end

    test "destroying it drops its loaded models" do
      {:ok, world} = create_test_world("graph_models_persistent_destroy", mode: :persistent)
      on_exit(fn -> File.rm_rf!(Brain.Worlds.world_path(world.id)) end)

      assert %{embeddings: %Nx.Tensor{}} = Embeddings.loaded!(world.id)
      assert %{model: %Axon{}} = TripleScorer.loaded!(world.id)

      :ok = WorldManager.destroy(world.id)

      # The files stay with the persisted world; with them gone, nothing of the
      # models is left in memory to answer for the world.
      for {model, missing} <- @models do
        File.rm!(model.path(world.id))
        assert_raise RuntimeError, missing, fn -> model.loaded!(world.id) end
      end
    end
  end

  describe "an ephemeral world" do
    test "is created with the trained models loaded and no files" do
      {:ok, world} = create_test_world("graph_models_ephemeral")

      for {model, _missing} <- @models do
        refute File.exists?(model.path(world.id))
      end

      assert Embeddings.loaded!(world.id) == Embeddings.read!(Embeddings.trained_path(), :trained)

      assert TripleScorer.loaded!(world.id).vocab ==
               TripleScorer.read!(TripleScorer.trained_path(), :trained).vocab
    end

    test "destroying it drops its loaded models" do
      {:ok, world} = create_test_world("graph_models_ephemeral_destroy")
      assert %{embeddings: %Nx.Tensor{}} = Embeddings.loaded!(world.id)
      assert %{model: %Axon{}} = TripleScorer.loaded!(world.id)

      :ok = WorldManager.destroy(world.id)

      for {model, missing} <- @models do
        assert_raise RuntimeError, missing, fn -> model.loaded!(world.id) end
      end
    end
  end
end
