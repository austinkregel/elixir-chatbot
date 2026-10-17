defmodule Mix.Tasks.World.SeedModelTest do
  @moduledoc """
  `mix world.seed_model` copies a trained model into worlds that already exist.

  It names its worlds explicitly, refuses a model it does not know, and leaves a
  named world holding a byte-identical copy of the trained model.
  """
  use World.AtlasCase, async: false
  import Brain.TestHelpers

  alias Brain.ML.EntityModel
  alias Mix.Tasks.World.SeedModel
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

    previous_shell = Mix.shell()
    Mix.shell(Mix.Shell.Process)
    on_exit(fn -> Mix.shell(previous_shell) end)

    :ok
  end

  test "raises without --world or --all" do
    assert_raise Mix.Error, ~r/pass --world <id> \(repeatable\) or --all/, fn ->
      SeedModel.run(["entity_model"])
    end
  end

  test "raises on a model it does not know, naming the models it does" do
    assert_raise Mix.Error, ~r/no model "not_a_model".*entity_model/, fn ->
      SeedModel.run(["not_a_model", "--world", "default"])
    end
  end

  test "seeds a named world with a byte-identical copy of the trained model" do
    {:ok, world} = create_test_world("seed_model_task", mode: :persistent)
    on_exit(fn -> WorldManager.purge(world.id) end)

    path = EntityModel.path(world.id)
    File.rm!(path)
    refute File.exists?(path)

    SeedModel.run(["entity_model", "--world", world.id])

    assert File.read!(path) == File.read!(EntityModel.trained_path())

    world_id = world.id
    assert_received {:mix_shell, :info, ["  seeded " <> ^world_id]}
  end

  test "seeds a named world with byte-identical copies of the Poincare embeddings and the triple scorer" do
    {:ok, world} = create_test_world("seed_graph_models_task", mode: :persistent)
    on_exit(fn -> WorldManager.purge(world.id) end)

    for {name, model} <- [
          {"poincare_embeddings", Brain.ML.Poincare.Embeddings},
          {"triple_scorer", Brain.ML.KnowledgeGraph.TripleScorer}
        ] do
      path = model.path(world.id)
      File.rm!(path)
      refute File.exists?(path)

      SeedModel.run([name, "--world", world.id])

      assert File.read!(path) == File.read!(model.trained_path()), "#{name}: copy differs from the trained file"

      world_id = world.id
      assert_received {:mix_shell, :info, ["  seeded " <> ^world_id]}
    end
  end

  # The suite's own VM has already started EXLA, which hid that the task, run
  # on its own, never started what rebuilding the triple scorer reads. A fresh
  # VM that starts nothing runs it as `mix world.seed_model` does.
  @tag timeout: 300_000
  test "seeds the triple scorer in a VM that starts nothing else" do
    world_id = "test_seed_fresh_vm_#{System.unique_integer([:positive])}"
    path = Brain.ML.KnowledgeGraph.TripleScorer.path(world_id)
    on_exit(fn -> File.rm_rf!(World.Persistence.world_path(world_id)) end)

    code = ~s|Mix.Tasks.World.SeedModel.run(["triple_scorer", "--world", #{inspect(world_id)}])|

    {output, status} =
      System.cmd("mix", ["run", "--no-start", "--no-compile", "--no-deps-check", "-e", code],
        cd: Brain.umbrella_root(),
        env: [{"MIX_ENV", "test"}],
        stderr_to_stdout: true
      )

    assert status == 0, output
    assert output =~ "seeded #{world_id}"
    assert File.read!(path) == File.read!(Brain.ML.KnowledgeGraph.TripleScorer.trained_path())
  end
end
