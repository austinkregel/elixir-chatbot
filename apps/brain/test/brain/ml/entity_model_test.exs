defmodule Brain.ML.EntityModelTest do
  @moduledoc """
  Each world holds its own copy of the entity model. A world with no copy
  raises naming the world and the command that seeds it; seeding copies the
  trained model byte for byte; an ephemeral world is seeded in memory and
  forgotten when dropped; and a model file of an earlier shape is refused.

  Every test runs in a world of its own that nothing else uses, and removes
  that world's directory and loaded model when it exits.
  """

  use ExUnit.Case, async: false

  alias Brain.Analysis.RunProvenance
  alias Brain.ML.EntityModel

  setup do
    world_id = "entity_model_test_#{System.unique_integer([:positive])}"

    # `path/1` is <world>/models/entity_model.term.
    world_dir = world_id |> EntityModel.path() |> Path.dirname() |> Path.dirname()

    on_exit(fn ->
      EntityModel.drop(world_id)
      File.rm_rf!(world_dir)
    end)

    {:ok, world_id: world_id}
  end

  describe "loaded!/1" do
    test "raises naming the world and the command that seeds it when the world has no copy", %{world_id: world_id} do
      refute File.exists?(EntityModel.path(world_id))

      error = assert_raise RuntimeError, fn -> EntityModel.loaded!(world_id) end
      message = Exception.message(error)

      assert message =~ inspect(world_id)
      assert message =~ EntityModel.path(world_id)
      assert message =~ "mix world.seed_model entity_model --world #{world_id}"
    end
  end

  describe "seed!/1" do
    test "copies the trained model byte for byte and makes it the world's loaded model", %{world_id: world_id} do
      trained = EntityModel.trained_path()

      assert :ok = EntityModel.seed!(world_id)

      copy = EntityModel.path(world_id)
      assert File.regular?(copy)
      assert RunProvenance.sha256_file!(copy) == RunProvenance.sha256_file!(trained)

      expected = EntityModel.read!(trained, :trained)
      assert EntityModel.loaded!(world_id) == expected

      # Forgetting the loaded model leaves the copy, which is read back from disk.
      :ok = EntityModel.drop(world_id)
      assert EntityModel.loaded!(world_id) == expected
    end
  end

  describe "seed_in_memory!/1" do
    test "makes loaded!/1 return the trained model with no file on disk", %{world_id: world_id} do
      assert :ok = EntityModel.seed_in_memory!(world_id)

      refute File.exists?(EntityModel.path(world_id))
      assert EntityModel.loaded!(world_id) == EntityModel.read!(EntityModel.trained_path(), :trained)
    end
  end

  describe "drop/1" do
    test "forgets a world's loaded model", %{world_id: world_id} do
      :ok = EntityModel.seed_in_memory!(world_id)
      assert %{} = EntityModel.loaded!(world_id)

      assert :ok = EntityModel.drop(world_id)

      # Nothing on disk and nothing loaded: the world has no model again.
      assert_raise RuntimeError, ~r/has no entity model/, fn -> EntityModel.loaded!(world_id) end
    end
  end

  describe "read!/2" do
    setup do
      dir = Path.join(System.tmp_dir!(), "entity_model_test_read_#{System.unique_integer([:positive])}")
      File.mkdir_p!(dir)
      on_exit(fn -> File.rm_rf!(dir) end)
      {:ok, dir: dir}
    end

    test "raises for a model trained without :feature_totals", %{dir: dir} do
      # Every key the current trainer writes but :feature_totals.
      earlier = %{
        tag_vocabulary: %{"O" => 0, "B-location" => 1},
        feature_weights: %{},
        smoothing: 1.0,
        transition_weights: %{},
        tag_priors: %{}
      }

      file = Path.join(dir, "entity_model.term")
      File.write!(file, :erlang.term_to_binary(earlier))

      error = assert_raise RuntimeError, fn -> EntityModel.read!(file, :trained) end
      message = Exception.message(error)

      assert message =~ file
      assert message =~ ":feature_totals"
    end
  end
end
