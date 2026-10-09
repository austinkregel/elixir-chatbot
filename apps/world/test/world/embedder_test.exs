defmodule World.EmbedderTest do
  @moduledoc """
  A world embeds with its own copy of the trained embedder vocabulary, seeded
  when `World.Manager` creates it. A world with no copy raises naming the
  world: there is no fallback to the trained vocabulary or to another world.
  """
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias World.Embedder
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

  # A world id with nothing loaded for it and no directory on disk.
  defp unseeded_world_id, do: "test_unseeded_embedder_#{System.unique_integer([:positive])}"

  describe "get_status/1" do
    test "reports a world with nothing loaded as not initialized" do
      assert Embedder.get_status(unseeded_world_id()) == %{
               ready: false,
               phase: :not_initialized,
               phase_label: "Not initialized",
               vocabulary_size: 0,
               built_at: nil,
               episode_count: 0
             }
    end
  end

  describe "embed/2" do
    test "raises naming a world that has no embedder copy" do
      world_id = unseeded_world_id()

      assert_raise RuntimeError, ~r/world "#{world_id}" has no embedder vocabulary/, fn ->
        Embedder.embed(world_id, "hello")
      end
    end

    test "embeds in a world created with its seeded copy" do
      {:ok, world} = create_test_world("embedder_seeded")

      trained = Embedder.read!(Embedder.trained_path(), :trained)
      assert map_size(trained.vocabulary) > 0

      assert {:ok, embedding} = Embedder.embed(world.id, "hello there, how is the weather")
      assert length(embedding) == map_size(trained.vocabulary)
      assert Enum.all?(embedding, &is_float/1)

      assert Embedder.ready?(world.id)
      assert Embedder.get_status(world.id).vocabulary_size == map_size(trained.vocabulary)
    end

    test "raises once the world that held the copy is destroyed" do
      {:ok, world} = create_test_world("embedder_destroyed")
      assert {:ok, _embedding} = Embedder.embed(world.id, "hello")

      :ok = WorldManager.destroy(world.id)

      assert_raise RuntimeError, ~r/world "#{world.id}" has no embedder vocabulary/, fn ->
        Embedder.embed(world.id, "hello")
      end
    end
  end

  describe "cosine_similarity/2" do
    test "is 1.0 for a vector with itself" do
      assert_in_delta Embedder.cosine_similarity([0.6, 0.8], [0.6, 0.8]), 1.0, 1.0e-9
    end

    test "is 0.0 for orthogonal vectors" do
      assert Embedder.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    end

    test "is 0.0 for vectors of different lengths or empty vectors" do
      assert Embedder.cosine_similarity([1.0, 0.0], [1.0]) == 0.0
      assert Embedder.cosine_similarity([], []) == 0.0
    end
  end
end
