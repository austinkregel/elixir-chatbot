defmodule Brain.ML.Poincare.EmbeddingsWorldTest do
  @moduledoc """
  Each world holds its own copy of the Poincare embeddings and every lookup
  names its world. A world with no copy raises naming the world and the command
  that seeds it; two worlds holding different embeddings answer differently;
  seeding copies the trained embeddings byte for byte; an ephemeral world is
  seeded in memory and forgotten when dropped; and a file whose embeddings do
  not have the stored width is refused.

  Every test runs in `test_` worlds of its own that nothing else uses, and
  removes their directories and loaded embeddings when it exits.
  """

  use ExUnit.Case, async: false

  alias Brain.Analysis.RunProvenance
  alias Brain.ML.Poincare.Embeddings

  @pairs_a [{"dog", "animal"}, {"cat", "animal"}, {"oak", "tree"}, {"tree", "plant"}, {"rose", "plant"}]
  @pairs_b [{"dog", "pet"}, {"animal", "pet"}, {"oak", "plant"}, {"cat", "tree"}]

  setup do
    world_a = "test_poincare_a_#{System.unique_integer([:positive])}"
    world_b = "test_poincare_b_#{System.unique_integer([:positive])}"

    on_exit(fn ->
      for world_id <- [world_a, world_b] do
        Embeddings.drop(world_id)
        # `path/1` is <world>/models/poincare_embeddings.term.
        File.rm_rf!(world_id |> Embeddings.path() |> Path.dirname() |> Path.dirname())
      end
    end)

    {:ok, world_a: world_a, world_b: world_b}
  end

  describe "a world without a copy" do
    test "raises naming the world and the command that seeds it", %{world_a: world_a} do
      refute File.exists?(Embeddings.path(world_a))

      for lookup <- [
            fn -> Embeddings.loaded!(world_a) end,
            fn -> Embeddings.lookup(world_a, "dog") end,
            fn -> Embeddings.entity_distance(world_a, "dog", "animal") end,
            fn -> Embeddings.all_embeddings(world_a) end
          ] do
        error = assert_raise RuntimeError, lookup
        message = Exception.message(error)

        assert message =~ inspect(world_a)
        assert message =~ Embeddings.path(world_a)
        assert message =~ "mix world.seed_model poincare_embeddings --world #{world_a}"
      end
    end
  end

  describe "two worlds" do
    test "each reads its own copy", %{world_a: world_a, world_b: world_b} do
      write_copy!(world_a, @pairs_a)
      write_copy!(world_b, @pairs_b)

      assert {:ok, distance_a} = Embeddings.entity_distance(world_a, "dog", "cat")
      assert {:ok, distance_b} = Embeddings.entity_distance(world_b, "dog", "cat")
      assert is_float(distance_a) and is_float(distance_b)
      assert distance_a != distance_b

      # "rose" is only in world a's hierarchy, "pet" only in world b's.
      assert {:ok, %Nx.Tensor{}} = Embeddings.lookup(world_a, "rose")
      assert {:error, :not_found} = Embeddings.lookup(world_b, "rose")
      assert {:error, :not_found} = Embeddings.lookup(world_a, "pet")
      assert {:ok, %Nx.Tensor{}} = Embeddings.lookup(world_b, "pet")

      {:ok, all_a} = Embeddings.all_embeddings(world_a)
      {:ok, all_b} = Embeddings.all_embeddings(world_b)
      assert all_a |> Map.keys() |> Enum.sort() == ~w(animal cat dog oak plant rose tree)
      assert all_b |> Map.keys() |> Enum.sort() == ~w(animal cat dog oak pet plant tree)
    end
  end

  describe "entity_distance/3" do
    test "returns {:ok, distance} for two embedded entities and {:error, :not_found} otherwise",
         %{world_a: world_a} do
      write_copy!(world_a, @pairs_a)

      assert {:ok, distance} = Embeddings.entity_distance(world_a, "dog", "animal")
      assert is_float(distance) and distance >= 0.0

      assert {:error, :not_found} = Embeddings.entity_distance(world_a, "dog", "no_such_entity")
      assert {:error, :not_found} = Embeddings.entity_distance(world_a, "no_such_entity", "dog")
    end
  end

  describe "seed!/1" do
    test "copies the trained embeddings byte for byte and makes them the world's", %{world_a: world_a} do
      trained = Embeddings.trained_path()

      assert :ok = Embeddings.seed!(world_a)

      copy = Embeddings.path(world_a)
      assert File.regular?(copy)
      assert RunProvenance.sha256_file!(copy) == RunProvenance.sha256_file!(trained)

      expected = Embeddings.read!(trained, :trained)
      assert Embeddings.loaded!(world_a) == expected

      # Forgetting the loaded embeddings leaves the copy, which is read back from disk.
      :ok = Embeddings.drop(world_a)
      assert Embeddings.loaded!(world_a) == expected
    end
  end

  describe "seed_in_memory!/1 and drop/1" do
    test "an in-memory copy has no file, and dropping it leaves the world without one", %{world_a: world_a} do
      assert :ok = Embeddings.seed_in_memory!(world_a)

      refute File.exists?(Embeddings.path(world_a))
      assert Embeddings.loaded!(world_a) == Embeddings.read!(Embeddings.trained_path(), :trained)

      assert :ok = Embeddings.drop(world_a)
      assert_raise RuntimeError, ~r/has no Poincare embeddings/, fn -> Embeddings.loaded!(world_a) end
    end
  end

  describe "read!/2" do
    @tag :tmp_dir
    test "raises when the embeddings do not have the stored dim", %{tmp_dir: tmp_dir} do
      {:ok, embeddings, to_idx, from_idx} = Embeddings.train(@pairs_a, dim: 3, epochs: 2, seed: 9)
      file = Path.join(tmp_dir, "poincare_embeddings.term")
      :ok = Embeddings.save(embeddings, to_idx, from_idx, 5, file)

      error = assert_raise RuntimeError, fn -> Embeddings.read!(file, :trained) end
      assert Exception.message(error) =~ "stores dim 5"
      assert Exception.message(error) =~ "{7, 3}"
    end

    @tag :tmp_dir
    test "raises naming the training command when there is no trained file", %{tmp_dir: tmp_dir} do
      file = Path.join(tmp_dir, "missing.term")

      error = assert_raise RuntimeError, fn -> Embeddings.read!(file, :trained) end
      assert Exception.message(error) =~ "mix train_poincare"
    end
  end

  # Trains embeddings from `pairs` and writes them as the world's own copy.
  defp write_copy!(world_id, pairs) do
    {:ok, embeddings, to_idx, from_idx} = Embeddings.train(pairs, dim: 3, epochs: 10, seed: 9)
    :ok = Embeddings.save(embeddings, to_idx, from_idx, 3, Embeddings.path(world_id))
  end
end
