defmodule Brain.ML.KnowledgeGraph.TripleScorerWorldTest do
  @moduledoc """
  Each world holds its own copy of the triple scorer, every score names its
  world, and `EntityVectorCache` keeps each world's entity and type vectors
  apart, computed from that world's scorer.

  A world with no copy raises naming the world and the command that seeds it;
  two worlds holding different scorers score and encode differently; seeding
  copies the trained scorer byte for byte; and seeding or dropping a world's
  scorer forgets the vectors computed from the copy it replaces, and only that
  world's.

  Every test runs in `test_` worlds of its own that nothing else uses, and
  removes their directories, loaded scorers and cached vectors when it exits.
  """

  use ExUnit.Case, async: false

  alias Brain.Analysis.RunProvenance
  alias Brain.ML.KnowledgeGraph.{EntityVectorCache, TripleScorer}

  @triples_a [
    {"dog", "IsA", "animal"},
    {"cat", "IsA", "animal"},
    {"paris", "AtLocation", "france"}
  ]

  @triples_b [
    {"dog", "Likes", "bone"},
    {"cat", "Likes", "fish"},
    {"berlin", "AtLocation", "germany"},
    {"rome", "AtLocation", "italy"}
  ]

  setup do
    world_a = "test_triple_scorer_a_#{System.unique_integer([:positive])}"
    world_b = "test_triple_scorer_b_#{System.unique_integer([:positive])}"

    on_exit(fn ->
      for world_id <- [world_a, world_b] do
        TripleScorer.drop(world_id)
        # `path/1` is <world>/models/triple_scorer.term.
        File.rm_rf!(world_id |> TripleScorer.path() |> Path.dirname() |> Path.dirname())
      end
    end)

    {:ok, world_a: world_a, world_b: world_b}
  end

  describe "a world without a copy" do
    test "raises naming the world and the command that seeds it", %{world_a: world_a} do
      refute File.exists?(TripleScorer.path(world_a))
      refute TripleScorer.held?(world_a)

      for use_scorer <- [
            fn -> TripleScorer.loaded!(world_a) end,
            fn -> TripleScorer.score(world_a, "dog", "IsA", "animal") end,
            fn -> TripleScorer.score_batch(world_a, [{"dog", "IsA", "animal"}]) end,
            fn -> TripleScorer.relation_coverage(world_a) end,
            fn -> TripleScorer.current_model_version(world_a) end,
            fn -> EntityVectorCache.get_or_compute(world_a, "dog") end,
            fn -> EntityVectorCache.get_or_compute_type(world_a, "animal") end
          ] do
        error = assert_raise RuntimeError, use_scorer
        message = Exception.message(error)

        assert message =~ inspect(world_a)
        assert message =~ TripleScorer.path(world_a)
        assert message =~ "mix world.seed_model triple_scorer --world #{world_a}"
      end
    end
  end

  describe "two worlds" do
    test "each scores with its own copy", %{world_a: world_a, world_b: world_b} do
      write_copy!(world_a, @triples_a, "version_a")
      write_copy!(world_b, @triples_b, "version_b")

      assert TripleScorer.held?(world_a)
      assert TripleScorer.held?(world_b)

      assert {:ok, score_a} = TripleScorer.score(world_a, "dog", "IsA", "animal")
      assert {:ok, score_b} = TripleScorer.score(world_b, "dog", "IsA", "animal")
      assert is_float(score_a) and score_a > 0.0 and score_a < 1.0
      assert is_float(score_b) and score_b > 0.0 and score_b < 1.0
      assert score_a != score_b

      assert {:ok, [batch_a]} = TripleScorer.score_batch(world_a, [{"dog", "IsA", "animal"}])
      assert batch_a == score_a

      assert TripleScorer.relation_coverage(world_a) == %{"IsA" => 2, "AtLocation" => 1}
      assert TripleScorer.relation_coverage(world_b) == %{"Likes" => 2, "AtLocation" => 2}
      assert TripleScorer.current_model_version(world_a) == "version_a"
      assert TripleScorer.current_model_version(world_b) == "version_b"
    end

    test "EntityVectorCache keeps each world's vectors, computed from its own scorer",
         %{world_a: world_a, world_b: world_b} do
      write_copy!(world_a, @triples_a, "version_a")
      write_copy!(world_b, @triples_b, "version_b")

      assert {:ok, %Nx.Tensor{} = vector_a} = EntityVectorCache.get_or_compute(world_a, "dog")
      assert {:ok, %Nx.Tensor{} = vector_b} = EntityVectorCache.get_or_compute(world_b, "dog")
      assert Nx.to_flat_list(vector_a) != Nx.to_flat_list(vector_b)

      # A second lookup is the cached vector.
      assert {:ok, ^vector_a} = EntityVectorCache.get_or_compute(world_a, "dog")

      assert :miss = EntityVectorCache.get_cached_type(world_a, "animal")
      assert {:ok, %Nx.Tensor{} = type_a} = EntityVectorCache.get_or_compute_type(world_a, "animal")
      assert {:ok, ^type_a} = EntityVectorCache.get_cached_type(world_a, "animal")
      assert :miss = EntityVectorCache.get_cached_type(world_b, "animal")
    end
  end

  describe "replacing a world's scorer" do
    test "forgets that world's cached vectors and no other world's", %{world_a: world_a, world_b: world_b} do
      write_copy!(world_a, @triples_a, "version_a")
      write_copy!(world_b, @triples_b, "version_b")

      {:ok, _} = EntityVectorCache.get_or_compute_type(world_a, "animal")
      {:ok, _} = EntityVectorCache.get_or_compute_type(world_b, "animal")

      assert :ok = TripleScorer.drop(world_a)

      assert :miss = EntityVectorCache.get_cached_type(world_a, "animal")
      assert {:ok, %Nx.Tensor{}} = EntityVectorCache.get_cached_type(world_b, "animal")
    end
  end

  describe "seed!/1" do
    test "copies the trained scorer byte for byte and makes it the world's", %{world_a: world_a} do
      trained = TripleScorer.trained_path()

      assert :ok = TripleScorer.seed!(world_a)

      copy = TripleScorer.path(world_a)
      assert File.regular?(copy)
      assert RunProvenance.sha256_file!(copy) == RunProvenance.sha256_file!(trained)

      expected = TripleScorer.read!(trained, :trained)
      assert TripleScorer.loaded!(world_a).vocab == expected.vocab
      assert TripleScorer.relation_coverage(world_a) == Map.get(expected, :relation_coverage, %{})
    end
  end

  describe "seed_in_memory!/1 and drop/1" do
    test "an in-memory copy has no file, and dropping it leaves the world without one", %{world_a: world_a} do
      assert :ok = TripleScorer.seed_in_memory!(world_a)

      refute File.exists?(TripleScorer.path(world_a))
      assert TripleScorer.held?(world_a)
      assert TripleScorer.loaded!(world_a).vocab == TripleScorer.read!(TripleScorer.trained_path(), :trained).vocab

      assert :ok = TripleScorer.drop(world_a)
      refute TripleScorer.held?(world_a)
      assert_raise RuntimeError, ~r/has no triple scorer/, fn -> TripleScorer.loaded!(world_a) end
    end
  end

  describe "read!/2" do
    @tag :tmp_dir
    test "raises when the config's vocab_size is not the vocabulary's", %{tmp_dir: tmp_dir} do
      {:ok, _model, params, vocab, config} = train(@triples_a)
      file = Path.join(tmp_dir, "triple_scorer.term")
      :ok = TripleScorer.save_model(params, vocab, %{config | vocab_size: map_size(vocab) + 1}, file)

      error = assert_raise RuntimeError, fn -> TripleScorer.read!(file, :trained) end
      assert Exception.message(error) =~ "vocab_size #{map_size(vocab) + 1}"
    end

    @tag :tmp_dir
    test "raises naming the training command when there is no trained file", %{tmp_dir: tmp_dir} do
      error = assert_raise RuntimeError, fn -> TripleScorer.read!(Path.join(tmp_dir, "missing.term"), :trained) end
      assert Exception.message(error) =~ "mix train_kg_lstm"
    end
  end

  defp train(triples) do
    TripleScorer.train(triples, epochs: 3, neg_ratio: 1, embedding_dim: 8, hidden_dim: 16, seed: 9)
  end

  # Trains a scorer on `triples` and writes it as the world's own copy.
  defp write_copy!(world_id, triples, model_version) do
    {:ok, _model, params, vocab, config} = train(triples)

    coverage = Enum.frequencies_by(triples, fn {_h, r, _t} -> r end)

    :ok =
      TripleScorer.save_model(params, vocab, config, TripleScorer.path(world_id),
        relation_coverage: coverage,
        model_version: model_version
      )
  end
end
