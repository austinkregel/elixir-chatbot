defmodule Brain.ML.KnowledgeGraph.TripleScorerExtendedTest do
  use ExUnit.Case, async: false

  alias Brain.ML.KnowledgeGraph.TripleScorer

  describe "GenServer lifecycle" do
    test "starts and reports not ready without model" do
      name = :"kg_test_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, pid} = TripleScorer.start_link(name: name, world_id: world)
      assert Process.alive?(pid)
      refute TripleScorer.ready?(name)
    end

    test "score returns error when not ready" do
      name = :"kg_score_test_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, _pid} = TripleScorer.start_link(name: name, world_id: world)
      assert {:error, :not_ready} = TripleScorer.score("Paris", "capital_of", "France", name)
    end

    test "score_batch returns error when not ready" do
      name = :"kg_batch_test_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, _pid} = TripleScorer.start_link(name: name, world_id: world)
      triples = [{"a", "r", "b"}]
      assert {:error, :not_ready} = TripleScorer.score_batch(triples, name)
    end

    test "ready? returns false for non-existent server" do
      refute TripleScorer.ready?(:"nonexistent_kg_#{:rand.uniform(100_000)}")
    end
  end

  describe "build_model/2" do
    test "builds model with defaults" do
      model = TripleScorer.build_model(100)
      assert %Axon{} = model
    end

    test "builds model with custom dimensions" do
      model = TripleScorer.build_model(200, embedding_dim: 32, hidden_dim: 64)
      assert %Axon{} = model
    end
  end

  describe "train/2 edge cases" do
    test "trains with minimal triples" do
      triples = [
        {"A", "R", "B"},
        {"C", "R", "D"}
      ]

      {:ok, model, params, vocab, config} = TripleScorer.train(triples,
        epochs: 5, neg_ratio: 1, embedding_dim: 8, hidden_dim: 16)

      assert %Axon{} = model
      assert is_map(params) or is_struct(params)
      assert is_map(vocab)
      assert is_map(config)
    end

    test "vocabulary includes special tokens" do
      triples = [{"cat", "IS_A", "animal"}]

      {:ok, _model, _params, vocab, _config} = TripleScorer.train(triples,
        epochs: 3, neg_ratio: 1, embedding_dim: 8, hidden_dim: 16)

      assert Map.has_key?(vocab, "[HEAD]")
      assert Map.has_key?(vocab, "[REL]")
      assert Map.has_key?(vocab, "[TAIL]")
      assert Map.has_key?(vocab, "[PAD]")
    end

    test "all scores are in (0, 1) range after training" do
      triples = [
        {"dog", "IS_A", "animal"},
        {"cat", "IS_A", "animal"},
        {"Paris", "CAPITAL_OF", "France"}
      ]

      {:ok, model, params, vocab, _config} = TripleScorer.train(triples,
        epochs: 10, neg_ratio: 2, embedding_dim: 16, hidden_dim: 32)

      test_texts = [
        "[HEAD] dog [REL] IS_A [TAIL] animal",
        "[HEAD] cat [REL] IS_A [TAIL] Paris"
      ]

      for text <- test_texts do
        {input, mask} = TripleScorer.encode_single_public(text, vocab)
        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        score = Nx.squeeze(output) |> Nx.to_number()
        assert score > 0.0 and score < 1.0, "Score #{score} out of range for: #{text}"
      end
    end
  end

  describe "save_model/4 and load_model/1" do
    @tag :tmp_dir
    test "roundtrip preserves model", %{tmp_dir: tmp_dir} do
      triples = [{"A", "R", "B"}, {"C", "R", "D"}]

      {:ok, _model, params, vocab, config} = TripleScorer.train(triples,
        epochs: 3, neg_ratio: 1, embedding_dim: 8, hidden_dim: 16)

      path = Path.join(tmp_dir, "kg_test.term")
      TripleScorer.save_model(params, vocab, config, path)

      assert File.exists?(path)

      {:ok, loaded} = TripleScorer.load_model(path)
      assert Map.has_key?(loaded, :params)
      assert Map.has_key?(loaded, :vocab)
      assert Map.has_key?(loaded, :config)
    end

    test "load returns error for missing file" do
      result = TripleScorer.load_model("/tmp/nonexistent_kg_#{:rand.uniform(100_000)}.term")
      assert {:error, _} = result
    end
  end

  describe "encode_single_public/2" do
    test "encodes and pads text correctly" do
      vocab = %{"[PAD]" => 0, "[HEAD]" => 1, "[REL]" => 2, "[TAIL]" => 3, "cat" => 4}

      {input, mask} = TripleScorer.encode_single_public("[HEAD] cat [REL] is [TAIL] animal", vocab)

      assert Nx.shape(input) == {1, elem(Nx.shape(input), 1)}
      assert elem(Nx.shape(mask), 0) == 1
      assert elem(Nx.shape(mask), 1) == elem(Nx.shape(input), 1)

      # First tokens should be non-pad (mask = 1)
      first_mask = Nx.to_flat_list(mask) |> hd()
      assert first_mask == 1
    end

    test "handles unknown tokens with OOV index" do
      vocab = %{"[PAD]" => 0, "[HEAD]" => 1, "[REL]" => 2, "[TAIL]" => 3}

      {input, mask} = TripleScorer.encode_single_public("[HEAD] unknown_entity [REL] r [TAIL] other", vocab)
      assert is_struct(input, Nx.Tensor)
      assert is_struct(mask, Nx.Tensor)
    end
  end
end
