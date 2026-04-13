defmodule Brain.ML.KnowledgeGraph.TripleScorerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.KnowledgeGraph.TripleScorer

  @sample_triples [
    {"John", "LIKES", "coffee"},
    {"Mary", "LIKES", "tea"},
    {"John", "WORKS_AT", "Acme"},
    {"Mary", "LIVES_IN", "Paris"},
    {"Acme", "LOCATED_IN", "New York"},
    {"Bob", "LIKES", "music"},
    {"Alice", "WORKS_AT", "Google"},
    {"Bob", "LIVES_IN", "London"},
    {"Google", "LOCATED_IN", "California"},
    {"Alice", "LIKES", "books"},
    {"John", "LIVES_IN", "Boston"},
    {"Mary", "WORKS_AT", "Microsoft"}
  ]

  describe "build_model/2" do
    test "builds a valid Axon model" do
      model = TripleScorer.build_model(100)
      assert %Axon{} = model
    end
  end

  describe "train/2" do
    @tag :convergence
    @tag timeout: 120_000
    test "trains successfully on sample triples" do
      {:ok, _model, params, vocab, config} = TripleScorer.train(@sample_triples,
        epochs: 30,
        neg_ratio: 3,
        embedding_dim: 32,
        hidden_dim: 32
      )

      assert is_map(vocab)
      assert map_size(vocab) > 0
      assert config.vocab_size == map_size(vocab)
    end
  end

  describe "scoring" do
    @tag :convergence
    @tag timeout: 180_000
    test "positive triples score higher than negatives after training" do
      {:ok, model, params, vocab, _config} = TripleScorer.train(@sample_triples,
        epochs: 200,
        neg_ratio: 3,
        embedding_dim: 16,
        hidden_dim: 32,
        learning_rate: 0.005
      )

      pos_scores = Enum.map(@sample_triples, fn {h, r, t} ->
        text = "[HEAD] #{h} [REL] #{r} [TAIL] #{t}"
        {input, mask} = TripleScorer.encode_single_public(text, vocab)

        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        Nx.squeeze(output) |> Nx.to_number()
      end)

      neg_triples = [
        {"John", "LIKES", "Acme"},
        {"coffee", "WORKS_AT", "Mary"},
        {"tea", "LOCATED_IN", "John"}
      ]

      neg_scores = Enum.map(neg_triples, fn {h, r, t} ->
        text = "[HEAD] #{h} [REL] #{r} [TAIL] #{t}"
        {input, mask} = TripleScorer.encode_single_public(text, vocab)

        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        Nx.squeeze(output) |> Nx.to_number()
      end)

      avg_pos = Enum.sum(pos_scores) / length(pos_scores)
      avg_neg = Enum.sum(neg_scores) / length(neg_scores)

      assert avg_pos > avg_neg,
        "Avg positive score (#{Float.round(avg_pos, 3)}) should be > " <>
        "avg negative score (#{Float.round(avg_neg, 3)})"
    end

    test "score range is in (0, 1)" do
      {:ok, model, params, vocab, _config} = TripleScorer.train(@sample_triples,
        epochs: 5,
        neg_ratio: 2,
        embedding_dim: 16,
        hidden_dim: 16
      )

      for {h, r, t} <- @sample_triples do
        text = "[HEAD] #{h} [REL] #{r} [TAIL] #{t}"
        {input, mask} = TripleScorer.encode_single_public(text, vocab)

        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        score = Nx.squeeze(output) |> Nx.to_number()

        assert score > 0.0, "Score should be > 0, got #{score}"
        assert score < 1.0, "Score should be < 1, got #{score}"
      end
    end
  end

  describe "serialization" do
    @tag timeout: 60_000
    test "save and load roundtrip" do
      {:ok, _model, params, vocab, config} = TripleScorer.train(@sample_triples,
        epochs: 5,
        neg_ratio: 2,
        embedding_dim: 16,
        hidden_dim: 16
      )

      tmp_path = Path.join(System.tmp_dir!(), "kg_lstm_test_#{:rand.uniform(100000)}.term")

      try do
        TripleScorer.save_model(params, vocab, config, tmp_path)
        {:ok, loaded} = TripleScorer.load_model(tmp_path)

        assert loaded.config == config
        assert loaded.vocab == vocab
      after
        File.rm(tmp_path)
      end
    end
  end
end
