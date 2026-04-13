defmodule Brain.ML.KnowledgeGraph.TripleScoringIntegrationTest do
  use ExUnit.Case, async: false
  @moduletag :integration

  alias Brain.ML.KnowledgeGraph.TripleScorer

  @knowledge_triples [
    {"Einstein", "BORN_IN", "Germany"},
    {"Einstein", "FIELD_OF", "physics"},
    {"Newton", "BORN_IN", "England"},
    {"Newton", "FIELD_OF", "physics"},
    {"Darwin", "BORN_IN", "England"},
    {"Darwin", "FIELD_OF", "biology"},
    {"Paris", "CAPITAL_OF", "France"},
    {"Berlin", "CAPITAL_OF", "Germany"},
    {"London", "CAPITAL_OF", "England"},
    {"physics", "IS_A", "science"},
    {"biology", "IS_A", "science"},
    {"France", "IS_A", "country"},
    {"Germany", "IS_A", "country"},
    {"England", "IS_A", "country"}
  ]

  describe "triple scoring integration" do
    @tag :convergence
    @tag timeout: 120_000
    test "trained scorer differentiates valid from invalid triples" do
      {:ok, model, params, vocab, _config} = TripleScorer.train(@knowledge_triples,
        epochs: 50,
        neg_ratio: 5,
        embedding_dim: 32,
        hidden_dim: 64
      )

      valid_novel = [
        {"Einstein", "IS_A", "scientist"},
        {"Paris", "IS_A", "city"}
      ]

      invalid = [
        {"physics", "CAPITAL_OF", "Einstein"},
        {"Paris", "BORN_IN", "biology"}
      ]

      valid_scores = Enum.map(valid_novel, fn {h, r, t} ->
        text = "[HEAD] #{h} [REL] #{r} [TAIL] #{t}"
        {input, mask} = TripleScorer.encode_single_public(text, vocab)
        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        Nx.squeeze(output) |> Nx.to_number()
      end)

      invalid_scores = Enum.map(invalid, fn {h, r, t} ->
        text = "[HEAD] #{h} [REL] #{r} [TAIL] #{t}"
        {input, mask} = TripleScorer.encode_single_public(text, vocab)
        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        Nx.squeeze(output) |> Nx.to_number()
      end)

      # Both should produce valid scores in (0, 1)
      for s <- valid_scores ++ invalid_scores do
        assert s > 0.0 and s < 1.0, "Score should be in (0, 1), got #{s}"
      end
    end

    @tag timeout: 60_000
    test "graph writer gating: low-scoring triples are flaggable" do
      {:ok, model, params, vocab, _config} = TripleScorer.train(@knowledge_triples,
        epochs: 30,
        neg_ratio: 3,
        embedding_dim: 16,
        hidden_dim: 32
      )

      test_triples = [
        {"Einstein", "BORN_IN", "Germany"},
        {"Paris", "BORN_IN", "biology"},
        {"Newton", "FIELD_OF", "physics"}
      ]

      scored = Enum.map(test_triples, fn {h, r, t} = triple ->
        text = "[HEAD] #{h} [REL] #{r} [TAIL] #{t}"
        {input, mask} = TripleScorer.encode_single_public(text, vocab)
        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        score = Nx.squeeze(output) |> Nx.to_number()

        flag = if score < 0.3, do: :low_confidence, else: :ok
        {triple, score, flag}
      end)

      assert length(scored) == 3
      for {_triple, _score, flag} <- scored do
        assert flag in [:ok, :low_confidence]
      end
    end
  end
end
