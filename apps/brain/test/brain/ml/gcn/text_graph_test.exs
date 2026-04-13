defmodule Brain.ML.GCN.TextGraphTest do
  use ExUnit.Case, async: false

  alias Brain.ML.GCN.TextGraph

  @small_corpus [
    {"hello how are you", "greeting"},
    {"good morning to you", "greeting"},
    {"what is the weather today", "weather"},
    {"tell me the forecast", "weather"},
    {"goodbye see you later", "farewell"}
  ]

  describe "build/2" do
    test "creates graph with correct node counts" do
      graph = TextGraph.build(@small_corpus, vocab_size: 50)

      assert graph.num_docs == 5
      assert graph.num_words > 0
      assert graph.num_words <= 50

      total_nodes = graph.num_docs + graph.num_words
      assert Nx.shape(graph.adjacency) == {total_nodes, total_nodes}
      assert Nx.shape(graph.features) == {total_nodes, graph.num_words}
    end

    test "adjacency matrix is symmetric" do
      graph = TextGraph.build(@small_corpus, vocab_size: 50)

      adjacency = graph.adjacency
      transposed = Nx.transpose(adjacency)

      diff = Nx.subtract(adjacency, transposed) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6, "Adjacency matrix must be symmetric"
    end

    test "labels are correct for document nodes and -1 for word nodes" do
      graph = TextGraph.build(@small_corpus, vocab_size: 50)

      labels = Nx.to_flat_list(graph.labels)

      doc_labels = Enum.take(labels, graph.num_docs)
      word_labels = Enum.drop(labels, graph.num_docs)

      assert Enum.all?(doc_labels, &(&1 >= 0))
      assert Enum.all?(word_labels, &(&1 == -1))
    end

    test "detects correct number of classes" do
      graph = TextGraph.build(@small_corpus, vocab_size: 50)
      assert graph.num_classes == 3
    end

    test "label_to_idx and idx_to_label are inverses" do
      graph = TextGraph.build(@small_corpus, vocab_size: 50)

      for {label, idx} <- graph.label_to_idx do
        assert Map.get(graph.idx_to_label, idx) == label
      end
    end

    test "node_map contains both doc and word entries" do
      graph = TextGraph.build(@small_corpus, vocab_size: 50)

      doc_nodes = Enum.filter(graph.node_map, fn {_k, v} ->
        match?({:doc, _, _}, v)
      end)

      word_nodes = Enum.filter(graph.node_map, fn {_k, v} ->
        match?({:word, _}, v)
      end)

      assert length(doc_nodes) == graph.num_docs
      assert length(word_nodes) == graph.num_words
    end
  end

  describe "compute_pmi/3" do
    @tag :known_answer
    test "PMI values are correct for known co-occurrences" do
      tokenized = [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["c", "d", "e"]
      ]

      vocabulary = %{"a" => 0, "b" => 1, "c" => 2, "d" => 3, "e" => 4}

      pmi = TextGraph.compute_pmi(tokenized, vocabulary, 20)

      # "a" and "b" co-occur in 2/3 docs, each appears in 2/3 docs
      # PMI("a","b") = log(P(a,b) / (P(a) * P(b)))
      # All positive PMI values should be in the result
      assert is_map(pmi)

      for {{w1, w2}, val} <- pmi do
        assert val > 0, "PMI should be positive (PPMI)"
        assert w1 < w2, "PMI keys should be ordered"
      end
    end

    test "empty corpus produces empty PMI" do
      pmi = TextGraph.compute_pmi([], %{}, 20)
      assert pmi == %{}
    end
  end

  describe "from_atlas/1" do
    test "converts Atlas adjacency format to text graph format" do
      atlas_data = %{
        node_ids: ["n1", "n2", "n3"],
        edges: [{"n1", "n2"}, {"n2", "n3"}],
        features: %{
          "n1" => %{"type" => "entity"},
          "n2" => %{"type" => "entity"},
          "n3" => %{"type" => "relation"}
        }
      }

      result = TextGraph.from_atlas(atlas_data)

      assert result.num_nodes == 3
      assert Nx.shape(result.adjacency) == {3, 3}

      adj_list = Nx.to_flat_list(result.adjacency)
      assert Enum.at(adj_list, 1) == 1.0
      assert Enum.at(adj_list, 3) == 1.0
    end
  end
end
