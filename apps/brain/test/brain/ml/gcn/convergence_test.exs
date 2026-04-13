defmodule Brain.ML.GCN.ConvergenceTest do
  use ExUnit.Case, async: false
  @moduletag :convergence

  alias Brain.ML.GCN.{Layer, Model, TextGraph}

  describe "synthetic graph convergence" do
    @tag timeout: 120_000
    test "GCN converges on planted-partition graph" do
      # 3 communities of 8 nodes each with distinct features
      documents = generate_planted_partition_corpus(3, 8)

      text_graph = TextGraph.build(documents, vocab_size: 100)

      assert text_graph.num_classes == 3

      {:ok, _model, params, _adj_norm, config} = Model.train(text_graph,
        epochs: 200,
        hidden_dim: 32,
        learning_rate: 0.01
      )

      model = Model.build_model(config.num_features, config.num_classes,
        hidden_dim: config.hidden_dim,
        dropout: 0.0
      )

      adjacency_norm = Layer.normalize_adjacency(text_graph.adjacency)

      output = Axon.predict(model, params, %{
        "features" => text_graph.features,
        "adjacency" => adjacency_norm
      })

      predictions = Nx.argmax(output, axis: 1)
      labels = text_graph.labels
      mask = Nx.not_equal(labels, -1) |> Nx.as_type(:f32)

      correct = Nx.equal(predictions, Nx.max(labels, 0))
      |> Nx.as_type(:f32)
      |> Nx.multiply(mask)
      |> Nx.sum()
      |> Nx.to_number()

      total = Nx.sum(mask) |> Nx.to_number()
      accuracy = correct / max(total, 1) * 100

      assert accuracy > 60.0,
        "Train accuracy should be > 60% on planted partition, got #{Float.round(accuracy, 1)}%"
    end

    @tag timeout: 120_000
    test "overfitting sanity check - achieves near-100% on tiny dataset" do
      documents = [
        {"alpha beta gamma", "class_a"},
        {"alpha beta delta", "class_a"},
        {"epsilon zeta eta", "class_b"},
        {"epsilon zeta theta", "class_b"},
        {"iota kappa lambda", "class_c"},
        {"iota kappa mu", "class_c"}
      ]

      text_graph = TextGraph.build(documents, vocab_size: 50)

      {:ok, _model, params, _adj_norm, config} = Model.train(text_graph,
        epochs: 200,
        hidden_dim: 16,
        learning_rate: 0.01,
        dropout: 0.0
      )

      model = Model.build_model(config.num_features, config.num_classes,
        hidden_dim: config.hidden_dim,
        dropout: 0.0
      )

      adjacency_norm = Layer.normalize_adjacency(text_graph.adjacency)

      output = Axon.predict(model, params, %{
        "features" => text_graph.features,
        "adjacency" => adjacency_norm
      })

      predictions = Nx.argmax(output, axis: 1)
      labels = text_graph.labels
      mask = Nx.not_equal(labels, -1) |> Nx.as_type(:f32)

      correct = Nx.equal(predictions, Nx.max(labels, 0))
      |> Nx.as_type(:f32)
      |> Nx.multiply(mask)
      |> Nx.sum()
      |> Nx.to_number()

      total = Nx.sum(mask) |> Nx.to_number()
      accuracy = correct / max(total, 1) * 100

      assert accuracy > 90.0,
        "Overfitting sanity: should achieve > 90% train accuracy, got #{Float.round(accuracy, 1)}%"
    end
  end

  defp generate_planted_partition_corpus(num_communities, docs_per_community) do
    community_words = %{
      0 => ~w(alpha beta gamma delta epsilon),
      1 => ~w(zeta eta theta iota kappa),
      2 => ~w(lambda mu nu xi omicron)
    }

    for c <- 0..(num_communities - 1),
        i <- 1..docs_per_community do
      words = Map.get(community_words, c, ~w(unknown))
      selected = Enum.take_random(words, 3) |> Enum.join(" ")
      {selected, "class_#{c}"}
    end
  end
end
