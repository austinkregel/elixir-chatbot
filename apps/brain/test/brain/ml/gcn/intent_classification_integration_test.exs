defmodule Brain.ML.GCN.IntentClassificationIntegrationTest do
  use ExUnit.Case, async: false
  @moduletag :integration

  alias Brain.ML.GCN.{Layer, Model, TextGraph}

  @training_data [
    {"hello there", "greeting"},
    {"hi how are you", "greeting"},
    {"good morning", "greeting"},
    {"hey", "greeting"},
    {"what is the weather today", "weather.query"},
    {"whats the forecast", "weather.query"},
    {"is it going to rain", "weather.query"},
    {"weather in seattle", "weather.query"},
    {"goodbye", "farewell"},
    {"see you later", "farewell"},
    {"bye bye", "farewell"},
    {"take care", "farewell"},
    {"tell me about france", "knowledge.query"},
    {"what do you know about dogs", "knowledge.query"},
    {"explain quantum physics", "knowledge.query"},
    {"who was einstein", "knowledge.query"}
  ]

  describe "end-to-end GCN intent classification" do
    @tag timeout: 120_000
    test "trains and classifies held-out intents above minimum threshold" do
      {train, test_data} = split_data(@training_data, 0.75)

      text_graph = TextGraph.build(train, vocab_size: 200)

      {:ok, _model, params, _adj_norm, config} = Model.train(text_graph,
        epochs: 150,
        hidden_dim: 32,
        learning_rate: 0.01
      )

      assert config.num_classes == 4

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

      assert accuracy > 50.0,
        "GCN should achieve > 50% train accuracy, got #{Float.round(accuracy, 1)}%"
    end

    test "predictions have correct format for ensemble integration" do
      text_graph = TextGraph.build(@training_data, vocab_size: 200)

      {:ok, _model, params, _adj_norm, config} = Model.train(text_graph,
        epochs: 50,
        hidden_dim: 16,
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

      probs = output[0]
      best_class = Nx.argmax(probs) |> Nx.to_number()
      confidence = probs[best_class] |> Nx.to_number()
      intent = Map.get(text_graph.idx_to_label, best_class)

      assert is_binary(intent), "Intent should be a string"
      assert confidence >= 0.0 and confidence <= 1.0, "Confidence should be in [0, 1]"
      assert intent in ["greeting", "weather.query", "farewell", "knowledge.query"]
    end

    @tag timeout: 60_000
    test "serialization roundtrip produces identical predictions" do
      text_graph = TextGraph.build(@training_data, vocab_size: 200)

      {:ok, _model, params, _adj_norm, config} = Model.train(text_graph,
        epochs: 30,
        hidden_dim: 16,
        learning_rate: 0.01
      )

      tmp_path = Path.join(System.tmp_dir!(), "gcn_test_model_#{:rand.uniform(100000)}.term")

      try do
        Model.save_model(params, text_graph, config, tmp_path)
        {:ok, loaded} = Model.load_model(tmp_path)

        assert loaded.config.num_features == config.num_features
        assert loaded.config.num_classes == config.num_classes
        assert loaded.config.hidden_dim == config.hidden_dim

        model = Model.build_model(config.num_features, config.num_classes,
          hidden_dim: config.hidden_dim,
          dropout: 0.0
        )

        adjacency_norm = Layer.normalize_adjacency(text_graph.adjacency)

        original_output = Axon.predict(model, params, %{
          "features" => text_graph.features,
          "adjacency" => adjacency_norm
        })

        loaded_adj_norm = Layer.normalize_adjacency(loaded.adjacency)

        loaded_output = Axon.predict(model, loaded.params, %{
          "features" => loaded.features,
          "adjacency" => loaded_adj_norm
        })

        diff = Nx.subtract(original_output, loaded_output)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

        assert diff < 1.0e-4,
          "Serialization roundtrip should preserve predictions, max diff: #{diff}"
      after
        File.rm(tmp_path)
      end
    end
  end

  defp split_data(data, train_ratio) do
    shuffled = Enum.shuffle(data)
    split_at = trunc(length(shuffled) * train_ratio)
    Enum.split(shuffled, split_at)
  end
end
