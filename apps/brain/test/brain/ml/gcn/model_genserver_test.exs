defmodule Brain.ML.GCN.ModelGenServerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.GCN.{Model, TextGraph}

  @tag timeout: 30_000
  describe "GenServer lifecycle" do
    test "starts and reports not ready without a model" do
      name = :"gcn_test_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, pid} = Model.start_link(name: name, world_id: world)
      assert Process.alive?(pid)
      refute Model.ready?(name)
    end

    test "classify returns error when not ready" do
      name = :"gcn_test_classify_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, _pid} = Model.start_link(name: name, world_id: world)

      assert {:error, :not_ready} = Model.classify("test text", name)
    end

    test "ready? returns false for non-existent server" do
      refute Model.ready?(:"nonexistent_gcn_#{:rand.uniform(100_000)}")
    end
  end

  describe "build_model/3" do
    test "builds model with default options" do
      model = Model.build_model(50, 5)
      assert %Axon{} = model
    end

    test "builds model with custom hidden_dim" do
      model = Model.build_model(100, 10, hidden_dim: 64)
      assert %Axon{} = model
    end

    test "builds model with custom dropout" do
      model = Model.build_model(30, 3, dropout: 0.3)
      assert %Axon{} = model
    end

    test "model produces correct output shape" do
      num_features = 20
      num_classes = 4
      n = 10

      model = Model.build_model(num_features, num_classes, hidden_dim: 16, dropout: 0.0)

      template = %{
        "features" => Nx.template({n, num_features}, :f32),
        "adjacency" => Nx.template({n, n}, :f32)
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      features = Nx.broadcast(0.1, {n, num_features})
      adjacency = Nx.eye(n)

      output = predict_fn.(params, %{"features" => features, "adjacency" => adjacency})
      assert Nx.shape(output) == {n, num_classes}

      # Softmax: all rows should sum to ~1
      row_sums = Nx.sum(output, axes: [1]) |> Nx.to_flat_list()
      for sum <- row_sums do
        assert_in_delta sum, 1.0, 0.01
      end
    end
  end

  describe "save_model/4 and load_model/1" do
    @tag :tmp_dir
    test "roundtrip serialization preserves model data", %{tmp_dir: tmp_dir} do
      documents = [
        {"hello world", "greeting"},
        {"good morning", "greeting"},
        {"what is the weather", "weather"},
        {"how hot is it", "weather"},
        {"play some music", "music"},
        {"turn on the radio", "music"}
      ]

      text_graph = TextGraph.build(documents, vocab_size: 50)

      {:ok, _model, params, _adj_norm, config} = Model.train(text_graph,
        epochs: 5, hidden_dim: 16, learning_rate: 0.01)

      path = Path.join(tmp_dir, "test_gcn_model.term")
      Model.save_model(params, text_graph, config, path)
      assert File.exists?(path)

      {:ok, loaded} = Model.load_model(path)
      assert Map.has_key?(loaded, :params)
      assert Map.has_key?(loaded, :config)
    end

    test "load_model returns error for missing file" do
      result = Model.load_model("/tmp/nonexistent_gcn_#{:rand.uniform(100_000)}.term")
      assert {:error, _} = result
    end
  end

  describe "train/2 edge cases" do
    test "trains with minimal dataset" do
      documents = [
        {"hello", "a"},
        {"world", "b"}
      ]

      text_graph = TextGraph.build(documents, vocab_size: 20)

      result = Model.train(text_graph, epochs: 3, hidden_dim: 8, learning_rate: 0.01)
      assert {:ok, _model, _params, _adj_norm, _config} = result
    end

    test "trains with single-class data" do
      documents = [
        {"hello world", "only_class"},
        {"good morning", "only_class"},
        {"how are you", "only_class"}
      ]

      text_graph = TextGraph.build(documents, vocab_size: 20)
      assert text_graph.num_classes == 1

      result = Model.train(text_graph, epochs: 3, hidden_dim: 8)
      assert {:ok, _model, _params, _adj_norm, _config} = result
    end
  end
end
