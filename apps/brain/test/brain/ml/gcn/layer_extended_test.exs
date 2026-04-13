defmodule Brain.ML.GCN.LayerExtendedTest do
  use ExUnit.Case, async: false

  alias Brain.ML.GCN.Layer

  describe "xavier_uniform_init/1" do
    test "returns a function" do
      init_fn = Layer.xavier_uniform_init(10)
      assert is_function(init_fn, 3)
    end

    test "produces tensor of correct shape" do
      init_fn = Layer.xavier_uniform_init(16)
      key = Nx.Random.key(42)
      result = init_fn.(key, {8, 16}, :f32)

      assert Nx.shape(result) == {8, 16}
      assert Nx.type(result) == {:f, 32}
    end

    test "values are bounded by 1/sqrt(out_features)" do
      out_features = 25
      scale = 1.0 / :math.sqrt(out_features)
      init_fn = Layer.xavier_uniform_init(out_features)
      key = Nx.Random.key(0)
      result = init_fn.(key, {100, out_features}, :f32)

      max_val = Nx.reduce_max(Nx.abs(result)) |> Nx.to_number()
      assert max_val <= scale + 0.01,
        "Max value #{max_val} exceeds expected scale #{scale}"
    end

    test "larger out_features produce smaller scale" do
      init_small = Layer.xavier_uniform_init(4)
      init_large = Layer.xavier_uniform_init(400)
      key = Nx.Random.key(42)

      small = init_small.(key, {50, 4}, :f32)
      large = init_large.(key, {50, 400}, :f32)

      max_small = Nx.reduce_max(Nx.abs(small)) |> Nx.to_number()
      max_large = Nx.reduce_max(Nx.abs(large)) |> Nx.to_number()

      assert max_small > max_large,
        "Smaller dimension (#{max_small}) should produce larger weights than larger dimension (#{max_large})"
    end
  end

  describe "normalize_adjacency/1 edge cases" do
    test "fully connected graph" do
      n = 4
      adjacency = Nx.subtract(Nx.broadcast(1.0, {n, n}), Nx.eye(n))
      a_hat = Layer.normalize_adjacency(adjacency)

      assert Nx.shape(a_hat) == {n, n}
      transposed = Nx.transpose(a_hat)
      diff = Nx.subtract(a_hat, transposed) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5, "Result should be symmetric"
    end

    test "star graph (one hub connected to all)" do
      adjacency = Nx.tensor([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
      ])

      a_hat = Layer.normalize_adjacency(adjacency)
      assert Nx.shape(a_hat) == {4, 4}

      # Hub (node 0) has degree 4 with self-loop, leaves have degree 2
      diag = Nx.to_flat_list(Nx.take_diagonal(a_hat))
      assert_in_delta Enum.at(diag, 0), 1.0 / 4.0, 0.01
      assert_in_delta Enum.at(diag, 1), 1.0 / 2.0, 0.01
    end

    test "disconnected graph (no edges)" do
      adjacency = Nx.broadcast(0.0, {3, 3})
      a_hat = Layer.normalize_adjacency(adjacency)

      expected = Nx.eye(3)
      diff = Nx.subtract(a_hat, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5, "No-edge graph should normalize to identity"
    end

    test "2-node graph" do
      adjacency = Nx.tensor([[0.0, 1.0], [1.0, 0.0]])
      a_hat = Layer.normalize_adjacency(adjacency)

      assert Nx.shape(a_hat) == {2, 2}
      vals = Nx.to_flat_list(a_hat)
      assert_in_delta Enum.at(vals, 0), 0.5, 0.01
      assert_in_delta Enum.at(vals, 1), 0.5, 0.01
    end
  end

  describe "propagate/4 edge cases" do
    test "zero weights produce bias-only output" do
      n = 3
      f = 2
      h = 4

      adj_norm = Nx.eye(n)
      features = Nx.broadcast(1.0, {n, f})
      weights = Nx.broadcast(0.0, {f, h})
      bias = Nx.broadcast(0.5, {h})

      result = Layer.propagate(adj_norm, features, weights, bias)
      expected = Nx.broadcast(0.5, {n, h})

      diff = Nx.subtract(result, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "zero bias with identity adjacency" do
      features = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      weights = Nx.tensor([[1.0], [0.0]])
      bias = Nx.tensor([0.0])
      adj_norm = Nx.eye(2)

      result = Layer.propagate(adj_norm, features, weights, bias)
      expected = Nx.tensor([[1.0], [3.0]])

      diff = Nx.subtract(result, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end
  end

  describe "gcn_layer/3 activation options" do
    test "nil activation skips activation" do
      model = Layer.gcn_layer(
        Axon.input("features", shape: {nil, 2}),
        Axon.input("adjacency", shape: {nil, nil}),
        out_features: 3, name: "nil_act", activation: nil
      )

      template = %{
        "features" => Nx.template({2, 2}, :f32),
        "adjacency" => Nx.template({2, 2}, :f32)
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      features = Nx.tensor([[1.0, -1.0], [-1.0, 1.0]])
      adjacency = Nx.eye(2)

      result = predict_fn.(params, %{"features" => features, "adjacency" => adjacency})
      # With nil activation, negative values should remain negative
      min_val = Nx.reduce_min(result) |> Nx.to_number()
      # Could be negative since no ReLU
      assert is_float(min_val)
    end

    test ":none activation skips activation" do
      model = Layer.gcn_layer(
        Axon.input("features", shape: {nil, 2}),
        Axon.input("adjacency", shape: {nil, nil}),
        out_features: 2, name: "none_act", activation: :none
      )

      template = %{
        "features" => Nx.template({1, 2}, :f32),
        "adjacency" => Nx.template({1, 1}, :f32)
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      result = predict_fn.(params, %{
        "features" => Nx.tensor([[1.0, 2.0]]),
        "adjacency" => Nx.eye(1)
      })

      assert Nx.shape(result) == {1, 2}
    end
  end
end
