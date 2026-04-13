defmodule Brain.ML.GCN.LayerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.GCN.Layer

  describe "normalize_adjacency/1" do
    @tag :known_answer
    test "produces correct symmetric normalization for a 3-node graph" do
      # Graph: A-B, B-C (linear chain)
      adjacency = Nx.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
      ])

      a_hat = Layer.normalize_adjacency(adjacency)

      # A + I:
      # [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
      # Degree: [2, 3, 2]
      # D^(-1/2): [1/sqrt(2), 1/sqrt(3), 1/sqrt(2)]

      assert Nx.shape(a_hat) == {3, 3}

      a_hat_vals = Nx.to_flat_list(a_hat)

      # Verify symmetry
      for i <- 0..2, j <- 0..2 do
        val_ij = Enum.at(a_hat_vals, i * 3 + j)
        val_ji = Enum.at(a_hat_vals, j * 3 + i)
        assert_in_delta val_ij, val_ji, 1.0e-5, "A_hat must be symmetric at (#{i},#{j})"
      end

      # Diagonal: D^(-1/2) * 1 * D^(-1/2) = 1/degree
      assert_in_delta Enum.at(a_hat_vals, 0), 1.0 / 2.0, 1.0e-5
      assert_in_delta Enum.at(a_hat_vals, 4), 1.0 / 3.0, 1.0e-5
      assert_in_delta Enum.at(a_hat_vals, 8), 1.0 / 2.0, 1.0e-5

      # Off-diagonal A-B: 1/sqrt(2) * 1 * 1/sqrt(3) = 1/sqrt(6)
      assert_in_delta Enum.at(a_hat_vals, 1), 1.0 / :math.sqrt(6), 1.0e-5
    end

    @tag :property
    test "identity adjacency preserves self-loop-only normalization" do
      n = 5
      adjacency = Nx.eye(n)
      a_hat = Layer.normalize_adjacency(adjacency)

      # A + I = 2*I, D = 2*I, D^(-1/2) = (1/sqrt(2))*I
      # Result = (1/sqrt(2))*I * 2*I * (1/sqrt(2))*I = I
      expected = Nx.eye(n)

      assert Nx.shape(a_hat) == {n, n}
      assert_all_close(a_hat, expected, atol: 1.0e-5)
    end

    @tag :property
    test "result is always symmetric" do
      adjacency = Nx.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0]
      ])

      a_hat = Layer.normalize_adjacency(adjacency)
      transposed = Nx.transpose(a_hat)

      assert_all_close(a_hat, transposed, atol: 1.0e-6)
    end
  end

  describe "propagate/4" do
    @tag :known_answer
    test "computes A_hat @ (X @ W) + b correctly" do
      adjacency_norm = Nx.tensor([
        [0.5, 0.5],
        [0.5, 0.5]
      ])

      features = Nx.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
      ])

      weights = Nx.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
      ])

      bias = Nx.tensor([0.0, 0.0, 0.0])

      result = Layer.propagate(adjacency_norm, features, weights, bias)

      # X @ W = [[1, 0, 0], [0, 1, 0]]
      # A_hat @ (X @ W) = [[0.5, 0.5, 0], [0.5, 0.5, 0]]
      expected = Nx.tensor([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0]
      ])

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    @tag :property
    test "output shape is {N, H} for N nodes and H output features" do
      n = 10
      f = 5
      h = 3

      adjacency_norm = Nx.eye(n)
      features = Nx.broadcast(1.0, {n, f})
      weights = Nx.broadcast(0.1, {f, h})
      bias = Nx.broadcast(0.0, {h})

      result = Layer.propagate(adjacency_norm, features, weights, bias)

      assert Nx.shape(result) == {n, h}
    end
  end

  describe "gcn_layer/3 (Axon custom layer)" do
    @tag :property
    test "identity adjacency behaves like a standard dense layer" do
      n = 5
      features_dim = 3
      out_dim = 2

      model = Layer.gcn_layer(
        Axon.input("features", shape: {nil, features_dim}),
        Axon.input("adjacency", shape: {nil, nil}),
        out_features: out_dim,
        name: "test_gcn",
        activation: :relu
      )

      template = %{
        "features" => Nx.template({n, features_dim}, :f32),
        "adjacency" => Nx.template({n, n}, :f32)
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      features = Nx.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
      ])

      identity_adj = Nx.eye(n)

      result = predict_fn.(params, %{"features" => features, "adjacency" => identity_adj})
      assert Nx.shape(result) == {n, out_dim}
    end

    @tag :property
    test "permutation equivariance" do
      n = 4
      features_dim = 3
      out_dim = 2

      model = Layer.gcn_layer(
        Axon.input("features", shape: {nil, features_dim}),
        Axon.input("adjacency", shape: {nil, nil}),
        out_features: out_dim,
        name: "perm_gcn",
        activation: :relu
      )

      template = %{
        "features" => Nx.template({n, features_dim}, :f32),
        "adjacency" => Nx.template({n, n}, :f32)
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      features = Nx.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
      ])

      adjacency = Nx.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
      ])

      result_original = predict_fn.(params, %{"features" => features, "adjacency" => adjacency})

      perm = [2, 0, 3, 1]
      perm_matrix = Nx.tensor(
        for i <- perm do
          for j <- 0..3, do: if(j == i, do: 1.0, else: 0.0)
        end
      )

      perm_features = Nx.dot(perm_matrix, features)
      perm_adjacency = perm_matrix |> Nx.dot(adjacency) |> Nx.dot(Nx.transpose(perm_matrix))

      result_permuted = predict_fn.(params, %{"features" => perm_features, "adjacency" => perm_adjacency})

      expected_permuted = Nx.dot(perm_matrix, result_original)

      assert_all_close(result_permuted, expected_permuted, atol: 1.0e-4)
    end

    @tag :known_answer
    test "disconnected nodes depend only on self-features" do
      n = 3
      features_dim = 2
      out_dim = 2

      model = Layer.gcn_layer(
        Axon.input("features", shape: {nil, features_dim}),
        Axon.input("adjacency", shape: {nil, nil}),
        out_features: out_dim,
        name: "disc_gcn",
        activation: nil
      )

      template = %{
        "features" => Nx.template({n, features_dim}, :f32),
        "adjacency" => Nx.template({n, n}, :f32)
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      features = Nx.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
      ])

      no_edges = Nx.broadcast(0.0, {3, 3})
      adj_norm = Layer.normalize_adjacency(no_edges)

      result = predict_fn.(params, %{"features" => features, "adjacency" => adj_norm})
      assert Nx.shape(result) == {n, out_dim}
    end

    @tag :known_answer
    test "single node graph works" do
      model = Layer.gcn_layer(
        Axon.input("features", shape: {nil, 2}),
        Axon.input("adjacency", shape: {nil, nil}),
        out_features: 3,
        name: "single_gcn",
        activation: :relu
      )

      template = %{
        "features" => Nx.template({1, 2}, :f32),
        "adjacency" => Nx.template({1, 1}, :f32)
      }

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      features = Nx.tensor([[1.0, 2.0]])
      adjacency = Nx.tensor([[0.0]])
      adj_norm = Layer.normalize_adjacency(adjacency)

      result = predict_fn.(params, %{"features" => features, "adjacency" => adj_norm})
      assert Nx.shape(result) == {1, 3}
    end
  end

  defp assert_all_close(actual, expected, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)

    diff = Nx.subtract(actual, expected) |> Nx.abs()
    max_diff = Nx.reduce_max(diff) |> Nx.to_number()

    assert max_diff < atol,
      "Tensors not close enough. Max diff: #{max_diff}, tolerance: #{atol}"
  end
end
