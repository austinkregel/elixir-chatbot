defmodule Brain.ML.GCN.Layer do
  @moduledoc """
  Graph Convolutional Network layer implementation for Axon/Nx.

  Implements the GCN propagation rule from Kipf & Welling (2017):

      H' = σ(A_hat @ H @ W + b)

  where A_hat = D^(-1/2)(A + I)D^(-1/2) is the symmetrically normalized
  adjacency matrix with self-loops.
  """

  import Nx.Defn

  @doc """
  Compute the symmetric normalization of an adjacency matrix.

  Given A, returns D^(-1/2)(A + I)D^(-1/2) where D is the degree matrix
  of A + I.
  """
  defn normalize_adjacency(adjacency) do
    n = Nx.axis_size(adjacency, 0)
    identity = Nx.eye(n)
    a_hat = Nx.add(adjacency, identity)

    degree = Nx.sum(a_hat, axes: [1])
    d_inv_sqrt = Nx.rsqrt(Nx.max(degree, 1.0e-7))
    d_mat = Nx.multiply(Nx.eye(n), Nx.new_axis(d_inv_sqrt, 1))

    d_mat
    |> Nx.dot([1], a_hat, [0])
    |> Nx.dot([1], d_mat, [0])
  end

  @doc """
  Single GCN layer forward pass: A_hat @ H @ W + b, followed by activation.

  ## Parameters
    - `adjacency_norm` - Pre-normalized adjacency matrix {N, N}
    - `features` - Node feature matrix {N, F}
    - `weights` - Weight matrix {F, H}
    - `bias` - Bias vector {H}
  """
  defn propagate(adjacency_norm, features, weights, bias) do
    features
    |> Nx.dot([1], weights, [0])
    |> then(&Nx.dot(adjacency_norm, [1], &1, [0]))
    |> Nx.add(bias)
  end

  @doc """
  Build a single GCN layer as an Axon custom layer.

  The adjacency matrix is passed as a frozen input (not trainable).
  Returns the Axon node representing the layer output.

  ## Options
    - `:out_features` - Output dimension (required)
    - `:name` - Layer name (required)
    - `:activation` - Activation function atom, e.g. `:relu` (default: `:relu`)
  """
  def gcn_layer(feature_input, adjacency_input, opts \\ []) do
    out_features = Keyword.fetch!(opts, :out_features)
    name = Keyword.fetch!(opts, :name)
    activation = Keyword.get(opts, :activation, :relu)

    linear = feature_input
    |> Axon.dense(out_features, name: "#{name}_dense", use_bias: true)

    propagated = Axon.layer(
      fn linear_out, adj, _opts ->
        Nx.dot(adj, [1], linear_out, [0])
      end,
      [linear, adjacency_input],
      name: "#{name}_propagate",
      op_name: :gcn_propagate
    )

    case activation do
      nil -> propagated
      :none -> propagated
      act -> Axon.activation(propagated, act, name: "#{name}_activation")
    end
  end

  @doc """
  Initialize weights with Xavier uniform initialization.
  Scale = 1 / sqrt(out_features).
  """
  def xavier_uniform_init(out_features) do
    scale = 1.0 / :math.sqrt(out_features)
    fn key, shape, _type ->
      {val, _key} = Nx.Random.uniform(key, -scale, scale, shape: shape)
      val
    end
  end
end
