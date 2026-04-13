defmodule Brain.ML.Poincare.Distance do
  @moduledoc """
  Poincare ball distance functions.

  Implements the distance function for the Poincare ball model of hyperbolic space:

      d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))

  Points must satisfy ||x|| < 1 (inside the unit ball).
  """

  import Nx.Defn

  @doc """
  Compute the Poincare ball distance between two points (or batches of points).

  ## Parameters
    - `u` - Point(s) in the Poincare ball, shape {d} or {batch, d}
    - `v` - Point(s) in the Poincare ball, shape {d} or {batch, d}

  ## Returns
    Scalar or tensor of distances.
  """
  defn distance(u, v) do
    diff_sq = Nx.sum(Nx.pow(Nx.subtract(u, v), 2), axes: [-1])
    u_sq = Nx.sum(Nx.pow(u, 2), axes: [-1])
    v_sq = Nx.sum(Nx.pow(v, 2), axes: [-1])

    denom = Nx.multiply(Nx.subtract(1.0, u_sq), Nx.subtract(1.0, v_sq))
    arg = Nx.add(1.0, Nx.divide(Nx.multiply(2.0, diff_sq), Nx.max(denom, 1.0e-10)))

    Nx.acosh(Nx.max(arg, 1.0 + 1.0e-7))
  end

  @doc """
  Compute distance from the origin to a point.

  d(0, v) = 2 * arctanh(||v||)
  """
  defn distance_from_origin(v) do
    norm = Nx.sqrt(Nx.sum(Nx.pow(v, 2), axes: [-1]))
    Nx.multiply(2.0, arctanh(Nx.min(norm, 1.0 - 1.0e-7)))
  end

  @doc """
  Hyperbolic arctanh: arctanh(x) = 0.5 * ln((1 + x) / (1 - x))
  """
  defn arctanh(x) do
    x_clamped = Nx.min(Nx.max(x, -1.0 + 1.0e-7), 1.0 - 1.0e-7)
    Nx.multiply(0.5, Nx.log(Nx.divide(Nx.add(1.0, x_clamped), Nx.subtract(1.0, x_clamped))))
  end

  @doc """
  Check if a point (or batch of points) is inside the Poincare ball.
  """
  defn inside_ball?(x) do
    norms = Nx.sqrt(Nx.sum(Nx.pow(x, 2), axes: [-1]))
    Nx.less(norms, 1.0)
  end
end
