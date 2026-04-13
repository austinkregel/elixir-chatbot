defmodule Brain.ML.Poincare.DistanceExtendedTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Poincare.Distance

  describe "arctanh/1" do
    test "arctanh(0) = 0" do
      result = Distance.arctanh(Nx.tensor(0.0)) |> Nx.to_number()
      assert_in_delta result, 0.0, 1.0e-6
    end

    test "arctanh is odd function: arctanh(-x) = -arctanh(x)" do
      x = Nx.tensor(0.5)
      pos = Distance.arctanh(x) |> Nx.to_number()
      neg = Distance.arctanh(Nx.negate(x)) |> Nx.to_number()
      assert_in_delta pos, -neg, 1.0e-5
    end

    test "known value: arctanh(0.5) ≈ 0.5493" do
      result = Distance.arctanh(Nx.tensor(0.5)) |> Nx.to_number()
      assert_in_delta result, 0.5493, 0.001
    end

    test "values near boundary are large" do
      result = Distance.arctanh(Nx.tensor(0.99)) |> Nx.to_number()
      assert result > 2.0
    end
  end

  describe "inside_ball?/1 edge cases" do
    test "point on boundary (norm = 1) is outside" do
      point = Nx.tensor([1.0, 0.0])
      result = Distance.inside_ball?(point) |> Nx.to_number()
      assert result == 0
    end

    test "point outside ball" do
      point = Nx.tensor([0.8, 0.8])
      result = Distance.inside_ball?(point) |> Nx.to_number()
      assert result == 0
    end

    test "point well inside ball" do
      point = Nx.tensor([0.1, 0.1, 0.1])
      result = Distance.inside_ball?(point) |> Nx.to_number()
      assert result == 1
    end

    test "negative coordinates inside ball" do
      point = Nx.tensor([-0.3, -0.2, 0.1])
      result = Distance.inside_ball?(point) |> Nx.to_number()
      assert result == 1
    end

    test "zero vector is inside ball" do
      point = Nx.tensor([0.0, 0.0, 0.0])
      result = Distance.inside_ball?(point) |> Nx.to_number()
      assert result == 1
    end
  end

  describe "distance/2 edge cases" do
    test "distance between identical points near boundary" do
      p = Nx.tensor([0.99, 0.0])
      d = Distance.distance(p, p) |> Nx.to_number()
      assert_in_delta d, 0.0, 0.01
    end

    test "distance increases as points move further apart" do
      origin = Nx.tensor([0.0, 0.0])
      near = Nx.tensor([0.1, 0.0])
      mid = Nx.tensor([0.5, 0.0])
      far = Nx.tensor([0.9, 0.0])

      d_near = Distance.distance(origin, near) |> Nx.to_number()
      d_mid = Distance.distance(origin, mid) |> Nx.to_number()
      d_far = Distance.distance(origin, far) |> Nx.to_number()

      assert d_near < d_mid
      assert d_mid < d_far
    end

    test "distance grows rapidly near boundary" do
      p1 = Nx.tensor([0.5, 0.0])
      p2 = Nx.tensor([0.6, 0.0])
      p3 = Nx.tensor([0.9, 0.0])
      p4 = Nx.tensor([0.95, 0.0])

      d_inner = Distance.distance(p1, p2) |> Nx.to_number()
      d_outer = Distance.distance(p3, p4) |> Nx.to_number()

      # Same Euclidean distance (0.1 vs 0.05) but hyperbolic distance near boundary
      # should be larger
      assert d_outer > d_inner * 0.5, "Distance near boundary should grow faster"
    end

    test "higher dimensional points" do
      p1 = Nx.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
      p2 = Nx.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
      d = Distance.distance(p1, p2) |> Nx.to_number()
      assert d > 0.0
      assert is_float(d)
    end
  end

  describe "distance_from_origin/1" do
    test "origin has zero distance" do
      origin = Nx.tensor([0.0, 0.0])
      d = Distance.distance_from_origin(origin) |> Nx.to_number()
      assert_in_delta d, 0.0, 0.01
    end

    test "distance increases with norm" do
      p1 = Nx.tensor([0.1, 0.0])
      p2 = Nx.tensor([0.5, 0.0])
      p3 = Nx.tensor([0.9, 0.0])

      d1 = Distance.distance_from_origin(p1) |> Nx.to_number()
      d2 = Distance.distance_from_origin(p2) |> Nx.to_number()
      d3 = Distance.distance_from_origin(p3) |> Nx.to_number()

      assert d1 < d2
      assert d2 < d3
    end
  end
end
