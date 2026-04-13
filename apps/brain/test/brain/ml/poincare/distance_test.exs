defmodule Brain.ML.Poincare.DistanceTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Poincare.Distance

  describe "distance/2" do
    @tag :known_answer
    test "distance at origin matches hand-computed value" do
      u = Nx.tensor([0.0, 0.0])
      v = Nx.tensor([0.5, 0.0])

      # d([0,0], [0.5, 0]) = arcosh(1 + 2*0.25/(1*0.75)) = arcosh(1.6667)
      expected = :math.acosh(1.0 + 2.0 * 0.25 / (1.0 * 0.75))
      result = Distance.distance(u, v) |> Nx.to_number()

      assert_in_delta result, expected, 1.0e-4
    end

    @tag :known_answer
    test "distance near boundary is very large" do
      u = Nx.tensor([0.99, 0.0])
      v = Nx.tensor([0.0, 0.99])

      result = Distance.distance(u, v) |> Nx.to_number()
      assert result > 5.0, "Points near boundary should have large distance, got #{result}"
    end

    @tag :property
    test "self-distance is zero" do
      points = [
        Nx.tensor([0.0, 0.0]),
        Nx.tensor([0.5, 0.3]),
        Nx.tensor([0.1, -0.2]),
        Nx.tensor([-0.7, 0.1])
      ]

      for p <- points do
        d = Distance.distance(p, p) |> Nx.to_number()
        assert_in_delta d, 0.0, 1.0e-3, "Self-distance should be 0, got #{d}"
      end
    end

    @tag :property
    test "symmetry: d(u, v) = d(v, u)" do
      u = Nx.tensor([0.3, 0.4])
      v = Nx.tensor([-0.2, 0.5])

      d_uv = Distance.distance(u, v) |> Nx.to_number()
      d_vu = Distance.distance(v, u) |> Nx.to_number()

      assert_in_delta d_uv, d_vu, 1.0e-5
    end

    @tag :property
    test "non-negativity: d(u, v) >= 0" do
      pairs = [
        {Nx.tensor([0.1, 0.2]), Nx.tensor([0.3, -0.1])},
        {Nx.tensor([0.0, 0.0]), Nx.tensor([0.5, 0.5])},
        {Nx.tensor([-0.3, 0.4]), Nx.tensor([0.4, -0.3])}
      ]

      for {u, v} <- pairs do
        d = Distance.distance(u, v) |> Nx.to_number()
        assert d >= 0.0, "Distance should be non-negative, got #{d}"
      end
    end

    @tag :property
    test "triangle inequality: d(u, w) <= d(u, v) + d(v, w)" do
      u = Nx.tensor([0.1, 0.2])
      v = Nx.tensor([0.3, -0.1])
      w = Nx.tensor([-0.2, 0.4])

      d_uv = Distance.distance(u, v) |> Nx.to_number()
      d_vw = Distance.distance(v, w) |> Nx.to_number()
      d_uw = Distance.distance(u, w) |> Nx.to_number()

      assert d_uw <= d_uv + d_vw + 1.0e-5,
        "Triangle inequality violated: #{d_uw} > #{d_uv} + #{d_vw}"
    end

    @tag :known_answer
    test "numerical stability near boundary" do
      u = Nx.tensor([0.9999, 0.0])
      v = Nx.tensor([0.0, 0.9999])

      result = Distance.distance(u, v) |> Nx.to_number()
      assert is_float(result)
      refute result == :nan
      refute result == :infinity
      assert result > 0.0
    end

    @tag :property
    test "batch computation gives same results as individual" do
      u_batch = Nx.tensor([
        [0.1, 0.2],
        [0.3, -0.1],
        [-0.2, 0.4]
      ])

      v_batch = Nx.tensor([
        [0.3, 0.4],
        [-0.1, 0.2],
        [0.5, -0.3]
      ])

      batch_result = Distance.distance(u_batch, v_batch) |> Nx.to_flat_list()

      individual_results = for i <- 0..2 do
        u = u_batch[i]
        v = v_batch[i]
        Distance.distance(u, v) |> Nx.to_number()
      end

      for {batch_val, indiv_val} <- Enum.zip(batch_result, individual_results) do
        assert_in_delta batch_val, indiv_val, 1.0e-4
      end
    end
  end

  describe "distance_from_origin/1" do
    @tag :known_answer
    test "equals 2 * arctanh(||v||)" do
      v = Nx.tensor([0.3, 0.4])
      norm = :math.sqrt(0.3 * 0.3 + 0.4 * 0.4)
      expected = 2.0 * :math.atanh(norm)

      result = Distance.distance_from_origin(v) |> Nx.to_number()
      assert_in_delta result, expected, 1.0e-4
    end

    @tag :known_answer
    test "origin point has zero distance from origin" do
      v = Nx.tensor([0.0, 0.0, 0.0])
      result = Distance.distance_from_origin(v) |> Nx.to_number()
      assert_in_delta result, 0.0, 1.0e-4
    end
  end

  describe "inside_ball?/1" do
    test "origin is inside the ball" do
      assert Distance.inside_ball?(Nx.tensor([0.0, 0.0])) |> Nx.to_number() == 1
    end

    test "points with norm < 1 are inside" do
      assert Distance.inside_ball?(Nx.tensor([0.5, 0.3])) |> Nx.to_number() == 1
    end
  end
end
