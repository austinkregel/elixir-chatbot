defmodule Brain.ML.Poincare.OptimizerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Poincare.Optimizer

  describe "riemannian_grad/2" do
    @tag :known_answer
    test "gradient scaling matches formula" do
      euclidean_grad = Nx.tensor([1.0, 2.0])
      theta = Nx.tensor([0.5, 0.0])

      # ||theta||^2 = 0.25
      # scale = (1 - 0.25)^2 / 4 = 0.75^2 / 4 = 0.5625 / 4 = 0.140625
      expected_scale = :math.pow(1.0 - 0.25, 2) / 4.0

      result = Optimizer.riemannian_grad(euclidean_grad, theta)
      result_list = Nx.to_flat_list(result)

      assert_in_delta Enum.at(result_list, 0), 1.0 * expected_scale, 1.0e-5
      assert_in_delta Enum.at(result_list, 1), 2.0 * expected_scale, 1.0e-5
    end

    @tag :known_answer
    test "at origin, scale is 1/4" do
      grad = Nx.tensor([1.0, 1.0])
      theta = Nx.tensor([0.0, 0.0])

      result = Optimizer.riemannian_grad(grad, theta)
      result_list = Nx.to_flat_list(result)

      assert_in_delta Enum.at(result_list, 0), 0.25, 1.0e-5
      assert_in_delta Enum.at(result_list, 1), 0.25, 1.0e-5
    end
  end

  describe "project_to_ball/2" do
    @tag :known_answer
    test "points outside the ball are projected to boundary" do
      theta = Nx.tensor([1.5, 0.0])
      result = Optimizer.project_to_ball(theta)
      norm = Nx.sqrt(Nx.sum(Nx.pow(result, 2))) |> Nx.to_number()

      assert norm < 1.0, "Projected point should be inside ball, norm: #{norm}"
      assert_in_delta norm, 1.0 - 1.0e-5, 1.0e-4
    end

    @tag :known_answer
    test "points inside the ball are unchanged" do
      theta = Nx.tensor([0.3, 0.4])
      result = Optimizer.project_to_ball(theta)

      diff = Nx.subtract(result, theta)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

      assert diff < 1.0e-5, "Points inside ball should be unchanged"
    end

    @tag :property
    test "gradient step always stays in ball" do
      theta = Nx.tensor([0.8, 0.5])
      grad = Nx.tensor([5.0, 5.0])

      updated = Nx.subtract(theta, Nx.multiply(0.1, grad))
      projected = Optimizer.project_to_ball(updated)

      norm = Nx.sqrt(Nx.sum(Nx.pow(projected, 2))) |> Nx.to_number()
      assert norm < 1.0, "After projection, norm should be < 1, got #{norm}"
    end
  end

  describe "step/3" do
    test "full step stays in ball" do
      theta = Nx.tensor([0.5, 0.3])
      grad = Nx.tensor([1.0, 1.0])
      lr = 0.1

      result = Optimizer.step(theta, grad, lr)
      norm = Nx.sqrt(Nx.sum(Nx.pow(result, 2))) |> Nx.to_number()

      assert norm < 1.0, "Step result should be inside ball, norm: #{norm}"
    end

    test "zero gradient produces no movement" do
      theta = Nx.tensor([0.3, 0.4])
      grad = Nx.tensor([0.0, 0.0])

      result = Optimizer.step(theta, grad, 0.1)

      diff = Nx.subtract(result, theta)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

      assert diff < 1.0e-5
    end
  end
end
