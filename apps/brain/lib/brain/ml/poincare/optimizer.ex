defmodule Brain.ML.Poincare.Optimizer do
  @moduledoc """
  Riemannian SGD optimizer for Poincare embeddings.

  In the Poincare ball model, the Euclidean gradient must be rescaled by the
  inverse of the metric tensor, then the updated point must be projected back
  into the ball to maintain the constraint ||x|| < 1.

  The Riemannian gradient is:

      grad_R = ((1 - ||theta||^2)^2 / 4) * grad_E

  After the update step, points are projected back into the ball with a small
  epsilon margin.
  """

  import Nx.Defn

  @doc """
  Convert a Euclidean gradient to a Riemannian gradient in the Poincare ball.

  Rescales by ((1 - ||theta||^2)^2 / 4) which is the inverse of the
  Poincare metric tensor.
  """
  defn riemannian_grad(euclidean_grad, theta) do
    norm_sq = Nx.sum(Nx.pow(theta, 2), axes: [-1], keep_axes: true)
    scale = Nx.divide(Nx.pow(Nx.subtract(1.0, norm_sq), 2), 4.0)
    Nx.multiply(euclidean_grad, scale)
  end

  @doc """
  Project a point back into the Poincare ball.

  Ensures ||theta|| < 1 - eps for numerical stability.
  """
  defn project_to_ball(theta, eps \\ 1.0e-5) do
    norm = Nx.sqrt(Nx.sum(Nx.pow(theta, 2), axes: [-1], keep_axes: true))
    max_norm = 1.0 - eps
    scale = Nx.min(Nx.divide(max_norm, Nx.max(norm, 1.0e-10)), 1.0)
    Nx.multiply(theta, scale)
  end

  @doc """
  Perform a single Riemannian SGD update step.

  1. Convert Euclidean gradient to Riemannian gradient
  2. Apply learning rate
  3. Update parameters: theta_new = theta - lr * grad_R
  4. Project back into the ball

  ## Parameters
    - `theta` - Current embedding position
    - `euclidean_grad` - Euclidean gradient
    - `learning_rate` - Learning rate scalar
  """
  defn step(theta, euclidean_grad, learning_rate) do
    riem_grad = riemannian_grad(euclidean_grad, theta)
    updated = Nx.subtract(theta, Nx.multiply(learning_rate, riem_grad))
    project_to_ball(updated)
  end

  @doc """
  Full training step: compute loss, get gradients, update embeddings.

  ## Parameters
    - `embeddings` - All embeddings tensor {num_entities, dim}
    - `positive_pairs` - Tensor of positive pair indices {batch, 2}
    - `negative_indices` - Tensor of negative entity indices {batch, num_negatives}
    - `learning_rate` - Learning rate
  """
  def train_step(embeddings, positive_pairs, negative_indices, learning_rate) do
    {loss, grad} = compute_loss_and_grad(embeddings, positive_pairs, negative_indices)

    riem_grad = riemannian_grad(grad, embeddings)
    updated = Nx.subtract(embeddings, Nx.multiply(learning_rate, riem_grad))
    projected = project_to_ball(updated)

    {projected, loss}
  end

  defn compute_loss_and_grad(embeddings, positive_pairs, negative_indices) do
    Nx.Defn.value_and_grad(embeddings, fn emb ->
      negative_sampling_loss(emb, positive_pairs, negative_indices)
    end)
  end

  @doc """
  Negative sampling loss from the Poincare embeddings paper.

  For each positive (u, v) pair:
    L = -log(exp(-d(u,v)) / (exp(-d(u,v)) + sum_neg(exp(-d(u, n)))))
  """
  defn negative_sampling_loss(embeddings, positive_pairs, negative_indices) do
    u_idx = positive_pairs[[.., 0]]
    v_idx = positive_pairs[[.., 1]]

    u = Nx.take(embeddings, u_idx)
    v = Nx.take(embeddings, v_idx)

    pos_dist = pairwise_distance(u, v)
    pos_score = Nx.exp(Nx.negate(pos_dist))

    neg_entities = Nx.take(embeddings, negative_indices)
    u_expanded = Nx.new_axis(u, 1)

    neg_dist = batched_distance(u_expanded, neg_entities)
    neg_scores = Nx.sum(Nx.exp(Nx.negate(neg_dist)), axes: [1])

    denom = Nx.add(pos_score, neg_scores)
    loss = Nx.negate(Nx.log(Nx.divide(pos_score, Nx.max(denom, 1.0e-10))))

    Nx.mean(loss)
  end

  defn pairwise_distance(u, v) do
    diff_sq = Nx.sum(Nx.pow(Nx.subtract(u, v), 2), axes: [-1])
    u_sq = Nx.sum(Nx.pow(u, 2), axes: [-1])
    v_sq = Nx.sum(Nx.pow(v, 2), axes: [-1])

    denom = Nx.multiply(Nx.subtract(1.0, u_sq), Nx.subtract(1.0, v_sq))
    arg = Nx.add(1.0, Nx.divide(Nx.multiply(2.0, diff_sq), Nx.max(denom, 1.0e-10)))

    Nx.acosh(Nx.max(arg, 1.0 + 1.0e-7))
  end

  defn batched_distance(u_expanded, neg_entities) do
    diff_sq = Nx.sum(Nx.pow(Nx.subtract(u_expanded, neg_entities), 2), axes: [-1])
    u_sq = Nx.sum(Nx.pow(u_expanded, 2), axes: [-1])
    v_sq = Nx.sum(Nx.pow(neg_entities, 2), axes: [-1])

    denom = Nx.multiply(Nx.subtract(1.0, u_sq), Nx.subtract(1.0, v_sq))
    arg = Nx.add(1.0, Nx.divide(Nx.multiply(2.0, diff_sq), Nx.max(denom, 1.0e-10)))

    Nx.acosh(Nx.max(arg, 1.0 + 1.0e-7))
  end
end
