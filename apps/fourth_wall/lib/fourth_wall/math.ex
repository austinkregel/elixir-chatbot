defmodule FourthWall.Math do
  @moduledoc """
  Shared mathematical utilities for the ChatBot umbrella.

  Extracted from duplicated implementations across brain and world apps.
  """

  @doc """
  Compute cosine similarity between two list-based embedding vectors.

  Returns a float in [-1, 1]. For orthogonal vectors returns 0.0,
  for identical vectors returns 1.0, for opposite vectors returns -1.0.

  Handles edge cases:
  - Empty vectors: 0.0
  - Mismatched lengths: 0.0
  - Zero magnitude: 0.0
  """
  def cosine_similarity(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b) do
    if length(vec_a) != length(vec_b) or vec_a == [] do
      0.0
    else
      dot_product =
        Enum.zip(vec_a, vec_b)
        |> Enum.reduce(0, fn {a, b}, acc -> acc + a * b end)

      mag_a = :math.sqrt(Enum.reduce(vec_a, 0, fn val, acc -> acc + val * val end))
      mag_b = :math.sqrt(Enum.reduce(vec_b, 0, fn val, acc -> acc + val * val end))

      if mag_a > 0 and mag_b > 0 do
        dot_product / (mag_a * mag_b)
      else
        0.0
      end
    end
  end

  def cosine_similarity(_, _), do: 0.0
end
