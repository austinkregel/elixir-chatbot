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
      cosine_similarity!(vec_a, vec_b)
    end
  end

  def cosine_similarity(_, _), do: 0.0

  @doc """
  Compute cosine similarity between two list-based embedding vectors that must
  come from the same embedding space, such as a vector cached with a world's
  vocabulary and a query embedded with it.

  Vectors of different lengths come from different spaces (a vocabulary that
  was rebuilt or replaced in between), so their similarity has no answer:
  that raises `ArgumentError` naming both lengths, where `cosine_similarity/2`
  would return 0.0.

  Returns a float in [-1, 1]; 0.0 for empty vectors or a zero-magnitude one.
  """
  def cosine_similarity!(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b) do
    length_a = length(vec_a)
    length_b = length(vec_b)

    if length_a != length_b do
      raise ArgumentError,
            "cosine similarity of vectors of different lengths (#{length_a} and #{length_b}): " <>
              "they were not embedded in the same space"
    end

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
