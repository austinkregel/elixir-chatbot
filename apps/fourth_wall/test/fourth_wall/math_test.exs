defmodule FourthWall.MathTest do
  use ExUnit.Case, async: false

  alias FourthWall.Math

  describe "cosine_similarity/2" do
    test "orthogonal vectors return 0.0" do
      assert Math.cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0
      assert Math.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    end

    test "identical vectors return 1.0" do
      assert_in_delta Math.cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0, 1.0e-10
      assert_in_delta Math.cosine_similarity([0.5, 0.5], [0.5, 0.5]), 1.0, 1.0e-10
    end

    test "opposite vectors return -1.0" do
      assert_in_delta Math.cosine_similarity([1, 0], [-1, 0]), -1.0, 1.0e-10
    end

    test "empty vectors return 0.0 without crash" do
      assert Math.cosine_similarity([], []) == 0.0
    end

    test "mismatched lengths return 0.0" do
      assert Math.cosine_similarity([1, 2], [1, 2, 3]) == 0.0
      assert Math.cosine_similarity([1], []) == 0.0
    end
  end

  describe "cosine_similarity!/2" do
    test "is the cosine similarity of vectors of the same length" do
      assert_in_delta Math.cosine_similarity!([1, 2, 3], [1, 2, 3]), 1.0, 1.0e-10
      assert Math.cosine_similarity!([1, 0, 0], [0, 1, 0]) == 0.0
      assert_in_delta Math.cosine_similarity!([1, 0], [-1, 0]), -1.0, 1.0e-10
    end

    test "is 0.0 for empty vectors and a zero-magnitude one" do
      assert Math.cosine_similarity!([], []) == 0.0
      assert Math.cosine_similarity!([0.0, 0.0], [1.0, 0.0]) == 0.0
    end

    test "raises naming both lengths for vectors of different lengths" do
      assert_raise ArgumentError, ~r/different lengths \(2 and 3\)/, fn ->
        Math.cosine_similarity!([1, 2], [1, 2, 3])
      end

      assert_raise ArgumentError, ~r/different lengths \(1 and 0\)/, fn ->
        Math.cosine_similarity!([1], [])
      end
    end
  end
end
