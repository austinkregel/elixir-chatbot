defmodule Brain.Analysis.DocumentProfileTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.{DocumentProfile, ChunkProfile}

  # Helper: build a minimal ChunkProfile with a given feature vector and modality
  defp profile(vector, modality \\ :declarative, speech_act \\ :assertive) do
    %ChunkProfile{
      feature_vector: vector,
      modality: modality,
      speech_act_category: speech_act,
      domain: :unknown
    }
  end

  describe "aggregate/2 with empty input" do
    test "returns an empty profile with doc_id" do
      result = DocumentProfile.aggregate([], doc_id: "doc_1")
      assert result.doc_id == "doc_1"
      assert result.chunk_count == 0
      assert result.mean_vector == []
      assert result.variance_vector == []
      assert result.skew_vector == []
      assert result.entity_slices == %{}
    end
  end

  describe "aggregate/2 with a single chunk" do
    test "mean equals the input vector" do
      vec = [1.0, 2.0, 3.0, 4.0, 5.0]
      profiles = [profile(vec)]

      result = DocumentProfile.aggregate(profiles)
      assert result.chunk_count == 1
      assert length(result.mean_vector) == 5

      Enum.zip(result.mean_vector, vec)
      |> Enum.each(fn {got, expected} ->
        assert_in_delta got, expected, 1.0e-6
      end)
    end

    test "variance is zero for a single chunk" do
      vec = [1.0, 2.0, 3.0]
      result = DocumentProfile.aggregate([profile(vec)])

      Enum.each(result.variance_vector, fn v ->
        assert_in_delta v, 0.0, 1.0e-6
      end)
    end
  end

  describe "aggregate/2 with multiple chunks" do
    test "mean is the element-wise average (uniform weighting)" do
      p1 = profile([1.0, 0.0, 4.0])
      p2 = profile([3.0, 2.0, 0.0])
      result = DocumentProfile.aggregate([p1, p2])

      assert result.chunk_count == 2
      [m1, m2, m3] = result.mean_vector
      assert_in_delta m1, 2.0, 1.0e-6
      assert_in_delta m2, 1.0, 1.0e-6
      assert_in_delta m3, 2.0, 1.0e-6
    end

    test "variance captures spread across chunks" do
      p1 = profile([0.0, 10.0])
      p2 = profile([10.0, 10.0])
      result = DocumentProfile.aggregate([p1, p2])

      [var1, var2] = result.variance_vector
      assert var1 > 0.0, "dim 1 should have non-zero variance"
      assert_in_delta var2, 0.0, 1.0e-6, "dim 2 should have zero variance"
    end

    test "token_counts weight chunks by length" do
      short = profile([0.0, 0.0])
      long = profile([10.0, 10.0])

      result = DocumentProfile.aggregate([short, long], token_counts: [1, 9])

      [m1, m2] = result.mean_vector
      assert m1 > 5.0, "mean should be pulled toward the longer chunk"
      assert_in_delta m1, 9.0, 1.0e-6
      assert_in_delta m2, 9.0, 1.0e-6
    end
  end

  describe "aggregate/2 — dimension preservation" do
    test "output vector dimension matches input" do
      dim = 280
      vec = List.duplicate(0.5, dim)
      result = DocumentProfile.aggregate([profile(vec)])

      assert length(result.mean_vector) == dim
      assert length(result.variance_vector) == dim
      assert length(result.skew_vector) == dim
    end
  end

  describe "aggregate/2 — rhetorical mode" do
    test "assertion-dominated document" do
      profiles = for _ <- 1..8, do: profile([1.0], :declarative, :assertive)
      profiles = profiles ++ [profile([1.0], :interrogative)]

      result = DocumentProfile.aggregate(profiles)

      assert result.rhetorical_mode.assertion_share > 0.8
      assert result.rhetorical_mode.question_share > 0.0
    end

    test "question-heavy document" do
      profiles = for _ <- 1..5, do: profile([1.0], :interrogative)

      result = DocumentProfile.aggregate(profiles)
      assert result.rhetorical_mode.question_share == 1.0
    end
  end

  describe "aggregate/2 — entity slicing" do
    test "produces entity-specific mean vectors" do
      p1 = profile([1.0, 0.0, 0.0])
      p2 = profile([0.0, 1.0, 0.0])
      p3 = profile([0.0, 0.0, 1.0])

      entity_lists = [
        [%{value: "France"}],
        [%{value: "France"}, %{value: "Paris"}],
        [%{value: "Paris"}]
      ]

      result = DocumentProfile.aggregate([p1, p2, p3], entity_lists: entity_lists)

      assert Map.has_key?(result.entity_slices, "france")
      assert Map.has_key?(result.entity_slices, "paris")

      france_vec = result.entity_slices["france"]
      assert length(france_vec) == 3

      paris_vec = result.entity_slices["paris"]
      assert length(paris_vec) == 3
    end

    test "skips entity slicing when entity_lists is nil" do
      result = DocumentProfile.aggregate([profile([1.0, 2.0])])
      assert result.entity_slices == %{}
    end
  end

  describe "aggregate/2 — skips profiles with empty feature vectors" do
    test "empty vector profiles are excluded from statistics" do
      valid = profile([1.0, 2.0, 3.0])
      empty = %ChunkProfile{feature_vector: []}

      result = DocumentProfile.aggregate([valid, empty])

      assert result.chunk_count == 2
      assert length(result.mean_vector) == 3
      assert_in_delta hd(result.mean_vector), 1.0, 1.0e-6
    end
  end

  describe "similarity/2" do
    test "identical profiles have similarity 1.0" do
      vec = [1.0, 2.0, 3.0]
      dp = DocumentProfile.aggregate([profile(vec)])

      assert_in_delta DocumentProfile.similarity(dp, dp), 1.0, 1.0e-6
    end

    test "orthogonal profiles have similarity 0.0" do
      dp1 = DocumentProfile.aggregate([profile([1.0, 0.0])])
      dp2 = DocumentProfile.aggregate([profile([0.0, 1.0])])

      assert_in_delta DocumentProfile.similarity(dp1, dp2), 0.0, 1.0e-6
    end
  end

  describe "deviation_from/2" do
    test "deviation from self is 0.0" do
      dp = DocumentProfile.aggregate([profile([1.0, 2.0, 3.0])])

      assert_in_delta DocumentProfile.deviation_from(dp, dp.mean_vector), 0.0, 1.0e-6
    end

    test "deviation from orthogonal is 1.0" do
      dp = DocumentProfile.aggregate([profile([1.0, 0.0])])
      centroid = [0.0, 1.0]

      assert_in_delta DocumentProfile.deviation_from(dp, centroid), 1.0, 1.0e-6
    end
  end

  describe "vector_dimension/1" do
    test "returns the dimension of the mean vector" do
      dp = DocumentProfile.aggregate([profile(List.duplicate(0.5, 50))])
      assert DocumentProfile.vector_dimension(dp) == 50
    end

    test "returns 0 for empty profile" do
      dp = DocumentProfile.aggregate([])
      assert DocumentProfile.vector_dimension(dp) == 0
    end
  end
end
