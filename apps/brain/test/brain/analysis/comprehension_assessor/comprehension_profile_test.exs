defmodule Brain.Analysis.ComprehensionAssessor.ComprehensionProfileTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.ComprehensionAssessor.ComprehensionProfile

  defp equal_weights do
    dims = ComprehensionProfile.dimension_names()
    weight = 1.0 / length(dims)
    Map.new(dims, fn d -> {d, weight} end)
  end

  defp high_scores do
    for dim <- ComprehensionProfile.dimension_names(), into: %{} do
      {dim, {0.9, %{test: true}}}
    end
  end

  defp low_scores do
    for dim <- ComprehensionProfile.dimension_names(), into: %{} do
      {dim, {0.1, %{test: true}}}
    end
  end

  describe "build/2" do
    test "high scores produce :comprehended verdict" do
      profile = ComprehensionProfile.build(high_scores(), equal_weights())

      assert profile.verdict == :comprehended
      assert profile.learnable == true
      assert profile.composite_score >= 0.7
      assert profile.gaps == []
      assert is_binary(profile.id)
    end

    test "low scores produce :garbled verdict" do
      profile = ComprehensionProfile.build(low_scores(), equal_weights())

      assert profile.verdict == :garbled
      assert profile.learnable == false
      assert profile.composite_score < 0.2
      assert length(profile.gaps) > 0
    end

    test "structural_coherence hard gate overrides other high scores" do
      scores =
        high_scores()
        |> Map.put(:structural_coherence, {0.15, %{reason: "garbled"}})

      profile = ComprehensionProfile.build(scores, equal_weights())

      assert profile.verdict == :garbled
      assert profile.learnable == false
      # composite is the structural_coherence score since it's a hard gate
      assert profile.composite_score == 0.15
    end

    test "partial scores produce :partial verdict" do
      scores =
        for dim <- ComprehensionProfile.dimension_names(), into: %{} do
          {dim, {0.5, %{}}}
        end

      profile = ComprehensionProfile.build(scores, equal_weights())

      assert profile.verdict == :partial
      assert profile.learnable == true
      assert profile.composite_score >= 0.4 and profile.composite_score < 0.7
    end

    test "opaque scores produce :opaque verdict" do
      scores =
        for dim <- ComprehensionProfile.dimension_names(), into: %{} do
          {dim, {0.25, %{}}}
        end

      profile = ComprehensionProfile.build(scores, equal_weights())

      assert profile.verdict == :opaque
      assert profile.learnable == false
      assert profile.composite_score >= 0.2 and profile.composite_score < 0.4
    end

    test "gaps list dimensions with score below 0.4" do
      scores =
        for dim <- ComprehensionProfile.dimension_names(), into: %{} do
          if dim == :temporal_grounding do
            {dim, {0.1, %{reason: "no temporal info"}}}
          else
            {dim, {0.9, %{}}}
          end
        end

      profile = ComprehensionProfile.build(scores, equal_weights())

      assert length(profile.gaps) == 1
      [gap] = profile.gaps
      assert gap.dimension == :temporal_grounding
      assert gap.score == 0.1
      assert is_binary(gap.description)
    end

    test "weighted dimensions affect composite score" do
      # Give high weight to a dimension with a low score
      scores = %{
        referential_clarity: {0.1, %{}},
        actor_identification: {0.9, %{}},
        propositional_content: {0.9, %{}},
        temporal_grounding: {0.9, %{}},
        contextual_sufficiency: {0.9, %{}},
        epistemic_grounding: {0.9, %{}},
        structural_coherence: {0.9, %{}},
        illocutionary_clarity: {0.9, %{}}
      }

      # Equal weights
      profile_equal = ComprehensionProfile.build(scores, equal_weights())

      # Heavy weight on the low-scoring dimension
      heavy_weights =
        equal_weights()
        |> Map.put(:referential_clarity, 0.5)

      profile_heavy = ComprehensionProfile.build(scores, heavy_weights)

      assert profile_heavy.composite_score < profile_equal.composite_score
    end
  end

  describe "dimension_score/2" do
    test "returns score for existing dimension" do
      profile = ComprehensionProfile.build(high_scores(), equal_weights())
      assert ComprehensionProfile.dimension_score(profile, :referential_clarity) == 0.9
    end

    test "returns nil for non-existent dimension" do
      profile = ComprehensionProfile.build(high_scores(), equal_weights())
      assert ComprehensionProfile.dimension_score(profile, :nonexistent) == nil
    end
  end

  describe "dimension_names/0" do
    test "returns all 8 dimension names" do
      names = ComprehensionProfile.dimension_names()
      assert length(names) == 8
      assert :referential_clarity in names
      assert :structural_coherence in names
    end
  end
end
