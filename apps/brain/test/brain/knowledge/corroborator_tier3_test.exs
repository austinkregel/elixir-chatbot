defmodule Brain.Knowledge.CorroboratorTier3Test do
  @moduledoc """
  Tests for the KG-LSTM triple scoring integration in Corroborator.
  """

  use ExUnit.Case, async: false

  alias Brain.Knowledge.Corroborator
  alias Brain.Knowledge.Types.{Finding, SourceInfo}

  describe "corroborate/2 with KG-LSTM unavailable" do
    test "still produces candidates without KG scoring" do
      source1 = SourceInfo.new("http://source1.com", title: "Source 1")
      source2 = SourceInfo.new("http://source2.com", title: "Source 2")

      findings = [
        Finding.new("Paris is the capital of France", "Paris", source1, confidence: 0.9),
        Finding.new("The capital of France is Paris", "Paris", source2, confidence: 0.85)
      ]

      {:ok, candidates} = Corroborator.corroborate(findings)
      assert is_list(candidates)
    end

    test "handles empty findings" do
      {:ok, candidates} = Corroborator.corroborate([])
      assert candidates == []
    end

    test "single finding below min_sources returns empty" do
      source = SourceInfo.new("http://source.com", title: "Source")
      findings = [Finding.new("Some fact", "entity", source, confidence: 0.9)]

      {:ok, candidates} = Corroborator.corroborate(findings)
      assert candidates == []
    end

    test "include_uncorroborated returns single-source findings" do
      source = SourceInfo.new("http://source.com", title: "Source")
      findings = [Finding.new("Some fact", "entity", source, confidence: 0.9)]

      {:ok, candidates} = Corroborator.corroborate(findings, include_uncorroborated: true)
      assert length(candidates) >= 1
    end
  end

  describe "compare_claims/2" do
    test "identical claims have high similarity" do
      {:ok, similarity} = Corroborator.compare_claims(
        "Paris is the capital of France",
        "Paris is the capital of France"
      )

      assert similarity >= 0.9
    end

    test "different claims have lower similarity" do
      {:ok, similarity} = Corroborator.compare_claims(
        "Paris is the capital of France",
        "Tokyo is the capital of Japan"
      )

      assert similarity < 1.0
    end

    test "empty strings" do
      {:ok, similarity} = Corroborator.compare_claims("", "")
      assert is_number(similarity)
    end
  end
end
