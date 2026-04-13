defmodule Brain.Knowledge.CorroboratorTest do
  alias Brain.Knowledge.Types
  use ExUnit.Case, async: false

  alias Brain.Knowledge.Corroborator
  alias Types.{Finding, SourceInfo}

  describe "corroborate/2" do
    test "groups similar claims from different sources" do
      findings = [
        build_finding("Paris is the capital of France", "France", "source1.com"),
        build_finding("France's capital city is Paris", "France", "source2.com"),
        build_finding("Berlin is the capital of Germany", "Germany", "source1.com")
      ]

      {:ok, all_candidates} = Corroborator.corroborate(findings, include_uncorroborated: true)
      assert all_candidates != []
      {:ok, filtered_candidates} = Corroborator.corroborate(findings)

      paris_candidates =
        Enum.filter(all_candidates, fn c ->
          String.contains?(c.finding.claim, "Paris") or
            String.contains?(c.finding.claim, "capital of France")
        end)

      assert paris_candidates != []

      berlin_in_filtered =
        Enum.find(filtered_candidates, &String.contains?(&1.finding.claim, "Berlin"))

      if berlin_in_filtered do
        assert berlin_in_filtered.corroborating_sources == []
      end
    end

    test "includes uncorroborated when option set" do
      findings = [build_finding("Single source claim", "Entity", "source1.com")]

      {:ok, candidates} = Corroborator.corroborate(findings, include_uncorroborated: true)

      assert length(candidates) == 1
    end

    test "returns empty list for empty input" do
      {:ok, candidates} = Corroborator.corroborate([])
      assert candidates == []
    end

    test "respects custom similarity threshold" do
      findings = [
        build_finding("The population is 100", "City", "source1.com"),
        build_finding("The population is around 100", "City", "source2.com")
      ]

      {:ok, candidates} = Corroborator.corroborate(findings, similarity_threshold: 0.5)

      assert candidates != []
    end
  end

  describe "compare_claims/2" do
    test "similar claims have high similarity" do
      {:ok, similarity} =
        Corroborator.compare_claims(
          "Paris is the capital of France",
          "The capital of France is Paris"
        )

      assert similarity > 0.5
    end

    test "different claims have low similarity" do
      {:ok, similarity} =
        Corroborator.compare_claims(
          "Paris is the capital of France",
          "Tokyo is a large city in Japan"
        )

      assert similarity < 0.5
    end
  end

  describe "find_conflicts/2" do
    test "detects negation conflicts" do
      finding = build_finding("Paris is the capital", "Paris", "source1.com")

      existing = [build_finding("Paris is not the capital", "Paris", "source2.com")]

      conflicts = Corroborator.find_conflicts(finding, existing)

      assert length(conflicts) == 1
    end

    test "detects number disagreements" do
      finding = build_finding("The population is 14 million", "Tokyo", "source1.com")

      existing = [build_finding("The population is 37 million", "Tokyo", "source2.com")]

      conflicts = Corroborator.find_conflicts(finding, existing)

      assert length(conflicts) == 1
    end

    test "does not flag similar claims as conflicts" do
      finding = build_finding("Paris is the capital of France", "Paris", "source1.com")

      existing = [build_finding("Paris is the capital of France", "Paris", "source2.com")]

      conflicts = Corroborator.find_conflicts(finding, existing)

      assert conflicts == []
    end
  end

  defp build_finding(claim, entity, domain) do
    source = SourceInfo.new("https://#{domain}/article", reliability_score: 0.7)

    Finding.new(claim, entity, source, entity_type: "location", confidence: 0.7)
  end
end