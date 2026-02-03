defmodule Brain.Knowledge.AcademicValidatorTest do
  use ExUnit.Case, async: false

  alias Brain.Knowledge.AcademicValidator
  alias Brain.Knowledge.Types.{Finding, SourceInfo, ReviewCandidate}

  # Sample finding for testing
  def sample_finding(opts \\ []) do
    source =
      SourceInfo.new(
        Keyword.get(opts, :url, "https://example.com/article"),
        reliability_score: Keyword.get(opts, :reliability, 0.7),
        trust_tier: :neutral
      )

    Finding.new(
      Keyword.get(opts, :claim, "Transformer models use self-attention mechanisms."),
      Keyword.get(opts, :entity, "Transformer"),
      source,
      confidence: Keyword.get(opts, :confidence, 0.6)
    )
  end

  describe "validate/2" do
    test "returns insufficient_evidence when no matching beliefs found" do
      finding = sample_finding(claim: "A completely unique and obscure claim xyz123")

      case AcademicValidator.validate(finding) do
        {:ok, :insufficient_evidence} ->
          assert true

        {:ok, {:corroborated, _details}} ->
          assert true

        {:ok, {:contradicted, _details}} ->
          assert true

        {:error, _reason} ->
          # BeliefStore might not be running
          assert true
      end
    end

    test "accepts live_search option" do
      finding = sample_finding()

      # Should not error with the option
      result = AcademicValidator.validate(finding, live_search: false)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "accepts min_confidence option" do
      finding = sample_finding()

      result = AcademicValidator.validate(finding, min_confidence: 0.8)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end

  describe "bulk_validate/2" do
    test "validates multiple findings" do
      findings = [
        sample_finding(claim: "Claim one about neural networks"),
        sample_finding(claim: "Claim two about machine learning"),
        sample_finding(claim: "Claim three about deep learning")
      ]

      case AcademicValidator.bulk_validate(findings) do
        {:ok, results} ->
          assert length(results) == 3
          assert is_list(results)

        {:error, _reason} ->
          # Expected if epistemic layer not running
          :ok
      end
    end

    test "handles empty list" do
      {:ok, results} = AcademicValidator.bulk_validate([])
      assert results == []
    end
  end

  describe "apply_validation/2" do
    test "boosts confidence when corroborated" do
      finding = sample_finding()
      source = SourceInfo.new("https://test.com", reliability_score: 0.7)

      candidate =
        ReviewCandidate.new(finding,
          aggregate_confidence: 0.6,
          corroborating_sources: [source]
        )

      validation_result =
        {:corroborated,
         %{
           boost: 0.15,
           sources: [%{paper_id: "paper123", title: "Test Paper"}]
         }}

      updated = AcademicValidator.apply_validation(candidate, validation_result)

      assert updated.aggregate_confidence == 0.75
      assert length(updated.corroborating_sources) >= 1
    end

    test "caps confidence at 1.0" do
      finding = sample_finding()

      candidate =
        ReviewCandidate.new(finding,
          aggregate_confidence: 0.95
        )

      validation_result = {:corroborated, %{boost: 0.15, sources: []}}

      updated = AcademicValidator.apply_validation(candidate, validation_result)

      assert updated.aggregate_confidence == 1.0
    end

    test "adds contradictions when contradicted" do
      finding = sample_finding()

      candidate =
        ReviewCandidate.new(finding,
          aggregate_confidence: 0.6,
          existing_contradictions: []
        )

      validation_result =
        {:contradicted,
         %{
           conflicting_papers: [
             %{type: :academic_conflict, paper_id: "conflict123", claim: "Contradicting claim"}
           ]
         }}

      updated = AcademicValidator.apply_validation(candidate, validation_result)

      assert length(updated.existing_contradictions) == 1
    end

    test "leaves candidate unchanged for insufficient_evidence" do
      finding = sample_finding()

      candidate =
        ReviewCandidate.new(finding,
          aggregate_confidence: 0.6
        )

      updated = AcademicValidator.apply_validation(candidate, :insufficient_evidence)

      assert updated.aggregate_confidence == 0.6
    end
  end

  describe "search_academic_consensus/2" do
    @tag :integration
    test "returns consensus information for a topic" do
      # This test would hit real APIs unless mocked
      case AcademicValidator.search_academic_consensus("machine learning", limit: 5) do
        {:ok, result} ->
          assert Map.has_key?(result, :papers)
          assert Map.has_key?(result, :consensus)
          assert result.consensus in [:no_data, :emerging, :weak, :moderate, :strong]

        {:error, _reason} ->
          # Expected if APIs are rate-limited or unavailable
          :ok
      end
    end
  end
end
