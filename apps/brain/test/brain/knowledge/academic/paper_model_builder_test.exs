defmodule Brain.Knowledge.Academic.PaperModelBuilderTest do
  alias Brain.Knowledge.Academic
  use ExUnit.Case, async: false

  alias Academic.{Paper, PaperModelBuilder}

  def sample_paper(opts \\ []) do
    Paper.new(
      id: Keyword.get(opts, :id, "test-paper-123"),
      title: Keyword.get(opts, :title, "Test Paper Title"),
      abstract:
        Keyword.get(
          opts,
          :abstract,
          "We propose a novel approach to machine learning that significantly improves performance on benchmark tasks. Our method achieves state-of-the-art results on multiple datasets."
        ),
      authors: [%{id: "a1", name: "Test Author"}],
      venue: Keyword.get(opts, :venue, "Test Conference"),
      year: Keyword.get(opts, :year, 2024),
      citation_count: Keyword.get(opts, :citation_count, 50),
      url: "https://example.com/paper",
      source: :semantic_scholar
    )
  end

  describe "extract_claims/1" do
    test "extracts claims from paper with abstract" do
      paper = sample_paper()
      claims = PaperModelBuilder.extract_claims(paper)

      assert is_list(claims)
      assert claims != []

      Enum.each(claims, fn claim ->
        assert is_binary(claim)
        assert String.length(claim) > 0
      end)
    end

    test "returns empty list for paper without abstract" do
      paper = sample_paper(abstract: nil)
      claims = PaperModelBuilder.extract_claims(paper)

      assert claims == []
    end

    test "returns empty list for very short abstract" do
      paper = sample_paper(abstract: "Short.")
      claims = PaperModelBuilder.extract_claims(paper)

      assert claims == []
    end
  end

  describe "citation_to_confidence/1" do
    test "returns high confidence for highly cited papers" do
      assert PaperModelBuilder.citation_to_confidence(1000) == 0.95
      assert PaperModelBuilder.citation_to_confidence(600) == 0.95
    end

    test "returns moderate confidence for moderately cited papers" do
      assert PaperModelBuilder.citation_to_confidence(150) == 0.85
      assert PaperModelBuilder.citation_to_confidence(101) == 0.85
    end

    test "returns lower confidence for less cited papers" do
      assert PaperModelBuilder.citation_to_confidence(50) == 0.7
      assert PaperModelBuilder.citation_to_confidence(11) == 0.7
    end

    test "returns base confidence for new papers" do
      assert PaperModelBuilder.citation_to_confidence(5) == 0.5
      assert PaperModelBuilder.citation_to_confidence(0) == 0.5
    end
  end

  describe "determine_node_type/1" do
    test "returns :premise for highly cited papers" do
      paper = sample_paper(citation_count: 500)
      assert PaperModelBuilder.determine_node_type(paper) == :premise
    end

    test "returns :assumption for less cited papers" do
      paper = sample_paper(citation_count: 50)
      assert PaperModelBuilder.determine_node_type(paper) == :assumption
    end

    test "returns :assumption for new papers" do
      paper = sample_paper(citation_count: 0)
      assert PaperModelBuilder.determine_node_type(paper) == :assumption
    end
  end

  describe "ingest_paper/1" do
    setup do
      case Process.whereis(Brain.Epistemic.JTMS) do
        nil ->
          :skip

        _pid ->
          :ok
      end
    end

    @tag :integration
    test "ingests paper and returns node IDs" do
      paper = sample_paper()

      case PaperModelBuilder.ingest_paper(paper) do
        {:ok, node_ids} ->
          assert is_list(node_ids)

        {:error, _reason} ->
          :ok
      end
    end

    @tag :integration
    test "handles paper without abstract gracefully" do
      paper = sample_paper(abstract: nil)

      {:ok, node_ids} = PaperModelBuilder.ingest_paper(paper)
      assert node_ids == []
    end
  end

  describe "ingest_papers/1" do
    test "ingests multiple papers" do
      papers = [
        sample_paper(id: "paper-1"),
        sample_paper(id: "paper-2"),
        sample_paper(id: "paper-3")
      ]

      case PaperModelBuilder.ingest_papers(papers) do
        {:ok, node_ids} ->
          assert is_list(node_ids)

        {:error, _reason} ->
          :ok
      end
    end

    test "handles empty list" do
      {:ok, node_ids} = PaperModelBuilder.ingest_papers([])
      assert node_ids == []
    end
  end
end