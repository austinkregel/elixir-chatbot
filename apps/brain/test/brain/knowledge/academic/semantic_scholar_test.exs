defmodule Brain.Knowledge.Academic.SemanticScholarTest do
  use ExUnit.Case, async: false

  alias Brain.Knowledge.Academic.Paper

  describe "Paper struct" do
    test "can convert to finding with valid abstract" do
      paper =
        Paper.new(
          id: "abc123",
          title: "Attention Is All You Need",
          abstract:
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms. This model achieves state-of-the-art results.",
          authors: [%{id: "1", name: "Ashish Vaswani"}],
          venue: "NeurIPS",
          year: 2017,
          citation_count: 50_000,
          source: :semantic_scholar
        )

      finding = Paper.to_finding(paper)

      assert finding != nil
      assert finding.claim =~ "Transformer"
      assert finding.confidence >= 0.9
    end

    test "returns nil for paper without abstract" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          abstract: nil,
          source: :semantic_scholar
        )

      assert Paper.to_finding(paper) == nil
    end

    test "returns nil for paper with short abstract" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          abstract: "Short.",
          source: :semantic_scholar
        )

      assert Paper.to_finding(paper) == nil
    end

    test "formats author string correctly for single author" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          authors: [%{id: "1", name: "John Doe"}],
          source: :semantic_scholar
        )

      assert Paper.author_string(paper) == "John Doe"
    end

    test "formats author string correctly for two authors" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          authors: [%{id: "1", name: "John Doe"}, %{id: "2", name: "Jane Smith"}],
          source: :semantic_scholar
        )

      assert Paper.author_string(paper) == "John Doe and Jane Smith"
    end

    test "formats author string correctly for multiple authors" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          authors: [
            %{id: "1", name: "Ashish Vaswani"},
            %{id: "2", name: "Noam Shazeer"},
            %{id: "3", name: "Niki Parmar"}
          ],
          source: :semantic_scholar
        )

      assert Paper.author_string(paper) =~ "Vaswani"
      assert Paper.author_string(paper) =~ "3 authors"
    end

    test "returns arxiv_id when present" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          external_ids: %{"ArXiv" => "1706.03762"},
          source: :semantic_scholar
        )

      assert Paper.arxiv_id(paper) == "1706.03762"
    end

    test "returns nil when arxiv_id not present" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          external_ids: %{},
          source: :semantic_scholar
        )

      assert Paper.arxiv_id(paper) == nil
    end

    test "returns doi when present" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          external_ids: %{"DOI" => "10.1234/test"},
          source: :semantic_scholar
        )

      assert Paper.doi(paper) == "10.1234/test"
    end

    test "generates pdf_url from arxiv_id" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          external_ids: %{"ArXiv" => "1706.03762"},
          source: :semantic_scholar
        )

      assert Paper.pdf_url(paper) == "https://arxiv.org/pdf/1706.03762.pdf"
    end

    test "converts to source_info" do
      paper =
        Paper.new(
          id: "test",
          title: "Test Paper",
          url: "https://example.com/paper",
          source: :semantic_scholar
        )

      source = Paper.to_source_info(paper)

      assert source.url == "https://example.com/paper"
      assert source.title == "Test Paper"
      assert source.reliability_score == 0.95
      assert source.trust_tier == :verified
    end
  end

  describe "search/2 (with snapshot)" do
    setup do
      # Load the snapshot for this test (server is started globally in test_helper.exs)
      {:ok, _} = Brain.Test.HTTPSnapshot.use_snapshot("semantic_scholar/search_transformer")
      :ok
    end

    test "returns papers for a valid query" do
      alias Brain.Knowledge.Academic.SemanticScholar

      {:ok, papers} = SemanticScholar.search("transformer attention", limit: 3)

      assert is_list(papers)
      assert length(papers) == 3

      [paper | _] = papers
      assert %Paper{} = paper
      assert paper.source == :semantic_scholar
      assert is_binary(paper.title) and paper.title != ""
    end

    test "papers have expected fields from snapshot" do
      alias Brain.Knowledge.Academic.SemanticScholar

      {:ok, papers} = SemanticScholar.search("transformer attention", limit: 3)

      [first | _] = papers
      assert is_integer(first.citation_count) and first.citation_count > 0
      assert is_binary(first.venue) and first.venue != ""
    end
  end
end
