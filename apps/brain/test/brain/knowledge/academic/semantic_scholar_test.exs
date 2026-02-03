defmodule Brain.Knowledge.Academic.SemanticScholarTest do
  use ExUnit.Case, async: true

  alias Brain.Knowledge.Academic.Paper

  # Tests that don't require API calls
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
          citation_count: 50000,
          source: :semantic_scholar
        )

      finding = Paper.to_finding(paper)

      assert finding != nil
      assert finding.claim =~ "Transformer"
      # High citation count should give high confidence
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

  # Integration tests that hit real APIs (tagged for optional running)
  describe "search/2 (integration)" do
    @tag :integration
    @tag :external_api
    test "returns papers for a valid query" do
      alias Brain.Knowledge.Academic.SemanticScholar

      {:ok, papers} = SemanticScholar.search("transformer attention", limit: 3)

      assert is_list(papers)
      assert length(papers) > 0

      [paper | _] = papers
      assert %Paper{} = paper
      assert paper.source == :semantic_scholar
    end
  end
end
