defmodule Brain.Knowledge.Academic.ArxivTest do
  use ExUnit.Case, async: true

  alias Brain.Knowledge.Academic.Arxiv

  describe "get_pdf_url/1" do
    test "returns correct PDF URL for arXiv ID" do
      url = Arxiv.get_pdf_url("1706.03762")
      assert url == "https://arxiv.org/pdf/1706.03762.pdf"
    end

    test "normalizes arxiv: prefix" do
      url = Arxiv.get_pdf_url("arxiv:1706.03762")
      assert url == "https://arxiv.org/pdf/1706.03762.pdf"
    end

    test "normalizes arXiv: prefix (case variation)" do
      url = Arxiv.get_pdf_url("arXiv:1706.03762")
      assert url == "https://arxiv.org/pdf/1706.03762.pdf"
    end

    test "handles full URL" do
      url = Arxiv.get_pdf_url("https://arxiv.org/abs/1706.03762")
      assert url == "https://arxiv.org/pdf/1706.03762.pdf"
    end
  end

  describe "get_abs_url/1" do
    test "returns correct abstract URL for arXiv ID" do
      url = Arxiv.get_abs_url("1706.03762")
      assert url == "https://arxiv.org/abs/1706.03762"
    end

    test "normalizes arxiv: prefix" do
      url = Arxiv.get_abs_url("arxiv:1706.03762")
      assert url == "https://arxiv.org/abs/1706.03762"
    end
  end

  describe "search/2 (integration)" do
    @tag :integration
    @tag :external_api
    test "returns papers for a valid query" do
      alias Brain.Knowledge.Academic.Paper

      {:ok, papers} = Arxiv.search("transformer attention", limit: 3)

      assert is_list(papers)

      if papers != [] do
        [paper | _] = papers
        assert %Paper{} = paper
        assert paper.source == :arxiv
      end
    end
  end
end