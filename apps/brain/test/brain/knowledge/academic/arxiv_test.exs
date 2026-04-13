defmodule Brain.Knowledge.Academic.ArxivTest do
  use ExUnit.Case, async: false

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

  describe "search/2 (with snapshot)" do
    setup do
      # Load the snapshot for this test (server is started globally in test_helper.exs)
      {:ok, _} = Brain.Test.HTTPSnapshot.use_snapshot("arxiv/search_transformer")
      :ok
    end

    test "returns papers for a valid query" do
      alias Brain.Knowledge.Academic.Paper

      {:ok, papers} = Arxiv.search("transformer attention", limit: 3)

      assert is_list(papers)
      assert length(papers) == 3

      [paper | _] = papers
      assert %Paper{} = paper
      assert paper.source == :arxiv
      assert is_binary(paper.title) and paper.title != ""
    end
  end
end
