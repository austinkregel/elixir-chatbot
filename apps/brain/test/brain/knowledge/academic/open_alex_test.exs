defmodule Brain.Knowledge.Academic.OpenAlexTest do
  use ExUnit.Case, async: false

  describe "search/2 (with snapshot)" do
    setup do
      # Load the snapshot for this test (server is started globally in test_helper.exs)
      {:ok, _} = Brain.Test.HTTPSnapshot.use_snapshot("open_alex/search_transformer")
      :ok
    end

    test "returns papers for a valid query" do
      alias Brain.Knowledge.Academic.OpenAlex
      alias Brain.Knowledge.Academic.Paper

      {:ok, papers} = OpenAlex.search("transformer attention", limit: 3)

      assert is_list(papers)
      assert length(papers) == 3

      [paper | _] = papers
      assert %Paper{} = paper
      assert paper.source == :openalex
      assert is_binary(paper.title) and paper.title != ""
    end
  end

  describe "search_cs/2 (with snapshot)" do
    setup do
      # Use the CS-filtered snapshot (server is started globally in test_helper.exs)
      {:ok, _} = Brain.Test.HTTPSnapshot.use_snapshot("open_alex/search_cs")
      :ok
    end

    test "applies CS concept filter" do
      alias Brain.Knowledge.Academic.OpenAlex

      {:ok, papers} = OpenAlex.search_cs("machine learning", limit: 3)

      assert is_list(papers)
    end
  end
end
