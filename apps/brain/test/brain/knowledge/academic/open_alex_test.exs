defmodule Brain.Knowledge.Academic.OpenAlexTest do
  use ExUnit.Case, async: false

  describe "search/2 (with snapshot)" do
    setup do
      # Loaded for this test only: the snapshot server is global and its
      # matcher is a substring match, so a snapshot left loaded would answer a
      # later test's request with a stale body.
      Brain.Test.Singletons.use_http_snapshots!(["open_alex/search_transformer"])
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
      # The CS-filtered snapshot, loaded for this test only. Without the
      # restore it stayed loaded alongside `open_alex/search_transformer`,
      # and both match `/works`.
      Brain.Test.Singletons.use_http_snapshots!(["open_alex/search_cs"])
    end

    test "applies CS concept filter" do
      alias Brain.Knowledge.Academic.OpenAlex

      {:ok, papers} = OpenAlex.search_cs("machine learning", limit: 3)

      assert is_list(papers)
    end
  end
end
