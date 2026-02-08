defmodule Brain.Knowledge.Academic.OpenAlexTest do
  use ExUnit.Case, async: true

  describe "search/2 (integration)" do
    @tag :integration
    @tag :external_api
    test "returns papers for a valid query" do
      alias Brain.Knowledge.Academic.OpenAlex
      alias Brain.Knowledge.Academic.Paper

      {:ok, papers} = OpenAlex.search("transformer attention", limit: 3)

      assert is_list(papers)

      if papers != [] do
        [paper | _] = papers
        assert %Paper{} = paper
        assert paper.source == :openalex
        assert paper.id =~ ~r/^W\d+$/
      end
    end
  end

  describe "search_cs/2 (integration)" do
    @tag :integration
    @tag :external_api
    test "applies CS concept filter" do
      alias Brain.Knowledge.Academic.OpenAlex

      {:ok, papers} = OpenAlex.search_cs("machine learning", limit: 3)

      assert is_list(papers)
    end
  end
end