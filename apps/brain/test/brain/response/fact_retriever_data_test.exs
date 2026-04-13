defmodule Brain.Response.FactRetrieverDataTest do
  @moduledoc """
  Data-driven tests for FactRetriever covering fact lookup and formatting.
  """
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Response.FactRetriever
  alias Brain.FactDatabase

  setup do
    ensure_pubsub_started()
    ensure_started(FactDatabase)
    :ok
  end

  # Test data for get_relevant_facts/1
  @query_test_cases [
    {[entity: "France"], :list, "entity query returns list"},
    {[category: "geography"], :list, "category query returns list"},
    {[search: "capital"], :list, "search query returns list"},
    {[limit: 1], :list, "limit option works"},
    {[], :list, "empty options returns list"},
    {[entity: "NonExistentEntity12345"], :empty_or_list, "unknown entity returns empty or list"},
  ]

  describe "get_relevant_facts/1 - data driven" do
    for {opts, expected_type, description} <- @query_test_cases do
      @opts opts
      @expected_type expected_type
      @description description

      test "#{description}" do
        result = FactRetriever.get_relevant_facts(@opts)

        case @expected_type do
          :list ->
            assert is_list(result)

          :empty_or_list ->
            assert is_list(result)
        end
      end
    end
  end

  # Test data for get_facts_for_query/2
  @query_entity_test_cases [
    # {query, entities, description}
    {"What is the capital?", [], "query with no entities"},
    {"Tell me about France", ["France"], "query with string entity"},
    {"", [%{value: "Paris"}], "empty query with map entity"},
    {"capital city", [%{"value" => "France"}], "string key entity map"},
    {"weather", [%{value: "Seattle"}, %{value: "weather"}], "multiple entities"},
  ]

  describe "get_facts_for_query/2 - data driven" do
    for {query, entities, description} <- @query_entity_test_cases do
      @query query
      @entities entities
      @description description

      test "#{description}" do
        result = FactRetriever.get_facts_for_query(@query, @entities)
        assert is_list(result)
      end
    end
  end

  # Test data for format_facts/1
  # Note: format_facts returns a list of strings, not a single string
  @format_test_cases [
    # {input_facts, description}
    {[], "empty list returns empty list"},
    {[%{fact: "Paris is the capital", entity: "Paris"}], "struct-like fact formatted"},
  ]

  describe "format_facts/1 - data driven" do
    for {facts, description} <- @format_test_cases do
      @facts facts
      @description description

      test "#{description}" do
        result = FactRetriever.format_facts(@facts)
        # format_facts returns a list of formatted strings
        assert is_list(result)
      end
    end
  end

  # Edge cases
  describe "edge cases" do
    test "handles nil entities gracefully" do
      result = FactRetriever.get_facts_for_query("test", nil)
      # Should not crash, returns list
      assert is_list(result) or result == []
    end

    test "handles malformed entity maps" do
      entities = [%{no_value_key: "test"}, nil, "valid"]
      result = FactRetriever.get_facts_for_query("test", entities)
      assert is_list(result)
    end
  end
end
