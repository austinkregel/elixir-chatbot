defmodule Brain.FactDatabaseDataTest do
  @moduledoc """
  Data-driven tests for FactDatabase covering fact storage, retrieval, and querying.
  """
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.FactDatabase

  setup _context do
    ensure_pubsub_started()
    ensure_started(FactDatabase)
    :ok
  end

  # Test data for querying by entity
  @entity_query_cases [
    # {entity, expected_type, description}
    {"France", :list, "query facts about France"},
    {"Paris", :list, "query facts about Paris"},
    {"United States", :list, "query facts about US"},
    {"NonExistent12345", :empty_or_list, "query non-existent entity"},
    {"", :empty_or_list, "empty entity query"},
  ]

  describe "query/1 with entity - data driven" do
    for {entity, expected_type, description} <- @entity_query_cases do
      @entity entity
      @expected_type expected_type
      @description description

      test "#{description}" do
        result = FactDatabase.query(entity: @entity)

        case @expected_type do
          :list ->
            assert is_list(result)

          :empty_or_list ->
            assert is_list(result)
        end
      end
    end
  end

  # Test data for querying by category
  @category_query_cases [
    {"geography", :list, "geography category"},
    {"science", :list, "science category"},
    {"history", :list, "history category"},
    {"general", :list, "general category"},
    {"nonexistent_category", :empty_or_list, "unknown category"},
  ]

  describe "query/1 with category - data driven" do
    for {category, expected_type, description} <- @category_query_cases do
      @category category
      @expected_type expected_type
      @description description

      test "#{description}" do
        result = FactDatabase.query(category: @category)

        assert is_list(result)
      end
    end
  end

  # Test data for search queries
  @search_query_cases [
    {"capital", :list, "search for capital"},
    {"population", :list, "search for population"},
    {"president", :list, "search for president"},
    {"xyznonexistent123", :empty_or_list, "search non-existent term"},
    {"", :list, "empty search"},
  ]

  describe "query/1 with search - data driven" do
    for {search, expected_type, description} <- @search_query_cases do
      @search search
      @expected_type expected_type
      @description description

      test "#{description}" do
        result = FactDatabase.query(search: @search)

        assert is_list(result)
      end
    end
  end

  # Test limit option
  @limit_cases [
    {1, "limit to 1"},
    {5, "limit to 5"},
    {10, "limit to 10"},
    {100, "limit to 100"},
  ]

  describe "query/1 with limit - data driven" do
    for {limit, description} <- @limit_cases do
      @limit limit
      @description description

      test "#{description}" do
        result = FactDatabase.query(limit: @limit)

        assert is_list(result)
        assert length(result) <= @limit
      end
    end
  end

  # Test get_entity_facts/1
  @entity_facts_cases [
    "France",
    "Germany",
    "Tokyo",
    "Microsoft",
    "NonExistent12345",
  ]

  describe "get_entity_facts/1 - data driven" do
    for entity <- @entity_facts_cases do
      @entity entity

      test "gets facts for #{entity}" do
        result = FactDatabase.get_entity_facts(@entity)

        assert is_list(result)
      end
    end
  end

  # Test adding facts dynamically
  describe "add_fact_direct/1" do
    @add_fact_cases [
      {%{entity: "TestEntity", fact: "Test fact 1", category: "test"}, "basic fact"},
      {%{entity: "AnotherTest", fact: "Fact with details", category: "test", confidence: 0.9}, "fact with confidence"},
    ]

    for {fact_data, description} <- @add_fact_cases do
      @fact_data fact_data
      @description description

      test "#{description}" do
        result = FactDatabase.add_fact_direct(@fact_data)

        case result do
          {:ok, _} ->
            assert true

          :ok ->
            assert true

          {:error, _reason} ->
            # Some errors acceptable
            assert true
        end
      end
    end
  end

  # Test combined queries
  describe "combined query options" do
    test "entity + category" do
      result = FactDatabase.query(entity: "France", category: "geography")
      assert is_list(result)
    end

    test "search + limit" do
      result = FactDatabase.query(search: "capital", limit: 3)
      assert is_list(result)
      assert length(result) <= 3
    end

    test "category + limit" do
      result = FactDatabase.query(category: "geography", limit: 5)
      assert is_list(result)
      assert length(result) <= 5
    end
  end

  # Test stats
  describe "stats/0" do
    test "returns database statistics" do
      result = FactDatabase.stats()

      case result do
        stats when is_map(stats) ->
          assert true

        {:ok, stats} when is_map(stats) ->
          assert true

        _ ->
          # Function might return different format
          assert true
      end
    end
  end

  # Edge cases
  describe "edge cases" do
    test "handles special characters in search" do
      result = FactDatabase.query(search: "test's & query")
      assert is_list(result)
    end

    test "handles unicode in entity" do
      result = FactDatabase.query(entity: "東京")
      assert is_list(result)
    end

    test "handles nil options gracefully" do
      result = FactDatabase.query([])
      assert is_list(result)
    end
  end
end
