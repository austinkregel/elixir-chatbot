defmodule Brain.Response.FactRetriever do
  @moduledoc """
  Retrieves relevant facts from the FactDatabase for use in responses.

  This module provides a simple interface for querying facts based on:
  - Entity mentions in the conversation
  - Intent classification (factual questions)
  - Keyword search in user queries
  """

  alias Brain.FactDatabase
  require Logger

  @doc """
  Retrieves relevant facts for a given query or entity.

  Options:
  - `:entity` - Entity name to search for facts about
  - `:category` - Filter by category (geography, science, history, general)
  - `:search` - Search term to find in fact text
  - `:limit` - Maximum number of facts to return (default: 3)
  """
  def get_relevant_facts(opts \\ []) do
    try do
      FactDatabase.query(opts)
    rescue
      e ->
        Logger.warning("Failed to query FactDatabase", %{error: Exception.message(e)})
        []
    catch
      :exit, _ ->
        Logger.debug("FactDatabase not available")
        []
    end
  end

  @doc """
  Extracts entities from a query and retrieves facts about them.

  This is a convenience function that combines entity extraction
  with fact retrieval.

  Parameters:
  - query: The original query text (for keyword search)
  - entities: List of entity maps or entity name strings
  """
  def get_facts_for_query(query \\ "", entities \\ []) do
    query_str = if is_binary(query), do: query, else: ""
    entities = if is_list(entities), do: entities, else: []

    entity_facts =
      entities
      |> Enum.filter(&(not is_nil(&1)))
      |> Enum.map(fn entity ->
        entity_name =
          cond do
            is_binary(entity) -> entity
            is_map(entity) -> entity[:value] || entity["value"] || ""
            true -> ""
          end

        if entity_name != "" do
          get_relevant_facts(entity: entity_name, limit: 2)
        else
          []
        end
      end)
      |> List.flatten()
      |> Enum.uniq_by(fn
        %{id: id} -> id
        %{"id" => id} -> id
        fact when is_struct(fact) -> Map.get(fact, :id, fact)
        other -> other
      end)

    results =
      if entity_facts == [] and query_str != "" do
        get_relevant_facts(search: query_str, limit: 3)
      else
        entity_facts
      end

    record_fact_retrieval(length(results))
    results
  end

  defp record_fact_retrieval(result_count) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(Brain.Metrics.Aggregator, {:record_fact_retrieval, result_count})
    end
  end

  @doc """
  Formats facts for use in responses.

  Returns a list of formatted fact strings.
  """
  def format_facts(facts, max_count \\ 3) do
    facts
    |> Enum.take(max_count)
    |> Enum.map(&format_single_fact/1)
  end

  @doc """
  Formats a single fact for display.
  """
  def format_single_fact(%Brain.FactDatabase.Fact{} = fact) do
    # If fact already mentions the entity, return as-is
    # Otherwise, prepend entity name
    fact_tokens = fact.fact |> Brain.ML.Tokenizer.tokenize_normalized() |> MapSet.new()
    entity_tokens = fact.entity |> Brain.ML.Tokenizer.tokenize_normalized() |> MapSet.new()

    if not MapSet.disjoint?(fact_tokens, entity_tokens) do
      fact.fact
    else
      "#{fact.entity}: #{fact.fact}"
    end
  end

  def format_single_fact(fact) when is_map(fact) do
    # Backwards compatibility for raw maps
    fact_text = Map.get(fact, "fact", Map.get(fact, :fact, ""))
    entity = Map.get(fact, "entity", Map.get(fact, :entity, ""))

    fact_tokens = fact_text |> Brain.ML.Tokenizer.tokenize_normalized() |> MapSet.new()
    entity_tokens = entity |> Brain.ML.Tokenizer.tokenize_normalized() |> MapSet.new()

    if not MapSet.disjoint?(fact_tokens, entity_tokens) do
      fact_text
    else
      "#{entity}: #{fact_text}"
    end
  end

  def format_single_fact(_), do: ""

  @doc """
  Checks if FactDatabase is available.
  """
  def available? do
    case Process.whereis(FactDatabase) do
      nil -> false
      _pid -> true
    end
  end
end
