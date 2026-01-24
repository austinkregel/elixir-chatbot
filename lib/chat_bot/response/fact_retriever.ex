defmodule ChatBot.Response.FactRetriever do
  @moduledoc """
  Retrieves relevant facts from the FactDatabase for use in responses.
  
  This module provides a simple interface for querying facts based on:
  - Entity mentions in the conversation
  - Intent classification (factual questions)
  - Keyword search in user queries
  """

  alias ChatBot.FactDatabase
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
    
    # Try to get facts for each mentioned entity
    entity_facts =
      entities
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
      |> Enum.uniq_by(& &1["id"])

    # Also try keyword search if no entity facts found or query is provided
    if entity_facts == [] and query_str != "" do
      get_relevant_facts(search: query_str, limit: 3)
    else
      entity_facts
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
  def format_single_fact(fact) when is_map(fact) do
    fact_text = Map.get(fact, "fact", "")
    entity = Map.get(fact, "entity", "")
    
    # If fact already mentions the entity, return as-is
    # Otherwise, prepend entity name
    if String.contains?(String.downcase(fact_text), String.downcase(entity)) do
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
