defmodule ChatBot.Response.MemoryAugmented do
  @moduledoc """
  Generates contextually appropriate responses by:
  1. Finding similar past interactions via TF-IDF similarity
  2. Extracting successful response patterns from those interactions
  3. Adapting patterns to current context using slot filling

  This module provides an alternative to template-based responses
  by leveraging episodic memory of past successful conversations.
  """

  alias ChatBot.Memory.{Store, Embedder}
  alias ChatBot.ML.Tokenizer

  require Logger

  # Minimum similarity threshold for using a memory-based response
  @similarity_threshold 0.6

  # Maximum number of episodes to consider
  @max_episodes 5

  @doc """
  Attempts to generate a response from similar past interactions.

  Returns:
  - {:ok, response, metadata} - Successfully generated response
  - :no_memory_match - No suitable past interactions found
  - :embedder_not_ready - Embedder not available
  """
  def generate(intent, entities, context \\ %{}) do
    if Embedder.ready?() and Process.whereis(Store) do
      do_generate(intent, entities, context)
    else
      :embedder_not_ready
    end
  end

  @doc """
  Finds similar past interactions for debugging/inspection.
  """
  def find_similar_episodes(intent, entities, limit \\ @max_episodes) do
    query = build_semantic_query(intent, entities)

    case Store.query_similar(query, limit) do
      {:ok, episodes} -> {:ok, episodes}
      error -> error
    end
  end

  # Private implementation

  defp do_generate(intent, entities, context) do
    # Build query from current state
    query = build_semantic_query(intent, entities)

    # Find similar successful past exchanges
    case Store.query_similar(query, @max_episodes) do
      {:ok, [_ | _] = episodes} ->
        # Extract and adapt response pattern
        adapt_from_episodes(episodes, entities, context)

      _ ->
        :no_memory_match
    end
  rescue
    e ->
      Logger.warning("Memory-augmented generation failed: #{Exception.message(e)}")
      :no_memory_match
  end

  defp build_semantic_query(intent, entities) do
    # Combine intent with entity values for rich query
    entity_text =
      entities
      |> Enum.map(fn e -> e[:value] || e["value"] || "" end)
      |> Enum.filter(&(&1 != ""))
      |> Enum.join(" ")

    if entity_text == "" do
      intent || ""
    else
      "#{intent || ""} #{entity_text}"
    end
  end

  defp adapt_from_episodes(episodes, current_entities, _context) do
    # Filter for episodes with positive outcomes
    positive_episodes =
      episodes
      |> Enum.filter(fn {ep, sim} ->
        sim >= @similarity_threshold and positive_outcome?(ep)
      end)

    case positive_episodes do
      [] ->
        :no_memory_match

      candidates ->
        # Find best match
        {best_episode, similarity} =
          Enum.max_by(candidates, fn {_ep, sim} -> sim end)

        # Extract response pattern and substitute current entities
        case extract_response_pattern(best_episode) do
          nil ->
            :no_memory_match

          pattern ->
            filled = substitute_entities(pattern, current_entities)

            Logger.debug("Memory-augmented response generated", %{
              episode_id: best_episode.id,
              similarity: similarity,
              pattern_length: String.length(pattern),
              filled_length: String.length(filled)
            })

            {:ok, filled,
             %{
               source: :memory,
               episode_id: best_episode.id,
               similarity: similarity,
               original_pattern: pattern
             }}
        end
    end
  end

  defp positive_outcome?(episode) do
    # Check episode tags or outcome for positive indicators
    tags = episode.tags || []
    outcome = episode.outcome || ""

    cond do
      # Explicit positive tag
      "successful" in tags -> true
      "positive" in tags -> true
      # Negative indicators
      "failed" in tags -> false
      "negative" in tags -> false
      # Check outcome text
      String.contains?(outcome, "success") -> true
      String.contains?(outcome, "error") -> false
      # Default to considering it positive
      true -> true
    end
  end

  defp extract_response_pattern(episode) do
    # The outcome field typically contains the bot's response
    outcome = episode.outcome

    cond do
      is_binary(outcome) and String.length(outcome) > 0 ->
        # Clean up the pattern for reuse
        clean_pattern(outcome)

      is_map(episode) and Map.has_key?(episode, :response) ->
        clean_pattern(episode.response)

      true ->
        nil
    end
  end

  defp clean_pattern(text) when is_binary(text) do
    text
    |> String.trim()
    |> Tokenizer.collapse_whitespace_public()
  end

  defp clean_pattern(_), do: nil

  defp substitute_entities(pattern, entities) when is_list(entities) do
    # Replace placeholder tokens with current entity values
    # Placeholders are in format @entity_type
    Enum.reduce(entities, pattern, fn entity, acc ->
      entity_type = entity[:entity_type]
      value = entity[:value] || ""

      if entity_type && value != "" do
        placeholder = "@#{entity_type}"

        # Use tokenizer to find and replace (not regex for the main logic)
        replace_placeholder_tokens(acc, placeholder, value)
      else
        acc
      end
    end)
  end

  defp substitute_entities(pattern, _), do: pattern

  defp replace_placeholder_tokens(text, placeholder, value) do
    # Tokenize and replace matching tokens
    tokens = Tokenizer.tokenize_words(text)
    normalized_placeholder = Tokenizer.normalize(placeholder)

    replaced_tokens =
      Enum.map(tokens, fn token ->
        if Tokenizer.normalize(token) == normalized_placeholder do
          value
        else
          token
        end
      end)

    Enum.join(replaced_tokens, " ")
  end
end
