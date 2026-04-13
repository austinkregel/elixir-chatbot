defmodule Brain.Response.MemoryAugmented do
  @moduledoc "Generates contextually appropriate responses by:\n1. Finding similar past interactions via TF-IDF similarity\n2. Extracting successful response patterns from those interactions\n3. Adapting patterns to current context using slot filling\n\nThis module provides an alternative to template-based responses\nby leveraging episodic memory of past successful conversations.\n"

  alias Brain.Memory.{Store, Embedder}
  alias Brain.ML.Tokenizer

  require Logger
  @similarity_threshold 0.6
  @max_episodes 5

  @doc "Attempts to generate a response from similar past interactions.\n\nReturns:\n- {:ok, response, metadata} - Successfully generated response\n- :no_memory_match - No suitable past interactions found\n- :embedder_not_ready - Embedder not available\n"
  def generate(intent, entities, context \\ %{}) do
    if Embedder.ready?() and Process.whereis(Store) do
      do_generate(intent, entities, context)
    else
      :embedder_not_ready
    end
  end

  @doc "Finds similar past interactions for debugging/inspection.\n"
  def find_similar_episodes(intent, entities, limit \\ @max_episodes) do
    query = build_semantic_query(intent, entities)

    case Store.query_similar(query, limit) do
      {:ok, episodes} -> {:ok, episodes}
      error -> error
    end
  end

  defp do_generate(intent, entities, context) do
    query = build_semantic_query(intent, entities)

    case Store.query_similar(query, @max_episodes) do
      {:ok, [_ | _] = episodes} ->
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
    positive_episodes =
      episodes
      |> Enum.filter(fn {ep, sim} ->
        sim >= @similarity_threshold and positive_outcome?(ep)
      end)

    case positive_episodes do
      [] ->
        :no_memory_match

      candidates ->
        {best_episode, similarity} =
          Enum.max_by(candidates, fn {_ep, sim} -> sim end)

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

  @positive_tags MapSet.new(~w(successful positive completed resolved))
  @negative_tags MapSet.new(~w(failed negative error rejected))
  @positive_outcome_tokens MapSet.new(~w(success successful completed resolved))
  @negative_outcome_tokens MapSet.new(~w(error failed failure rejected))

  defp positive_outcome?(episode) do
    tags = MapSet.new(episode.tags || [])

    cond do
      not MapSet.disjoint?(tags, @positive_tags) -> true
      not MapSet.disjoint?(tags, @negative_tags) -> false
      true ->
        outcome = episode.outcome || ""

        if outcome == "" do
          true
        else
          outcome_tokens = Tokenizer.tokenize_normalized(outcome) |> MapSet.new()
          pos = not MapSet.disjoint?(outcome_tokens, @positive_outcome_tokens)
          neg = not MapSet.disjoint?(outcome_tokens, @negative_outcome_tokens)

          cond do
            pos and not neg -> true
            neg and not pos -> false
            true -> true
          end
        end
    end
  end

  defp extract_response_pattern(episode) do
    outcome = episode.outcome

    cond do
      is_binary(outcome) and String.length(outcome) > 0 ->
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

  defp clean_pattern(_) do
    nil
  end

  defp substitute_entities(pattern, entities) when is_list(entities) do
    Enum.reduce(entities, pattern, fn entity, acc ->
      entity_type = entity[:entity_type]
      value = entity[:value] || ""

      if entity_type && value != "" do
        placeholder = "@#{entity_type}"
        replace_placeholder_tokens(acc, placeholder, value)
      else
        acc
      end
    end)
  end

  defp substitute_entities(pattern, _) do
    pattern
  end

  defp replace_placeholder_tokens(text, placeholder, value) do
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