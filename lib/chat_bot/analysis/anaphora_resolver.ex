defmodule ChatBot.Analysis.AnaphoraResolver do
  @moduledoc """
  Resolves anaphoric references (pronouns, demonstratives) using
  Gazetteer-based lookup and conversation history.

  This module uses tokenization (no regex) and Gazetteer entity lookup
  to identify anaphoric terms, then resolves them to referents from
  conversation history based on recency and entity type compatibility.
  """

  alias ChatBot.ML.Tokenizer
  alias ChatBot.ML.Gazetteer

  require Logger

  # Entity types that match different anaphora types
  @entity_type_compatibility %{
    "pronoun_object" => ~w(device song music-artist topic location item),
    "pronoun_subject" => ~w(person music-artist),
    "demonstrative" => ~w(device song topic item location action),
    "location_reference" => ~w(location room city place-name geo-location),
    "indefinite" => ~w(device song topic item),
    "identity_reference" => ~w(device song topic item action),
    "possessive" => ~w(person device)
  }

  # Recency decay factor for scoring
  @recency_decay 0.15

  @doc """
  Resolves anaphoric references in text using conversation history.

  Returns:
  - {:resolved, resolutions} - List of {token_index, original_term, resolved_entity}
  - {:no_anaphora, []} - No anaphoric terms found
  """
  def resolve(text, conversation_history) when is_binary(text) do
    # Tokenize without regex
    tokens = Tokenizer.tokenize_normalized(text, expand_contractions: true)

    # Lookup anaphoric terms via Gazetteer
    anaphora_spans = lookup_anaphora_spans(tokens)

    if length(anaphora_spans) > 0 do
      # Extract candidate entities from history
      candidates = extract_candidate_entities(conversation_history)

      # Resolve each anaphoric reference
      resolutions =
        Enum.map(anaphora_spans, fn {idx, anaphora_type, term} ->
          best_match = score_and_rank_candidates(candidates, anaphora_type)
          {idx, term, best_match}
        end)
        |> Enum.filter(fn {_, _, match} -> match != nil end)

      if length(resolutions) > 0 do
        {:resolved, resolutions}
      else
        {:no_anaphora, []}
      end
    else
      {:no_anaphora, []}
    end
  end

  @doc """
  Resolves references and returns substituted text with entities.

  This is useful when you want to expand pronouns before intent classification.
  """
  def resolve_and_substitute(text, conversation_history) do
    case resolve(text, conversation_history) do
      {:resolved, resolutions} ->
        # Build substituted text
        tokens = Tokenizer.tokenize_words(text)

        substituted_tokens =
          tokens
          |> Enum.with_index()
          |> Enum.map(fn {token, idx} ->
            case Enum.find(resolutions, fn {res_idx, _, _} -> res_idx == idx end) do
              {_, _, entity} when not is_nil(entity) ->
                entity[:value] || entity["value"] || token

              _ ->
                token
            end
          end)

        substituted_text = Enum.join(substituted_tokens, " ")

        # Extract resolved entities for use in analysis
        resolved_entities =
          Enum.map(resolutions, fn {_, _, entity} -> entity end)
          |> Enum.filter(&(&1 != nil))

        {:ok, substituted_text, resolved_entities}

      {:no_anaphora, []} ->
        {:ok, text, []}
    end
  end

  # Lookup anaphoric terms using Gazetteer
  defp lookup_anaphora_spans(tokens) do
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      # Normalize for lookup
      normalized = Tokenizer.normalize(token)

      case Gazetteer.lookup(normalized) do
        {:ok, %{entity_type: "anaphora", metadata: meta}} ->
          anaphora_type = meta["anaphora_type"] || meta[:anaphora_type]
          [{idx, anaphora_type, token}]

        _ ->
          []
      end
    end)
  end

  # Extract candidate entities from conversation history
  defp extract_candidate_entities(history) when is_list(history) do
    history
    |> Enum.with_index(1)
    |> Enum.flat_map(fn {context, turns_ago} ->
      entities = extract_entities_from_context(context)

      Enum.map(entities, fn entity ->
        recency_score = calculate_recency_score(turns_ago)
        {entity, recency_score, turns_ago}
      end)
    end)
  end

  defp extract_candidate_entities(_), do: []

  defp extract_entities_from_context(context) when is_map(context) do
    # Get entities from context
    entities = Map.get(context, :entities) || Map.get(context, "entities") || %{}

    case entities do
      list when is_list(list) ->
        list

      map when is_map(map) ->
        # Convert map format to list format
        Enum.map(map, fn {type, value} ->
          %{entity: type, value: value}
        end)

      _ ->
        []
    end
  end

  defp extract_entities_from_context(_), do: []

  defp calculate_recency_score(turns_ago) do
    # Exponential decay
    :math.exp(-@recency_decay * (turns_ago - 1))
  end

  # Score and rank candidates based on type compatibility and recency
  defp score_and_rank_candidates(candidates, anaphora_type) when is_list(candidates) do
    compatible_types = Map.get(@entity_type_compatibility, anaphora_type, [])

    candidates
    |> Enum.map(fn {entity, recency_score, _turns_ago} ->
      entity_type = entity[:entity_type]
      type_score = if entity_type in compatible_types, do: 1.0, else: 0.3

      # Combined score: weighted average of recency and type compatibility
      combined_score = recency_score * 0.4 + type_score * 0.6

      {entity, combined_score}
    end)
    |> Enum.filter(fn {_, score} -> score > 0.3 end)
    |> Enum.max_by(fn {_, score} -> score end, fn -> nil end)
    |> case do
      {entity, _score} -> entity
      nil -> nil
    end
  end

  defp score_and_rank_candidates(_, _), do: nil
end
