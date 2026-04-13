defmodule Brain.Analysis.AnaphoraResolver do
  @moduledoc "Resolves anaphoric references (pronouns, demonstratives) using\nGazetteer-based lookup and conversation history.\n\nThis module uses tokenization (no regex) and Gazetteer entity lookup\nto identify anaphoric terms, then resolves them to referents from\nconversation history based on recency and entity type compatibility.\n"

  alias Brain.ML.Tokenizer
  alias Brain.ML.Gazetteer

  require Logger

  @entity_type_compatibility %{
    "pronoun_object" => ~w(device song music-artist topic location item),
    "pronoun_subject" => ~w(person music-artist),
    "demonstrative" => ~w(device song topic item location action),
    "location_reference" => ~w(location room city place-name geo-location),
    "indefinite" => ~w(device song topic item),
    "identity_reference" => ~w(device song topic item action),
    "possessive" => ~w(person device)
  }
  @recency_decay 0.15

  @doc "Resolves anaphoric references in text using conversation history.\n\nReturns:\n- {:resolved, resolutions} - List of {token_index, original_term, resolved_entity}\n- {:no_anaphora, []} - No anaphoric terms found\n"
  def resolve(text, conversation_history) when is_binary(text) do
    tokens = Tokenizer.tokenize_normalized(text, expand_contractions: true)
    anaphora_spans = lookup_anaphora_spans(tokens)

    if anaphora_spans != [] do
      candidates = extract_candidate_entities(conversation_history)

      resolutions =
        Enum.map(anaphora_spans, fn {idx, anaphora_type, term} ->
          best_match = score_and_rank_candidates(candidates, anaphora_type)
          {idx, term, best_match}
        end)
        |> Enum.filter(fn {_, _, match} -> match != nil end)

      if resolutions != [] do
        {:resolved, resolutions}
      else
        {:no_anaphora, []}
      end
    else
      {:no_anaphora, []}
    end
  end

  @doc "Resolves references and returns substituted text with entities.\n\nThis is useful when you want to expand pronouns before intent classification.\n"
  def resolve_and_substitute(text, conversation_history) do
    case resolve(text, conversation_history) do
      {:resolved, resolutions} ->
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

        resolved_entities =
          Enum.map(resolutions, fn {_, _, entity} -> entity end)
          |> Enum.filter(&(&1 != nil))

        {:ok, substituted_text, resolved_entities}

      {:no_anaphora, []} ->
        {:ok, text, []}
    end
  end

  defp lookup_anaphora_spans(tokens) do
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
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

  defp extract_candidate_entities(_) do
    []
  end

  defp extract_entities_from_context(context) when is_map(context) do
    entities = Map.get(context, :entities) || Map.get(context, "entities") || %{}

    case entities do
      list when is_list(list) ->
        list

      map when is_map(map) ->
        Enum.map(map, fn {type, value} ->
          %{entity: type, value: value}
        end)

      _ ->
        []
    end
  end

  defp extract_entities_from_context(_) do
    []
  end

  defp calculate_recency_score(turns_ago) do
    :math.exp(-@recency_decay * (turns_ago - 1))
  end

  defp score_and_rank_candidates(candidates, anaphora_type) when is_list(candidates) do
    static_compatible = Map.get(@entity_type_compatibility, anaphora_type, [])

    candidates
    |> Enum.map(fn {entity, recency_score, _turns_ago} ->
      entity_type = to_string(entity[:entity_type] || "")

      type_score =
        cond do
          entity_type in static_compatible ->
            1.0

          type_compatible_via_hierarchy?(entity_type, static_compatible) ->
            0.85

          true ->
            0.3
        end

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

  defp score_and_rank_candidates(_, _) do
    nil
  end

  defp type_compatible_via_hierarchy?(entity_type, compatible_types) do
    alias Brain.Analysis.TypeHierarchy

    Enum.any?(compatible_types, fn expected ->
      TypeHierarchy.compatible?(entity_type, expected)
    end)
  end
end
