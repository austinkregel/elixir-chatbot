defmodule Brain.Analysis.AnaphoraResolver do
  @moduledoc """
  Resolves anaphoric references -- third-person personal pronouns and
  demonstratives -- to the entities of the conversation's earlier turns.

  A reference is a word the part-of-speech tagger (`Brain.ML.POSTagger`) tags
  `PRON` and whose closed-class declaration (`Brain.Lexicon.ClosedClass`)
  gives it the Universal Dependencies features of one: a personal pronoun of
  the third person (`PronType=Prs`, `Person=3`) that is not reflexive, or a
  demonstrative (`PronType=Dem`). First- and second-person pronouns name the
  speakers, and a reflexive's antecedent is in its own clause, so neither
  refers back to an earlier turn. The tag decides how the word is used:
  "that" tagged `SCONJ` ("I think that it works") is no reference.

  The history is chronological, the most recent turn last, and each turn's
  `:entities` is a list of entities with `:entity_type`, `:value` and
  `:confidence`. A reference resolves to the earlier entity that scores
  highest on how recent its turn is and on whether its type agrees with the
  reference. The referent is that entity, with the score of the resolution
  times the entity's own confidence as its confidence, and
  `source: :anaphora_resolution`. It has no position in the text it resolves
  a reference in.
  """

  alias Brain.Analysis.TypeHierarchy
  alias Brain.Lexicon.ClosedClass
  alias Brain.ML.{POSTagger, Tokenizer}

  # The entity types a reference agrees with, by the reference's UD Gender
  # when it has one, else its PronType. They are the types the anaphora
  # entries of `data/training/entities/anaphora.json` gave the same words:
  # Masc and Fem take those of the subject pronouns "he" and "she", Neut those
  # of the object pronoun "it", Dem those of the demonstratives, and Prs --
  # "they", "them" and "their", which have no gender -- those of the subject
  # and object pronouns together.
  @entity_type_compatibility %{
    "Masc" => ~w(person music_artist),
    "Fem" => ~w(person music_artist),
    "Neut" => ~w(device song music_artist topic location item),
    "Dem" => ~w(device song topic item location action),
    "Prs" => ~w(person music_artist device song topic location item)
  }
  @recency_decay 0.15

  @type referent :: %{
          entity_type: String.t(),
          value: term(),
          confidence: float(),
          source: :anaphora_resolution
        }

  @doc """
  The references in `text`, each as `{token_index, token, kind}`: the index
  of the token in `Brain.ML.Tokenizer.tokenize/1`'s tokens of `text`, the
  token, and the UD feature value that selects the entity types it agrees
  with (its Gender, else its PronType).
  """
  @spec references(String.t()) :: [{non_neg_integer(), String.t(), String.t()}]
  def references(text) when is_binary(text) do
    text
    |> Tokenizer.tokenize_words()
    |> POSTagger.predict(pos_model!())
    |> Enum.with_index()
    |> Enum.flat_map(fn {{token, tag}, idx} ->
      case reference_kind(tag, ClosedClass.features(token)) do
        nil -> []
        kind -> [{idx, token, kind}]
      end
    end)
  end

  defp reference_kind("PRON", %{"PronType" => "Dem"}), do: "Dem"

  defp reference_kind("PRON", %{"PronType" => "Prs", "Person" => "3"} = features) do
    if Map.get(features, "Reflex") == "Yes", do: nil, else: Map.get(features, "Gender", "Prs")
  end

  defp reference_kind(_tag, _features), do: nil

  @doc """
  Resolves the references in `text` to the entities of `history`.

  Returns `{:resolved, [{token_index, token, referent}]}` for the references
  that resolve, or `{:no_anaphora, []}` when none does.
  """
  @spec resolve(String.t(), [map()]) ::
          {:resolved, [{non_neg_integer(), String.t(), referent()}]} | {:no_anaphora, []}
  def resolve(text, history) when is_binary(text) and is_list(history) do
    candidates = candidates(history)

    resolutions =
      text
      |> references()
      |> Enum.flat_map(fn {idx, token, kind} ->
        case best_referent(candidates, kind) do
          nil -> []
          referent -> [{idx, token, referent}]
        end
      end)

    if resolutions == [], do: {:no_anaphora, []}, else: {:resolved, resolutions}
  end

  @doc """
  Resolves the references in `text` and returns `{:ok, substituted, referents}`:
  `text` with each resolved reference replaced by its referent's value, and
  the referents. The rest of `text` is left as it is.
  """
  @spec resolve_and_substitute(String.t(), [map()]) :: {:ok, String.t(), [referent()]}
  def resolve_and_substitute(text, history) do
    case resolve(text, history) do
      {:resolved, resolutions} ->
        tokens = Tokenizer.tokenize(text)
        graphemes = String.graphemes(text)

        substituted =
          resolutions
          |> Enum.sort_by(fn {idx, _, _} -> idx end, :desc)
          |> Enum.reduce(graphemes, fn {idx, _token, referent}, acc ->
            %{start_pos: start_pos, end_pos: end_pos} = Enum.at(tokens, idx)
            Enum.take(acc, start_pos) ++ [to_string(referent.value)] ++ Enum.drop(acc, end_pos + 1)
          end)
          |> Enum.join()

        {:ok, substituted, Enum.map(resolutions, fn {_, _, referent} -> referent end)}

      {:no_anaphora, []} ->
        {:ok, text, []}
    end
  end

  @doc """
  The entities of `history` a reference can resolve to, each as
  `{entity, recency_score, turns_ago}`. The history is chronological: its
  last turn is 1 turn ago and scores a recency of 1.0. Every turn's
  `:entities` must be a list of entities with a string `:entity_type`, a
  `:value` and a numeric `:confidence`; anything else raises.
  """
  @spec candidates([map()]) :: [{map(), float(), pos_integer()}]
  def candidates(history) when is_list(history) do
    history
    |> Enum.reverse()
    |> Enum.with_index(1)
    |> Enum.flat_map(fn {turn, turns_ago} ->
      recency = :math.exp(-@recency_decay * (turns_ago - 1))
      turn |> turn_entities!() |> Enum.map(&{&1, recency, turns_ago})
    end)
  end

  defp turn_entities!(%{entities: entities}) when is_list(entities), do: Enum.map(entities, &candidate!/1)

  defp turn_entities!(turn) do
    raise ArgumentError,
          "AnaphoraResolver: a history turn's :entities must be a list of entities, got: #{inspect(turn)}"
  end

  defp candidate!(%{entity_type: type, value: _value, confidence: confidence} = entity)
       when is_binary(type) and is_number(confidence),
       do: entity

  defp candidate!(entity) do
    raise ArgumentError,
          "AnaphoraResolver: a history entity needs a string :entity_type, a :value and a numeric " <>
            ":confidence, got: #{inspect(entity)}"
  end

  defp best_referent(candidates, kind) do
    compatible = Map.fetch!(@entity_type_compatibility, kind)

    candidates
    |> Enum.map(fn {entity, recency_score, _turns_ago} ->
      type_score =
        cond do
          entity.entity_type in compatible -> 1.0
          Enum.any?(compatible, &TypeHierarchy.compatible?(entity.entity_type, &1)) -> 0.85
          true -> 0.3
        end

      {entity, recency_score * 0.4 + type_score * 0.6}
    end)
    |> Enum.filter(fn {_, score} -> score > 0.3 end)
    |> Enum.max_by(fn {_, score} -> score end, fn -> nil end)
    |> case do
      {entity, score} ->
        %{
          entity_type: entity.entity_type,
          value: entity.value,
          confidence: score * entity.confidence,
          source: :anaphora_resolution
        }

      nil ->
        nil
    end
  end

  defp pos_model! do
    case POSTagger.get_model() do
      {:ok, model} ->
        model

      {:error, reason} ->
        raise "AnaphoraResolver: no usable part-of-speech model at #{POSTagger.model_path()}: #{inspect(reason)}"
    end
  end
end
