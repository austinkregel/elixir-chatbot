defmodule Brain.Analysis.NoveltyDetector do
  @moduledoc """
  Detects novel intent candidates based on classifier uncertainty.

  A candidate is considered novel if:
  - Best score is below a threshold (low confidence)
  - Margin between best and second is small (ambiguous)
  - The utterance is substantive (not pure smalltalk/backchannel)

  A novel candidate is considered *researchable* only if it contains
  concepts that could yield factual knowledge from external sources.
  Social/phatic inputs ("Hello, I'm Austin") are novel but not researchable.
  """

  alias Brain.ML.Tokenizer

  require Logger

  @default_novelty_threshold 0.5
  @default_margin_threshold 0.2
  @min_researchable_tokens 4

  @doc """
  Determines if an utterance represents a novel intent candidate.

  Returns `{:novel, novelty_score}` if novel, `:not_novel` otherwise.

  ## Options
    - `:novelty_threshold` - Maximum best_score to consider novel (default: 0.5)
    - `:margin_threshold` - Minimum margin required to avoid being novel (default: 0.2)
  """
  def is_novel?(best_score, margin, opts \\ []) do
    novelty_threshold = Keyword.get(opts, :novelty_threshold, @default_novelty_threshold)
    margin_threshold = Keyword.get(opts, :margin_threshold, @default_margin_threshold)
    entities = Keyword.get(opts, :entities, [])

    is_low_confidence = best_score < novelty_threshold
    is_ambiguous = margin < margin_threshold

    has_graph_unknown =
      entities != [] and
        Enum.any?(entities, fn e -> Map.get(e, :graph_known) == false end)

    cond do
      is_low_confidence or is_ambiguous or has_graph_unknown ->
        base_score = (1.0 - best_score) * 0.7 + (1.0 - margin) * 0.3
        graph_boost = if has_graph_unknown, do: 0.15, else: 0.0
        {:novel, min(base_score + graph_boost, 1.0)}

      true ->
        :not_novel
    end
  end

  @doc """
  Checks if an utterance is substantive enough to warrant review.

  Filters out pure smalltalk, backchannels, and other non-substantive utterances.
  """
  def is_substantive?(speech_act, _intent) do
    if speech_act.category == :expressive do
      not well_handled_expressive?(speech_act.sub_type)
    else
      true
    end
  end

  @doc """
  Determines whether a novel input contains researchable concepts.

  An input is researchable when it references external knowledge that
  could produce factual findings from web/academic sources. This filters
  out social/phatic inputs (greetings, introductions, backchannels) that
  are "novel" only because the classifier is uncertain, not because they
  contain knowledge worth investigating.

  Uses the outputs of trained models (entity extraction, speech act
  classification, POS tagging) rather than string matching.
  """
  @spec is_researchable?(String.t(), list(map()), map()) :: boolean()
  def is_researchable?(text, entities, speech_act) do
    has_depth = content_depth_sufficient?(text)
    has_entities = has_researchable_entities?(entities)
    is_knowledge_oriented = knowledge_oriented_act?(speech_act)

    has_depth and (has_entities or is_knowledge_oriented)
  end

  defp well_handled_expressive?(sub_type) do
    sub_type in [:greeting, :farewell, :thanks, :apology, :backchannel, :acknowledgment]
  end

  defp content_depth_sufficient?(text) do
    tokens = Tokenizer.tokenize_words(text)
    content_weight = lexicon_content_weight(tokens)

    if content_weight > 0.0 do
      content_weight >= 2.5
    else
      length(tokens) >= @min_researchable_tokens
    end
  end

  defp lexicon_content_weight(tokens) do
    if Process.whereis(Brain.ML.Lexicon) do
      tokens
      |> Enum.map(fn token ->
        lower = String.downcase(token)
        pos_tags = Brain.ML.Lexicon.pos(lower)

        cond do
          :noun in pos_tags or :verb in pos_tags -> 1.0
          :adj in pos_tags or :adj_satellite in pos_tags -> 0.7
          :adv in pos_tags -> 0.3
          true -> 0.1
        end
      end)
      |> Enum.sum()
    else
      0.0
    end
  end

  defp has_researchable_entities?(entities) when is_list(entities) do
    entities
    |> Enum.filter(fn entity ->
      confidence = Map.get(entity, :confidence, 0)
      confidence > 0.3
    end)
    |> Enum.any?(fn entity ->
      entity_type =
        Map.get(entity, :entity_type) || Map.get(entity, "entity_type") || Map.get(entity, :type)

      graph_known = Map.get(entity, :graph_known, true)

      researchable_entity_type?(entity_type) or not graph_known
    end)
  end

  defp has_researchable_entities?(_), do: false

  defp researchable_entity_type?(nil), do: false
  defp researchable_entity_type?("unknown"), do: false

  defp researchable_entity_type?(type) when is_atom(type) do
    researchable_entity_type?(Atom.to_string(type))
  end

  defp researchable_entity_type?(type) when is_binary(type) do
    normalized = String.downcase(type)

    if normalized in ["person", "pronoun"] do
      false
    else
      if Process.whereis(Brain.ML.Lexicon) do
        chain = Brain.ML.Lexicon.hypernym_chain(normalized, :noun, max_depth: 8)
        non_researchable = ~w(person pronoun function_word)
        not Enum.any?(chain, &(&1 in non_researchable))
      else
        true
      end
    end
  end

  defp knowledge_oriented_act?(%{category: :directive}), do: true

  defp knowledge_oriented_act?(%{category: :assertive} = act) do
    Map.get(act, :sub_type) not in [nil, :greeting, :farewell, :backchannel, :acknowledgment]
  end

  defp knowledge_oriented_act?(_), do: false
end
