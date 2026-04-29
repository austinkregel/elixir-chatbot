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
        raw_score = min(base_score + graph_boost, 1.0)

        text = Keyword.get(opts, :text, "")
        analysis_chunk = Keyword.get(opts, :analysis_chunk)
        final_score = maybe_kg_downweight(raw_score, text, analysis_chunk)

        {:novel, final_score}

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

  @doc """
  KG-aware redundancy downweight. Reduces the novelty score when the input
  closely matches an existing high-scoring belief, but does NOT suppress it
  entirely. State changes (sentiment/tense/negation flip) are detected and
  bypass the downweight.
  """
  def maybe_kg_downweight(novelty_score, text, analysis_chunk) do
    config = Application.get_env(:brain, :kg_signals, [])

    unless Keyword.get(config, :enabled, true) and Keyword.get(config, :novelty_downweight, true) do
      novelty_score
    else
      do_kg_downweight(novelty_score, text, analysis_chunk)
    end
  end

  defp do_kg_downweight(novelty_score, text, _analysis_chunk) when text == "" or is_nil(text) do
    novelty_score
  end

  defp do_kg_downweight(novelty_score, text, analysis_chunk) do
    unless Brain.Epistemic.BeliefStore.ready?() do
      novelty_score
    else
      case find_matching_beliefs(text) do
        [] ->
          novelty_score

        matching_beliefs ->
          best_match = Enum.max_by(matching_beliefs, fn {_belief, sim} -> sim end)
          {belief, similarity} = best_match

          if is_state_change?(belief, analysis_chunk) do
            :telemetry.execute([:brain, :novelty, :state_change_detected], %{similarity: similarity}, %{})
            novelty_score
          else
            if belief_still_active?(belief) do
              downweight = similarity * 0.5
              downweighted = novelty_score * (1.0 - downweight)
              :telemetry.execute([:brain, :novelty, :downweighted], %{
                original: novelty_score,
                downweighted: downweighted,
                similarity: similarity
              }, %{})
              downweighted
            else
              novelty_score
            end
          end
      end
    end
  rescue
    _ -> novelty_score
  end

  defp find_matching_beliefs(text) do
    case Brain.Memory.Embedder.embed(text) do
      {:ok, query_embedding} ->
        beliefs = Brain.Epistemic.BeliefStore.query_beliefs()

        beliefs
        |> Enum.filter(fn b ->
          b.subject != :system and b.predicate != :consolidated_knowledge
        end)
        |> Enum.flat_map(fn belief ->
          belief_text = "#{belief.subject} #{belief.predicate} #{belief.object}"

          case Brain.Memory.Embedder.embed(belief_text) do
            {:ok, belief_embedding} ->
              sim = FourthWall.Math.cosine_similarity(query_embedding, belief_embedding)
              if sim >= 0.85, do: [{belief, sim}], else: []

            _ ->
              []
          end
        end)
        |> Enum.take(5)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp is_state_change?(belief, analysis_chunk) do
    if is_nil(analysis_chunk) do
      false
    else
      chunk_sentiment = Map.get(analysis_chunk, :sentiment)
      chunk_tense = Map.get(analysis_chunk, :tense)

      belief_metadata = Map.get(belief, :metadata, %{})
      belief_sentiment = Map.get(belief_metadata, :sentiment)

      sentiment_differs =
        chunk_sentiment != nil and belief_sentiment != nil and
          sentiment_polarity(chunk_sentiment) != sentiment_polarity(belief_sentiment)

      tense_differs =
        chunk_tense != nil and
          Map.get(belief_metadata, :tense) != nil and
          chunk_tense != Map.get(belief_metadata, :tense)

      has_negation = Map.get(analysis_chunk, :has_negation, false)

      sentiment_differs or tense_differs or has_negation
    end
  end

  defp sentiment_polarity(sentiment) when is_map(sentiment) do
    label = Map.get(sentiment, :label) || Map.get(sentiment, "label")

    cond do
      label in [:positive, "positive"] -> :positive
      label in [:negative, "negative"] -> :negative
      true -> :neutral
    end
  end

  defp sentiment_polarity(_), do: :neutral

  defp belief_still_active?(belief) do
    config = Application.get_env(:brain, :kg_signals, [])
    window_days = Keyword.get(config, :novelty_retraction_window, 7)

    if belief.node_id do
      case Brain.Epistemic.JTMS.get_node(belief.node_id) do
        {:ok, node} ->
          label = Map.get(node, :label, :in)
          label == :in

        _ ->
          true
      end
    else
      cutoff = DateTime.add(DateTime.utc_now(), -window_days * 86400, :second)

      case belief.last_confirmed do
        nil -> true
        dt -> DateTime.compare(dt, cutoff) == :gt
      end
    end
  rescue
    _ -> true
  end
end
