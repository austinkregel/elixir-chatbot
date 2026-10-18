defmodule Brain.Epistemic.ConsolidationBridge do
  @moduledoc """
  Bridges Memory.Consolidation SemanticFacts to the Epistemic BeliefStore.

  When memory consolidation creates a SemanticFact from clustered episodes,
  this module converts it to world-scoped beliefs with provenance links
  back to the source episodes.

  ## Triple extraction (phase 2 of KG signal strengthening)

  Instead of flattening every SemanticFact into a single
  `subject: :system, predicate: :consolidated_knowledge` belief, this module
  now extracts structured `(subject, predicate, object)` triples via SRL
  on the source episode utterances. Each triple becomes a separate Belief
  with proper subject/predicate/object, enabling `BeliefStore.by_subject`
  and `by_predicate` index queries.

  When SRL extraction succeeds and the predicate is in the TripleScorer's
  trained vocabulary, confidence blends the episode count prior with the
  structural plausibility score. OOV predicates use count prior only (no
  noisy mid-range default).

  Falls back to the legacy single-belief shape when SRL produces zero frames.
  """

  require Logger

  alias Brain.Epistemic.{BeliefStore, Types}
  alias Brain.ML.KnowledgeGraph.{PredicateNormalizer, TripleScorer}
  alias Types.Belief

  @doc """
  Converts a SemanticFact to beliefs and adds them to BeliefStore.

  Attempts SRL-based triple extraction first. Falls back to the legacy
  single-belief path if SRL produces no frames.

  `world_id` is the world the semantic fact and its evidence episodes belong to;
  its triple scorer scores the extracted triples and its relation coverage
  normalizes their predicates, and the beliefs are held in it.

  Returns `{:ok, belief_ids}` or `{:error, reason}`. A missing world, or a world
  with no copy of the triple scorer, raises.
  """
  def bridge_semantic_fact(_semantic_fact, world_id)
      when not (is_binary(world_id) and world_id != "") do
    raise ArgumentError,
          "ConsolidationBridge.bridge_semantic_fact needs the world the semantic fact belongs to, got: #{inspect(world_id)}"
  end

  def bridge_semantic_fact(semantic_fact, world_id) do
    # Loaded outside the bridge's rescue, so a world with no copy of the scorer
    # raises rather than reading as a failed bridge.
    if kg_signals_enabled?(), do: TripleScorer.loaded!(world_id)

    do_bridge_semantic_fact(semantic_fact, world_id)
  end

  defp do_bridge_semantic_fact(semantic_fact, world_id) do
    if BeliefStore.ready?() do
      evidence_count = length(semantic_fact.evidence_ids || [])
      count_prior = evidence_count / (evidence_count + 2)
      provenance_base = build_provenance(semantic_fact)

      if kg_signals_enabled?() do
        triples = extract_triples(semantic_fact, world_id)

        if triples == [] do
          build_legacy_belief(semantic_fact, count_prior, provenance_base, world_id)
        else
          build_triple_beliefs(triples, count_prior, provenance_base, semantic_fact, world_id)
        end
      else
        build_legacy_belief(semantic_fact, count_prior, provenance_base, world_id)
      end
    else
      {:error, :belief_store_not_ready}
    end
  rescue
    e ->
      Logger.debug("ConsolidationBridge failed: #{Exception.message(e)}")
      {:error, :bridge_failed}
  end

  @doc """
  Extract `(subject, predicate, object)` triples from a SemanticFact.

  First tries SRL on constituent episode utterances (fetched via evidence_ids),
  then falls back to SRL on the raw representation text. The episodes are read
  from `world_id`.
  """
  def extract_triples(semantic_fact, world_id) do
    episode_triples = extract_from_episodes(semantic_fact.evidence_ids || [], world_id)

    if episode_triples != [] do
      episode_triples
    else
      extract_from_representation(semantic_fact.representation)
    end
  end

  defp extract_from_episodes(evidence_ids, world_id) do
    evidence_ids
    |> Enum.flat_map(fn id ->
      case Brain.Memory.Store.get_episode(id, world_id: world_id) do
        {:ok, episode} ->
          text = episode_text(episode)
          if text != "", do: run_srl(text), else: []

        _ ->
          []
      end
    end)
    |> Enum.uniq()
  end

  defp extract_from_representation(nil), do: []
  defp extract_from_representation(""), do: []

  defp extract_from_representation(representation) do
    run_srl(representation)
  end

  defp run_srl(text) do
    tokens = Brain.ML.Tokenizer.tokenize_words(text)

    if length(tokens) >= 2 do
      bio_tags = generate_bio_tags(tokens)
      frames = Brain.Analysis.SemanticRoleLabeler.label(tokens, bio_tags)
      Brain.Analysis.SemanticRoleLabeler.to_triples(frames)
    else
      []
    end
  rescue
    _ -> []
  end

  defp generate_bio_tags(tokens) do
    if Brain.ML.POSTagger.model_exists?() do
      case Brain.ML.POSTagger.load_model() do
        {:ok, model} ->
          tags = Brain.ML.POSTagger.predict(tokens, model)
          if is_list(tags) and length(tags) == length(tokens), do: pos_to_bio(tags), else: fallback_bio(tokens)

        _ ->
          fallback_bio(tokens)
      end
    else
      fallback_bio(tokens)
    end
  end

  defp pos_to_bio(pos_tags) do
    pos_tags
    |> Enum.with_index()
    |> Enum.map(fn {tag, idx} ->
      cond do
        tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] -> "B-V"
        idx == 0 -> "B-ARG0"
        tag in ["NN", "NNS", "NNP", "NNPS", "PRP"] -> "B-ARG1"
        true -> "O"
      end
    end)
  end

  defp fallback_bio(tokens) do
    Enum.with_index(tokens)
    |> Enum.map(fn {_token, idx} ->
      if idx == 0, do: "B-ARG0", else: "O"
    end)
  end

  defp build_triple_beliefs(triples, count_prior, provenance_base, semantic_fact, world_id) do
    blend_weight = consolidation_blend_weight()
    model_version = TripleScorer.current_model_version(world_id)
    coverage = TripleScorer.relation_coverage(world_id)

    belief_ids =
      Enum.flat_map(triples, fn {subject, predicate, object} ->
        {confidence, score_provenance} =
          compute_blended_confidence(
            {subject, predicate, object},
            count_prior,
            blend_weight,
            coverage,
            world_id
          )

        provenance =
          provenance_base ++
            score_provenance ++
            if(model_version, do: ["kg_score_model:#{model_version}"], else: [])

        belief =
          Belief.new(
            subject,
            normalize_predicate_for_belief(predicate, coverage),
            object,
            source: :consolidated,
            confidence: confidence,
            provenance: provenance
          )

        case BeliefStore.add_belief(world_id, belief) do
          {:ok, belief_id} ->
            Logger.debug("ConsolidationBridge: created triple belief",
              semantic_fact_id: semantic_fact.id,
              belief_id: belief_id,
              subject: subject,
              predicate: predicate,
              object: object,
              confidence: Float.round(confidence, 3)
            )

            [belief_id]

          _ ->
            []
        end
      end)

    if belief_ids == [] do
      build_legacy_belief(semantic_fact, count_prior, provenance_base, world_id)
    else
      {:ok, belief_ids}
    end
  end

  defp compute_blended_confidence({subject, predicate, object}, count_prior, blend_weight, coverage, world_id) do
    case PredicateNormalizer.normalize(predicate, coverage) do
      {:ok, canonical, _kind} ->
        {:ok, structural_score} =
          TripleScorer.score(world_id, to_string(subject), canonical, to_string(object))

        blended = count_prior * blend_weight + structural_score * (1 - blend_weight)
        {blended, ["kg_score:#{Float.round(structural_score, 3)}"]}

      {:error, :oov} ->
        {count_prior, []}

      {:error, _} ->
        {count_prior, []}
    end
  end

  defp build_legacy_belief(semantic_fact, count_prior, provenance_base, world_id) do
    belief =
      Belief.new(
        :system,
        :consolidated_knowledge,
        semantic_fact.representation,
        source: :consolidated,
        confidence: count_prior,
        provenance: provenance_base ++ ["semantic_fact:#{semantic_fact.id}"]
      )

    case BeliefStore.add_belief(world_id, belief) do
      {:ok, belief_id} ->
        Logger.debug("ConsolidationBridge: created legacy belief from semantic fact",
          semantic_fact_id: semantic_fact.id,
          belief_id: belief_id,
          confidence: Float.round(count_prior, 3)
        )

        {:ok, [belief_id]}

      error ->
        error
    end
  end

  defp build_provenance(semantic_fact) do
    episode_provenance =
      (semantic_fact.evidence_ids || [])
      |> Enum.map(fn id -> "episode:#{id}" end)

    episode_provenance ++ ["semantic_fact:#{semantic_fact.id}"]
  end

  defp episode_text(episode) do
    cond do
      is_binary(Map.get(episode, :state)) -> episode.state
      is_binary(Map.get(episode, :action)) -> episode.action
      true -> ""
    end
  end

  defp normalize_predicate_for_belief(predicate, coverage) when is_binary(predicate) do
    case PredicateNormalizer.normalize(predicate, coverage) do
      {:ok, canonical, _} -> String.to_atom(String.downcase(canonical))
      _ -> String.to_atom(String.downcase(predicate))
    end
  rescue
    _ -> :consolidated_knowledge
  end

  defp normalize_predicate_for_belief(predicate, _coverage) when is_atom(predicate), do: predicate
  defp normalize_predicate_for_belief(_, _coverage), do: :consolidated_knowledge

  defp kg_signals_enabled? do
    config = Application.get_env(:brain, :kg_signals, [])
    Keyword.get(config, :enabled, true)
  end

  defp consolidation_blend_weight do
    config = Application.get_env(:brain, :kg_signals, [])
    Keyword.get(config, :consolidation_blend, 0.6)
  end
end
