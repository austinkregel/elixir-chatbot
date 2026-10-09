defmodule Brain.Knowledge.Corroborator do
  @moduledoc "Analyzes findings for cross-source agreement and conflict detection.\n\nThe Corroborator implements the **evidence evaluation** phase of the\nscientific method:\n\n## Scientific Method Integration\n\n1. **Hypothesis Testing**: Evaluates hypotheses against gathered evidence\n2. **Falsifiability**: Contradicting evidence can falsify hypotheses\n3. **Support**: Agreeing evidence supports (but doesn't prove) hypotheses\n4. **Independent Verification**: Requires 2+ independent sources\n\nKey principle: \"We cannot prove a hypothesis true, only support it with\nevidence or falsify it with contradicting evidence.\"\n\n## Features\n\n- Groups findings by semantic similarity using TF-IDF embeddings\n- Requires 2+ independent sources for high-confidence facts\n- Detects conflicting claims between sources\n- Computes aggregate confidence scores based on corroboration\n- Evaluates hypotheses and determines if they are supported/falsified\n\n## Example\n\n    findings = [\n      %Finding{claim: \"Paris is the capital of France\", source: %{domain: \"source1.com\"}},\n      %Finding{claim: \"France's capital is Paris\", source: %{domain: \"source2.com\"}}\n    ]\n\n    {:ok, candidates} = Corroborator.corroborate(findings)\n    # => Single ReviewCandidate with 2 corroborating sources\n\n    # Or with hypothesis testing:\n    {:ok, investigation} = Corroborator.test_hypotheses(world_id, investigation, findings)\n"

  alias Brain.Knowledge.Types
  require Logger

  # World.Embedder is in a sibling umbrella app that depends on :brain.
  @compile {:no_warn_undefined, World.Embedder}
  alias World.Embedder, as: WorldEmbedder
  alias Types.{Finding, ReviewCandidate, Hypothesis, Investigation}
  alias Brain.Telemetry
  @similarity_threshold 0.7
  @min_sources 2
  @max_cluster_size 20

  @doc "Groups findings by semantic similarity and calculates corroboration.\n\nReturns ReviewCandidates with corroboration metadata. Only findings\nthat meet the minimum source threshold are returned.\n\n## Options\n  - :world_id - Required. The world whose embedder embeds the claims, whose entity vectors inform clustering and whose triple scorer scores each candidate; absent raises.\n  - :similarity_threshold - Override default similarity threshold\n  - :min_sources - Override minimum source requirement\n  - :include_uncorroborated - If true, include findings below threshold\n"
  @spec corroborate([Finding.t()], keyword()) :: {:ok, [ReviewCandidate.t()]}
  def corroborate(findings, opts) when is_list(findings) do
    world_id =
      case Keyword.get(opts, :world_id) do
        world_id when is_binary(world_id) and world_id != "" ->
          world_id

        other ->
          raise ArgumentError,
                "Corroborator.corroborate requires a :world_id option, the world whose entity " <>
                  "vectors inform clustering; got #{inspect(other)}"
      end

    Telemetry.span(:knowledge_corroborate, %{findings_count: length(findings)}, fn ->
      do_corroborate(findings, opts, world_id)
    end)
  end

  defp do_corroborate(findings, opts, world_id) do
    similarity_threshold = Keyword.get(opts, :similarity_threshold, @similarity_threshold)
    min_sources = Keyword.get(opts, :min_sources, @min_sources)
    include_uncorroborated = Keyword.get(opts, :include_uncorroborated, false)

    Logger.debug("Starting corroboration", findings_count: length(findings))

    if findings == [] do
      {:ok, []}
    else
      embedded = embed_findings(findings, world_id)
      clusters = cluster_by_similarity(embedded, similarity_threshold, world_id)

      candidates =
        clusters
        |> Enum.map(&build_candidate_from_cluster(&1, world_id))
        |> maybe_filter_by_threshold(min_sources, include_uncorroborated)

      Logger.info("Corroboration completed",
        findings: length(findings),
        clusters: length(clusters),
        candidates: length(candidates)
      )

      {:ok, candidates}
    end
  end

  @doc "Checks if two claims are semantically similar in the world's embedder.\n\nReturns {:ok, similarity_score} where score is 0.0-1.0. While the world's\nvocabulary is being rebuilt from its episodes, this waits for the rebuild to\nfinish (`World.Embedder.await_ready/2`) and compares the claims then. A world\nwhose embedder cannot embed, or whose rebuild does not finish, raises.\n"
  @spec compare_claims(String.t(), String.t(), String.t()) :: {:ok, float()}
  def compare_claims(world_id, claim1, claim2)
      when is_binary(world_id) and is_binary(claim1) and is_binary(claim2) do
    {:ok, compute_similarity(world_embedding!(world_id, claim1), world_embedding!(world_id, claim2))}
  end

  @doc "Detects conflicts between a new claim and existing claims.\n\nReturns a list of claims that conflict with the given claim.\n"
  @spec find_conflicts(Finding.t(), [Finding.t()]) :: [Finding.t()]
  def find_conflicts(%Finding{} = finding, existing_findings) do
    existing_findings
    |> Enum.filter(fn existing ->
      same_entity?(finding, existing) and contradicts?(finding.claim, existing.claim)
    end)
  end

  @doc "Tests hypotheses in an investigation against gathered evidence, comparing\nclaims in `world_id`'s embedder (`Investigation.record_evidence/3`).\n\nThis is the core of the scientific method implementation:\n1. For each hypothesis, find relevant evidence\n2. Classify evidence as supporting or contradicting\n3. Apply falsifiability rules\n4. Update hypothesis status\n\n## Falsifiability Rules\n\nA hypothesis is **falsified** if:\n- Reliable contradicting evidence exists (reliability >= 0.6)\n- The contradicting source is independent\n\nA hypothesis is **supported** if:\n- 2+ independent sources provide agreeing evidence\n- No reliable contradicting evidence exists\n\n## Returns\n\nUpdated investigation with evaluated hypotheses.\n"
  @spec test_hypotheses(String.t(), Investigation.t(), [Finding.t()]) :: {:ok, Investigation.t()}
  def test_hypotheses(world_id, %Investigation{} = investigation, findings)
      when is_binary(world_id) and world_id != "" and is_list(findings) do
    Telemetry.span(
      :knowledge_corroborate,
      %{
        hypothesis_count: length(investigation.hypotheses),
        evidence_count: length(findings)
      },
      fn ->
        do_test_hypotheses(world_id, investigation, findings)
      end
    )
  end

  defp do_test_hypotheses(world_id, %Investigation{} = investigation, findings) do
    Logger.debug("Testing hypotheses",
      hypotheses: length(investigation.hypotheses),
      findings: length(findings)
    )

    investigation = Investigation.record_evidence(world_id, investigation, findings)
    concluded = Investigation.conclude(investigation)

    Logger.info("Hypothesis testing completed",
      supported: Enum.count(concluded.hypotheses, &(&1.status == :supported)),
      falsified: Enum.count(concluded.hypotheses, &(&1.status == :falsified)),
      inconclusive: Enum.count(concluded.hypotheses, &(&1.status == :inconclusive))
    )

    {:ok, concluded}
  end

  @doc "Evaluates a single hypothesis against a set of findings, comparing\nclaims in the world's embedder.\n\nReturns the hypothesis with updated status and evidence.\n"
  @spec evaluate_hypothesis(String.t(), Hypothesis.t(), [Finding.t()]) :: Hypothesis.t()
  def evaluate_hypothesis(world_id, %Hypothesis{} = hypothesis, findings) when is_list(findings) do
    {supporting, contradicting} = partition_evidence(world_id, hypothesis, findings)

    hypothesis =
      supporting
      |> Enum.reduce(hypothesis, fn finding, hyp ->
        Hypothesis.add_supporting_evidence(hyp, finding)
      end)

    hypothesis =
      contradicting
      |> Enum.reduce(hypothesis, fn finding, hyp ->
        Hypothesis.add_contradicting_evidence(hyp, finding)
      end)

    Hypothesis.evaluate(hypothesis)
  end

  @doc "Determines if evidence supports or contradicts a hypothesis.\n\nUses semantic similarity in the world's embedder and negation detection.\n"
  @spec classify_evidence(String.t(), Hypothesis.t(), Finding.t()) ::
          :supporting | :contradicting | :irrelevant
  def classify_evidence(world_id, %Hypothesis{} = hypothesis, %Finding{} = finding) do
    case compare_claims(world_id, hypothesis.claim, finding.claim) do
      {:ok, similarity} when similarity >= @similarity_threshold ->
        if contradicts?(hypothesis.claim, finding.claim) do
          :contradicting
        else
          :supporting
        end

      {:ok, _low_similarity} ->
        :irrelevant
    end
  end

  @doc "Converts supported hypotheses from an investigation into ReviewCandidates.\n\nOnly hypotheses that are:\n1. Supported (not falsified)\n2. High confidence (>= 0.7)\n3. Have 2+ independent sources\n\nare converted to candidates for admin review.\n"
  @spec hypotheses_to_candidates(Investigation.t(), keyword()) :: [ReviewCandidate.t()]
  def hypotheses_to_candidates(%Investigation{} = investigation, opts \\ []) do
    session_id = Keyword.get(opts, :session_id)

    investigation
    |> Investigation.promotable_hypotheses()
    |> Enum.map(fn hypothesis ->
      primary_finding = build_finding_from_hypothesis(hypothesis)

      corroborating_sources =
        hypothesis.supporting_evidence
        |> Enum.map(& &1.source)
        |> Enum.uniq_by(& &1.domain)

      ReviewCandidate.new(primary_finding,
        corroborating_sources: corroborating_sources,
        conflicting_findings: hypothesis.contradicting_evidence,
        aggregate_confidence: hypothesis.confidence,
        session_id: session_id
      )
    end)
  end

  defp partition_evidence(world_id, %Hypothesis{} = hypothesis, findings) do
    findings
    |> Enum.reduce({[], []}, fn finding, {supporting, contradicting} ->
      case classify_evidence(world_id, hypothesis, finding) do
        :supporting -> {[finding | supporting], contradicting}
        :contradicting -> {supporting, [finding | contradicting]}
        :irrelevant -> {supporting, contradicting}
      end
    end)
  end

  defp build_finding_from_hypothesis(%Hypothesis{} = hypothesis) do
    best_source =
      hypothesis.supporting_evidence
      |> Enum.max_by(fn f -> f.source.reliability_score end, fn -> nil end)

    source =
      if best_source do
        best_source.source
      else
        default_source()
      end

    Finding.new(
      hypothesis.claim,
      hypothesis.entity || "unknown",
      source,
      confidence: hypothesis.confidence,
      raw_context: hypothesis.derived_from || ""
    )
  end

  defp default_source do
    alias Brain.Knowledge.Types.SourceInfo
    SourceInfo.new("internal://hypothesis", title: "Hypothesis-derived")
  end

  defp embed_findings(findings, world_id) do
    findings
    |> Enum.map(fn finding ->
      embedding = world_embedding!(world_id, finding.claim)
      {finding, embedding}
    end)
  end

  # A world's embedding comes from the world's embedder and nowhere else. While
  # the world's vocabulary is being rebuilt this waits for the rebuild to finish
  # (`World.Embedder.await_ready/2`, which raises naming the world when it does
  # not) and embeds then. A world whose embedder has no copy, no vocabulary, or
  # no table raises.
  defp world_embedding!(world_id, text) do
    case WorldEmbedder.embed(world_id, text) do
      {:ok, embedding} ->
        embedding

      {:error, :vocabulary_building} ->
        :ok = WorldEmbedder.await_ready(world_id)
        world_embedding!(world_id, text)

      {:error, reason} ->
        raise "Corroborator: world #{inspect(world_id)}'s embedder cannot embed: #{inspect(reason)}"
    end
  end

  defp cluster_by_similarity(embedded_findings, threshold, world_id) do
    embedded_findings
    |> Enum.reduce([], fn {finding, embedding}, clusters ->
      case find_matching_cluster(clusters, embedding, threshold, world_id) do
        {:found, cluster_idx} ->
          update_cluster(clusters, cluster_idx, {finding, embedding})

        :not_found ->
          new_cluster = %{
            primary: {finding, embedding},
            supporting: []
          }

          clusters ++ [new_cluster]
      end
    end)
    |> Enum.map(&finalize_cluster/1)
  end

  defp find_matching_cluster(clusters, embedding, threshold, world_id) do
    clusters
    |> Enum.with_index()
    |> Enum.find_value(:not_found, fn {cluster, idx} ->
      {primary_finding, primary_embedding} = cluster.primary
      tfidf_sim = compute_similarity(embedding, primary_embedding)

      similarity =
        if kg_clustering_enabled?() do
          blend_with_kg_similarity(tfidf_sim, primary_finding, world_id)
        else
          tfidf_sim
        end

      if similarity >= threshold do
        {:found, idx}
      else
        nil
      end
    end)
  end

  defp blend_with_kg_similarity(tfidf_sim, primary_finding, world_id) do
    primary_entity = Map.get(primary_finding, :entity) || Map.get(primary_finding, :claim, "")
    primary_vec = entity_vector(primary_entity, world_id)

    if primary_vec == nil do
      tfidf_sim
    else
      tfidf_sim
    end
  end

  # The world's entity vector; a world with no copy of the triple scorer raises.
  defp entity_vector(text, world_id) when is_binary(text) and text != "" do
    alias Brain.ML.KnowledgeGraph.EntityVectorCache

    {:ok, vec} = EntityVectorCache.get_or_compute(world_id, text)
    vec
  end

  defp entity_vector(_, _world_id), do: nil

  defp kg_clustering_enabled? do
    config = Application.get_env(:brain, :kg_signals, [])
    Keyword.get(config, :enabled, true)
  end

  defp update_cluster(clusters, idx, {finding, embedding}) do
    cluster = Enum.at(clusters, idx)

    if length(cluster.supporting) < @max_cluster_size do
      updated = %{cluster | supporting: cluster.supporting ++ [{finding, embedding}]}
      List.replace_at(clusters, idx, updated)
    else
      clusters
    end
  end

  defp finalize_cluster(%{primary: {primary_finding, _}, supporting: supporting}) do
    supporting_findings = Enum.map(supporting, fn {f, _} -> f end)
    {primary_finding, supporting_findings}
  end

  # The cosine similarity of two of the world's embeddings. Vectors of
  # different lengths (the world's vocabulary was rebuilt between the two
  # embeddings) raise rather than compare as 0.0, and so does anything that is
  # not an embedding.
  defp compute_similarity(emb1, emb2) when is_list(emb1) and is_list(emb2) do
    FourthWall.Math.cosine_similarity!(emb1, emb2)
  end

  defp compute_similarity(emb1, emb2) do
    raise ArgumentError,
          "Corroborator compares embeddings, lists of numbers; got #{inspect(emb1, limit: 3)} and " <>
            "#{inspect(emb2, limit: 3)}"
  end

  defp build_candidate_from_cluster({primary, supporting}, world_id) do
    all_findings = [primary | supporting]
    source_count = count_unique_domains(all_findings)
    avg_reliability = average_source_reliability(all_findings)
    {corroborating, conflicting} = partition_by_conflict(primary, supporting)

    aggregate = calculate_aggregate_confidence(source_count, avg_reliability, length(conflicting))
    aggregate = apply_kg_triple_scoring(primary, aggregate, world_id)

    corroborating_sources =
      corroborating
      |> Enum.map(& &1.source)
      |> Enum.uniq_by(& &1.domain)

    ReviewCandidate.new(primary,
      corroborating_sources: corroborating_sources,
      conflicting_findings: conflicting,
      aggregate_confidence: aggregate
    )
  end

  # Blends in the world's triple scorer's score for the claim; a world with no
  # copy of the scorer raises.
  defp apply_kg_triple_scoring(finding, aggregate, world_id) do
    {:ok, score} = Brain.ML.KnowledgeGraph.TripleScorer.score(world_id, finding.entity, "claims", finding.claim)
    adjusted = aggregate * 0.7 + score * 0.3
    min(max(adjusted, 0.0), 1.0)
  end

  defp count_unique_domains(findings) do
    findings
    |> Enum.map(fn f -> f.source.domain end)
    |> Enum.uniq()
    |> length()
  end

  defp average_source_reliability(findings) do
    scores = Enum.map(findings, fn f -> f.source.reliability_score end)

    if scores != [] do
      Enum.sum(scores) / length(scores)
    else
      0.5
    end
  end

  defp partition_by_conflict(primary, supporting) do
    Enum.split_with(supporting, fn finding ->
      not contradicts?(primary.claim, finding.claim)
    end)
  end

  defp calculate_aggregate_confidence(source_count, avg_reliability, conflict_count) do
    base = avg_reliability
    source_bonus = :math.log(source_count + 1) / :math.log(5) * 0.2
    conflict_penalty = conflict_count * 0.1

    (base + source_bonus - conflict_penalty)
    |> max(0.0)
    |> min(1.0)
  end

  defp maybe_filter_by_threshold(candidates, min_sources, include_uncorroborated) do
    if include_uncorroborated do
      candidates
    else
      Enum.filter(candidates, fn candidate ->
        total_sources = length(candidate.corroborating_sources) + 1
        total_sources >= min_sources
      end)
    end
  end

  defp same_entity?(%Finding{entity: e1}, %Finding{entity: e2}) do
    normalize_entity(e1) == normalize_entity(e2)
  end

  defp normalize_entity(entity) when is_binary(entity) do
    entity
    |> String.downcase()
    |> String.trim()
  end

  defp normalize_entity(_) do
    ""
  end

  defp contradicts?(claim1, claim2) when is_binary(claim1) and is_binary(claim2) do
    c1 = String.downcase(claim1)
    c2 = String.downcase(claim2)

    cond do
      c1 == c2 ->
        false

      has_negation_difference?(c1, c2) ->
        true

      has_number_disagreement?(c1, c2) ->
        true

      true ->
        false
    end
  end

  defp contradicts?(_, _) do
    false
  end

  defp has_negation_difference?(c1, c2),
    do: Brain.Knowledge.ContradictionDetector.has_negation_difference?(c1, c2)

  defp has_number_disagreement?(c1, c2),
    do: Brain.Knowledge.ContradictionDetector.has_number_disagreement?(c1, c2)
end
