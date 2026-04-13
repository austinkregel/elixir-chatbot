defmodule Brain.Knowledge.Corroborator do
  @moduledoc "Analyzes findings for cross-source agreement and conflict detection.\n\nThe Corroborator implements the **evidence evaluation** phase of the\nscientific method:\n\n## Scientific Method Integration\n\n1. **Hypothesis Testing**: Evaluates hypotheses against gathered evidence\n2. **Falsifiability**: Contradicting evidence can falsify hypotheses\n3. **Support**: Agreeing evidence supports (but doesn't prove) hypotheses\n4. **Independent Verification**: Requires 2+ independent sources\n\nKey principle: \"We cannot prove a hypothesis true, only support it with\nevidence or falsify it with contradicting evidence.\"\n\n## Features\n\n- Groups findings by semantic similarity using TF-IDF embeddings\n- Requires 2+ independent sources for high-confidence facts\n- Detects conflicting claims between sources\n- Computes aggregate confidence scores based on corroboration\n- Evaluates hypotheses and determines if they are supported/falsified\n\n## Example\n\n    findings = [\n      %Finding{claim: \"Paris is the capital of France\", source: %{domain: \"source1.com\"}},\n      %Finding{claim: \"France's capital is Paris\", source: %{domain: \"source2.com\"}}\n    ]\n\n    {:ok, candidates} = Corroborator.corroborate(findings)\n    # => Single ReviewCandidate with 2 corroborating sources\n\n    # Or with hypothesis testing:\n    {:ok, investigation} = Corroborator.test_hypotheses(investigation, findings)\n"

  alias Brain.Knowledge.Types
  alias Brain.ML.Tokenizer
  require Logger

  alias Brain.Memory.Embedder
  alias Types.{Finding, ReviewCandidate, Hypothesis, Investigation}
  alias Brain.Telemetry
  @similarity_threshold 0.7
  @min_sources 2
  @max_cluster_size 20

  @doc "Groups findings by semantic similarity and calculates corroboration.\n\nReturns ReviewCandidates with corroboration metadata. Only findings\nthat meet the minimum source threshold are returned.\n\n## Options\n  - :similarity_threshold - Override default similarity threshold\n  - :min_sources - Override minimum source requirement\n  - :include_uncorroborated - If true, include findings below threshold\n"
  @spec corroborate([Finding.t()], keyword()) :: {:ok, [ReviewCandidate.t()]}
  def corroborate(findings, opts \\ []) when is_list(findings) do
    Telemetry.span(:knowledge_corroborate, %{findings_count: length(findings)}, fn ->
      do_corroborate(findings, opts)
    end)
  end

  defp do_corroborate(findings, opts) do
    similarity_threshold = Keyword.get(opts, :similarity_threshold, @similarity_threshold)
    min_sources = Keyword.get(opts, :min_sources, @min_sources)
    include_uncorroborated = Keyword.get(opts, :include_uncorroborated, false)

    Logger.debug("Starting corroboration", findings_count: length(findings))

    if findings == [] do
      {:ok, []}
    else
      embedded = embed_findings(findings)
      clusters = cluster_by_similarity(embedded, similarity_threshold)

      candidates =
        clusters
        |> Enum.map(&build_candidate_from_cluster/1)
        |> maybe_filter_by_threshold(min_sources, include_uncorroborated)

      Logger.info("Corroboration completed",
        findings: length(findings),
        clusters: length(clusters),
        candidates: length(candidates)
      )

      {:ok, candidates}
    end
  end

  @doc "Checks if two claims are semantically similar.\n\nReturns {:ok, similarity_score} where score is 0.0-1.0.\n"
  @spec compare_claims(String.t(), String.t()) :: {:ok, float()} | {:error, term()}
  def compare_claims(claim1, claim2) when is_binary(claim1) and is_binary(claim2) do
    if Embedder.ready?() do
      with {:ok, emb1} <- Embedder.embed(claim1),
           {:ok, emb2} <- Embedder.embed(claim2) do
        similarity = cosine_similarity(emb1, emb2)
        {:ok, similarity}
      end
    else
      {:ok, token_overlap_similarity(claim1, claim2)}
    end
  end

  @doc "Detects conflicts between a new claim and existing claims.\n\nReturns a list of claims that conflict with the given claim.\n"
  @spec find_conflicts(Finding.t(), [Finding.t()]) :: [Finding.t()]
  def find_conflicts(%Finding{} = finding, existing_findings) do
    existing_findings
    |> Enum.filter(fn existing ->
      same_entity?(finding, existing) and contradicts?(finding.claim, existing.claim)
    end)
  end

  @doc "Tests hypotheses in an investigation against gathered evidence.\n\nThis is the core of the scientific method implementation:\n1. For each hypothesis, find relevant evidence\n2. Classify evidence as supporting or contradicting\n3. Apply falsifiability rules\n4. Update hypothesis status\n\n## Falsifiability Rules\n\nA hypothesis is **falsified** if:\n- Reliable contradicting evidence exists (reliability >= 0.6)\n- The contradicting source is independent\n\nA hypothesis is **supported** if:\n- 2+ independent sources provide agreeing evidence\n- No reliable contradicting evidence exists\n\n## Returns\n\nUpdated investigation with evaluated hypotheses.\n"
  @spec test_hypotheses(Investigation.t(), [Finding.t()]) :: {:ok, Investigation.t()}
  def test_hypotheses(%Investigation{} = investigation, findings) when is_list(findings) do
    Telemetry.span(
      :knowledge_corroborate,
      %{
        hypothesis_count: length(investigation.hypotheses),
        evidence_count: length(findings)
      },
      fn ->
        do_test_hypotheses(investigation, findings)
      end
    )
  end

  defp do_test_hypotheses(%Investigation{} = investigation, findings) do
    Logger.debug("Testing hypotheses",
      hypotheses: length(investigation.hypotheses),
      findings: length(findings)
    )

    investigation = Investigation.record_evidence(investigation, findings)
    concluded = Investigation.conclude(investigation)

    Logger.info("Hypothesis testing completed",
      supported: Enum.count(concluded.hypotheses, &(&1.status == :supported)),
      falsified: Enum.count(concluded.hypotheses, &(&1.status == :falsified)),
      inconclusive: Enum.count(concluded.hypotheses, &(&1.status == :inconclusive))
    )

    {:ok, concluded}
  end

  @doc "Evaluates a single hypothesis against a set of findings.\n\nReturns the hypothesis with updated status and evidence.\n"
  @spec evaluate_hypothesis(Hypothesis.t(), [Finding.t()]) :: Hypothesis.t()
  def evaluate_hypothesis(%Hypothesis{} = hypothesis, findings) when is_list(findings) do
    {supporting, contradicting} = partition_evidence(hypothesis, findings)

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

  @doc "Determines if evidence supports or contradicts a hypothesis.\n\nUses semantic similarity and negation detection.\n"
  @spec classify_evidence(Hypothesis.t(), Finding.t()) ::
          :supporting | :contradicting | :irrelevant
  def classify_evidence(%Hypothesis{} = hypothesis, %Finding{} = finding) do
    case compare_claims(hypothesis.claim, finding.claim) do
      {:ok, similarity} when similarity >= @similarity_threshold ->
        if contradicts?(hypothesis.claim, finding.claim) do
          :contradicting
        else
          :supporting
        end

      {:ok, _low_similarity} ->
        :irrelevant

      {:error, _} ->
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

  defp partition_evidence(%Hypothesis{} = hypothesis, findings) do
    findings
    |> Enum.reduce({[], []}, fn finding, {supporting, contradicting} ->
      case classify_evidence(hypothesis, finding) do
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

  defp embed_findings(findings) do
    findings
    |> Enum.map(fn finding ->
      embedding = get_embedding(finding.claim)
      {finding, embedding}
    end)
  end

  defp get_embedding(text) do
    if Embedder.ready?() do
      case Embedder.embed(text) do
        {:ok, embedding} -> embedding
        _ -> compute_simple_embedding(text)
      end
    else
      compute_simple_embedding(text)
    end
  end

  defp compute_simple_embedding(text) do
    words = tokenize(text)
    word_counts = Enum.frequencies(words)
    total = Enum.sum(Map.values(word_counts))

    if total > 0 do
      word_counts
      |> Enum.map(fn {word, count} -> {word, count / total} end)
      |> Map.new()
    else
      %{}
    end
  end

  defp cluster_by_similarity(embedded_findings, threshold) do
    embedded_findings
    |> Enum.reduce([], fn {finding, embedding}, clusters ->
      case find_matching_cluster(clusters, embedding, threshold) do
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

  defp find_matching_cluster(clusters, embedding, threshold) do
    clusters
    |> Enum.with_index()
    |> Enum.find_value(:not_found, fn {cluster, idx} ->
      {_primary_finding, primary_embedding} = cluster.primary
      similarity = compute_similarity(embedding, primary_embedding)

      if similarity >= threshold do
        {:found, idx}
      else
        nil
      end
    end)
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

  defp compute_similarity(emb1, emb2) when is_list(emb1) and is_list(emb2) do
    cosine_similarity(emb1, emb2)
  end

  defp compute_similarity(emb1, emb2) when is_map(emb1) and is_map(emb2) do
    map_similarity(emb1, emb2)
  end

  defp compute_similarity(_, _) do
    0.0
  end

  defp cosine_similarity(v1, v2) when is_list(v1) and is_list(v2) do
    if length(v1) == length(v2) and v1 != [] do
      dot = Enum.zip(v1, v2) |> Enum.map(fn {a, b} -> a * b end) |> Enum.sum()
      norm1 = :math.sqrt(Enum.map(v1, &(&1 * &1)) |> Enum.sum())
      norm2 = :math.sqrt(Enum.map(v2, &(&1 * &1)) |> Enum.sum())

      if norm1 > 0 and norm2 > 0 do
        dot / (norm1 * norm2)
      else
        0.0
      end
    else
      0.0
    end
  end

  defp map_similarity(map1, map2) do
    keys1 = MapSet.new(Map.keys(map1))
    keys2 = MapSet.new(Map.keys(map2))

    intersection = MapSet.intersection(keys1, keys2) |> MapSet.size()
    union = MapSet.union(keys1, keys2) |> MapSet.size()

    if union > 0 do
      intersection / union
    else
      0.0
    end
  end

  defp token_overlap_similarity(text1, text2) do
    words1 = tokenize(text1) |> MapSet.new()
    words2 = tokenize(text2) |> MapSet.new()

    intersection = MapSet.intersection(words1, words2) |> MapSet.size()
    union = MapSet.union(words1, words2) |> MapSet.size()

    if union > 0 do
      intersection / union
    else
      0.0
    end
  end

  defp tokenize(text) do
    text
    |> Tokenizer.tokenize_normalized(min_length: 3)
  end

  defp build_candidate_from_cluster({primary, supporting}) do
    all_findings = [primary | supporting]
    source_count = count_unique_domains(all_findings)
    avg_reliability = average_source_reliability(all_findings)
    {corroborating, conflicting} = partition_by_conflict(primary, supporting)

    aggregate = calculate_aggregate_confidence(source_count, avg_reliability, length(conflicting))
    aggregate = apply_kg_triple_scoring(primary, aggregate)

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

  defp apply_kg_triple_scoring(finding, aggregate) do
    if Brain.ML.KnowledgeGraph.TripleScorer.ready?() do
      case Brain.ML.KnowledgeGraph.TripleScorer.score(finding.entity, "claims", finding.claim) do
        {:ok, score} ->
          adjusted = aggregate * 0.7 + score * 0.3
          min(max(adjusted, 0.0), 1.0)

        {:error, _reason} ->
          aggregate
      end
    else
      log_once(:kg_lstm_not_ready, "KG-LSTM TripleScorer not ready; skipping graph-aware scoring")

      :telemetry.execute(
        [:chat_bot, :ml, :kg_lstm, :unavailable],
        %{count: 1},
        %{stage: :corroboration}
      )

      aggregate
    end
  rescue
    e ->
      Logger.warning("KG-LSTM scoring failed: #{Exception.message(e)}")
      aggregate
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

  defp log_once(key, message) do
    pt_key = {__MODULE__, :logged, key}
    unless :persistent_term.get(pt_key, false) do
      Logger.info(message)
      :persistent_term.put(pt_key, true)
    end
  end
end
