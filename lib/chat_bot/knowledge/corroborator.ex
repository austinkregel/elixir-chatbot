defmodule ChatBot.Knowledge.Corroborator do
  @moduledoc """
  Analyzes findings for cross-source agreement and conflict detection.

  The Corroborator:
  - Groups findings by semantic similarity using TF-IDF embeddings
  - Requires 2+ independent sources for high-confidence facts
  - Detects conflicting claims between sources
  - Computes aggregate confidence scores based on corroboration

  ## Example

      findings = [
        %Finding{claim: "Paris is the capital of France", source: %{domain: "source1.com"}},
        %Finding{claim: "France's capital is Paris", source: %{domain: "source2.com"}}
      ]

      {:ok, candidates} = Corroborator.corroborate(findings)
      # => Single ReviewCandidate with 2 corroborating sources
  """

  require Logger

  alias ChatBot.Memory.Embedder
  alias ChatBot.Knowledge.Types.{Finding, ReviewCandidate}
  alias ChatBot.Telemetry

  # Cosine similarity threshold for considering claims as "the same"
  @similarity_threshold 0.70

  # Minimum number of sources required for a finding to be considered corroborated
  @min_sources 2

  # Maximum cluster size to prevent runaway clustering
  @max_cluster_size 20

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Groups findings by semantic similarity and calculates corroboration.

  Returns ReviewCandidates with corroboration metadata. Only findings
  that meet the minimum source threshold are returned.

  ## Options
    - :similarity_threshold - Override default similarity threshold
    - :min_sources - Override minimum source requirement
    - :include_uncorroborated - If true, include findings below threshold
  """
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

    if length(findings) == 0 do
      {:ok, []}
    else
      # 1. Embed all claims
      embedded = embed_findings(findings)

      # 2. Cluster by similarity
      clusters = cluster_by_similarity(embedded, similarity_threshold)

      # 3. Build ReviewCandidates with corroboration info
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

  @doc """
  Checks if two claims are semantically similar.

  Returns {:ok, similarity_score} where score is 0.0-1.0.
  """
  @spec compare_claims(String.t(), String.t()) :: {:ok, float()} | {:error, term()}
  def compare_claims(claim1, claim2) when is_binary(claim1) and is_binary(claim2) do
    if Embedder.ready?() do
      with {:ok, emb1} <- Embedder.embed(claim1),
           {:ok, emb2} <- Embedder.embed(claim2) do
        similarity = cosine_similarity(emb1, emb2)
        {:ok, similarity}
      end
    else
      # Fall back to simple token overlap
      {:ok, token_overlap_similarity(claim1, claim2)}
    end
  end

  @doc """
  Detects conflicts between a new claim and existing claims.

  Returns a list of claims that conflict with the given claim.
  """
  @spec find_conflicts(Finding.t(), [Finding.t()]) :: [Finding.t()]
  def find_conflicts(%Finding{} = finding, existing_findings) do
    existing_findings
    |> Enum.filter(fn existing ->
      same_entity?(finding, existing) and contradicts?(finding.claim, existing.claim)
    end)
  end

  # ============================================================================
  # Private Functions - Embedding
  # ============================================================================

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
    # Simple fallback: normalized word frequency vector
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

  # ============================================================================
  # Private Functions - Clustering
  # ============================================================================

  defp cluster_by_similarity(embedded_findings, threshold) do
    # Greedy clustering: assign each finding to the first cluster it matches
    # or create a new cluster

    embedded_findings
    |> Enum.reduce([], fn {finding, embedding}, clusters ->
      case find_matching_cluster(clusters, embedding, threshold) do
        {:found, cluster_idx} ->
          update_cluster(clusters, cluster_idx, {finding, embedding})

        :not_found ->
          # Create new cluster with this finding as the primary
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
    # Map-based embeddings (fallback mode)
    map_similarity(emb1, emb2)
  end

  defp compute_similarity(_, _), do: 0.0

  defp cosine_similarity(v1, v2) when is_list(v1) and is_list(v2) do
    if length(v1) == length(v2) and length(v1) > 0 do
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
    # Jaccard-like similarity for word frequency maps
    keys1 = MapSet.new(Map.keys(map1))
    keys2 = MapSet.new(Map.keys(map2))

    intersection = MapSet.intersection(keys1, keys2) |> MapSet.size()
    union = MapSet.union(keys1, keys2) |> MapSet.size()

    if union > 0, do: intersection / union, else: 0.0
  end

  defp token_overlap_similarity(text1, text2) do
    words1 = tokenize(text1) |> MapSet.new()
    words2 = tokenize(text2) |> MapSet.new()

    intersection = MapSet.intersection(words1, words2) |> MapSet.size()
    union = MapSet.union(words1, words2) |> MapSet.size()

    if union > 0, do: intersection / union, else: 0.0
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.split(~r/[^\w]+/, trim: true)
    |> Enum.filter(&(String.length(&1) > 2))
  end

  # ============================================================================
  # Private Functions - Candidate Building
  # ============================================================================

  defp build_candidate_from_cluster({primary, supporting}) do
    # Calculate aggregate confidence based on:
    # - Number of independent sources
    # - Source reliability scores
    # - Claim consistency

    all_findings = [primary | supporting]
    source_count = count_unique_domains(all_findings)
    avg_reliability = average_source_reliability(all_findings)

    # Detect conflicts within the cluster
    {corroborating, conflicting} = partition_by_conflict(primary, supporting)

    aggregate = calculate_aggregate_confidence(source_count, avg_reliability, length(conflicting))

    # Extract corroborating source info
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

  defp count_unique_domains(findings) do
    findings
    |> Enum.map(fn f -> f.source.domain end)
    |> Enum.uniq()
    |> length()
  end

  defp average_source_reliability(findings) do
    scores = Enum.map(findings, fn f -> f.source.reliability_score end)

    if length(scores) > 0 do
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
    # Base confidence from reliability
    base = avg_reliability

    # Bonus for multiple sources (diminishing returns)
    source_bonus = :math.log(source_count + 1) / :math.log(5) * 0.2

    # Penalty for conflicts
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
        # Primary + corroborating sources
        total_sources = length(candidate.corroborating_sources) + 1
        total_sources >= min_sources
      end)
    end
  end

  # ============================================================================
  # Private Functions - Conflict Detection
  # ============================================================================

  defp same_entity?(%Finding{entity: e1}, %Finding{entity: e2}) do
    normalize_entity(e1) == normalize_entity(e2)
  end

  defp normalize_entity(entity) when is_binary(entity) do
    entity
    |> String.downcase()
    |> String.trim()
  end

  defp normalize_entity(_), do: ""

  defp contradicts?(claim1, claim2) when is_binary(claim1) and is_binary(claim2) do
    # Check for explicit contradictions
    # This is a simplified heuristic - could be enhanced with NLI models

    c1 = String.downcase(claim1)
    c2 = String.downcase(claim2)

    cond do
      # Same claim (not a contradiction)
      c1 == c2 ->
        false

      # Negation patterns
      has_negation_difference?(c1, c2) ->
        true

      # Number disagreement (e.g., "14 million" vs "37 million")
      has_number_disagreement?(c1, c2) ->
        true

      # Default: not a contradiction
      true ->
        false
    end
  end

  defp contradicts?(_, _), do: false

  defp has_negation_difference?(c1, c2) do
    negation_words = ["not", "no", "never", "none", "neither", "nor", "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"]

    c1_has_negation = Enum.any?(negation_words, &String.contains?(c1, &1))
    c2_has_negation = Enum.any?(negation_words, &String.contains?(c2, &1))

    # XOR: one has negation, other doesn't
    c1_has_negation != c2_has_negation
  end

  defp has_number_disagreement?(c1, c2) do
    # Extract numbers from both claims
    numbers1 = extract_numbers(c1)
    numbers2 = extract_numbers(c2)

    # If both have numbers and they're significantly different
    if length(numbers1) > 0 and length(numbers2) > 0 do
      # Check if any corresponding numbers differ by more than 20%
      Enum.any?(Enum.zip(numbers1, numbers2), fn {n1, n2} ->
        min_val = min(n1, n2)
        max_val = max(n1, n2)
        min_val > 0 and (max_val - min_val) / min_val > 0.2
      end)
    else
      false
    end
  end

  defp extract_numbers(text) do
    ~r/\d+(?:,\d{3})*(?:\.\d+)?/
    |> Regex.scan(text)
    |> List.flatten()
    |> Enum.map(&parse_number/1)
    |> Enum.reject(&is_nil/1)
  end

  defp parse_number(str) do
    str
    |> String.replace(",", "")
    |> Float.parse()
    |> case do
      {num, _} -> num
      :error -> nil
    end
  end
end
