defmodule Brain.Knowledge.AcademicValidator do
  @moduledoc """
  Validates findings against peer-reviewed academic literature.

  Uses the epistemic layer (BeliefStore, FactDatabase) to cross-reference
  claims and provide confidence boosts when corroborated by academic sources.

  ## Validation Outcomes

  - `:corroborated` - Finding matches academic literature (confidence boost)
  - `:contradicted` - Finding conflicts with academic sources (triggers review)
  - `:insufficient_evidence` - Not enough academic data to validate

  ## Example

      {:ok, result} = AcademicValidator.validate(finding)
      # => {:corroborated, %{boost: 0.15, sources: [...]}}

      {:ok, results} = AcademicValidator.bulk_validate(findings)
  """

  require Logger

  alias Brain.Epistemic.BeliefStore
  alias Brain.FactDatabase
  alias Brain.Knowledge.Corroborator
  alias Brain.Knowledge.Types.{Finding, ReviewCandidate}
  alias Brain.Knowledge.Academic.{SemanticScholar, OpenAlex, PaperModelBuilder}

  # Similarity threshold for considering claims as matching
  @similarity_threshold 0.7

  # High confidence threshold for strong corroboration
  @high_confidence_threshold 0.8

  # Confidence boost amounts
  @high_confidence_boost 0.15
  @moderate_confidence_boost 0.10
  @low_confidence_boost 0.05

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Validates a finding against peer-reviewed academic literature.

  Checks both the BeliefStore (for previously ingested papers) and
  optionally searches live academic APIs.

  ## Options
    - :live_search - Search live APIs if no match in BeliefStore (default: false)
    - :min_confidence - Minimum confidence for matches (default: 0.6)

  ## Returns
    - {:ok, {:corroborated, details}} - Finding matches academic literature
    - {:ok, {:contradicted, details}} - Finding conflicts with academic sources
    - {:ok, :insufficient_evidence} - Not enough academic data
    - {:error, reason} - Validation failed
  """
  @spec validate(Finding.t(), keyword()) ::
          {:ok, {:corroborated, map()} | {:contradicted, map()} | :insufficient_evidence}
          | {:error, term()}
  def validate(%Finding{} = finding, opts \\ []) do
    live_search = Keyword.get(opts, :live_search, false)
    min_confidence = Keyword.get(opts, :min_confidence, 0.6)

    Logger.debug("Validating finding against academic sources",
      claim: String.slice(finding.claim, 0, 50),
      entity: finding.entity
    )

    # First check BeliefStore for matching academic beliefs
    case check_belief_store(finding.claim, min_confidence) do
      {:ok, {:corroborated, _} = result} ->
        {:ok, result}

      {:ok, {:contradicted, _} = result} ->
        {:ok, result}

      {:ok, :no_match} ->
        if live_search do
          # Search live APIs for validation
          search_live_sources(finding, opts)
        else
          {:ok, :insufficient_evidence}
        end

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Validates multiple findings efficiently.

  Groups similar findings and batches API requests where possible.

  ## Returns
    - {:ok, [validation_result]} - List of results in same order as input
  """
  @spec bulk_validate([Finding.t()], keyword()) :: {:ok, [term()]}
  def bulk_validate(findings, opts \\ []) when is_list(findings) do
    results =
      findings
      |> Enum.map(fn finding ->
        case validate(finding, opts) do
          {:ok, result} -> result
          {:error, _} -> :validation_error
        end
      end)

    {:ok, results}
  end

  @doc """
  Applies validation results to enhance a ReviewCandidate.

  Boosts confidence and adds academic corroborating sources.
  """
  @spec apply_validation(ReviewCandidate.t(), term()) :: ReviewCandidate.t()
  def apply_validation(%ReviewCandidate{} = candidate, {:corroborated, details}) do
    boost = Map.get(details, :boost, @low_confidence_boost)
    sources = Map.get(details, :sources, [])

    new_confidence =
      (candidate.aggregate_confidence + boost)
      |> min(1.0)

    # Add academic sources to corroborating sources
    updated_sources = candidate.corroborating_sources ++ sources

    %{candidate | aggregate_confidence: new_confidence, corroborating_sources: updated_sources}
  end

  def apply_validation(%ReviewCandidate{} = candidate, {:contradicted, details}) do
    academic_contradictions = Map.get(details, :conflicting_papers, [])

    # Convert papers to findings for the existing_contradictions field
    existing = candidate.existing_contradictions ++ academic_contradictions

    %{candidate | existing_contradictions: existing}
  end

  def apply_validation(candidate, _), do: candidate

  @doc """
  Searches academic sources for papers related to a topic.

  Used for exploratory validation where we want to understand
  what the academic consensus is on a topic.
  """
  @spec search_academic_consensus(String.t(), keyword()) ::
          {:ok, %{papers: [map()], consensus: atom()}}
  def search_academic_consensus(topic, opts \\ []) do
    limit = Keyword.get(opts, :limit, 20)

    # Search multiple sources
    papers = search_all_sources(topic, limit: limit)

    if papers == [] do
      {:ok, %{papers: [], consensus: :no_data}}
    else
      # Analyze papers for consensus
      consensus = analyze_consensus(papers)
      {:ok, %{papers: papers, consensus: consensus}}
    end
  end

  # ============================================================================
  # Private Functions - BeliefStore Validation
  # ============================================================================

  defp check_belief_store(claim, min_confidence) do
    case BeliefStore.query_beliefs(predicate: :paper_claim, min_confidence: min_confidence) do
      {:ok, beliefs} when beliefs != [] ->
        find_matching_beliefs(claim, beliefs)

      {:ok, []} ->
        {:ok, :no_match}

      {:error, :not_ready} ->
        {:ok, :no_match}

      {:error, reason} ->
        Logger.warning("BeliefStore query failed", error: inspect(reason))
        {:ok, :no_match}
    end
  rescue
    # BeliefStore might not be running
    _ -> {:ok, :no_match}
  end

  defp find_matching_beliefs(claim, beliefs) do
    # Compare claim against each belief
    matches =
      beliefs
      |> Enum.map(fn belief ->
        case compare_with_belief(claim, belief) do
          {:match, similarity} -> {belief, similarity}
          :no_match -> nil
        end
      end)
      |> Enum.reject(&is_nil/1)
      |> Enum.sort_by(fn {_, similarity} -> similarity end, :desc)

    case matches do
      [] ->
        {:ok, :no_match}

      [{best_match, similarity} | rest] ->
        # Check for contradictions in the matches
        if has_contradiction?(claim, best_match) do
          {:ok,
           {:contradicted,
            %{
              conflicting_papers:
                Enum.map([{best_match, similarity} | rest], fn {b, _} ->
                  build_contradiction_info(b)
                end),
              similarity: similarity
            }}}
        else
          # Calculate boost based on confidence of matching beliefs
          boost = calculate_boost(best_match, length(rest) + 1)

          sources =
            [{best_match, similarity} | rest]
            |> Enum.take(3)
            |> Enum.map(fn {b, _} -> build_source_info(b) end)

          {:ok, {:corroborated, %{boost: boost, sources: sources, similarity: similarity}}}
        end
    end
  end

  defp compare_with_belief(claim, belief) do
    case Corroborator.compare_claims(claim, belief.object) do
      {:ok, similarity} when similarity >= @similarity_threshold ->
        {:match, similarity}

      _ ->
        :no_match
    end
  rescue
    # Corroborator might not be available
    _ -> :no_match
  end

  defp has_contradiction?(claim, belief) do
    # Simple heuristic for contradiction detection
    c1 = String.downcase(claim)
    c2 = String.downcase(belief.object)

    negation_words = ["not", "no", "never", "cannot", "doesn't", "don't", "isn't"]

    c1_has_negation = Enum.any?(negation_words, &String.contains?(c1, &1))
    c2_has_negation = Enum.any?(negation_words, &String.contains?(c2, &1))

    # XOR: one has negation, other doesn't
    c1_has_negation != c2_has_negation
  end

  defp calculate_boost(best_match, total_matches) do
    base_boost =
      cond do
        best_match.confidence >= @high_confidence_threshold -> @high_confidence_boost
        best_match.confidence >= 0.6 -> @moderate_confidence_boost
        true -> @low_confidence_boost
      end

    # Small bonus for multiple corroborating sources
    multi_source_bonus = min((total_matches - 1) * 0.02, 0.05)

    base_boost + multi_source_bonus
  end

  defp build_source_info(belief) do
    %{
      paper_id: get_in(belief.metadata, [:paper_id]),
      title: get_in(belief.metadata, [:title]),
      venue: get_in(belief.metadata, [:venue]),
      year: get_in(belief.metadata, [:year]),
      citation_count: get_in(belief.metadata, [:citation_count]),
      confidence: belief.confidence
    }
  end

  defp build_contradiction_info(belief) do
    %{
      type: :academic_conflict,
      paper_id: get_in(belief.metadata, [:paper_id]),
      claim: belief.object,
      confidence: belief.confidence,
      citation_count: get_in(belief.metadata, [:citation_count])
    }
  end

  # ============================================================================
  # Private Functions - Live API Search
  # ============================================================================

  defp search_live_sources(finding, opts) do
    # Build search query from finding
    query = build_search_query(finding)
    limit = Keyword.get(opts, :limit, 5)

    # Search Semantic Scholar (primary source)
    papers =
      case SemanticScholar.search(query, limit: limit) do
        {:ok, papers} -> papers
        {:error, _} -> []
      end

    # Supplement with OpenAlex if needed
    papers =
      if length(papers) < 3 do
        case OpenAlex.search_cs(query, limit: limit) do
          {:ok, more_papers} -> papers ++ more_papers
          {:error, _} -> papers
        end
      else
        papers
      end

    if papers == [] do
      {:ok, :insufficient_evidence}
    else
      # Ingest papers into epistemic model
      PaperModelBuilder.ingest_papers(papers)

      # Analyze results
      analyze_paper_results(finding.claim, papers)
    end
  end

  defp build_search_query(finding) do
    # Use entity and key terms from claim
    terms = [finding.entity]

    # Extract key nouns from claim
    words =
      finding.claim
      |> String.split()
      |> Enum.filter(&(String.length(&1) > 4))
      |> Enum.take(5)

    (terms ++ words)
    |> Enum.join(" ")
    |> String.slice(0, 100)
  end

  defp analyze_paper_results(claim, papers) do
    # Check if any papers support or contradict the claim
    supporting =
      Enum.filter(papers, fn paper ->
        case paper.abstract do
          nil -> false
          abstract -> claim_supported?(claim, abstract)
        end
      end)

    if supporting != [] do
      # Calculate boost based on citation counts
      avg_citations =
        supporting
        |> Enum.map(& &1.citation_count)
        |> Enum.sum()
        |> Kernel./(length(supporting))

      boost =
        cond do
          avg_citations > 100 -> @high_confidence_boost
          avg_citations > 10 -> @moderate_confidence_boost
          true -> @low_confidence_boost
        end

      sources =
        supporting
        |> Enum.take(3)
        |> Enum.map(fn paper ->
          %{
            paper_id: paper.id,
            title: paper.title,
            venue: paper.venue,
            year: paper.year,
            citation_count: paper.citation_count
          }
        end)

      {:ok, {:corroborated, %{boost: boost, sources: sources}}}
    else
      {:ok, :insufficient_evidence}
    end
  end

  defp claim_supported?(claim, abstract) do
    # Simple keyword overlap check
    claim_words =
      claim
      |> String.downcase()
      |> String.split()
      |> Enum.filter(&(String.length(&1) > 3))
      |> MapSet.new()

    abstract_words =
      abstract
      |> String.downcase()
      |> String.split()
      |> Enum.filter(&(String.length(&1) > 3))
      |> MapSet.new()

    overlap = MapSet.intersection(claim_words, abstract_words) |> MapSet.size()
    total = MapSet.size(claim_words)

    total > 0 and overlap / total >= 0.3
  end

  # ============================================================================
  # Private Functions - Consensus Analysis
  # ============================================================================

  defp search_all_sources(topic, opts) do
    limit = Keyword.get(opts, :limit, 10)

    # Search multiple sources in parallel
    tasks = [
      Task.async(fn -> SemanticScholar.search(topic, limit: div(limit, 2)) end),
      Task.async(fn -> OpenAlex.search_cs(topic, limit: div(limit, 2)) end)
    ]

    tasks
    |> Task.await_many(20_000)
    |> Enum.flat_map(fn
      {:ok, papers} -> papers
      _ -> []
    end)
  rescue
    _ -> []
  end

  defp analyze_consensus(papers) do
    # Simple consensus analysis based on citation counts
    total_citations = papers |> Enum.map(& &1.citation_count) |> Enum.sum()
    paper_count = length(papers)

    cond do
      paper_count >= 10 and total_citations > 1000 -> :strong
      paper_count >= 5 and total_citations > 100 -> :moderate
      paper_count >= 3 -> :weak
      true -> :emerging
    end
  end
end
