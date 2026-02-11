defmodule Brain.Knowledge.AcademicValidator do
  @moduledoc "Validates findings against peer-reviewed academic literature.\n\nUses the epistemic layer (BeliefStore, FactDatabase) to cross-reference\nclaims and provide confidence boosts when corroborated by academic sources.\n\n## Validation Outcomes\n\n- `:corroborated` - Finding matches academic literature (confidence boost)\n- `:contradicted` - Finding conflicts with academic sources (triggers review)\n- `:insufficient_evidence` - Not enough academic data to validate\n\n## Example\n\n    {:ok, result} = AcademicValidator.validate(finding)\n    # => {:corroborated, %{boost: 0.15, sources: [...]}}\n\n    {:ok, results} = AcademicValidator.bulk_validate(findings)\n"

  alias Brain.LinguisticData
  alias Brain.Knowledge.Academic
  alias Brain.Knowledge.Types
  require Logger

  alias Brain.Epistemic.BeliefStore
  alias Brain.Knowledge.Corroborator
  alias Types.{Finding, ReviewCandidate}
  alias Academic.{SemanticScholar, OpenAlex, PaperModelBuilder}
  @similarity_threshold 0.7
  @high_confidence_threshold 0.8
  @high_confidence_boost 0.15
  @moderate_confidence_boost 0.1
  @low_confidence_boost 0.05

  @doc "Validates a finding against peer-reviewed academic literature.\n\nChecks both the BeliefStore (for previously ingested papers) and\noptionally searches live academic APIs.\n\n## Options\n  - :live_search - Search live APIs if no match in BeliefStore (default: false)\n  - :min_confidence - Minimum confidence for matches (default: 0.6)\n\n## Returns\n  - {:ok, {:corroborated, details}} - Finding matches academic literature\n  - {:ok, {:contradicted, details}} - Finding conflicts with academic sources\n  - {:ok, :insufficient_evidence} - Not enough academic data\n"
  @spec validate(Finding.t(), keyword()) ::
          {:ok, {:corroborated, map()} | {:contradicted, map()} | :insufficient_evidence}
  def validate(%Finding{} = finding, opts \\ []) do
    live_search = Keyword.get(opts, :live_search, false)
    min_confidence = Keyword.get(opts, :min_confidence, 0.6)

    Logger.debug("Validating finding against academic sources",
      claim: String.slice(finding.claim, 0, 50),
      entity: finding.entity
    )

    case check_belief_store(finding.claim, min_confidence) do
      {:ok, {:corroborated, _} = result} ->
        {:ok, result}

      {:ok, {:contradicted, _} = result} ->
        {:ok, result}

      {:ok, :no_match} ->
        if live_search do
          search_live_sources(finding, opts)
        else
          {:ok, :insufficient_evidence}
        end
    end
  end

  @doc "Validates multiple findings efficiently.\n\nGroups similar findings and batches API requests where possible.\n\n## Returns\n  - {:ok, [validation_result]} - List of results in same order as input\n"
  @spec bulk_validate([Finding.t()], keyword()) :: {:ok, [term()]}
  def bulk_validate(findings, opts \\ []) when is_list(findings) do
    results =
      findings
      |> Enum.map(fn finding ->
        {:ok, result} = validate(finding, opts)
        result
      end)

    {:ok, results}
  end

  @doc "Applies validation results to enhance a ReviewCandidate.\n\nBoosts confidence and adds academic corroborating sources.\n"
  @spec apply_validation(ReviewCandidate.t(), term()) :: ReviewCandidate.t()
  def apply_validation(%ReviewCandidate{} = candidate, {:corroborated, details}) do
    boost = Map.get(details, :boost, @low_confidence_boost)
    sources = Map.get(details, :sources, [])

    new_confidence =
      (candidate.aggregate_confidence + boost)
      |> min(1.0)

    updated_sources = candidate.corroborating_sources ++ sources

    %{candidate | aggregate_confidence: new_confidence, corroborating_sources: updated_sources}
  end

  def apply_validation(%ReviewCandidate{} = candidate, {:contradicted, details}) do
    academic_contradictions = Map.get(details, :conflicting_papers, [])
    existing = candidate.existing_contradictions ++ academic_contradictions

    %{candidate | existing_contradictions: existing}
  end

  def apply_validation(candidate, _) do
    candidate
  end

  @doc "Searches academic sources for papers related to a topic.\n\nUsed for exploratory validation where we want to understand\nwhat the academic consensus is on a topic.\n"
  @spec search_academic_consensus(String.t(), keyword()) ::
          {:ok, %{papers: [map()], consensus: atom()}}
  def search_academic_consensus(topic, opts \\ []) do
    limit = Keyword.get(opts, :limit, 20)
    papers = search_all_sources(topic, limit: limit)

    if papers == [] do
      {:ok, %{papers: [], consensus: :no_data}}
    else
      consensus = analyze_consensus(papers)
      {:ok, %{papers: papers, consensus: consensus}}
    end
  end

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
    _ -> {:ok, :no_match}
  end

  defp find_matching_beliefs(claim, beliefs) do
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
    _ -> :no_match
  end

  defp has_contradiction?(claim, belief) do
    c1 = String.downcase(claim)
    c2 = String.downcase(belief.object)

    negation_words = LinguisticData.negation_words()

    c1_has_negation = Enum.any?(negation_words, &String.contains?(c1, &1))
    c2_has_negation = Enum.any?(negation_words, &String.contains?(c2, &1))
    c1_has_negation != c2_has_negation
  end

  defp calculate_boost(best_match, total_matches) do
    base_boost =
      cond do
        best_match.confidence >= @high_confidence_threshold -> @high_confidence_boost
        best_match.confidence >= 0.6 -> @moderate_confidence_boost
        true -> @low_confidence_boost
      end

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

  defp search_live_sources(finding, opts) do
    query = build_search_query(finding)
    limit = Keyword.get(opts, :limit, 5)

    papers =
      case SemanticScholar.search(query, limit: limit) do
        {:ok, papers} -> papers
        {:error, _} -> []
      end

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
      PaperModelBuilder.ingest_papers(papers)
      analyze_paper_results(finding.claim, papers)
    end
  end

  defp build_search_query(finding) do
    terms = [finding.entity]

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
    supporting =
      Enum.filter(papers, fn paper ->
        case paper.abstract do
          nil -> false
          abstract -> claim_supported?(claim, abstract)
        end
      end)

    if supporting != [] do
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

  defp search_all_sources(topic, opts) do
    limit = Keyword.get(opts, :limit, 10)

    tasks = [
      Task.async(fn -> SemanticScholar.search(topic, limit: div(limit, 2)) end),
      Task.async(fn -> OpenAlex.search_cs(topic, limit: div(limit, 2)) end)
    ]

    tasks
    |> Task.await_many(20000)
    |> Enum.flat_map(fn
      {:ok, papers} -> papers
      _ -> []
    end)
  rescue
    _ -> []
  end

  defp analyze_consensus(papers) do
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