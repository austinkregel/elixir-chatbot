defmodule Brain.Knowledge.Academic.PaperModelBuilder do
  @moduledoc """
  Builds a running epistemic model from academic papers.

  Each paper's claims become JTMS nodes with justifications.
  Citations create dependency links between papers.
  Contradictions are automatically detected and handled.

  ## Architecture

  1. **Paper Ingested** → Extract claims from abstract
  2. **Claims → JTMS Nodes** → High-citation papers become premises
  3. **Claims → BeliefStore** → Stored with academic provenance
  4. **Contradiction Check** → Compare against existing beliefs
  5. **High-Confidence → FactDatabase** → Promoted after corroboration

  ## Node Types

  - `:premise` - Established facts (citation_count > 100)
  - `:assumption` - New claims from recent papers
  - `:derived` - Claims supported by multiple papers
  - `:contradiction` - Conflicting claims between papers

  ## Example

      {:ok, node_ids} = PaperModelBuilder.ingest_paper(paper)
      {:ok, all_node_ids} = PaperModelBuilder.ingest_papers(papers)
  """

  require Logger

  alias Brain.Analysis.Pipeline
  alias Brain.Epistemic.{JTMS, BeliefStore}
  alias Brain.Epistemic.Types.Belief
  alias Brain.Knowledge.Academic.Paper
  alias Brain.Knowledge.Corroborator

  # Minimum abstract length to extract claims from
  @min_abstract_length 100

  # Citation thresholds for node type determination
  @premise_threshold 100
  @high_confidence_threshold 500

  # Similarity threshold for contradiction detection
  @contradiction_similarity_threshold 0.7

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Ingests a single paper into the epistemic model.

  Creates JTMS nodes for claims, stores beliefs with academic provenance,
  and checks for contradictions with existing knowledge.

  ## Returns
    - {:ok, node_ids} - List of created JTMS node IDs
    - {:error, reason} - If ingestion failed
  """
  @spec ingest_paper(Paper.t()) :: {:ok, [String.t()]} | {:error, term()}
  def ingest_paper(%Paper{} = paper) do
    Logger.debug("Ingesting paper into epistemic model",
      paper_id: paper.id,
      title: paper.title,
      citation_count: paper.citation_count
    )

    try do
      # 1. Extract claims from abstract
      claims = extract_claims(paper)

      if claims == [] do
        Logger.debug("No claims extracted from paper", paper_id: paper.id)
        {:ok, []}
      else
        # 2. Create JTMS nodes for each claim
        node_ids = create_jtms_nodes(claims, paper)

        # 3. Store as beliefs with academic provenance
        store_beliefs(claims, paper)

        # 4. Check for contradictions with existing beliefs
        check_contradictions(claims, node_ids, paper)

        Logger.info("Paper ingested into epistemic model",
          paper_id: paper.id,
          claims: length(claims),
          nodes: length(node_ids)
        )

        {:ok, node_ids}
      end
    rescue
      e ->
        Logger.error("Failed to ingest paper",
          paper_id: paper.id,
          error: Exception.message(e)
        )

        {:error, {:ingestion_failed, Exception.message(e)}}
    end
  end

  @doc """
  Ingests multiple papers into the epistemic model.

  Processes papers in sequence to properly detect cross-paper contradictions.
  """
  @spec ingest_papers([Paper.t()]) :: {:ok, [String.t()]} | {:error, term()}
  def ingest_papers(papers) when is_list(papers) do
    results =
      papers
      |> Enum.map(&ingest_paper/1)
      |> Enum.reduce({[], []}, fn
        {:ok, node_ids}, {all_nodes, errors} ->
          {all_nodes ++ node_ids, errors}

        {:error, reason}, {all_nodes, errors} ->
          {all_nodes, [reason | errors]}
      end)

    case results do
      {node_ids, []} ->
        {:ok, node_ids}

      {node_ids, errors} ->
        Logger.warning("Some papers failed to ingest",
          successful: length(node_ids),
          failed: length(errors)
        )

        {:ok, node_ids}
    end
  end

  @doc """
  Extracts claims from a paper's abstract using the NLP pipeline.

  Returns a list of claim strings extracted from the abstract.
  """
  @spec extract_claims(Paper.t()) :: [String.t()]
  def extract_claims(%Paper{abstract: nil}), do: []

  def extract_claims(%Paper{abstract: abstract}) when byte_size(abstract) < @min_abstract_length do
    []
  end

  def extract_claims(%Paper{abstract: abstract}) do
    # Use the Analysis Pipeline to extract claims
    case Pipeline.process(abstract, skip_entity_extraction: true) do
      %{analyses: analyses} when is_list(analyses) ->
        analyses
        |> Enum.filter(&is_assertive_claim?/1)
        |> Enum.map(&extract_claim_text/1)
        |> Enum.reject(&is_nil/1)
        |> Enum.reject(&(&1 == ""))
        |> Enum.uniq()

      _ ->
        # Fallback: split abstract into sentences
        fallback_extract_claims(abstract)
    end
  rescue
    e ->
      Logger.warning("Claim extraction failed, using fallback",
        error: Exception.message(e)
      )

      fallback_extract_claims(abstract)
  end

  @doc """
  Calculates confidence score based on citation count.
  """
  @spec citation_to_confidence(non_neg_integer()) :: float()
  def citation_to_confidence(count) when count > @high_confidence_threshold, do: 0.95
  def citation_to_confidence(count) when count > @premise_threshold, do: 0.85
  def citation_to_confidence(count) when count > 10, do: 0.70
  def citation_to_confidence(_), do: 0.50

  @doc """
  Determines the JTMS node type based on paper metadata.
  """
  @spec determine_node_type(Paper.t()) :: :premise | :assumption
  def determine_node_type(%Paper{citation_count: count}) when count > @premise_threshold do
    :premise
  end

  def determine_node_type(_paper), do: :assumption

  # ============================================================================
  # Private Functions - JTMS Integration
  # ============================================================================

  defp create_jtms_nodes(claims, paper) do
    node_type = determine_node_type(paper)

    claims
    |> Enum.map(fn claim ->
      metadata = build_node_metadata(paper, claim)

      case JTMS.create_node(claim, node_type: node_type, metadata: metadata) do
        {:ok, node_id} ->
          node_id

        {:error, reason} ->
          Logger.warning("Failed to create JTMS node",
            claim: String.slice(claim, 0, 50),
            error: inspect(reason)
          )

          nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp build_node_metadata(paper, claim) do
    %{
      source: :academic,
      paper_id: paper.id,
      title: paper.title,
      citation_count: paper.citation_count,
      venue: paper.venue,
      year: paper.year,
      claim: claim,
      api_source: paper.source
    }
  end

  # ============================================================================
  # Private Functions - BeliefStore Integration
  # ============================================================================

  defp store_beliefs(claims, paper) do
    confidence = citation_to_confidence(paper.citation_count)

    Enum.each(claims, fn claim ->
      belief =
        Belief.new(
          :world,
          :paper_claim,
          claim,
          source: :learned,
          confidence: confidence,
          provenance: [paper.id],
          metadata: %{
            paper_id: paper.id,
            title: paper.title,
            venue: paper.venue,
            year: paper.year,
            citation_count: paper.citation_count,
            api_source: paper.source
          }
        )

      case BeliefStore.add_belief(belief) do
        {:ok, _belief_id} ->
          :ok

        {:error, reason} ->
          Logger.warning("Failed to store belief",
            claim: String.slice(claim, 0, 50),
            error: inspect(reason)
          )
      end
    end)
  end

  # ============================================================================
  # Private Functions - Contradiction Detection
  # ============================================================================

  defp check_contradictions(claims, node_ids, paper) do
    # Query existing beliefs for potential contradictions
    case BeliefStore.query_beliefs(predicate: :paper_claim, min_confidence: 0.5) do
      {:ok, existing_beliefs} ->
        # Compare each new claim against existing beliefs
        Enum.each(Enum.zip(claims, node_ids), fn {claim, node_id} ->
          check_claim_contradictions(claim, node_id, existing_beliefs, paper)
        end)

      {:error, _} ->
        # BeliefStore not available, skip contradiction check
        :ok
    end
  end

  defp check_claim_contradictions(claim, node_id, existing_beliefs, paper) do
    # Filter beliefs from other papers (not self-contradiction)
    other_beliefs =
      Enum.reject(existing_beliefs, fn belief ->
        get_in(belief.metadata, [:paper_id]) == paper.id
      end)

    # Find potentially contradicting beliefs
    Enum.each(other_beliefs, fn belief ->
      existing_claim = belief.object

      case Corroborator.compare_claims(claim, existing_claim) do
        {:ok, similarity} when similarity > @contradiction_similarity_threshold ->
          # High similarity - check if they contradict
          if claims_contradict?(claim, existing_claim) do
            register_contradiction(node_id, belief, paper)
          end

        _ ->
          :ok
      end
    end)
  rescue
    # Corroborator might not be available
    _ -> :ok
  end

  defp claims_contradict?(claim1, claim2) do
    # Simple heuristic: check for negation patterns
    c1 = String.downcase(claim1)
    c2 = String.downcase(claim2)

    negation_words = [
      "not",
      "no",
      "never",
      "cannot",
      "doesn't",
      "don't",
      "isn't",
      "aren't",
      "wasn't",
      "weren't",
      "won't",
      "wouldn't",
      "shouldn't",
      "couldn't"
    ]

    c1_has_negation = Enum.any?(negation_words, &String.contains?(c1, &1))
    c2_has_negation = Enum.any?(negation_words, &String.contains?(c2, &1))

    # XOR: one has negation, other doesn't = likely contradiction
    c1_has_negation != c2_has_negation
  end

  defp register_contradiction(new_node_id, existing_belief, new_paper) do
    # Get the JTMS node ID for the existing belief if it has one
    existing_node_id = existing_belief.node_id

    if existing_node_id do
      # Register mutual contradiction in JTMS
      metadata = %{
        source: :knowledge_expansion,
        new_fact: %{
          claim: new_paper.title,
          paper_id: new_paper.id,
          citation_count: new_paper.citation_count
        },
        existing_belief: %{
          claim: existing_belief.object,
          paper_id: get_in(existing_belief.metadata, [:paper_id]),
          citation_count: get_in(existing_belief.metadata, [:citation_count])
        }
      }

      case JTMS.register_contradiction([new_node_id, existing_node_id], "academic_conflict") do
        {:ok, _contra_id} ->
          Logger.info("Academic contradiction registered",
            new_paper: new_paper.id,
            existing_paper: get_in(existing_belief.metadata, [:paper_id])
          )

        {:error, reason} ->
          Logger.warning("Failed to register contradiction", error: inspect(reason))
      end
    end
  end

  # ============================================================================
  # Private Functions - Claim Extraction
  # ============================================================================

  defp is_assertive_claim?(analysis) do
    # Consider assertive speech acts as claims
    case analysis do
      %{speech_act: %{category: :assertive}} -> true
      %{speech_act: %{category: :commissive}} -> false
      %{speech_act: %{category: :directive}} -> false
      %{speech_act: %{category: :expressive}} -> false
      _ -> false
    end
  end

  defp extract_claim_text(%{text: text}) when is_binary(text), do: String.trim(text)
  defp extract_claim_text(_), do: nil

  defp fallback_extract_claims(abstract) when is_binary(abstract) do
    # Simple sentence splitting as fallback
    abstract
    |> String.split(~r/[.!?]+/)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.filter(&is_substantive_sentence?/1)
    |> Enum.take(5)
  end

  defp is_substantive_sentence?(sentence) do
    # Filter out very short or likely non-claim sentences
    word_count = sentence |> String.split() |> length()
    word_count >= 5 and word_count <= 50
  end
end
