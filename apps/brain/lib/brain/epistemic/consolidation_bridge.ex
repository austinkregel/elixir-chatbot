defmodule Brain.Epistemic.ConsolidationBridge do
  @moduledoc """
  Bridges Memory.Consolidation SemanticFacts to the Epistemic BeliefStore.

  When memory consolidation creates a SemanticFact from clustered episodes,
  this module converts it to a world-scoped belief with provenance links
  back to the source episodes.

  Confidence uses a Bayesian prior: evidence_count / (evidence_count + 2).
  This requires 2 episodes for 0.5 confidence and 10 for ~0.83.
  """

  require Logger

  alias Brain.Epistemic.{BeliefStore, Types}
  alias Types.Belief

  @doc """
  Converts a SemanticFact to a belief and adds it to BeliefStore.

  Returns `{:ok, belief_id}` or `{:error, reason}`.
  """
  def bridge_semantic_fact(semantic_fact) do
    if BeliefStore.ready?() do
      evidence_count = length(semantic_fact.evidence_ids || [])
      confidence = evidence_count / (evidence_count + 2)

      provenance =
        (semantic_fact.evidence_ids || [])
        |> Enum.map(fn id -> "episode:#{id}" end)

      belief =
        Belief.new(
          :system,
          :consolidated_knowledge,
          semantic_fact.representation,
          source: :consolidated,
          confidence: confidence,
          provenance: provenance ++ ["semantic_fact:#{semantic_fact.id}"]
        )

      case BeliefStore.add_belief(belief) do
        {:ok, belief_id} ->
          Logger.debug("ConsolidationBridge: created belief from semantic fact",
            semantic_fact_id: semantic_fact.id,
            belief_id: belief_id,
            evidence_count: evidence_count,
            confidence: Float.round(confidence, 3)
          )

          {:ok, belief_id}

        error ->
          error
      end
    else
      {:error, :belief_store_not_ready}
    end
  rescue
    e ->
      Logger.debug("ConsolidationBridge failed: #{Exception.message(e)}")
      {:error, :bridge_failed}
  end
end
