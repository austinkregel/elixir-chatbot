defmodule Brain.Analysis.ComprehensionAssessor.ComprehensionProfile do
  @moduledoc """
  Represents the result of a comprehension assessment.

  Each dimension captures a score (0.0-1.0), evidence map, and current weight.
  The composite score is a weighted sum of dimension scores.
  The verdict is derived from the composite score:
    - :comprehended (>= 0.7) — system understands the text well
    - :partial (>= 0.4) — system has partial understanding
    - :opaque (>= 0.2) — system mostly doesn't understand
    - :garbled (< 0.2) — text appears malformed or non-natural-language

  `learnable` is true when verdict is :comprehended or :partial.
  """

  @type dimension_result :: %{
          score: float(),
          evidence: map(),
          weight: float()
        }

  @type verdict :: :comprehended | :partial | :opaque | :garbled

  @type gap :: %{
          dimension: atom(),
          description: String.t(),
          score: float()
        }

  @type t :: %__MODULE__{
          id: String.t(),
          dimensions: %{atom() => dimension_result()},
          composite_score: float(),
          verdict: verdict(),
          gaps: [gap()],
          learnable: boolean(),
          assessed_at: DateTime.t()
        }

  @enforce_keys [:id]
  defstruct [
    :id,
    dimensions: %{},
    composite_score: 0.0,
    verdict: :garbled,
    gaps: [],
    learnable: false,
    assessed_at: nil
  ]

  @dimension_names [
    :referential_clarity,
    :actor_identification,
    :propositional_content,
    :temporal_grounding,
    :contextual_sufficiency,
    :epistemic_grounding,
    :structural_coherence,
    :illocutionary_clarity
  ]

  @gap_descriptions %{
    referential_clarity: "Cannot identify what this text is about",
    actor_identification: "Cannot identify who is involved",
    propositional_content: "Cannot extract a specific claim being made",
    temporal_grounding: "Cannot determine when this applies",
    contextual_sufficiency: "Missing required context to understand this text",
    epistemic_grounding: "Cannot relate this to any existing knowledge",
    structural_coherence: "Text appears malformed or non-natural-language",
    illocutionary_clarity: "Cannot determine what kind of communication this is"
  }

  @doc "All recognized dimension names."
  def dimension_names, do: @dimension_names

  @doc """
  Builds a ComprehensionProfile from a map of dimension scores and the current weights.

  `dimension_scores` is a map of `%{atom() => {score, evidence}}`.
  `weights` is a map of `%{atom() => float()}` (should sum to 1.0).
  """
  def build(dimension_scores, weights) when is_map(dimension_scores) and is_map(weights) do
    dimensions =
      for {dim, {score, evidence}} <- dimension_scores, into: %{} do
        weight = Map.get(weights, dim, 1.0 / length(@dimension_names))

        {dim, %{score: score, evidence: evidence, weight: weight}}
      end

    # Check structural_coherence hard gate
    structural_score =
      case Map.get(dimensions, :structural_coherence) do
        %{score: s} -> s
        _ -> 1.0
      end

    {composite, verdict, learnable} =
      if structural_score < 0.2 do
        {structural_score, :garbled, false}
      else
        composite = compute_composite(dimensions)
        verdict = verdict_from_composite(composite)
        learnable = verdict in [:comprehended, :partial]
        {composite, verdict, learnable}
      end

    gaps = compute_gaps(dimensions)

    %__MODULE__{
      id: generate_id(),
      dimensions: dimensions,
      composite_score: composite,
      verdict: verdict,
      gaps: gaps,
      learnable: learnable,
      assessed_at: DateTime.utc_now()
    }
  end

  @doc "Returns the score for a specific dimension, or nil if not present."
  def dimension_score(%__MODULE__{dimensions: dims}, dimension) do
    case Map.get(dims, dimension) do
      %{score: s} -> s
      _ -> nil
    end
  end

  defp compute_composite(dimensions) do
    {weighted_sum, total_weight} =
      Enum.reduce(dimensions, {0.0, 0.0}, fn {_dim, %{score: score, weight: weight}},
                                              {sum, tw} ->
        {sum + score * weight, tw + weight}
      end)

    if total_weight > 0.0 do
      weighted_sum / total_weight
    else
      0.0
    end
  end

  defp verdict_from_composite(composite) when composite >= 0.7, do: :comprehended
  defp verdict_from_composite(composite) when composite >= 0.4, do: :partial
  defp verdict_from_composite(composite) when composite >= 0.2, do: :opaque
  defp verdict_from_composite(_composite), do: :garbled

  defp compute_gaps(dimensions) do
    dimensions
    |> Enum.filter(fn {_dim, %{score: score}} -> score < 0.4 end)
    |> Enum.map(fn {dim, %{score: score}} ->
      %{
        dimension: dim,
        description: Map.get(@gap_descriptions, dim, "Unknown dimension gap"),
        score: score
      }
    end)
    |> Enum.sort_by(& &1.score)
  end

  defp generate_id do
    :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
  end
end
