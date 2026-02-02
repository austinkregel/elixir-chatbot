defmodule Brain.Analysis.ActivationPool do
  @moduledoc """
  Manages global inhibition and activation normalization.

  Prevents runaway confidence by ensuring:
  - Sum of all activations <= 1.0 (enforced via normalization)
  - Diminishing returns kick in above threshold
  - Learned heuristics have lower boost caps than seeded ones
  - Multiple boosts from same source don't stack linearly

  This addresses the "activation inflation" risk where learned heuristics
  accumulate and everything fires at 0.8-0.9 activation.
  """

  alias Brain.Analysis.Interpretation

  @max_total_activation 1.0
  @diminishing_returns_threshold 0.7
  @minimum_secondary_activation 0.05

  # Maximum boost caps by source type
  @boost_caps %{
    seeded: 0.40,
    learned_global: 0.25,
    learned_cohort: 0.20,
    learned_user: 0.15
  }

  @doc """
  Normalizes a list of interpretations so their activations sum to <= 1.0.

  Uses softmax-like normalization when total exceeds the limit.
  Preserves relative rankings while preventing inflation.
  """
  def normalize(interpretations) when is_list(interpretations) do
    raw_sum = Enum.sum(Enum.map(interpretations, & &1.raw_activation))

    if raw_sum <= @max_total_activation do
      # No normalization needed, but ensure minimum for secondaries
      ensure_minimum_activation(interpretations)
    else
      # Apply softmax-like normalization
      normalized =
        Enum.map(interpretations, fn interp ->
          normalized_value = interp.raw_activation / raw_sum
          Interpretation.with_normalized_activation(interp, normalized_value)
        end)

      ensure_minimum_activation(normalized)
    end
  end

  @doc """
  Normalizes a single interpretation and its alternatives.
  """
  def normalize_with_alternatives(%Interpretation{} = interp) do
    # Collect all activations (primary + alternatives)
    all_activations = [interp.activation | Enum.map(interp.alternatives, & &1.activation)]
    total = Enum.sum(all_activations)

    if total <= @max_total_activation do
      interp
    else
      # Normalize primary
      normalized_primary = interp.activation / total

      # Normalize alternatives
      normalized_alts =
        Enum.map(interp.alternatives, fn alt ->
          %{alt | activation: alt.activation / total}
        end)

      %{interp | activation: normalized_primary, alternatives: normalized_alts}
    end
  end

  @doc """
  Applies a boost to a base activation with diminishing returns.

  Returns the new activation value, respecting:
  - Diminishing returns above threshold
  - Source-specific max boost caps
  - Linear stacking prevention
  """
  def apply_boost(base_activation, boost, source) when is_float(base_activation) do
    # Calculate headroom before diminishing returns
    headroom = max(0.0, @diminishing_returns_threshold - base_activation)

    # Apply diminishing returns: above threshold, boosts are heavily dampened
    effective_boost =
      if headroom > 0 do
        # Linear scaling up to threshold
        boost * (headroom / @diminishing_returns_threshold)
      else
        # Above threshold, only 10% of boost applies
        boost * 0.1
      end

    # Get max boost for this source type
    max_boost = Map.get(@boost_caps, source, 0.15)

    # Apply the boost, capped by source limit
    capped_boost = min(effective_boost, max_boost)

    # Final activation, capped at 1.0
    min(base_activation + capped_boost, 1.0)
  end

  @doc """
  Applies multiple boosts with anti-stacking logic.

  When multiple heuristics want to boost the same interpretation,
  returns diminished after the first.
  """
  def apply_stacked_boosts(base_activation, boosts) when is_list(boosts) do
    # Sort boosts by magnitude (largest first gets full effect)
    sorted_boosts = Enum.sort_by(boosts, fn {_source, amount} -> -amount end)

    Enum.reduce(sorted_boosts, {base_activation, 1.0}, fn {source, amount}, {acc, multiplier} ->
      # Each subsequent boost gets reduced effectiveness
      effective_amount = amount * multiplier
      new_activation = apply_boost(acc, effective_amount, source)
      # Reduce multiplier for next boost (diminishing stacking)
      {new_activation, multiplier * 0.5}
    end)
    |> elem(0)
  end

  @doc """
  Gets the maximum boost allowed for a given source type.
  """
  def max_boost_for(source), do: Map.get(@boost_caps, source, 0.15)

  @doc """
  Checks if the total activation is approaching the limit.

  Useful for stability self-reflection (ST1).
  """
  def approaching_limit?(interpretations, threshold \\ 0.9) when is_list(interpretations) do
    total = Enum.sum(Enum.map(interpretations, & &1.activation))
    total >= threshold * @max_total_activation
  end

  @doc """
  Returns activation statistics for debugging/monitoring.
  """
  def stats(interpretations) when is_list(interpretations) do
    activations = Enum.map(interpretations, & &1.activation)

    %{
      total: Enum.sum(activations),
      max: Enum.max(activations, fn -> 0.0 end),
      min: Enum.min(activations, fn -> 0.0 end),
      count: length(activations),
      at_limit: Enum.sum(activations) >= @max_total_activation * 0.99
    }
  end

  @doc """
  Returns the diminishing returns threshold.
  """
  def diminishing_threshold, do: @diminishing_returns_threshold

  # Private functions

  defp ensure_minimum_activation(interpretations) do
    # Ensure secondary interpretations maintain minimum activation
    # so they remain viable for backtracking
    Enum.map(interpretations, fn interp ->
      if interp.activation < @minimum_secondary_activation and interp.activation > 0 do
        Interpretation.with_normalized_activation(interp, @minimum_secondary_activation)
      else
        interp
      end
    end)
  end
end
