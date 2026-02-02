defmodule Brain.Analysis.NoveltyDetector do
  @moduledoc """
  Detects novel intent candidates based on classifier uncertainty.

  A candidate is considered novel if:
  - Best score is below a threshold (low confidence)
  - Margin between best and second is small (ambiguous)
  - The utterance is substantive (not pure smalltalk/backchannel)
  """

  require Logger

  @default_novelty_threshold 0.5
  @default_margin_threshold 0.2

  @doc """
  Determines if an utterance represents a novel intent candidate.

  Returns `{:novel, novelty_score}` if novel, `:not_novel` otherwise.

  ## Options
    - `:novelty_threshold` - Maximum best_score to consider novel (default: 0.5)
    - `:margin_threshold` - Minimum margin required to avoid being novel (default: 0.2)
  """
  def is_novel?(best_score, margin, opts \\ []) do
    novelty_threshold = Keyword.get(opts, :novelty_threshold, @default_novelty_threshold)
    margin_threshold = Keyword.get(opts, :margin_threshold, @default_margin_threshold)

    # Novel if:
    # 1. Best score is low (below threshold)
    # 2. Margin is small (ambiguous between intents)
    is_low_confidence = best_score < novelty_threshold
    is_ambiguous = margin < margin_threshold

    cond do
      is_low_confidence or is_ambiguous ->
        # Calculate novelty score: combination of low confidence and ambiguity
        novelty_score = (1.0 - best_score) * 0.7 + (1.0 - margin) * 0.3
        {:novel, novelty_score}

      true ->
        :not_novel
    end
  end

  @doc """
  Checks if an utterance is substantive enough to warrant review.

  Filters out pure smalltalk, backchannels, and other non-substantive utterances.
  """
  def is_substantive?(speech_act, _intent) do
    # Skip pure expressives that are well-handled (greetings, thanks, etc.)
    if speech_act.category == :expressive do
      # Allow some expressives through if they might be misclassified
      # (e.g., "thanks for the weather" might be misclassified as thanks when it's actually weather.query)
      not well_handled_expressive?(speech_act.sub_type)
    else
      # All directives and assertives are substantive
      true
    end
  end

  defp well_handled_expressive?(sub_type) do
    sub_type in [:greeting, :farewell, :thanks, :apology, :backchannel, :acknowledgment]
  end
end
