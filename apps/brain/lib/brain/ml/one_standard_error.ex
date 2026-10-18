defmodule Brain.ML.OneStandardError do
  @moduledoc """
  The one-standard-error rule for choosing a smoothing or prior strength by
  cross-validation (Breiman, Friedman, Olshen and Stone, *Classification and
  Regression Trees*, 1984; Hastie, Tibshirani and Friedman, *The Elements of
  Statistical Learning*, 2nd ed., 2009, §7.10).

  Every candidate is scored on each fold of one cross-validation, a higher
  score better. A candidate's cross-validated score is the mean of its fold
  scores, and its standard error is their sample standard deviation over the
  square root of the number of folds. The *best* candidate has the highest
  mean. The *chosen* one is the most regularized candidate whose mean is at
  least the best mean less the best candidate's standard error.

  On a flat surface the folds cannot tell the candidates near the top apart,
  and the one with the highest mean is picked by fold noise; nothing the
  folds measured says a weaker prior generalizes better. The rule takes the
  strongest prior the folds cannot tell from the best. Where the surface is
  not flat, the best candidate's lead is larger than its standard error and
  the rule keeps it.

  What "more regularized" means is the caller's: `choose/2` takes a function
  of a candidate whose larger value, in Erlang term order, is the more
  regularized candidate. Ties: on equal means the best is the more
  regularized; among the candidates within one standard error, on equal
  regularization the higher mean is chosen, then the earlier candidate.
  """

  @type stats :: %{mean: float(), standard_error: float()}

  @type result :: %{
          chosen: term(),
          best: term(),
          mean: float(),
          standard_error: float(),
          threshold: float(),
          scores: %{term() => stats()}
        }

  @doc """
  Chooses among `scored`, a list of `{candidate, fold_scores}` scored on the
  same folds (at least two, a higher score better), by the one-standard-error
  rule; `regularization` maps a candidate to a term whose larger value is the
  more regularized candidate.

  Returns `%{chosen, best, mean, standard_error, threshold, scores}`: the
  chosen and the best candidates, the best's mean and standard error, the
  threshold (`mean - standard_error`) the chosen candidate's mean reaches,
  and every candidate's mean and standard error.

  Raises with no candidates, with candidates scored on different numbers of
  folds, with fewer than two folds, and on a fold score that is not a number.
  """
  @spec choose([{term(), [number()]}], (term() -> term())) :: result()
  def choose([_ | _] = scored, regularization) when is_function(regularization, 1) do
    case scored |> Enum.map(fn {_candidate, scores} -> length(scores) end) |> Enum.uniq() do
      [n] when n >= 2 ->
        :ok

      [n] ->
        raise ArgumentError,
              "OneStandardError: the candidates were scored on #{n} fold(s); a standard error needs at least two"

      counts ->
        raise ArgumentError,
              "OneStandardError: the candidates were scored on different numbers of folds (#{inspect(counts)}); " <>
                "they must be scored on the folds of one cross-validation"
    end

    stats = Enum.map(scored, fn {candidate, scores} -> {candidate, stats!(candidate, scores)} end)

    {best, best_stats} = Enum.max_by(stats, fn {candidate, s} -> {s.mean, regularization.(candidate)} end)
    threshold = best_stats.mean - best_stats.standard_error

    {chosen, _stats} =
      stats
      |> Enum.filter(fn {_candidate, s} -> s.mean >= threshold end)
      |> Enum.max_by(fn {candidate, s} -> {regularization.(candidate), s.mean} end)

    %{
      chosen: chosen,
      best: best,
      mean: best_stats.mean,
      standard_error: best_stats.standard_error,
      threshold: threshold,
      scores: Map.new(stats)
    }
  end

  def choose([], _regularization), do: raise(ArgumentError, "OneStandardError: no candidates to choose from")

  defp stats!(candidate, scores) do
    for score <- scores, not is_number(score) do
      raise ArgumentError, "OneStandardError: #{inspect(candidate)} has the fold score #{inspect(score)}, which is not a number"
    end

    n = length(scores)
    mean = Enum.sum(scores) / n
    variance = (scores |> Enum.map(&((&1 - mean) * (&1 - mean))) |> Enum.sum()) / (n - 1)

    %{mean: mean, standard_error: :math.sqrt(variance / n)}
  end
end
