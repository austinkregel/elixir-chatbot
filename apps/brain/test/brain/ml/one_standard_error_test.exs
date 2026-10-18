defmodule Brain.ML.OneStandardErrorTest do
  use ExUnit.Case, async: true

  alias Brain.ML.OneStandardError

  # Five folds' scores per smoothing strength. The surface is flat from 0.001
  # to 1: the folds vary by more than the candidates differ, and 0.01 has the
  # highest mean only by a hair. Past 1 the scores fall well below.
  @flat [
    {0.001, [0.80, 0.84, 0.78, 0.86, 0.82]},
    {0.01, [0.81, 0.85, 0.78, 0.86, 0.82]},
    {0.1, [0.80, 0.85, 0.77, 0.86, 0.82]},
    {1, [0.79, 0.84, 0.77, 0.85, 0.81]},
    {10, [0.70, 0.74, 0.68, 0.75, 0.71]},
    {100, [0.50, 0.55, 0.49, 0.56, 0.52]}
  ]

  defp mean(scores), do: Enum.sum(scores) / length(scores)

  defp standard_error(scores) do
    m = mean(scores)
    n = length(scores)
    :math.sqrt((scores |> Enum.map(&((&1 - m) * (&1 - m))) |> Enum.sum()) / (n - 1) / n)
  end

  test "on a flat surface, the most regularized candidate within one standard error of the best is chosen" do
    rule = OneStandardError.choose(@flat, & &1)
    best_scores = @flat |> List.keyfind(0.01, 0) |> elem(1)

    assert rule.best == 0.01
    assert_in_delta rule.mean, mean(best_scores), 1.0e-12
    assert_in_delta rule.standard_error, standard_error(best_scores), 1.0e-12
    assert_in_delta rule.threshold, mean(best_scores) - standard_error(best_scores), 1.0e-12

    # 1 is within one standard error of 0.01's mean and 10 is not.
    assert mean(elem(List.keyfind(@flat, 1, 0), 1)) >= rule.threshold
    assert mean(elem(List.keyfind(@flat, 10, 0), 1)) < rule.threshold
    assert rule.chosen == 1

    for {candidate, scores} <- @flat do
      assert_in_delta rule.scores[candidate].mean, mean(scores), 1.0e-12
      assert_in_delta rule.scores[candidate].standard_error, standard_error(scores), 1.0e-12
    end
  end

  test "where the best leads by more than its standard error, the best is kept" do
    steep = [{0.001, [0.90, 0.91, 0.89, 0.90, 0.92]}, {0.01, [0.70, 0.71, 0.69, 0.70, 0.72]}, {0.1, [0.50, 0.51, 0.49, 0.52, 0.50]}]
    rule = OneStandardError.choose(steep, & &1)

    assert rule.best == 0.001
    assert rule.chosen == 0.001
  end

  test "regularization is the caller's: the smaller value is chosen when the caller ranks it more regularized" do
    rule = OneStandardError.choose(@flat, &(-&1))

    assert rule.best == 0.01
    assert rule.chosen == 0.001
  end

  test "pairs are ranked by the term the caller gives, and of one rank the higher mean wins, then the earlier candidate" do
    scored = [
      {{1, :few}, [0.80, 0.84, 0.78, 0.86, 0.82]},
      {{1, :many}, [0.81, 0.85, 0.79, 0.86, 0.83]},
      {{2, :few}, [0.80, 0.84, 0.78, 0.85, 0.82]},
      {{2, :many}, [0.80, 0.84, 0.78, 0.85, 0.82]}
    ]

    rule = OneStandardError.choose(scored, fn {strength, _bins} -> strength end)

    assert rule.best == {1, :many}
    assert rule.chosen == {2, :few}
  end

  test "equal means: the best is the more regularized" do
    rule = OneStandardError.choose([{1, [0.5, 0.6]}, {2, [0.6, 0.5]}], & &1)
    assert rule.best == 2
    assert rule.chosen == 2
  end

  test "no candidates, one fold, unequal folds or a score that is not a number raise" do
    assert_raise ArgumentError, ~r/no candidates/, fn -> OneStandardError.choose([], & &1) end
    assert_raise ArgumentError, ~r/1 fold\(s\); a standard error needs at least two/, fn -> OneStandardError.choose([{1, [0.5]}], & &1) end

    assert_raise ArgumentError, ~r/different numbers of folds/, fn ->
      OneStandardError.choose([{1, [0.5, 0.6]}, {2, [0.5, 0.6, 0.7]}], & &1)
    end

    assert_raise ArgumentError, ~r/fold score nil, which is not a number/, fn ->
      OneStandardError.choose([{1, [0.5, nil]}], & &1)
    end
  end
end
