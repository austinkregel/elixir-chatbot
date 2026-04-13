defmodule Brain.ML.Evaluation do
  @moduledoc "ML model evaluation framework.\n\nComputes precision, recall, F1, confusion matrices, and accuracy metrics\nfor classification tasks. All computation is pure Elixir (no Nx dependency).\n\n## Usage\n\n    predictions = [\"weather.query\", \"smalltalk.greetings\", \"weather.query\"]\n    actuals = [\"weather.query\", \"smalltalk.greetings\", \"music.play\"]\n\n    cm = Evaluation.confusion_matrix(predictions, actuals)\n    report = Evaluation.classification_report(cm)\n    Evaluation.format_report(report) |> IO.puts()\n"

  @doc "Compute a confusion matrix from parallel lists of predictions and actuals.\n\nReturns `%{actual_label => %{predicted_label => count}}`.\n"
  def confusion_matrix(predictions, actuals) when length(predictions) == length(actuals) do
    Enum.zip(actuals, predictions)
    |> Enum.reduce(%{}, fn {actual, predicted}, acc ->
      acc
      |> Map.update(actual, %{predicted => 1}, fn row ->
        Map.update(row, predicted, 1, &(&1 + 1))
      end)
    end)
  end

  @doc "Compute per-class precision, recall, F1, and support from a confusion matrix.\n\nReturns a map of `%{label => %{precision: float, recall: float, f1: float, support: int}}`.\n"
  def classification_report(cm) do
    all_labels =
      (Map.keys(cm) ++ Enum.flat_map(Map.values(cm), &Map.keys/1))
      |> Enum.uniq()
      |> Enum.sort()

    Enum.into(all_labels, %{}, fn label ->
      tp = get_in(cm, [label, label]) || 0
      support = cm |> Map.get(label, %{}) |> Map.values() |> Enum.sum()

      total_predicted =
        Enum.reduce(cm, 0, fn {_actual, row}, acc ->
          acc + Map.get(row, label, 0)
        end)

      precision =
        if total_predicted > 0 do
          tp / total_predicted
        else
          0.0
        end

      recall =
        if support > 0 do
          tp / support
        else
          0.0
        end

      f1 =
        if precision + recall > 0 do
          2 * precision * recall / (precision + recall)
        else
          0.0
        end

      {label, %{precision: precision, recall: recall, f1: f1, support: support}}
    end)
  end

  @doc "Compute overall accuracy from predictions and actuals.\n"
  def accuracy(predictions, actuals) when length(predictions) == length(actuals) do
    total = length(predictions)

    if total == 0 do
      0.0
    else
      correct =
        Enum.zip(predictions, actuals)
        |> Enum.count(fn {p, a} -> p == a end)

      correct / total
    end
  end

  @doc "Compute macro-averaged F1 (unweighted mean of per-class F1).\n"
  def macro_f1(report) when is_map(report) do
    f1_values = Enum.map(report, fn {_label, metrics} -> metrics.f1 end)

    if f1_values != [] do
      Enum.sum(f1_values) / length(f1_values)
    else
      0.0
    end
  end

  @doc "Compute weighted-average F1 (weighted by support).\n"
  def weighted_f1(report) when is_map(report) do
    total_support = Enum.reduce(report, 0, fn {_label, m}, acc -> acc + m.support end)

    if total_support > 0 do
      Enum.reduce(report, 0.0, fn {_label, m}, acc ->
        acc + m.f1 * m.support
      end) / total_support
    else
      0.0
    end
  end

  @doc "Format a classification report as a printable table string.\n"
  def format_report(report) when is_map(report) do
    sorted = Enum.sort_by(report, fn {label, _} -> label end)

    header =
      String.pad_trailing("Label", 30) <>
        String.pad_trailing("Precision", 12) <>
        String.pad_trailing("Recall", 12) <>
        String.pad_trailing("F1", 12) <>
        String.pad_trailing("Support", 10)

    separator = String.duplicate("-", 76)

    rows =
      Enum.map(sorted, fn {label, m} ->
        String.pad_trailing(to_string(label), 30) <>
          String.pad_trailing(format_pct(m.precision), 12) <>
          String.pad_trailing(format_pct(m.recall), 12) <>
          String.pad_trailing(format_pct(m.f1), 12) <>
          String.pad_trailing(to_string(m.support), 10)
      end)

    macro = macro_f1(report)
    weighted = weighted_f1(report)
    total_support = Enum.reduce(report, 0, fn {_l, m}, acc -> acc + m.support end)

    aggregates = [
      separator,
      String.pad_trailing("macro avg", 30) <>
        String.pad_trailing("", 12) <>
        String.pad_trailing("", 12) <>
        String.pad_trailing(format_pct(macro), 12) <>
        String.pad_trailing(to_string(total_support), 10),
      String.pad_trailing("weighted avg", 30) <>
        String.pad_trailing("", 12) <>
        String.pad_trailing("", 12) <>
        String.pad_trailing(format_pct(weighted), 12) <>
        String.pad_trailing(to_string(total_support), 10)
    ]

    Enum.join([header, separator | rows] ++ aggregates, "\n")
  end

  @doc "Build a full evaluation result map suitable for storage.\n"
  def build_result(task, predictions, actuals, opts \\ []) do
    cm = confusion_matrix(predictions, actuals)
    report = classification_report(cm)
    acc = accuracy(predictions, actuals)

    %{
      task: task,
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      accuracy: acc,
      macro_f1: macro_f1(report),
      weighted_f1: weighted_f1(report),
      per_class: serialize_report(report),
      confusion_matrix: serialize_confusion_matrix(cm),
      total_examples: length(predictions),
      notes: Keyword.get(opts, :notes)
    }
  end

  defp format_pct(val) when is_float(val) do
    "#{Float.round(val * 100, 1)}%"
  end

  defp format_pct(_) do
    "-"
  end

  defp serialize_report(report) do
    Enum.into(report, %{}, fn {label, m} ->
      {to_string(label),
       %{
         "precision" => Float.round(m.precision, 4),
         "recall" => Float.round(m.recall, 4),
         "f1" => Float.round(m.f1, 4),
         "support" => m.support
       }}
    end)
  end

  defp serialize_confusion_matrix(cm) do
    Enum.into(cm, %{}, fn {actual, row} ->
      {to_string(actual),
       Enum.into(row, %{}, fn {pred, count} ->
         {to_string(pred), count}
       end)}
    end)
  end
end