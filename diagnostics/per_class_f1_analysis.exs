#!/usr/bin/env elixir
#
# diagnostics/per_class_f1_analysis.exs
#
# Evaluates the on-disk GA-weighted intent_domain model using a stratified
# 80/20 split of the pre-computed training data. Reports per-class precision,
# recall, F1, support, and a confusion matrix — focused on identifying
# minority class weaknesses.
#
# Run with:  mix run diagnostics/per_class_f1_analysis.exs
#            VAL_RATIO=0.3 mix run diagnostics/per_class_f1_analysis.exs

val_ratio_str = System.get_env("VAL_RATIO") || "0.2"
val_ratio = String.to_float(val_ratio_str)

alias Brain.ML.{FeatureVectorClassifier, MicroClassifiers}

training_path = "data/classifiers/intent_domain.json"

IO.puts("\n╔══════════════════════════════════════════════════════════════╗")
IO.puts("║        PER-CLASS F1 ANALYSIS — intent_domain               ║")
IO.puts("╚══════════════════════════════════════════════════════════════╝")

# ── Load training data ────────────────────────────────────────────────

IO.puts("\nLoading training data from #{training_path}…")

all_data =
  training_path
  |> File.read!()
  |> Jason.decode!()
  |> Enum.flat_map(fn
    %{"feature_vector" => v, "label" => l} when is_list(v) and is_binary(l) -> [{v, l}]
    _ -> []
  end)

IO.puts("Total examples: #{length(all_data)}")
labels = all_data |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> Enum.sort()
IO.puts("Classes (#{length(labels)}): #{Enum.join(labels, ", ")}")

# ── Stratified split ─────────────────────────────────────────────────

:rand.seed(:exsplus, {42, 137, 256})

by_class = Enum.group_by(all_data, fn {_v, l} -> l end)

{train_set, val_set} =
  Enum.reduce(by_class, {[], []}, fn {_label, examples}, {train_acc, val_acc} ->
    shuffled = Enum.shuffle(examples)
    split_at = max(round(length(shuffled) * (1 - val_ratio)), 1)
    {train, val} = Enum.split(shuffled, split_at)
    {train_acc ++ train, val_acc ++ val}
  end)

IO.puts("\nStratified split: #{length(train_set)} train / #{length(val_set)} val (ratio=#{val_ratio})")

# ── Load on-disk model (GA-weighted) ─────────────────────────────────

IO.puts("\nLoading on-disk intent_domain model…")

model_path =
  case :code.priv_dir(:brain) do
    {:error, _} -> "apps/brain/priv/ml_models/micro/intent_domain.term"
    priv -> Path.join(to_string(priv), "ml_models/micro/intent_domain.term")
  end

on_disk_model = model_path |> File.read!() |> :erlang.binary_to_term()
IO.puts("Model input_dim: #{on_disk_model.input_dim}")
IO.puts("Model has weights: #{on_disk_model.weights != nil}")

if on_disk_model.weights do
  alive = Enum.count(on_disk_model.weights, &(&1 > 0.01))
  IO.puts("Active dimensions (weight > 0.01): #{alive}/#{length(on_disk_model.weights)}")
end

# ── Also train an unweighted model for comparison ────────────────────

IO.puts("\nTraining unweighted model on same train split…")
unweighted_model = FeatureVectorClassifier.train(train_set)

# ── Classify validation set ──────────────────────────────────────────

classify_all = fn model, data ->
  Enum.map(data, fn {vec, true_label} ->
    case FeatureVectorClassifier.classify(vec, model) do
      {:ok, pred, conf, details} ->
        %{true: true_label, pred: pred, conf: conf, margin: details.margin}

      :error ->
        %{true: true_label, pred: "ERROR", conf: 0.0, margin: 0.0}
    end
  end)
end

IO.puts("\nClassifying #{length(val_set)} validation examples…")
ga_results = classify_all.(on_disk_model, val_set)
unweighted_results = classify_all.(unweighted_model, val_set)

# ── Metrics computation ──────────────────────────────────────────────

compute_metrics = fn results, class_list ->
  total = length(results)
  correct = Enum.count(results, &(&1.true == &1.pred))
  accuracy = correct / max(total, 1) * 100

  per_class =
    Enum.map(class_list, fn cls ->
      support = Enum.count(results, &(&1.true == cls))
      tp = Enum.count(results, &(&1.true == cls and &1.pred == cls))
      fp = Enum.count(results, &(&1.pred == cls and &1.true != cls))
      fn_ = Enum.count(results, &(&1.true == cls and &1.pred != cls))

      precision = if tp + fp == 0, do: 0.0, else: tp / (tp + fp)
      recall = if tp + fn_ == 0, do: 0.0, else: tp / (tp + fn_)

      f1 =
        if precision + recall == 0.0,
          do: 0.0,
          else: 2 * precision * recall / (precision + recall)

      {cls, %{support: support, tp: tp, fp: fp, fn: fn_, precision: precision, recall: recall, f1: f1}}
    end)

  macro_f1 = Enum.sum(Enum.map(per_class, fn {_, m} -> m.f1 end)) / max(length(per_class), 1)

  weighted_f1 =
    Enum.sum(Enum.map(per_class, fn {_, m} -> m.f1 * m.support end)) /
      max(Enum.sum(Enum.map(per_class, fn {_, m} -> m.support end)), 1)

  %{
    accuracy: accuracy,
    correct: correct,
    total: total,
    macro_f1: macro_f1,
    weighted_f1: weighted_f1,
    per_class: per_class
  }
end

ga_metrics = compute_metrics.(ga_results, labels)
uw_metrics = compute_metrics.(unweighted_results, labels)

# ── Print results ────────────────────────────────────────────────────

print_table = fn name, metrics ->
  IO.puts("\n══════════════════════════════════════════════════════════════")
  IO.puts("  #{name}")
  IO.puts("══════════════════════════════════════════════════════════════")
  IO.puts("  Accuracy:     #{Float.round(metrics.accuracy, 1)}% (#{metrics.correct}/#{metrics.total})")
  IO.puts("  Macro F1:     #{Float.round(metrics.macro_f1 * 100, 1)}%")
  IO.puts("  Weighted F1:  #{Float.round(metrics.weighted_f1 * 100, 1)}%")

  IO.puts("\n  " <>
    String.pad_trailing("class", 16) <>
    String.pad_leading("support", 8) <>
    String.pad_leading("prec", 7) <>
    String.pad_leading("recall", 7) <>
    String.pad_leading("F1", 7) <>
    String.pad_leading("TP", 6) <>
    String.pad_leading("FP", 6) <>
    String.pad_leading("FN", 6)
  )
  IO.puts("  " <> String.duplicate("─", 62))

  metrics.per_class
  |> Enum.sort_by(fn {_, m} -> -m.support end)
  |> Enum.each(fn {cls, m} ->
    f1_indicator = cond do
      m.f1 >= 0.7 -> " ✓"
      m.f1 >= 0.4 -> " ~"
      m.f1 > 0.0  -> " !"
      true        -> " ✗"
    end

    IO.puts("  " <>
      String.pad_trailing(cls, 16) <>
      String.pad_leading(Integer.to_string(m.support), 8) <>
      String.pad_leading(:io_lib.format("~5.1f%", [m.precision * 100]) |> IO.iodata_to_binary(), 7) <>
      String.pad_leading(:io_lib.format("~5.1f%", [m.recall * 100]) |> IO.iodata_to_binary(), 7) <>
      String.pad_leading(:io_lib.format("~5.1f%", [m.f1 * 100]) |> IO.iodata_to_binary(), 7) <>
      String.pad_leading(Integer.to_string(m.tp), 6) <>
      String.pad_leading(Integer.to_string(m.fp), 6) <>
      String.pad_leading(Integer.to_string(m.fn), 6) <>
      f1_indicator
    )
  end)
end

print_table.("GA-Weighted Model (on-disk)", ga_metrics)
print_table.("Unweighted Model (baseline)", uw_metrics)

# ── Confusion matrix for GA model ────────────────────────────────────

IO.puts("\n══════════════════════════════════════════════════════════════")
IO.puts("  Confusion Matrix — GA-Weighted (rows=true, cols=predicted)")
IO.puts("══════════════════════════════════════════════════════════════")

# Use short labels for compactness
short = fn cls ->
  case cls do
    "smarthome" -> "smart"
    "smalltalk" -> "small"
    "knowledge" -> "knowl"
    "communication" -> "comms"
    "calendar" -> "calen"
    "reminder" -> "remin"
    "account" -> "accnt"
    other -> String.slice(other, 0, 5)
  end
end

sorted_labels = Enum.sort_by(labels, fn cls ->
  -(Enum.count(ga_results, &(&1.true == cls)))
end)

header =
  "  " <>
  String.pad_trailing("true\\pred", 10) <>
  Enum.map_join(sorted_labels, "", fn l -> String.pad_leading(short.(l), 6) end)

IO.puts(header)
IO.puts("  " <> String.duplicate("─", String.length(header)))

Enum.each(sorted_labels, fn true_cls ->
  row =
    Enum.map_join(sorted_labels, "", fn pred_cls ->
      count = Enum.count(ga_results, &(&1.true == true_cls and &1.pred == pred_cls))
      cell = if count == 0, do: ".", else: Integer.to_string(count)
      String.pad_leading(cell, 6)
    end)

  IO.puts("  " <> String.pad_trailing(short.(true_cls), 10) <> row)
end)

# ── Top confusable pairs ─────────────────────────────────────────────

IO.puts("\n══════════════════════════════════════════════════════════════")
IO.puts("  Top 15 Confusable Pairs (true → predicted)")
IO.puts("══════════════════════════════════════════════════════════════")

ga_results
|> Enum.reject(&(&1.true == &1.pred))
|> Enum.group_by(fn r -> {r.true, r.pred} end)
|> Enum.map(fn {pair, rs} -> {pair, length(rs)} end)
|> Enum.sort_by(fn {_, n} -> -n end)
|> Enum.take(15)
|> Enum.each(fn {{t, p}, n} ->
  IO.puts("  #{String.pad_trailing(t, 16)} → #{String.pad_trailing(p, 16)}  n=#{n}")
end)

# ── Minority class deep dive ─────────────────────────────────────────

IO.puts("\n══════════════════════════════════════════════════════════════")
IO.puts("  Minority Class Deep Dive (classes with < 100 train examples)")
IO.puts("══════════════════════════════════════════════════════════════")

minority_classes =
  by_class
  |> Enum.filter(fn {_l, examples} -> length(examples) < 100 end)
  |> Enum.sort_by(fn {_l, examples} -> length(examples) end)
  |> Enum.map(fn {l, _} -> l end)

Enum.each(minority_classes, fn cls ->
  ga_class = Enum.filter(ga_results, &(&1.true == cls))

  if ga_class != [] do
    correct = Enum.count(ga_class, &(&1.pred == cls))
    wrong = Enum.reject(ga_class, &(&1.pred == cls))

    avg_conf_correct =
      ga_class
      |> Enum.filter(&(&1.pred == cls))
      |> Enum.map(& &1.conf)
      |> then(fn
        [] -> 0.0
        confs -> Enum.sum(confs) / length(confs)
      end)

    avg_conf_wrong =
      wrong
      |> Enum.map(& &1.conf)
      |> then(fn
        [] -> 0.0
        confs -> Enum.sum(confs) / length(confs)
      end)

    misclassified_as =
      wrong
      |> Enum.group_by(& &1.pred)
      |> Enum.map(fn {pred, rs} -> {pred, length(rs)} end)
      |> Enum.sort_by(fn {_, n} -> -n end)

    IO.puts("\n  #{cls} (#{length(ga_class)} val examples, #{length(by_class[cls])} total)")
    IO.puts("    Correct: #{correct}/#{length(ga_class)} (#{Float.round(correct / length(ga_class) * 100, 1)}%)")
    IO.puts("    Avg confidence — correct: #{Float.round(avg_conf_correct, 3)}, wrong: #{Float.round(avg_conf_wrong, 3)}")

    if misclassified_as != [] do
      IO.puts("    Misclassified as:")
      Enum.each(misclassified_as, fn {pred, n} ->
        IO.puts("      → #{pred}: #{n}")
      end)
    end
  end
end)

# ── Improvement delta table ──────────────────────────────────────────

IO.puts("\n══════════════════════════════════════════════════════════════")
IO.puts("  GA vs Unweighted — Per-Class F1 Delta")
IO.puts("══════════════════════════════════════════════════════════════")

ga_map = Map.new(ga_metrics.per_class)
uw_map = Map.new(uw_metrics.per_class)

IO.puts("  " <>
  String.pad_trailing("class", 16) <>
  String.pad_leading("GA F1", 8) <>
  String.pad_leading("UW F1", 8) <>
  String.pad_leading("Δ F1", 8) <>
  String.pad_leading("support", 8)
)
IO.puts("  " <> String.duplicate("─", 48))

labels
|> Enum.sort_by(fn cls -> -(ga_map[cls].support) end)
|> Enum.each(fn cls ->
  ga_f1 = ga_map[cls].f1
  uw_f1 = uw_map[cls].f1
  delta = ga_f1 - uw_f1

  sign = if delta >= 0, do: "+", else: ""

  IO.puts("  " <>
    String.pad_trailing(cls, 16) <>
    String.pad_leading(:io_lib.format("~5.1f%", [ga_f1 * 100]) |> IO.iodata_to_binary(), 8) <>
    String.pad_leading(:io_lib.format("~5.1f%", [uw_f1 * 100]) |> IO.iodata_to_binary(), 8) <>
    String.pad_leading("#{sign}#{:io_lib.format("~4.1f", [delta * 100]) |> IO.iodata_to_binary()}pp", 8) <>
    String.pad_leading(Integer.to_string(ga_map[cls].support), 8)
  )
end)

IO.puts("\n  Macro F1:    GA #{Float.round(ga_metrics.macro_f1 * 100, 1)}%  vs  UW #{Float.round(uw_metrics.macro_f1 * 100, 1)}%")
IO.puts("  Weighted F1: GA #{Float.round(ga_metrics.weighted_f1 * 100, 1)}%  vs  UW #{Float.round(uw_metrics.weighted_f1 * 100, 1)}%")
IO.puts("")
