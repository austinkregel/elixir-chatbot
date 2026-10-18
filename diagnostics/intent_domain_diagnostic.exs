#!/usr/bin/env elixir
#
# diagnostics/intent_domain_diagnostic.exs
#
# Reads gold_standard.json, runs Pipeline.analyze_chunk + FeatureExtractor.extract
# on every entry, and feeds the resulting vector to the trained
# `:intent_domain` centroid in three configurations:
#
#   1) FULL  — the current 326-dim Tier 1+2 vector, against the
#              `intent_domain` model now on disk (trained on the same shape).
#   2) TIER1 — the same vector truncated to its first 262 dims (Tier 1 only),
#              against an in-memory centroid retrained on Tier 1 vectors only.
#   3) Baselines — majority-class and uniform-random reference numbers.
#
# Reports:
#   * accuracy + macro-F1 in each configuration
#   * full per-class confusion matrix (rows = true, cols = predicted)
#   * per-class precision / recall / support
#   * margin (cosine winner − cosine runner-up) histogram for correct
#     vs. incorrect predictions
#   * top-5 per-class confusable pairs with example texts
#
# Run with:  mix run diagnostics/intent_domain_diagnostic.exs

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

sample_size = (System.get_env("SAMPLE") || "500") |> String.to_integer()

gold_path =
  case :code.priv_dir(:brain) do
    {:error, _} -> "apps/brain/priv/evaluation/intent/gold_standard.json"
    priv -> Path.join(to_string(priv), "evaluation/intent/gold_standard.json")
  end

intent_to_domain = %{
  "account" => "account",
  "alarm" => "reminder",
  "calendar" => "calendar",
  "code" => "code",
  "communication" => "communication",
  "knowledge" => "knowledge",
  "meta" => "meta",
  "music" => "music",
  "navigation" => "navigation",
  "news" => "knowledge",
  "payment" => "account",
  "reminder" => "reminder",
  "search" => "knowledge",
  "smalltalk" => "smalltalk",
  "smarthome" => "smarthome",
  "statement" => "smalltalk",
  "timer" => "reminder",
  "todo" => "calendar",
  "weather" => "weather",
  "date" => "time",
  "time" => "time",
  "dialog" => "smalltalk",
  "display" => "smarthome",
  "status" => "meta",
  "web" => "knowledge",
  "analysis" => "knowledge"
}

alias Brain.Analysis.{FeatureExtractor, Pipeline}
alias Brain.Analysis.FeatureExtractor.ChunkFeatures
alias Brain.ML.{FeatureVectorClassifier, MicroClassifiers}

# ──────────────────────────────────────────────────────────────────────
# Load and sample gold standard
# ──────────────────────────────────────────────────────────────────────

IO.puts("\nLoading gold standard from #{gold_path}…")
gold_all = gold_path |> File.read!() |> Jason.decode!()
:rand.seed(:exsplus, {1, 2, 3})
gold = gold_all |> Enum.shuffle() |> Enum.take(sample_size)

IO.puts("Total gold examples: #{length(gold_all)}")
IO.puts("Sampled for diagnostic: #{length(gold)}")
IO.puts("Vector dimension (full): #{ChunkFeatures.vector_dimension()}")

# ──────────────────────────────────────────────────────────────────────
# Pipeline + feature extraction on every entry
# ──────────────────────────────────────────────────────────────────────

IO.puts("\nRunning Pipeline + FeatureExtractor (#{System.schedulers_online()} workers)…")
t0 = System.monotonic_time(:millisecond)

records =
  gold
  |> Task.async_stream(
    fn entry ->
      try do
        analysis = Pipeline.analyze_chunk(entry["text"])
        {full_vec, _wf} = FeatureExtractor.extract(analysis)

        raw = entry["intent"] |> String.split(".") |> List.first()
        true_domain = Map.get(intent_to_domain, raw, "other")

        %{
          text: entry["text"],
          intent: entry["intent"],
          true_domain: true_domain,
          full_vec: full_vec
        }
      rescue
        _ -> nil
      catch
        _, _ -> nil
      end
    end,
    max_concurrency: System.schedulers_online(),
    timeout: 60_000,
    on_timeout: :kill_task,
    ordered: false
  )
  |> Enum.flat_map(fn
    {:ok, nil} -> []
    {:ok, r} -> [r]
    _ -> []
  end)

IO.puts(
  "Extracted #{length(records)} feature vectors in #{System.monotonic_time(:millisecond) - t0} ms"
)

# Filter to vectors that are non-empty and consistent
records = Enum.filter(records, fn r -> is_list(r.full_vec) and r.full_vec != [] end)
full_dim = records |> List.first() |> Map.get(:full_vec) |> length()
IO.puts("Full-vector dim observed: #{full_dim}")

# ──────────────────────────────────────────────────────────────────────
# Configuration 1: FULL — query the on-disk intent_domain model
# ──────────────────────────────────────────────────────────────────────

IO.puts("\n— Configuration 1: FULL (#{full_dim}-dim, on-disk model) —")

full_results =
  Enum.map(records, fn r ->
    case MicroClassifiers.classify_vector(:intent_domain, r.full_vec) do
      {:ok, label, conf} -> Map.merge(r, %{pred: label, conf: conf})
      _ -> Map.merge(r, %{pred: "unknown", conf: 0.0})
    end
  end)

# ──────────────────────────────────────────────────────────────────────
# Configuration 2: TIER1 — truncate to first 262 dims, retrain centroids
# in-memory on the same training set, then classify.
# ──────────────────────────────────────────────────────────────────────

IO.puts("\n— Configuration 2: TIER1 (first 262 dims, in-memory retrain) —")

# Load training data file written by `mix gen_micro_data`
training_path = "data/classifiers/intent_domain.json"
IO.puts("Loading training data from #{training_path}…")

training =
  training_path
  |> File.read!()
  |> Jason.decode!()
  |> Enum.flat_map(fn
    %{"feature_vector" => v, "label" => l} when is_list(v) and is_binary(l) ->
      [{v, l}]

    _ ->
      []
  end)

IO.puts("Training records: #{length(training)}")

# Truncate every training vector to 262 dims and retrain
tier1_dim = 262

tier1_training =
  Enum.map(training, fn {v, l} ->
    {Enum.take(v, tier1_dim), l}
  end)

tier1_model = FeatureVectorClassifier.train(tier1_training)
IO.puts("Tier-1 model trained: input_dim=#{FeatureVectorClassifier.input_dim(tier1_model)}")

tier1_results =
  Enum.map(records, fn r ->
    truncated = Enum.take(r.full_vec, tier1_dim)

    case FeatureVectorClassifier.classify(truncated, tier1_model) do
      {:ok, label, conf, _details} -> Map.merge(r, %{pred: label, conf: conf})
      _ -> Map.merge(r, %{pred: "unknown", conf: 0.0})
    end
  end)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

accuracy = fn results ->
  total = length(results)
  hits = Enum.count(results, fn r -> r.pred == r.true_domain end)
  {hits, total, hits / max(total, 1) * 100}
end

per_class = fn results ->
  classes = results |> Enum.map(& &1.true_domain) |> Enum.uniq() |> Enum.sort()

  Enum.map(classes, fn cls ->
    rs = Enum.filter(results, &(&1.true_domain == cls))
    support = length(rs)
    tp = Enum.count(rs, &(&1.pred == cls))

    fp =
      Enum.count(results, fn r -> r.pred == cls and r.true_domain != cls end)

    precision = if tp + fp == 0, do: 0.0, else: tp / (tp + fp)
    recall = if support == 0, do: 0.0, else: tp / support

    f1 =
      if precision + recall == 0.0,
        do: 0.0,
        else: 2 * precision * recall / (precision + recall)

    {cls, %{support: support, tp: tp, fp: fp, precision: precision, recall: recall, f1: f1}}
  end)
end

confusion_matrix = fn results ->
  classes = results |> Enum.flat_map(fn r -> [r.true_domain, r.pred] end) |> Enum.uniq() |> Enum.sort()
  empty = for c <- classes, into: %{}, do: {c, for(c2 <- classes, into: %{}, do: {c2, 0})}

  Enum.reduce(results, empty, fn r, acc ->
    update_in(acc, [r.true_domain, r.pred], &((&1 || 0) + 1))
  end)
end

print_per_class = fn label, results ->
  IO.puts("\n#{label} — per-class precision / recall / F1 / support")
  IO.puts(String.pad_trailing("class", 16) <> "  prec  recall    f1   support")
  IO.puts(String.duplicate("─", 50))

  per_class.(results)
  |> Enum.sort_by(fn {_, m} -> -m.support end)
  |> Enum.each(fn {cls, m} ->
    IO.puts(
      String.pad_trailing(cls, 16) <>
        "  #{:io_lib.format("~5.2f", [m.precision]) |> List.to_string()}  " <>
        "#{:io_lib.format("~5.2f", [m.recall]) |> List.to_string()}  " <>
        "#{:io_lib.format("~5.2f", [m.f1]) |> List.to_string()}   #{m.support}"
    )
  end)
end

print_confusion = fn label, results ->
  IO.puts("\n#{label} — confusion matrix (rows = true, cols = predicted)")
  cm = confusion_matrix.(results)
  classes = cm |> Map.keys() |> Enum.sort()

  header = String.pad_trailing("true \\ pred", 14) <> Enum.map_join(classes, " ", &String.pad_leading(String.slice(&1, 0, 5), 5))
  IO.puts(header)
  IO.puts(String.duplicate("─", String.length(header)))

  Enum.each(classes, fn t ->
    row = Enum.map_join(classes, " ", fn p ->
      v = get_in(cm, [t, p]) || 0
      String.pad_leading(if(v == 0, do: ".", else: Integer.to_string(v)), 5)
    end)

    IO.puts(String.pad_trailing(t, 14) <> row)
  end)
end

baselines = fn results ->
  total = length(results)
  classes = results |> Enum.map(& &1.true_domain) |> Enum.frequencies()
  {majority_cls, majority_count} = Enum.max_by(classes, fn {_c, n} -> n end)

  uniform_random = 1.0 / max(map_size(classes), 1) * 100
  majority_acc = majority_count / max(total, 1) * 100

  IO.puts("Random (uniform over #{map_size(classes)} classes): #{Float.round(uniform_random, 1)}%")
  IO.puts("Majority-class baseline (\"#{majority_cls}\"):           #{Float.round(majority_acc, 1)}%")
end

margin_summary = fn label, results ->
  correct = Enum.filter(results, &(&1.pred == &1.true_domain))
  wrong = Enum.reject(results, &(&1.pred == &1.true_domain))

  avg = fn rs ->
    if rs == [], do: 0.0, else: Enum.sum(Enum.map(rs, & &1.conf)) / length(rs)
  end

  IO.puts("\n#{label} — confidence (already a margin proxy in [0,1])")
  IO.puts("  correct (n=#{length(correct)}): mean conf = #{Float.round(avg.(correct), 3)}")
  IO.puts("  wrong   (n=#{length(wrong)}): mean conf = #{Float.round(avg.(wrong), 3)}")
end

top_confusables = fn label, results, k ->
  IO.puts("\n#{label} — top #{k} confusable pairs (true → predicted, with examples)")

  pairs =
    results
    |> Enum.reject(&(&1.pred == &1.true_domain))
    |> Enum.group_by(fn r -> {r.true_domain, r.pred} end)
    |> Enum.map(fn {pair, rs} -> {pair, length(rs), Enum.take(rs, 2)} end)
    |> Enum.sort_by(fn {_, n, _} -> -n end)
    |> Enum.take(k)

  Enum.each(pairs, fn {{t, p}, n, examples} ->
    IO.puts("  #{t} → #{p} (n=#{n})")

    Enum.each(examples, fn ex ->
      IO.puts("      • \"#{ex.text}\"  (intent=#{ex.intent})")
    end)
  end)
end

# ──────────────────────────────────────────────────────────────────────
# Print everything for both configurations + baselines
# ──────────────────────────────────────────────────────────────────────

IO.puts("\n══════════════════════════════════════════════════════════════")
IO.puts("Baselines for sampled set (n=#{length(records)})")
IO.puts("══════════════════════════════════════════════════════════════")
baselines.(records)

Enum.each(
  [{"FULL (#{full_dim}-dim, on-disk model)", full_results},
   {"TIER1 (#{tier1_dim}-dim, in-memory retrain)", tier1_results}],
  fn {label, results} ->
    {hits, total, acc} = accuracy.(results)
    IO.puts("\n══════════════════════════════════════════════════════════════")
    IO.puts(label)
    IO.puts("══════════════════════════════════════════════════════════════")
    IO.puts("Accuracy: #{Float.round(acc, 1)}% (#{hits}/#{total})")
    print_per_class.(label, results)
    print_confusion.(label, results)
    margin_summary.(label, results)
    top_confusables.(label, results, 8)
  end
)

IO.puts("")
