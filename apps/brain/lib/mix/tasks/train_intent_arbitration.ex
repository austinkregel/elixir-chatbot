defmodule Mix.Tasks.TrainIntentArbitration do
  @moduledoc """
  Trains `Brain.Analysis.IntentArbitration`: how a disagreement between the
  `:intent_full` and `:intent_domain` classifiers is resolved, learned from
  out-of-fold predictions on the intent gold standard's training split.

      mix train_intent_arbitration --world <id>

  Run after `mix train_micro`: it reads the vectors `mix gen_micro_data` wrote
  for both classifiers (`data/classifiers/intent_full.json`,
  `intent_domain.json`), one per training row in the gold standard's order.

  The rows are folded as `Brain.Analysis.LexicalEvidence.fold_models/2` folds
  them, and for each fold the two classifiers are trained, as `mix
  train_micro` trains them, on the other folds' vectors. A row is then
  analyzed only by models fitted without it: its fold's lexical evidence and
  its fold's two classifiers.

  It runs in two stages:

  1. Before the application boots, every row's vector is classified by its
     fold's classifiers, and where they disagree the `:classifier` and
     `:domain` routes are compared with the row's gold intent. A model of
     those two routes is chosen and written. Boot loads the model and fails
     without it, so the stage gives the second its boot.
  2. The application boots, and every row is analyzed by the pipeline
     (`Brain.Analysis.Pipeline.analyze_chunk/2`, side effects off) with its
     fold's models (`:lexical_evidence`, `:intent_models`). Where the two
     classifiers disagree the pipeline records what every route gave
     (`ChunkAnalysis.intent_arbitration`), the speech-act route among them,
     and the model of all three routes is chosen and written in place of the
     first. A row whose analysis fails raises.

  The held-out split is never read.
  """

  use Mix.Task

  alias Brain.Analysis.{ActiveContexts, IntentArbitration, LexicalEvidence, Pipeline}
  alias Brain.Lexicon.IntentDomains
  alias Brain.ML.{EvaluationStore, FeatureVectorClassifier, MicroProvenance}
  alias Mix.Tasks.{TrainMicro, TrainSlotTypePreferences}

  @shortdoc "Train how intent_full and intent_domain disagreements are resolved"

  # The intent gold standard is English text.
  @gold_language "en"

  @impl Mix.Task
  def run(args) do
    {opts, _, invalid} = OptionParser.parse(args, strict: [world: :string])
    if invalid != [], do: Mix.raise("train_intent_arbitration: unknown options #{inspect(invalid)}")

    world_id =
      Keyword.get(opts, :world) ||
        Mix.raise("mix train_intent_arbitration needs --world <id>, the world the training rows are analyzed in")

    Mix.Task.run("app.config")
    TrainSlotTypePreferences.start!([:logger, :jason, :telemetry, :nx, :exla], [])

    rows = EvaluationStore.load_gold_standard("intent", :train)
    pairs = Enum.map(rows, &{&1["text"], &1["intent"]})
    full = vectors!("intent_full", rows, & &1["intent"])
    domain = vectors!("intent_domain", rows, &IntentDomains.consolidate(&1["intent"]))

    {folds, lexical} = LexicalEvidence.fold_models(pairs, LexicalEvidence.read!(@gold_language))

    Mix.shell().info("Training the two classifiers on each of #{map_size(lexical)} folds of #{length(rows)} rows...")

    models =
      Map.new(Map.keys(lexical), fn k ->
        others = fn labelled -> labelled |> Enum.zip(folds) |> Enum.reject(&(elem(&1, 1) == k)) |> Enum.map(&elem(&1, 0)) end

        {k,
         %{
           intent_full: TrainMicro.train_feature_vector_model(others.(full), "intent_full"),
           intent_domain: TrainMicro.train_feature_vector_model(others.(domain), "intent_domain")
         }}
      end)

    # Stage 1: the classifier and domain routes, from the classifications alone.
    first =
      rows
      |> Enum.zip(folds)
      |> Enum.zip(Enum.map(full, &elem(&1, 0)))
      |> Enum.flat_map(fn {{row, k}, vector} -> classified_example(row, k, vector, Map.fetch!(models, k)) end)

    {model, report} = IntentArbitration.select(first, routes: [:classifier, :domain], language: @gold_language)
    print("Stage 1 (classifications alone)", model, report)
    IntentArbitration.save!(model)

    # Stage 2: every route, from the pipeline's own analysis of each row.
    Mix.Task.run("app.start")

    Enum.each(lexical, fn {k, model} -> :persistent_term.put({__MODULE__, :lexical, k}, model) end)
    Enum.each(models, fn {k, model} -> :persistent_term.put({__MODULE__, :intent, k}, model) end)

    Mix.shell().info("Analyzing #{length(rows)} rows with their out-of-fold models...")

    second =
      rows
      |> Enum.zip(folds)
      |> Task.async_stream(&analyzed_example(&1, world_id), max_concurrency: System.schedulers_online(), timeout: :infinity, ordered: true)
      |> Enum.flat_map(fn {:ok, examples} -> examples end)

    {model, report} = IntentArbitration.select(second, routes: IntentArbitration.routes(), language: @gold_language)
    print("Stage 2 (the pipeline's routes)", model, report)
    IntentArbitration.save!(model)
    Mix.shell().info("\nIntent arbitration (#{@gold_language}) -> #{IntentArbitration.path(@gold_language)}")
  end

  # The training records of `name`, one per gold row in order, as `{vector,
  # label}`; a record that is not the row's, by count or label, raises.
  defp vectors!(name, rows, label_of) do
    path = MicroProvenance.training_data_path(name)
    records = path |> File.read!() |> Jason.decode!()

    if length(records) != length(rows) do
      Mix.raise("train_intent_arbitration: #{path} holds #{length(records)} records for #{length(rows)} training rows; regenerate it with `mix gen_micro_data --world <id>`")
    end

    records
    |> Enum.zip(rows)
    |> Enum.map(fn {%{"feature_vector" => [_ | _] = vector, "label" => label}, row} ->
      unless label == label_of.(row) do
        Mix.raise("train_intent_arbitration: #{path} labels #{inspect(row["text"])} #{inspect(label)}, not #{inspect(label_of.(row))}; regenerate it")
      end

      {vector, label}
    end)
  end

  # A row's example when its fold's classifiers disagree on its vector.
  defp classified_example(row, k, vector, %{intent_full: full, intent_domain: domain}) do
    {:ok, label, _score, %{top_k: top_k}} = FeatureVectorClassifier.classify(vector, full)
    {:ok, profile_domain, _score, _details} = FeatureVectorClassifier.classify(vector, domain)

    if Pipeline.registered_intent?(label) and IntentDomains.consolidate(label) != profile_domain do
      in_domain = Enum.find_value(top_k, fn {candidate, _} -> if Pipeline.registered_intent?(candidate) and IntentDomains.consolidate(candidate) == profile_domain, do: candidate end)

      [
        %{
          fold: k,
          classifier_domain: IntentDomains.consolidate(label),
          profile_domain: profile_domain,
          right: %{classifier: label == row["intent"], domain: in_domain == row["intent"]}
        }
      ]
    else
      []
    end
  end

  # A row's example when the pipeline, with the row's fold models, recorded a
  # disagreement.
  defp analyzed_example({row, k}, world_id) do
    analysis =
      Pipeline.analyze_chunk(row["text"],
        side_effects: false,
        world_id: world_id,
        active_contexts: ActiveContexts.for_row(row),
        lexical_evidence: :persistent_term.get({__MODULE__, :lexical, k}),
        intent_models: :persistent_term.get({__MODULE__, :intent, k})
      )

    case analysis.intent_arbitration do
      nil ->
        []

      %{classifier_domain: c, profile_domain: p, routes: routes} ->
        [%{fold: k, classifier_domain: c, profile_domain: p, right: Map.new(routes, fn {route, intent} -> {route, intent == row["intent"]} end)}]
    end
  end

  defp print(stage, model, report) do
    selection = model.selection

    Mix.shell().info("\n#{stage}: #{report.examples} disagreements over folds #{inspect(report.folds)}")

    report.candidates
    |> Enum.sort_by(fn {{granularity, strength}, _} -> {granularity, strength} end)
    |> Enum.each(fn {{granularity, strength}, stats} ->
      Mix.shell().info("  #{granularity} k=#{strength}: fold mean #{pct(stats.mean)} +- #{pct(stats.standard_error)}")
    end)

    Mix.shell().info(
      "One-standard-error rule: best #{inspect(selection.best)} at #{pct(selection.mean)} +- #{pct(selection.standard_error)}, " <>
        "threshold #{pct(selection.threshold)}, chosen #{inspect(selection.chosen)}"
    )

    Mix.shell().info("Overall rates: " <> Enum.map_join(model.routes, ", ", &"#{&1} #{pct(Map.fetch!(model.overall.right, &1) / model.overall.n)}"))

    for {group, counts} <- Enum.sort_by(model.groups, &(-elem(&1, 1).n)) do
      Mix.shell().info("  #{inspect(group)} n #{counts.n}: " <> Enum.map_join(model.routes, " ", &"#{&1} #{Map.fetch!(counts.right, &1)}"))
    end
  end

  defp pct(x), do: "#{Float.round(x * 100, 2)}%"
end
