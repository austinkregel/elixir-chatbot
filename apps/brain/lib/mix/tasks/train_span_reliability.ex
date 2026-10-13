defmodule Mix.Tasks.TrainSpanReliability do
  @shortdoc "Learn how reliable each entity source is per type, from the NER training split"

  @moduledoc """
  Fits `Brain.ML.SpanReliability` on the NER gold standard's training split and
  writes `span_reliability.<language>.term` directly under the models
  directory, where boot loads it.

      mix train_span_reliability --world <id>

  For every training row, every entity source's readings are taken
  (`Brain.ML.EntityExtractor.readings/2`) and compared with the row's gold spans
  (`Brain.ML.NerGold.spans!/1`). The entity model's readings of a row come from
  its out-of-fold model (`Brain.ML.EntityTrainer.fold_models/2`, with the
  trained model's smoothing), so the model is measured on sentences it was not
  trained on, as it meets them at runtime. The prior strength and the decision
  thresholds are chosen by cross-validation on the same rows, and so are the
  thresholds a span left unresolved is decided again with once the turn's
  intent is known (`Brain.ML.SpanReliability.select_redecision/3`), on the
  training rows joined to an intent training row. The held-out splits are
  never read.

  Printed: the cross-validated F1 per prior strength and the thresholds
  chosen, every measured `{source, type}` pair with its counts and
  reliability, every pair the gold standard cannot measure, with how many
  training readings it had, and the cross-validated F1 of deciding unresolved
  spans again with no span, with the first decision's thresholds and with the
  redecision thresholds chosen.

  ## Why the application is not started

  Boot loads this task's artifact and fails without it, so the task starts only
  what extraction reads: the lexicon, the slot type preferences the gazetteer
  readings are scored with (train them first with
  `mix train_slot_type_preferences`), the gazetteer (which reads reviewed
  entities from Atlas) and the type hierarchy.

  ## Options

      --world ID   Required. The world whose gazetteer overlay extraction reads.
  """

  use Mix.Task

  alias Brain.ML.{EntityExtractor, EntityModel, EntityTrainer, EvaluationStore, NerGold, SpanReliability}

  @requirements ["app.config"]

  # The NER gold standard is English text.
  @gold_language "en"

  # What `Brain.ML.EntityExtractor.readings/2` reads, in the order
  # `Brain.Application` starts them.
  @extraction_children [
    Brain.ML.Lexicon,
    Brain.ML.POSTagger.LexicalFeatures,
    Brain.Analysis.SlotTypePreferences.Loader,
    Brain.Lexicon.Loader,
    Brain.Lexicon.UserDefined,
    Brain.ML.InformalExpansions,
    Brain.ML.Gazetteer,
    Brain.Analysis.TypeHierarchy
  ]

  @impl Mix.Task
  def run(args) do
    {opts, _, invalid} = OptionParser.parse(args, strict: [world: :string])
    if invalid != [], do: Mix.raise("train_span_reliability: unknown options #{inspect(invalid)}")

    world_id =
      Keyword.get(opts, :world) ||
        Mix.raise("mix train_span_reliability needs --world <id>, the world whose gazetteer extraction reads")

    start_extraction!()

    rows = EvaluationStore.load_gold_standard("ner", :train)
    trained = EntityModel.read!(EntityModel.trained_path(), :trained)
    {folds, fold_models} = EntityTrainer.fold_models(rows, trained)

    Mix.shell().info("Reading #{length(rows)} training rows with every entity source...")

    examples =
      rows
      |> Enum.zip(folds)
      |> Enum.map(fn {row, fold} ->
        %{
          key: row,
          readings:
            EntityExtractor.readings(row["text"], world_id: world_id, entity_model: Map.fetch!(fold_models, fold)),
          gold: NerGold.spans!(row)
        }
      end)

    canonical = EntityExtractor.load_type_mappings()

    {model, report} =
      SpanReliability.select(examples,
        language: @gold_language,
        canonical: canonical,
        hierarchy: Brain.Analysis.TypeHierarchy
      )

    print_report(model, report)

    model = select_redecision(model, rows, examples, canonical)
    print_redecision(model)

    SpanReliability.save!(model)
    Mix.shell().info("\nSpan reliability (#{@gold_language}) -> #{SpanReliability.path(@gold_language)}")
  end

  # The thresholds a span left unresolved is decided again with, once the
  # turn's intent says something about its types
  # (`SpanReliability.select_redecision/3`). The evidence is what the intent
  # says (`SlotTypePreferences.intent_evidence/3`), so only the training rows
  # joined to an intent training row take part, and each fold's preferences
  # are fitted, with the loaded model's smoothing, on every training row but
  # that fold's.
  defp select_redecision(model, rows, examples, canonical) do
    alias Brain.Analysis.SlotTypePreferences
    alias Mix.Tasks.TrainSlotTypePreferences

    intents = "intent" |> EvaluationStore.load_gold_standard(:train) |> SlotTypePreferences.intents_by_text!()
    tag = TrainSlotTypePreferences.tagger!()
    preference_examples = Map.new(rows, &{&1["text"], SlotTypePreferences.example!(&1, Map.get(intents, &1["text"]), tag)})

    fit_opts = [
      language: @gold_language,
      canonical: canonical,
      hierarchy: Brain.Analysis.TypeHierarchy,
      listed: SlotTypePreferences.listed_types(),
      tag_vocabulary: TrainSlotTypePreferences.tag_vocabulary(),
      smoothing: SlotTypePreferences.loaded!(@gold_language).smoothing
    ]

    {joined, unjoined} =
      examples
      |> Enum.map(&Map.put(&1, :intent, Map.get(intents, &1.key["text"])))
      |> Enum.split_with(&(&1.intent != nil))

    Mix.shell().info("\nChoosing the redecision thresholds on the #{length(joined)} training rows joined to an intent...")

    SpanReliability.select_redecision(model, joined,
      canonical: canonical,
      hierarchy: Brain.Analysis.TypeHierarchy,
      evidence: fn training ->
        preferences =
          (training ++ unjoined)
          |> Enum.map(&Map.fetch!(preference_examples, &1.key["text"]))
          |> SlotTypePreferences.fit(fit_opts)

        fn example, labels -> SlotTypePreferences.intent_evidence(preferences, example.intent, labels) end
      end
    )
  end

  defp print_redecision(%SpanReliability{redecision: redecision}) do
    Mix.shell().info("Cross-validated span F1 of deciding unresolved spans again with the intent's evidence:")

    for {name, r} <- [none: redecision.cross_validation.none, first_thresholds: redecision.cross_validation.first_thresholds, chosen: redecision.cross_validation.chosen] do
      Mix.shell().info(
        "  #{pad(name, 17)} F1 #{pct(r.f1)}  min_score #{fmt(r.min_score)}  min_margin #{fmt(r.min_margin)}  " <>
          "decided #{r.decided}, correct #{r.correct}, gold spans #{r.gold}"
      )
    end

    Mix.shell().info("Chosen: min_score #{fmt(redecision.min_score)}, min_margin #{fmt(redecision.min_margin)}")
  end

  defp start_extraction! do
    for app <- [:logger, :jason, :telemetry, :phoenix_pubsub, :nx, :exla, :atlas] do
      {:ok, _} = Application.ensure_all_started(app)
    end

    {:ok, _} = Phoenix.PubSub.Supervisor.start_link(name: Brain.PubSub)

    case Supervisor.start_link(@extraction_children, strategy: :one_for_one) do
      {:ok, _pid} -> :ok
      other -> Mix.raise("train_span_reliability: could not start what extraction reads: #{inspect(other)}")
    end
  end

  defp print_report(model, report) do
    Mix.shell().info("\nCross-validated span F1 by prior strength:")

    report
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.each(fn {strength, r} ->
      Mix.shell().info(
        "  #{pad(strength, 6)} F1 #{pct(r.f1)}  min_score #{fmt(r.min_score)}  min_margin #{fmt(r.min_margin)}  " <>
          "decided #{r.decided}, correct #{r.correct}, gold spans #{r.gold}"
      )
    end)

    Mix.shell().info(
      "Chosen: prior strength #{model.prior_strength}, min_score #{fmt(model.min_score)}, " <>
        "min_margin #{fmt(model.min_margin)}"
    )

    Mix.shell().info(
      "\nOverall: #{model.overall.agree} of #{model.overall.n} measured readings agree (#{pct(model.overall.rate)})"
    )

    Mix.shell().info("\nMeasured pairs (n, exact, refined, wrong type, boundary, no entity -> reliability):")

    model.pairs
    |> Enum.sort_by(fn {{source, type}, _} -> {Atom.to_string(source), type} end)
    |> Enum.each(fn {{source, type}, p} ->
      wrong = p.wrong_type |> Map.values() |> Enum.sum()

      Mix.shell().info(
        "  #{pad(source, 16)} #{pad(type, 20)} n #{pad(p.n, 5)} exact #{pad(p.exact, 5)} refined #{pad(p.refined, 5)} " <>
          "wrong #{pad(wrong, 5)} boundary #{pad(p.boundary, 5)} none #{pad(p.no_entity, 5)} -> #{Float.round(p.reliability, 3)}"
      )
    end)

    Mix.shell().info("\nUnmeasurable pairs (no gold label is the type or refines it; readings stay unresolved):")

    case Enum.sort_by(model.unmeasurable, fn {{source, type}, _} -> {Atom.to_string(source), type} end) do
      [] ->
        Mix.shell().info("  none")

      pairs ->
        Enum.each(pairs, fn {{source, type}, %{n: n}} ->
          Mix.shell().info("  #{pad(source, 16)} #{pad(type, 20)} #{n} training readings")
        end)
    end
  end

  defp pad(value, width), do: value |> to_string() |> String.pad_trailing(width)
  defp pct(rate), do: "#{Float.round(rate * 100, 1)}%"
  defp fmt(:infinity), do: "infinity"
  defp fmt(value) when is_float(value), do: Float.round(value, 3) |> to_string()
  defp fmt(value), do: to_string(value)
end
