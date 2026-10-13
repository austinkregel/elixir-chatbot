defmodule Mix.Tasks.TrainSlotTypePreferences do
  @shortdoc "Learn which entity types fill each intent's slots, and the tag before each type, from the gold training splits"

  @moduledoc """
  Fits `Brain.Analysis.SlotTypePreferences` on the training splits of the NER
  and intent gold standards and writes `slot_type_preferences.<language>.term`
  directly under the models directory, where boot loads it.

      mix train_slot_type_preferences

  Every NER training row is tokenized and tagged as
  `Brain.ML.EntityExtractor` tags a sentence, its gold spans are placed with
  `Brain.ML.NerGold.spans!/1`, and it is joined by text to its intent when the
  intent row is a training row too. The smoothing is chosen by
  cross-validation on the same rows. The held-out splits are never read.

  Printed: how many rows joined, the cross-validated log-likelihood of every
  smoothing value tried and the values chosen, each domain's most probable
  labels, and each label's most likely preceding tags.

  ## Why the application is not started

  Boot loads this task's artifact and fails without it, so the task starts only
  what tagging and the type hierarchy read: WordNet, the tagger's lexical
  features and the type hierarchy. It reads no world and no database.
  """

  use Mix.Task

  alias Brain.Analysis.{SlotTypePreferences, TypeHierarchy}
  alias Brain.ML.{EntityExtractor, EvaluationStore, POSTagger}

  @requirements ["app.config"]

  # The gold standards are English text.
  @gold_language "en"

  # What tagging and the type hierarchy read, in the order `Brain.Application`
  # starts them.
  @children [
    Brain.ML.Lexicon,
    Brain.ML.POSTagger.LexicalFeatures,
    Brain.Analysis.TypeHierarchy
  ]

  @impl Mix.Task
  def run(args) do
    {_opts, _, invalid} = OptionParser.parse(args, strict: [])
    if invalid != [], do: Mix.raise("train_slot_type_preferences: unknown options #{inspect(invalid)}")

    start!()

    examples = training_examples()
    joined = Enum.count(examples, &(&1.intent != nil))
    Mix.shell().info("#{length(examples)} NER training rows, #{joined} joined to an intent training row")

    {model, report} =
      SlotTypePreferences.select(examples,
        language: @gold_language,
        canonical: EntityExtractor.load_type_mappings(),
        hierarchy: TypeHierarchy,
        listed: SlotTypePreferences.listed_types(),
        tag_vocabulary: tag_vocabulary()
      )

    print_report(model, report)
    SlotTypePreferences.save!(model)
    Mix.shell().info("\nSlot type preferences (#{@gold_language}) -> #{SlotTypePreferences.path(@gold_language)}")
  end

  @doc false
  # The training examples, as `run/1` fits them. Public for evaluation scripts.
  def training_examples do
    intents = "intent" |> EvaluationStore.load_gold_standard(:train) |> SlotTypePreferences.intents_by_text!()
    tag = tagger!()

    "ner"
    |> EvaluationStore.load_gold_standard(:train)
    |> Enum.map(fn row -> SlotTypePreferences.example!(row, Map.get(intents, row["text"]), tag) end)
  end

  @doc false
  # Every UD tag the tagger emits, and the marker for a span that starts the text.
  def tag_vocabulary, do: POSTagger.valid_tags() ++ [SlotTypePreferences.sentence_start()]

  @doc false
  # Tags token texts as `Brain.ML.EntityExtractor` does.
  def tagger! do
    case POSTagger.load_model() do
      {:ok, model} -> fn texts -> POSTagger.predict_tags(texts, model) end
      {:error, reason} -> Mix.raise("train_slot_type_preferences: the POS model is required: #{inspect(reason)}")
    end
  end

  @doc false
  def start! do
    for app <- [:logger, :jason, :telemetry, :nx, :exla] do
      {:ok, _} = Application.ensure_all_started(app)
    end

    case Supervisor.start_link(@children, strategy: :one_for_one) do
      {:ok, _pid} -> :ok
      other -> Mix.raise("train_slot_type_preferences: could not start what tagging reads: #{inspect(other)}")
    end
  end

  defp print_report(model, report) do
    Mix.shell().info(
      "\nCross-validated mean log-likelihood per observation " <>
        "(#{report.observations.slots} span labels, #{report.observations.tags} preceding tags):"
    )

    for key <- [:global, :domain, :intent, :hierarchy, :tag_global, :tag_type] do
      values = Enum.map_join(Map.fetch!(report, key), "  ", fn {value, ll} -> "#{value}:#{Float.round(ll, 4)}" end)
      Mix.shell().info("  #{pad(key, 11)} #{values}")
    end

    Mix.shell().info("Chosen: #{inspect(report.chosen)}")
    Mix.shell().info("Fitted: #{inspect(model.fitted)}")

    Mix.shell().info("\nMost probable labels per domain seen in training:")

    model.slot_counts.domain
    |> Map.keys()
    |> Enum.sort()
    |> Enum.each(fn domain ->
      top =
        model
        |> SlotTypePreferences.distribution({:domain, domain})
        |> Enum.sort_by(fn {label, p} -> {-p, label} end)
        |> Enum.take(6)
        |> Enum.map_join(", ", fn {label, p} -> "#{label} #{Float.round(p, 3)}" end)

      Mix.shell().info("  #{pad(domain, 14)} #{top}")
    end)

    Mix.shell().info("\nMost likely preceding tags per gold label seen in training:")

    model.tag_counts.labels
    |> Map.keys()
    |> Enum.sort()
    |> Kernel.++([:ordinary])
    |> Enum.each(fn label ->
      table = if label == :ordinary, do: model.tag_likelihoods.ordinary, else: model.tag_likelihoods.labels[label]

      top =
        table
        |> Enum.sort_by(fn {tag, p} -> {-p, to_string(tag)} end)
        |> Enum.take(4)
        |> Enum.map_join(", ", fn {tag, p} -> "#{tag} #{Float.round(p, 3)}" end)

      Mix.shell().info("  #{pad(label, 16)} #{top}")
    end)
  end

  defp pad(value, width), do: value |> to_string() |> String.pad_trailing(width)
end
