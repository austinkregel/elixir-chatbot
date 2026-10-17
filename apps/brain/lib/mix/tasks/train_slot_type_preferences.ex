defmodule Mix.Tasks.TrainSlotTypePreferences do
  @shortdoc "Learn which entity types fill each intent's slots, how often a gazetteer match is an ordinary word, and the tag before each type, from the gold training splits"

  @moduledoc """
  Fits `Brain.Analysis.SlotTypePreferences` on the training splits of the NER
  and intent gold standards and writes `slot_type_preferences.<language>.term`
  directly under the models directory, where boot loads it.

      mix train_slot_type_preferences --world <id>

  Every NER training row is tokenized and tagged as
  `Brain.ML.EntityExtractor` tags a sentence, its gold spans are placed with
  `Brain.ML.NerGold.spans!/1`, it is read by the gazetteer as the extractor
  reads it -- one `Brain.ML.EntityExtractor.gazetteer_key/1` per token,
  `Brain.ML.Gazetteer.lookup_spans/2` in the world `--world` names, before any
  reading is scored -- and it is joined by text to its intent when the intent
  row is a training row too. The smoothing is chosen by cross-validation on
  the same rows. The held-out splits are never read.

  The gazetteer readings are the world's: the base gazetteer, which every
  world shares, and the entities reviewers approved in that world. Every
  entity type the gazetteer holds, and every type of the world's approved
  entities, joins the model's label universe, so a reading of any of them can
  be weighed.

  Printed: how many rows joined, the cross-validated log-likelihood of every
  smoothing value tried and the values chosen, each domain's most probable
  labels, its ordinary-word rate and the rate at which its gold spans agree
  with no type its slot schemas list, and each label's most likely preceding
  tags.

  ## Why the application is not started

  Boot loads this task's artifact and fails without it, so the task starts only
  what tagging, the gazetteer and the type hierarchy read: WordNet, the
  tagger's lexical features, the database repository the gazetteer reads
  approved entities through (the repository alone: no migration or import
  runs), the owned lexicon the gazetteer reads ordinary words from, the
  gazetteer and the type hierarchy.

  ## Options

      --world ID   Required. The world whose gazetteer the rows are read with.
  """

  use Mix.Task

  alias Brain.Analysis.{SlotDetector, SlotTypePreferences, TypeHierarchy}
  alias Brain.Lexicon.IntentDomains
  alias Brain.ML.{EntityExtractor, EvaluationStore, Gazetteer, POSTagger}

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
    {opts, _, invalid} = OptionParser.parse(args, strict: [world: :string])
    if invalid != [], do: Mix.raise("train_slot_type_preferences: unknown options #{inspect(invalid)}")

    world_id =
      Keyword.get(opts, :world) ||
        Mix.raise("mix train_slot_type_preferences needs --world <id>, the world whose gazetteer the rows are read with")

    start!()
    start_gazetteer!()

    examples = training_examples(world_id)
    joined = Enum.count(examples, &(&1.intent != nil))
    Mix.shell().info("#{length(examples)} NER training rows, #{joined} joined to an intent training row (world #{world_id})")

    {model, report} =
      SlotTypePreferences.select(examples,
        language: @gold_language,
        canonical: EntityExtractor.load_type_mappings(),
        hierarchy: TypeHierarchy,
        listed: SlotTypePreferences.listed_types(),
        reading_types: reading_types(world_id),
        tag_vocabulary: tag_vocabulary()
      )

    print_report(model, report)
    SlotTypePreferences.save!(model)
    Mix.shell().info("\nSlot type preferences (#{@gold_language}) -> #{SlotTypePreferences.path(@gold_language)}")
  end

  @doc false
  # The training examples, as `run/1` fits them. Public for evaluation scripts.
  def training_examples(world_id) do
    intents = "intent" |> EvaluationStore.load_gold_standard(:train) |> SlotTypePreferences.intents_by_text!()
    tag = tagger!()
    gazetteer = gazetteer!(world_id)

    "ner"
    |> EvaluationStore.load_gold_standard(:train)
    |> Enum.map(fn row -> SlotTypePreferences.example!(row, Map.get(intents, row["text"]), tag, gazetteer) end)
  end

  @doc false
  # Every UD tag the tagger emits, and the marker for a span that starts the text.
  def tag_vocabulary, do: POSTagger.valid_tags() ++ [SlotTypePreferences.sentence_start()]

  @doc false
  # Every entity type the gazetteer can read a span as in the world: the base
  # gazetteer's types and those of the world's approved entities.
  def reading_types(world_id) do
    base =
      case Gazetteer.list_types() do
        [_ | _] = types -> types
        [] -> Mix.raise("train_slot_type_preferences: the gazetteer holds no entity types; did it load?")
      end

    overlay =
      for {_key, infos} <- Gazetteer.get_world_overlay(world_id),
          info <- List.wrap(infos),
          is_map(info),
          type = Map.get(info, :entity_type),
          is_binary(type),
          do: type

    Enum.uniq(base ++ overlay)
  end

  @doc false
  # Tags token texts as `Brain.ML.EntityExtractor` does.
  def tagger! do
    case POSTagger.load_model() do
      {:ok, model} -> fn texts -> POSTagger.predict_tags(texts, model) end
      {:error, reason} -> Mix.raise("train_slot_type_preferences: the POS model is required: #{inspect(reason)}")
    end
  end

  @doc false
  # The gazetteer's spans over token texts as `{first, last}` token indexes,
  # found as `Brain.ML.EntityExtractor` finds them, in `world_id`.
  def gazetteer!(world_id) when is_binary(world_id) and world_id != "" do
    fn texts ->
      texts
      |> Enum.map(&EntityExtractor.gazetteer_key/1)
      |> Gazetteer.lookup_spans(world_id: world_id)
      |> Enum.map(fn {first, last, _infos} -> {first, last} end)
    end
  end

  @doc false
  def start!, do: start!([:logger, :jason, :telemetry, :nx, :exla], @children)

  @doc false
  # Starts `apps` and supervises `children`, and nothing else: a training task
  # starts only what it reads rather than the brain application, whose boot
  # loads the artifacts training tasks write. Raises naming the children when
  # they do not start.
  def start!(apps, children) when is_list(apps) and is_list(children) do
    for app <- apps do
      {:ok, _} = Application.ensure_all_started(app)
    end

    case children do
      [] ->
        :ok

      _ ->
        case Supervisor.start_link(children, strategy: :one_for_one) do
          {:ok, _pid} -> :ok
          other -> Mix.raise("could not start #{inspect(children)} for training: #{inspect(other)}")
        end
    end
  end

  @doc false
  # The gazetteer, the repository it reads approved entities through, and the
  # owned lexicon it asks which state codes are ordinary words
  # (`Brain.ML.DataLoaders.ordinary_word?/1`). The repository is started
  # alone, not the database application, so nothing migrates or imports.
  def start_gazetteer! do
    for app <- [:ecto_sql, :postgrex] do
      {:ok, _} = Application.ensure_all_started(app)
    end

    case Supervisor.start_link([Atlas.Repo, Brain.Lexicon.UserDefined, Gazetteer], strategy: :one_for_one) do
      {:ok, _pid} -> :ok
      other -> Mix.raise("train_slot_type_preferences: could not start the gazetteer: #{inspect(other)}")
    end
  end

  defp print_report(model, report) do
    Mix.shell().info(
      "\nCross-validated mean log-likelihood per observation " <>
        "(#{report.observations.slots} span labels, #{report.observations.tags} preceding tags, " <>
        "#{report.observations.gazetteer_spans} gazetteer spans):"
    )

    for key <- SlotTypePreferences.smoothing_keys() do
      values = Enum.map_join(Map.fetch!(report, key), "  ", fn {value, ll} -> "#{value}:#{Float.round(ll, 4)}" end)
      Mix.shell().info("  #{pad(key, 16)} #{values}")
    end

    Mix.shell().info("Chosen: #{inspect(report.chosen)}")
    Mix.shell().info("Fitted: #{inspect(model.fitted)}")
    Mix.shell().info("Ordinary-word rate across every utterance: #{Float.round(SlotTypePreferences.ordinary_rate(model, :global), 4)}")

    Mix.shell().info("\nPer domain seen in training: ordinary-word rate, rate agreeing with no listed type, most probable labels:")

    model.slot_counts.domain
    |> Map.keys()
    |> Enum.sort()
    |> Enum.each(fn domain ->
      context = {:domain, domain}
      # A domain of `IntentDomains.domains/0`, a fixed set, so the atom is bounded.
      listed = domain |> String.to_atom() |> SlotDetector.entity_types_for_domain() |> MapSet.to_list()

      top =
        model
        |> SlotTypePreferences.distribution(context)
        |> Enum.sort_by(fn {label, p} -> {-p, label} end)
        |> Enum.take(6)
        |> Enum.map_join(", ", fn {label, p} -> "#{label} #{Float.round(p, 3)}" end)

      Mix.shell().info(
        "  #{pad(domain, 14)} ordinary #{Float.round(SlotTypePreferences.ordinary_rate(model, context), 3)}  " <>
          "unlisted #{Float.round(SlotTypePreferences.unlisted_rate(model, context, listed), 3)}  #{top}"
      )
    end)

    Mix.shell().info("\nPer intent seen with gazetteer spans: ordinary spans / gazetteer spans -> ordinary-word rate:")

    model.ordinary_counts.intent
    |> Enum.sort()
    |> Enum.each(fn {intent, {o, m}} ->
      rate = SlotTypePreferences.ordinary_rate(model, {:intent, intent})
      Mix.shell().info("  #{pad(intent, 40)} #{pad("#{o}/#{m}", 8)} -> #{Float.round(rate, 3)}  (#{IntentDomains.consolidate(intent)})")
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
