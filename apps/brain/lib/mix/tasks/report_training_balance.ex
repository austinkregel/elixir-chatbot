defmodule Mix.Tasks.ReportTrainingBalance do
  @shortdoc "Report per-class training data balance across all tasks"
  @moduledoc """
  Reports per-class example counts for intent, NER, sentiment, and speech act
  gold standards, flagging classes below a minimum threshold.

  ## Usage

      mix report_training_balance              # Report all tasks (min 10)
      mix report_training_balance --min 5      # Custom minimum threshold
      mix report_training_balance --task intent # Report a single task
      mix report_training_balance --json        # Output as JSON
      mix report_training_balance --show-available  # Also show un-migrated intent sources
  """

  use Mix.Task

  alias Brain.ML.EvaluationStore

  @tasks ~w(intent ner sentiment speech_act)

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          min: :integer,
          task: :string,
          json: :boolean,
          show_available: :boolean
        ]
      )

    min_threshold = Keyword.get(opts, :min, 10)
    task_filter = Keyword.get(opts, :task)
    json? = Keyword.get(opts, :json, false)
    show_available? = Keyword.get(opts, :show_available, false)

    tasks =
      if task_filter && task_filter in @tasks do
        [task_filter]
      else
        @tasks
      end

    reports =
      Enum.map(tasks, fn task ->
        {task, build_report(task, min_threshold)}
      end)

    if json? do
      json_output =
        Enum.into(reports, %{}, fn {task, report} ->
          {task, %{
            total: report.total,
            class_count: report.class_count,
            below_threshold: length(report.below_min),
            min_threshold: min_threshold,
            classes: report.classes
          }}
        end)

      IO.puts(Jason.encode!(json_output, pretty: true))
    else
      IO.puts("\n" <> String.duplicate("=", 70))
      IO.puts("  TRAINING DATA BALANCE REPORT")
      IO.puts("  Minimum threshold: #{min_threshold} examples per class")
      IO.puts(String.duplicate("=", 70))

      for {task, report} <- reports do
        print_task_report(task, report, min_threshold)
      end

      if show_available? and "intent" in tasks do
        print_available_intents(min_threshold)
      end

      print_summary(reports, min_threshold)
    end
  end

  defp build_report("intent", min_threshold) do
    gold = EvaluationStore.load_gold_standard("intent")
    counts = Enum.frequencies_by(gold, fn ex -> ex["intent"] end)
    build_class_report(counts, min_threshold)
  end

  defp build_report("ner", min_threshold) do
    gold = EvaluationStore.load_gold_standard("ner")

    counts =
      gold
      |> Enum.flat_map(fn ex ->
        entities = ex["entities"] || ex["expected"] || []
        Enum.map(entities, fn e -> e["type"] || e["entity_type"] || "unknown" end)
      end)
      |> Enum.frequencies()

    build_class_report(counts, min_threshold)
  end

  defp build_report("sentiment", min_threshold) do
    gold = EvaluationStore.load_gold_standard("sentiment")
    counts = Enum.frequencies_by(gold, fn ex -> ex["sentiment"] end)
    build_class_report(counts, min_threshold)
  end

  defp build_report("speech_act", min_threshold) do
    gold = EvaluationStore.load_gold_standard("speech_act")
    counts = Enum.frequencies_by(gold, fn ex -> ex["speech_act"] end)
    build_class_report(counts, min_threshold)
  end

  defp build_class_report(counts, min_threshold) do
    classes =
      counts
      |> Enum.sort_by(fn {_label, count} -> count end, :desc)
      |> Enum.map(fn {label, count} ->
        %{label: label, count: count, below_min: count < min_threshold}
      end)

    below_min = Enum.filter(classes, & &1.below_min)
    total = Enum.sum(Enum.map(classes, & &1.count))

    %{
      total: total,
      class_count: length(classes),
      classes: classes,
      below_min: below_min
    }
  end

  defp print_task_report(task, report, min_threshold) do
    task_label = String.upcase(String.replace(task, "_", " "))

    IO.puts("\n  #{task_label}")
    IO.puts("  " <> String.duplicate("-", 60))
    IO.puts("  Total examples: #{report.total}  |  Classes: #{report.class_count}  |  Below min: #{length(report.below_min)}")
    IO.puts("")

    header =
      "    " <>
        String.pad_trailing("CLASS", 35) <>
        String.pad_trailing("COUNT", 10) <>
        "STATUS"

    IO.puts(header)
    IO.puts("    " <> String.duplicate("-", 55))

    for %{label: label, count: count, below_min: below?} <- report.classes do
      status = if below?, do: "  << BELOW #{min_threshold}", else: ""
      label_str = String.slice(to_string(label || "nil"), 0, 33)

      IO.puts(
        "    " <>
          String.pad_trailing(label_str, 35) <>
          String.pad_trailing(to_string(count), 10) <>
          status
      )
    end
  end

  defp print_available_intents(min_threshold) do
    IO.puts("\n  AVAILABLE (UN-MIGRATED) INTENT SOURCES")
    IO.puts("  " <> String.duplicate("-", 60))

    try do
      available = Brain.ML.GoldStandardMigrator.list_available_intents()
      gold = EvaluationStore.load_gold_standard("intent")
      gold_counts = Enum.frequencies_by(gold, fn ex -> ex["intent"] end)

      for intent <- available do
        gold_count = Map.get(gold_counts, intent.name, 0)
        source_count = intent.example_count
        gap = max(0, min_threshold - gold_count)

        if gap > 0 do
          sources = Enum.join(intent.sources, ", ")

          IO.puts(
            "    " <>
              String.pad_trailing(intent.name, 35) <>
              "gold: #{gold_count}  source: #{source_count}  gap: #{gap}  (#{sources})"
          )
        end
      end
    rescue
      _ -> IO.puts("    (Could not load available intents)")
    end
  end

  defp print_summary(reports, min_threshold) do
    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("  SUMMARY")
    IO.puts(String.duplicate("=", 70))

    header =
      "    " <>
        String.pad_trailing("TASK", 15) <>
        String.pad_trailing("TOTAL", 10) <>
        String.pad_trailing("CLASSES", 10) <>
        String.pad_trailing("BELOW MIN", 12) <>
        "COVERAGE"

    IO.puts(header)
    IO.puts("    " <> String.duplicate("-", 57))

    for {task, report} <- reports do
      above = report.class_count - length(report.below_min)
      coverage = if report.class_count > 0, do: Float.round(above / report.class_count * 100, 1), else: 0.0

      IO.puts(
        "    " <>
          String.pad_trailing(task, 15) <>
          String.pad_trailing(to_string(report.total), 10) <>
          String.pad_trailing(to_string(report.class_count), 10) <>
          String.pad_trailing(to_string(length(report.below_min)), 12) <>
          "#{coverage}%"
      )
    end

    total_below =
      reports
      |> Enum.flat_map(fn {_task, r} -> r.below_min end)
      |> length()

    if total_below > 0 do
      IO.puts("\n  #{total_below} class(es) across all tasks have fewer than #{min_threshold} examples.")
      IO.puts("  Use `mix migrate_gold_standard`, `mix download_sentiment_corpus`,")
      IO.puts("  or `mix download_speech_act_corpus` to add more training data.")
    else
      IO.puts("\n  All classes meet the minimum threshold of #{min_threshold} examples.")
    end

    IO.puts("")
  end
end
