defmodule Mix.Tasks.TrainFreeTextCapture do
  @shortdoc "Learn where free-text slot values (a reminder's text) start and end, from the annotated intent examples"

  @moduledoc """
  Fits `Brain.Analysis.FreeTextCapture` on the free-text spans annotated in
  the hand-written intent examples and writes `free_text_capture.<language>.term`
  directly under the models directory, where boot loads it.

      mix train_free_text_capture

  The examples are the rows of every intent file in
  `Brain.Corpus.Provenance.authored_dir/0` that annotates a free-text slot,
  tokenized with `Brain.ML.Tokenizer` and tagged with `Brain.ML.POSTagger` as
  capture tags an utterance. Rows are split as the intent gold standard is:
  the candidates and their strengths are chosen by cross-validation of the
  training rows, and each slot's threshold is calibrated on the held-out rows.

  Printed: each slot's intents and row and span counts, the cross-validated
  exact-span and token precision, recall and F1 of every candidate, the one
  chosen, and per slot the held-out results with the best outcome taken and
  with the threshold applied, and every held-out row with its gold and
  captured span and the confidence.

  ## Why the application is not started

  Boot loads this task's artifact and fails without it, so the task starts only
  what tagging reads (`Mix.Tasks.TrainSlotTypePreferences.start!/0`).
  """

  use Mix.Task

  alias Brain.Analysis.{FreeTextCapture, SlotDetector}
  alias Brain.ML.EvaluationStore
  alias Mix.Tasks.TrainSlotTypePreferences

  @requirements ["app.config"]

  # The annotated examples are English text.
  @language "en"

  @impl Mix.Task
  def run(args) do
    {_opts, positional, invalid} = OptionParser.parse(args, strict: [])
    if invalid != [] or positional != [], do: Mix.raise("train_free_text_capture takes no arguments, got #{inspect(args)}")

    TrainSlotTypePreferences.start!()

    examples =
      FreeTextCapture.examples!(
        registry: SlotDetector.registry(),
        gold: EvaluationStore.load_gold_standard("intent"),
        held_out: EvaluationStore.load_gold_standard("intent", :held_out),
        tagger: TrainSlotTypePreferences.tagger!()
      )

    model = FreeTextCapture.select(examples, language: @language)
    print_report(model)
    FreeTextCapture.save!(model)
    Mix.shell().info("\nFree-text capture (#{@language}) -> #{FreeTextCapture.path(@language)}")
  end

  defp print_report(model) do
    Mix.shell().info("Slots:")

    for {slot, %{intents: intents, counts: counts}} <- Enum.sort(model.slots) do
      Mix.shell().info(
        "  #{slot}: #{Enum.join(intents, ", ")}; train #{counts.train_spans} spans in #{counts.train_rows} rows, " <>
          "held out #{counts.held_out_spans} spans in #{counts.held_out_rows} rows"
      )
    end

    Mix.shell().info("\nCross-validated on the training rows (exact span P/R/F1 | token P/R/F1):")

    model.report.cross_validation
    |> Enum.sort_by(fn {{kind, smoothing}, _} -> {Enum.find_index(FreeTextCapture.kinds(), &(&1 == kind)), smoothing || 0} end)
    |> Enum.each(fn {{kind, smoothing}, counts} ->
      Mix.shell().info("  #{pad("#{kind} #{smoothing}", 22)} #{prf(counts.exact)} | #{prf(counts.tokens)}")
    end)

    Mix.shell().info("Chosen: #{model.kind}#{if model.smoothing, do: " strength #{model.smoothing}", else: ""}")

    for {slot, held} <- Enum.sort(model.report.held_out) do
      Mix.shell().info("\nHeld out, #{slot} (threshold #{Float.round(Map.fetch!(model.thresholds, slot), 4)}):")
      Mix.shell().info("  best outcome   #{prf(held.decided.exact)} | #{prf(held.decided.tokens)}")
      Mix.shell().info("  thresholded    #{prf(held.thresholded.exact)} | #{prf(held.thresholded.tokens)}")

      for row <- held.rows do
        Mix.shell().info("    #{pad(inspect(row.text), 46)} gold #{pad(inspect(row.gold), 18)} captured #{pad(inspect(row.predicted), 18)} #{Float.round(row.confidence, 4)}")
      end
    end
  end

  defp prf(counts) do
    {p, r, f} = FreeTextCapture.prf(counts)
    "#{inspect(counts)} #{Float.round(p, 3)}/#{Float.round(r, 3)}/#{Float.round(f, 3)}"
  end

  defp pad(value, width), do: value |> to_string() |> String.pad_trailing(width)
end
