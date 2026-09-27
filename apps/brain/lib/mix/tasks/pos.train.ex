defmodule Mix.Tasks.Pos.Train do
  @shortdoc "Train the POS tagger on the UD English Web Treebank"
  @moduledoc """
  Trains, measures and saves `Brain.ML.POSTagger` from the committed EWT
  fixtures (see `Brain.Training.POS`), then prints the evaluation saved with
  the model.

      mix pos.train
      mix pos.train --out path/to/pos_model.term

  Fails, saving nothing, if the model does not beat the most-frequent-tag
  baseline on the held-out test split.
  """

  use Mix.Task

  @impl Mix.Task
  def run(args) do
    {opts, _, invalid} = OptionParser.parse(args, strict: [out: :string])
    if invalid != [], do: Mix.raise("pos.train: unknown options #{inspect(invalid)}")

    Mix.Task.run("app.config")
    {:ok, _} = Application.ensure_all_started(:exla)

    Mix.shell().info("Training the POS tagger on the EWT fixtures...")
    {path, model} = Brain.Training.POS.train_and_save!(out: opts[:out])
    print(model)
    Mix.shell().info("Saved #{path}")
  end

  defp print(%{training: t, evaluation: e}) do
    Mix.shell().info("Epochs run #{t.epochs_run}, best epoch #{t.best_epoch}, dev accuracy #{pct(t.dev_accuracy)}")
    Mix.shell().info("Test accuracy #{pct(e.accuracy)} over #{e.tokens} words (lookup baseline #{pct(e.lookup_baseline)})")
    Mix.shell().info("Unseen-word accuracy #{pct(e.oov_accuracy)} over #{e.oov_tokens} words")
    Mix.shell().info("Per tag (count, recall, precision):")

    e.per_tag
    |> Enum.sort_by(fn {_t, s} -> -s.count end)
    |> Enum.each(fn {tag, s} ->
      Mix.shell().info("  #{String.pad_trailing(tag, 6)} #{String.pad_leading("#{s.count}", 6)}  #{pct(s.recall)}  #{pct(s.precision)}")
    end)
  end

  defp pct(nil), do: "n/a"
  defp pct(x), do: "#{Float.round(x * 100, 2)}%"
end
