defmodule Mix.Tasks.Evaluation.ImportResults do
  @moduledoc """
  Imports into Atlas the evaluation results saved as JSON files in
  `priv/evaluation/results/` that Atlas does not hold
  (`Brain.ML.EvaluationStore.import_result_files!/1`).

  Results are read from Atlas only, so a result saved as a file while its
  Atlas write was lost is not seen until it is imported. Running it again
  imports nothing new.

      mix evaluation.import_results
      mix evaluation.import_results --dir path/to/results
  """
  @shortdoc "Import evaluation result files Atlas does not hold"

  use Mix.Task

  @impl Mix.Task
  def run(args) do
    {opts, _positional, invalid} = OptionParser.parse(args, strict: [dir: :string])

    if invalid != [] do
      Mix.raise("Unknown options: #{inspect(invalid)}. Usage: mix evaluation.import_results [--dir DIR]")
    end

    Mix.Task.run("app.config")
    {:ok, _apps} = Application.ensure_all_started(:atlas)

    %{files: files, imported: imported, held: held} =
      case Keyword.fetch(opts, :dir) do
        {:ok, dir} -> Brain.ML.EvaluationStore.import_result_files!(dir)
        :error -> Brain.ML.EvaluationStore.import_result_files!()
      end

    Mix.shell().info("#{files} evaluation result file(s): #{imported} imported into Atlas, #{held} held there already")
  end
end
