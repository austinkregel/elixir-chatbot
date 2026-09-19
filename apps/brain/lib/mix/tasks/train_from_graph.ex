defmodule Mix.Tasks.TrainFromGraph do
  @moduledoc """
  Run graph-to-training pipeline to integrate graph data into ML models.

  ## Usage

      mix train_from_graph              # Run all graph training updates
      mix train_from_graph --pos-only   # Just POS weight refresh
      mix train_from_graph --priors     # Just intent priors extraction

  ## Options

  - `--pos-only` - Only refresh POS tagger weights from pos_graph
  - `--priors` - Only extract intent priors from conversation_graph
  - `--blend RATIO` - Override blend ratio for POS weights (default: 0.3)

  The gazetteer is not trained from the graph: it learns only what a human
  reviewer approved, through `Brain.Knowledge.ReviewQueue`.
  """

  use Mix.Task

  @shortdoc "Integrate graph data into ML training models"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, invalid} =
      OptionParser.parse(args,
        strict: [
          pos_only: :boolean,
          priors: :boolean,
          blend: :float
        ],
        aliases: [p: :pos_only, i: :priors, b: :blend]
      )

    # An unknown option (such as the removed --gazetteer) would otherwise be
    # dropped and the task would run everything.
    if invalid != [], do: Mix.raise("train_from_graph: unknown options #{inspect(invalid)}")

    run_all = not (opts[:pos_only] || opts[:priors])

    if run_all or opts[:pos_only] do
      Mix.shell().info("Refreshing POS weights from pos_graph...")
      blend = opts[:blend] || 0.3

      case Brain.Graph.Training.refresh_pos_weights(blend: blend) do
        :ok -> Mix.shell().info("  POS weights updated (blend: #{blend})")
        {:error, reason} -> Mix.shell().error("  POS weight refresh failed: #{inspect(reason)}")
      end
    end

    if run_all or opts[:priors] do
      Mix.shell().info("Extracting intent priors from conversation_graph...")
      priors = Brain.Graph.Training.extract_intent_priors()
      count = map_size(priors)
      Mix.shell().info("  Extracted #{count} intent transition entries")
    end

    Mix.shell().info("Graph training pipeline complete.")
  end
end
