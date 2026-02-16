defmodule Mix.Tasks.TrainFromGraph do
  @moduledoc """
  Run graph-to-training pipeline to integrate graph data into ML models.

  ## Usage

      mix train_from_graph              # Run all graph training updates
      mix train_from_graph --pos-only   # Just POS weight refresh
      mix train_from_graph --gazetteer  # Just gazetteer sync
      mix train_from_graph --priors     # Just intent priors extraction

  ## Options

  - `--pos-only` - Only refresh POS tagger weights from pos_graph
  - `--gazetteer` - Only sync gazetteer from knowledge_graph
  - `--priors` - Only extract intent priors from conversation_graph
  - `--blend RATIO` - Override blend ratio for POS weights (default: 0.3)
  """

  use Mix.Task

  @shortdoc "Integrate graph data into ML training models"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          pos_only: :boolean,
          gazetteer: :boolean,
          priors: :boolean,
          blend: :float
        ],
        aliases: [p: :pos_only, g: :gazetteer, i: :priors, b: :blend]
      )

    run_all = not (opts[:pos_only] || opts[:gazetteer] || opts[:priors])

    if run_all or opts[:pos_only] do
      Mix.shell().info("Refreshing POS weights from pos_graph...")
      blend = opts[:blend] || 0.3

      case Brain.Graph.Training.refresh_pos_weights(blend: blend) do
        :ok -> Mix.shell().info("  POS weights updated (blend: #{blend})")
        {:error, reason} -> Mix.shell().error("  POS weight refresh failed: #{inspect(reason)}")
      end
    end

    if run_all or opts[:gazetteer] do
      Mix.shell().info("Syncing gazetteer from knowledge_graph...")

      case Brain.Graph.Training.sync_gazetteer() do
        :ok -> Mix.shell().info("  Gazetteer sync complete")
        {:error, reason} -> Mix.shell().error("  Gazetteer sync failed: #{inspect(reason)}")
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
