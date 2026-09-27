defmodule Mix.Tasks.TrainFromGraph do
  @moduledoc """
  Run graph-to-training pipeline to integrate graph data into ML models.

  ## Usage

      mix train_from_graph              # Extract intent priors
      mix train_from_graph --priors     # The same; the only step left

  ## Options

  - `--priors` - Extract intent priors from conversation_graph

  Neither the gazetteer nor the POS tagger is trained from the graph. The
  gazetteer learns only what a human reviewer approved, through
  `Brain.Knowledge.ReviewQueue`; the POS tagger is a neural model trained on
  the UD English Web Treebank (`mix pos.train`), with no transition table to
  blend graph counts into.
  """

  use Mix.Task

  @shortdoc "Integrate graph data into ML training models"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {_opts, _, invalid} = OptionParser.parse(args, strict: [priors: :boolean], aliases: [i: :priors])

    # An unknown option (such as the removed --gazetteer or --pos-only) would
    # otherwise be dropped silently.
    if invalid != [], do: Mix.raise("train_from_graph: unknown options #{inspect(invalid)}")

    Mix.shell().info("Extracting intent priors from conversation_graph...")
    priors = Brain.Graph.Training.extract_intent_priors()
    Mix.shell().info("  Extracted #{map_size(priors)} intent transition entries")

    Mix.shell().info("Graph training pipeline complete.")
  end
end
