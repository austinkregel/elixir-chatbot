defmodule Mix.Tasks.TrainFromGraph do
  @moduledoc """
  Run graph-to-training pipeline to integrate graph data into ML models.

  ## Usage

      mix train_from_graph --world default            # Extract intent priors
      mix train_from_graph --world default --priors   # The same; the only step left

  ## Options

  - `--world` - required: the world whose conversation_graph is read
  - `--priors` - Extract intent priors from the world's conversation_graph

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

    {opts, _, invalid} =
      OptionParser.parse(args, strict: [priors: :boolean, world: :string], aliases: [i: :priors])

    # An unknown option (such as the removed --gazetteer or --pos-only) would
    # otherwise be dropped silently.
    if invalid != [], do: Mix.raise("train_from_graph: unknown options #{inspect(invalid)}")

    world_id =
      Keyword.get(opts, :world) ||
        Mix.raise("train_from_graph needs --world <id>, the world whose conversation_graph is read")

    Mix.shell().info("Extracting intent priors from world #{world_id}'s conversation_graph...")
    priors = Brain.Graph.Training.extract_intent_priors(world_id)
    Mix.shell().info("  Extracted #{map_size(priors)} intent transition entries")

    Mix.shell().info("Graph training pipeline complete.")
  end
end
