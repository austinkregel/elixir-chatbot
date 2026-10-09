defmodule Mix.Tasks.World.SeedModel do
  @shortdoc "Copy a trained model into worlds"

  @moduledoc """
  Copies a trained model into worlds' own models directories, replacing the copy
  each held. A world created by `World.Manager` is seeded when it is created;
  this seeds worlds that already exist, and is how a retrained model reaches them.

      mix world.seed_model entity_model --world default
      mix world.seed_model entity_model --world default --world ehi7tgxYguk
      mix world.seed_model entity_model --all

  `--all` seeds every world persisted under `World.Persistence.base_path/0`
  (`World.Persistence.list_persisted_worlds/0`).
  One of `--world` or `--all` is required.

  ## Models

  - `entity_model` — `Brain.ML.EntityModel`, from `Brain.ML.EntityModel.trained_path/0`.
  - `embedder` — `World.Embedder`, from `World.Embedder.trained_path/0`.

  A running server keeps the model it loaded for a world until it restarts.
  """

  use Mix.Task

  alias Brain.ML.EntityModel
  alias World.Embedder
  alias World.Persistence

  @requirements ["app.config"]

  @seeders %{"entity_model" => &EntityModel.seed!/1, "embedder" => &Embedder.seed!/1}

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: [world: :keep, all: :boolean])
    if invalid != [], do: Mix.raise("world.seed_model: unknown options #{inspect(invalid)}")

    seed! =
      case positional do
        [model] ->
          Map.get(@seeders, model) ||
            Mix.raise("world.seed_model: no model #{inspect(model)}; the models are #{inspect(Map.keys(@seeders))}")

        _ ->
          Mix.raise("world.seed_model: name one model, as in `mix world.seed_model entity_model --world default`")
      end

    worlds =
      case {Keyword.get_values(opts, :world), Keyword.get(opts, :all, false)} do
        {[], true} -> Enum.map(Persistence.list_persisted_worlds(), & &1.id)
        {[_ | _] = worlds, false} -> worlds
        {[], false} -> Mix.raise("world.seed_model: pass --world <id> (repeatable) or --all")
        {_, true} -> Mix.raise("world.seed_model: pass --world or --all, not both")
      end

    if worlds == [], do: Mix.raise("world.seed_model: no worlds under #{Persistence.base_path()}")

    Enum.each(worlds, fn world_id ->
      seed!.(world_id)
      Mix.shell().info("  seeded #{world_id}")
    end)
  end
end
