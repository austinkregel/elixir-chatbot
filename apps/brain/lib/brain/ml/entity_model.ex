defmodule Brain.ML.EntityModel do
  @moduledoc """
  Each world's entity model: the BIO tagger `Brain.ML.EntityTrainer` trains,
  which `Brain.ML.EntityExtractor` runs on every extraction.

  Entities are nomens, so the model belongs to a world. A world holds its own
  copy at `<world>/models/entity_model.term` (`path/1`), seeded by copying the
  trained model (`trained_path/0`, where `mix train` writes it):

  - a persistent world is seeded when `World.Manager` creates it, and an existing
    world, or every world after a retrain, with `mix world.seed_model entity_model`;
  - an ephemeral world has no directory, so it is seeded in memory when it is
    created (`seed_in_memory!/1`) and dropped with it (`drop/1`).

  `loaded!/1` reads a world's copy the first time the world extracts and keeps it
  in `:persistent_term`. A world with no copy raises naming the world: there is
  no fallback to the trained model or to another world.
  """

  alias Brain.ML.WorldModelFile

  @file_name "entity_model.term"
  @model_keys [:tag_vocabulary, :feature_weights, :feature_totals, :smoothing, :transition_weights, :tag_priors]

  @doc "Where a world keeps its copy of the entity model."
  @spec path(String.t()) :: Path.t()
  def path(world_id) when is_binary(world_id), do: WorldModelFile.path(world_id, @file_name)

  @doc "Where training writes the entity model that worlds are seeded from."
  @spec trained_path() :: Path.t()
  def trained_path, do: WorldModelFile.trained_path(@file_name)

  @doc "The world's entity model, read from its copy the first time it is asked for."
  @spec loaded!(String.t()) :: map()
  def loaded!(world_id) when is_binary(world_id) do
    case :persistent_term.get(key(world_id), nil) do
      nil ->
        model = read!(path(world_id), world_id)
        :persistent_term.put(key(world_id), model)
        model

      model ->
        model
    end
  end

  @doc """
  Copies the trained model into the world's models directory, checks the copy
  byte for byte, and makes it the world's loaded model. Replaces any copy the
  world held, which is how a retrained model reaches a world.
  """
  @spec seed!(String.t()) :: :ok
  def seed!(world_id) when is_binary(world_id) do
    model = read!(trained_path(), :trained)
    :ok = WorldModelFile.copy!(world_id, @file_name)
    :persistent_term.put(key(world_id), model)
    :ok
  end

  @doc "Makes a copy of the trained model the loaded model of a world that has no directory."
  @spec seed_in_memory!(String.t()) :: :ok
  def seed_in_memory!(world_id) when is_binary(world_id) do
    :persistent_term.put(key(world_id), read!(trained_path(), :trained))
    :ok
  end

  @doc "Forgets the world's loaded model, for a world that is destroyed."
  @spec drop(String.t()) :: :ok
  def drop(world_id) when is_binary(world_id) do
    :persistent_term.erase(key(world_id))
    :ok
  end

  @doc """
  Reads and checks an entity model file. Raises when it is missing, or does not
  hold a model of the shape `Brain.ML.EntityTrainer` trains now (a model trained
  by an earlier version has no `:feature_totals`, for one).
  """
  @spec read!(Path.t(), String.t() | :trained) :: map()
  def read!(file, owner) do
    Brain.Telemetry.measure_model_load("entity_model", fn -> read_checked!(file, owner) end)
  end

  defp read_checked!(file, owner) do
    unless File.regular?(file) do
      raise missing_message(file, owner)
    end

    model = file |> File.read!() |> :erlang.binary_to_term()

    case model do
      %{} = model when is_map_key(model, :tag_vocabulary) ->
        case Enum.reject(@model_keys, &Map.has_key?(model, &1)) do
          [] ->
            model

          missing ->
            raise "EntityModel: #{file} lacks #{inspect(missing)}; it was trained by an earlier " <>
                    "Brain.ML.EntityTrainer. Retrain with `mix train --world <id>`, then seed the worlds."
        end

      other ->
        raise "EntityModel: #{file} does not hold an entity model, got #{inspect(other, limit: 3)}"
    end
  end

  defp missing_message(file, :trained),
    do: "EntityModel: no trained entity model at #{file}; train one with `mix train --world <id>`"

  defp missing_message(file, world_id),
    do:
      "EntityModel: world #{inspect(world_id)} has no entity model at #{file}; " <>
        "seed it with `mix world.seed_model entity_model --world #{world_id}`"

  defp key(world_id), do: {__MODULE__, world_id}
end
