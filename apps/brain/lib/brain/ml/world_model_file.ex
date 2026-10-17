defmodule Brain.ML.WorldModelFile do
  @moduledoc """
  The files behind a world-scoped model: where training writes the model worlds
  are seeded from (`trained_path/1`), where a world keeps its own copy
  (`path/2`, under `<world>/models/`), and the checked copy from one to the
  other (`copy!/2`).

  The module that owns a model (`Brain.ML.EntityModel`, `World.Embedder`) reads
  and checks the file and holds the loaded copy; this only places and copies it.
  """

  alias Brain.Analysis.RunProvenance

  @compile {:no_warn_undefined, World.Persistence}

  @doc "Where a world keeps its copy of the model saved as `file_name`."
  @spec path(String.t(), String.t()) :: Path.t()
  def path(world_id, file_name) when is_binary(world_id) and is_binary(file_name) do
    Path.join([World.Persistence.world_path(world_id), "models", file_name])
  end

  @doc "Where training writes the model saved as `file_name` that worlds are seeded from."
  @spec trained_path(String.t()) :: Path.t()
  def trained_path(file_name) when is_binary(file_name) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    Path.join(models_path, file_name)
  end

  @doc """
  Copies the trained `file_name` into the world's models directory, replacing
  any copy it held, and checks the copy byte for byte. Raises when they differ.
  """
  @spec copy!(String.t(), String.t()) :: :ok
  def copy!(world_id, file_name) when is_binary(world_id) and is_binary(file_name) do
    from = trained_path(file_name)
    to = path(world_id, file_name)

    File.mkdir_p!(Path.dirname(to))
    File.cp!(from, to)

    unless RunProvenance.sha256_file!(to) == RunProvenance.sha256_file!(from) do
      raise "WorldModelFile: the copy at #{to} does not match #{from} after seeding world #{inspect(world_id)}"
    end

    :ok
  end
end
