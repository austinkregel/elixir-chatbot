defmodule Brain.ML.CorpusManager do
  @moduledoc """
  Utility module for monitoring training data corpus size and composition.

  Provides corpus size tracking to keep total data under the 50 GB limit.
  """

  @max_corpus_bytes 50 * 1024 * 1024 * 1024

  @doc """
  Returns total corpus size in bytes across all data directories.
  """
  def corpus_size do
    data_dirs()
    |> Enum.map(&dir_size/1)
    |> Enum.sum()
  end

  @doc """
  Returns corpus size breakdown by category.
  """
  def size_by_category do
    %{
      training_data: dir_size(data_path()),
      ml_models: dir_size(Brain.priv_path("ml_models")),
      evaluation: dir_size(Brain.priv_path("evaluation")),
      training_worlds: dir_size(worlds_path()),
      knowledge: dir_size(Brain.priv_path("knowledge")),
      total: corpus_size()
    }
  end

  @doc """
  Check if adding `bytes` would stay under the corpus limit.
  """
  def can_add?(bytes) when is_integer(bytes) do
    corpus_size() + bytes <= @max_corpus_bytes
  end

  @doc """
  Returns the maximum corpus size in bytes.
  """
  def max_size, do: @max_corpus_bytes

  @doc """
  Returns corpus utilization as a percentage (0.0 to 100.0).
  """
  def utilization_percent do
    Float.round(corpus_size() / @max_corpus_bytes * 100, 2)
  end

  @doc """
  Format a byte count as a human-readable string.
  """
  def format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  def format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"
  def format_bytes(bytes) when bytes < 1024 * 1024 * 1024, do: "#{Float.round(bytes / (1024 * 1024), 1)} MB"
  def format_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024 * 1024), 2)} GB"

  # Private

  defp data_dirs do
    [
      data_path(),
      Brain.priv_path("ml_models"),
      Brain.priv_path("evaluation"),
      worlds_path(),
      Brain.priv_path("knowledge")
    ]
  end

  defp data_path do
    Path.join(Application.app_dir(:brain), "../../data") |> Path.expand()
  end

  defp worlds_path do
    case Application.get_env(:world, :training_worlds_path) do
      nil -> Path.join(Application.app_dir(:world, "priv"), "training_worlds")
      path -> path
    end
  rescue
    _ -> ""
  end

  defp dir_size(path) when is_binary(path) do
    if File.dir?(path) do
      path
      |> File.ls!()
      |> Enum.reduce(0, fn entry, acc ->
        full_path = Path.join(path, entry)

        case File.stat(full_path) do
          {:ok, %{type: :regular, size: size}} -> acc + size
          {:ok, %{type: :directory}} -> acc + dir_size(full_path)
          _ -> acc
        end
      end)
    else
      0
    end
  rescue
    _ -> 0
  end

  defp dir_size(_), do: 0
end
