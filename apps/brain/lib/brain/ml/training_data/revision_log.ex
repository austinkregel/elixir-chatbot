defmodule Brain.ML.TrainingData.RevisionLog do
  @moduledoc """
  Append-only JSONL log of training-data edits.

  Each entry records who changed what, when, and a hash-based summary so
  diffs can be reconstructed if needed. No GenServer — just file appends.
  """

  @log_path "data/training/_revisions.jsonl"

  @doc "Append a revision entry."
  @spec log(atom(), atom(), String.t(), String.t(), map()) :: :ok | {:error, term()}
  def log(source_id, action, before_hash, after_hash, metadata \\ %{}) do
    entry =
      %{
        ts: DateTime.utc_now() |> DateTime.to_iso8601(),
        source_id: to_string(source_id),
        action: to_string(action),
        before_hash: before_hash,
        after_hash: after_hash,
        metadata: metadata
      }
      |> Jason.encode!()

    path = log_path()
    File.mkdir_p!(Path.dirname(path))
    File.write(path, entry <> "\n", [:append])
  end

  @doc "Read recent revision entries (last N)."
  @spec recent(non_neg_integer()) :: [map()]
  def recent(n \\ 50) do
    path = log_path()

    case File.read(path) do
      {:ok, content} ->
        content
        |> String.split("\n", trim: true)
        |> Enum.reverse()
        |> Enum.take(n)
        |> Enum.flat_map(fn line ->
          case Jason.decode(line) do
            {:ok, entry} -> [entry]
            _ -> []
          end
        end)

      _ ->
        []
    end
  end

  @doc "Compute a short content hash for diffing."
  @spec content_hash(term()) :: String.t()
  def content_hash(data) do
    :crypto.hash(:sha256, Jason.encode!(data))
    |> Base.encode16(case: :lower)
    |> String.slice(0, 12)
  end

  defp log_path do
    Application.get_env(:brain, :ml)[:training_data_path]
    |> case do
      nil -> @log_path
      base -> Path.join(base, "training/_revisions.jsonl")
    end
  end
end
