defmodule Brain.ML.TrainingData.Catalog do
  @moduledoc """
  Single gateway to every training-data source file in the system.

  All reads and writes go through this module so that source-of-truth
  invariants (schema validation, revision logging, PubSub broadcast)
  are enforced in one place.
  """

  alias Brain.ML.TrainingData.{SourceDescriptors, Schemas, RevisionLog}

  require Logger

  # ── Read ─────────────────────────────────────────────────────────────

  @doc "List all source descriptors with live file stats."
  @spec list_sources() :: [map()]
  def list_sources do
    Enum.map(SourceDescriptors.all(), fn desc ->
      Map.merge(desc, file_stats(desc))
    end)
  end

  @doc "List sources grouped by category with live stats."
  @spec list_sources_by_category() :: [{String.t(), [map()]}]
  def list_sources_by_category do
    SourceDescriptors.by_category()
    |> Enum.map(fn {cat, descs} ->
      {cat, Enum.map(descs, fn d -> Map.merge(d, file_stats(d)) end)}
    end)
  end

  @doc "Read all records from a source file. Returns `{:ok, records}` or `{:error, reason}`."
  @spec read_source(atom()) :: {:ok, list() | map()} | {:error, term()}
  def read_source(source_id) do
    case SourceDescriptors.get(source_id) do
      nil -> {:error, :unknown_source}
      desc -> read_file(desc)
    end
  end

  @doc "Read records with pagination. Returns `{:ok, page, total}`."
  @spec read_source_page(atom(), non_neg_integer(), non_neg_integer(), keyword()) ::
          {:ok, list(), non_neg_integer()} | {:error, term()}
  def read_source_page(source_id, offset \\ 0, limit \\ 50, opts \\ []) do
    filter = Keyword.get(opts, :filter, "")

    case read_source(source_id) do
      {:ok, records} when is_list(records) ->
        filtered =
          if filter == "" do
            records
          else
            down = String.downcase(filter)
            Enum.filter(records, fn rec -> record_matches?(rec, down) end)
          end

        total = length(filtered)
        page = Enum.slice(filtered, offset, limit)
        {:ok, page, total}

      {:ok, %{} = map} ->
        entries = Map.to_list(map)

        filtered =
          if filter == "" do
            entries
          else
            down = String.downcase(filter)

            Enum.filter(entries, fn {k, v} ->
              String.contains?(String.downcase(to_string(k)), down) or
                record_matches?(v, down)
            end)
          end

        total = length(filtered)
        page = Enum.slice(filtered, offset, limit)
        {:ok, page, total}

      err ->
        err
    end
  end

  @doc "Compute class/label distribution for a source."
  @spec class_distribution(atom()) :: {:ok, map()} | {:error, term()}
  def class_distribution(source_id) do
    desc = SourceDescriptors.get(source_id)

    case desc do
      nil ->
        {:error, :unknown_source}

      %{record_kind: kind} when kind in [:registry_entry, :slot_schema_entry, :speech_act_map_entry, :entity_type_entry, :csv_row] ->
        {:error, :not_applicable}

      _ ->
        case read_source(source_id) do
          {:ok, records} when is_list(records) ->
            dist =
              records
              |> Enum.frequencies_by(fn rec -> label_for_record(rec, desc) end)
              |> Enum.sort_by(fn {_, count} -> -count end)

            {:ok, Map.new(dist)}

          {:ok, %{} = map} ->
            {:ok, %{"entries" => map_size(map)}}

          err ->
            err
        end
    end
  end

  @doc "Get record count for a source."
  @spec record_count(atom()) :: non_neg_integer()
  def record_count(source_id) do
    case read_source(source_id) do
      {:ok, records} when is_list(records) -> length(records)
      {:ok, %{} = map} -> map_size(map)
      _ -> 0
    end
  end

  # ── Write (Phase 2 — stub for now) ──────────────────────────────────

  @doc "Write records to a source file with validation and revision logging."
  @spec write_source(atom(), list() | map()) :: :ok | {:error, term()}
  def write_source(source_id, records) do
    case SourceDescriptors.get(source_id) do
      nil ->
        {:error, :unknown_source}

      desc ->
        if not SourceDescriptors.editable?(desc) do
          {:error, :read_only}
        else
          with :ok <- Schemas.validate_all(desc.record_kind, records) do
            before_hash =
              case read_source(source_id) do
                {:ok, old} -> RevisionLog.content_hash(old)
                _ -> "none"
              end

            case do_write(desc, records) do
              :ok ->
                after_hash = RevisionLog.content_hash(records)

                record_count =
                  case records do
                    l when is_list(l) -> length(l)
                    m when is_map(m) -> map_size(m)
                    _ -> 0
                  end

                RevisionLog.log(source_id, :write, before_hash, after_hash, %{
                  record_count: record_count
                })

                :ok

              err ->
                err
            end
          end
        end
    end
  end

  @doc "Add a single record to a list-based source."
  @spec add_record(atom(), map()) :: :ok | {:error, term()}
  def add_record(source_id, record) do
    desc = SourceDescriptors.get(source_id)

    cond do
      is_nil(desc) -> {:error, :unknown_source}
      not SourceDescriptors.editable?(desc) -> {:error, :read_only}
      true ->
        with :ok <- Schemas.validate(desc.record_kind, record) do
          case read_source(source_id) do
            {:ok, records} when is_list(records) ->
              write_source(source_id, records ++ [record])

            {:ok, %{} = map} when desc.record_kind in [:registry_entry, :slot_schema_entry, :speech_act_map_entry, :entity_type_entry] ->
              {key, val} = record
              write_source(source_id, Map.put(map, key, val))

            {:error, :enoent} ->
              write_source(source_id, [record])

            err ->
              err
          end
        end
    end
  end

  @doc "Delete a record at a given index from a list-based source."
  @spec delete_record(atom(), non_neg_integer()) :: :ok | {:error, term()}
  def delete_record(source_id, index) when is_integer(index) and index >= 0 do
    desc = SourceDescriptors.get(source_id)

    cond do
      is_nil(desc) -> {:error, :unknown_source}
      not SourceDescriptors.editable?(desc) -> {:error, :read_only}
      true ->
        case read_source(source_id) do
          {:ok, records} when is_list(records) and index < length(records) ->
            write_source(source_id, List.delete_at(records, index))

          {:ok, _} ->
            {:error, :index_out_of_range}

          err ->
            err
        end
    end
  end

  @doc "Update a record at a given index in a list-based source."
  @spec update_record(atom(), non_neg_integer(), map()) :: :ok | {:error, term()}
  def update_record(source_id, index, new_record) when is_integer(index) and index >= 0 do
    desc = SourceDescriptors.get(source_id)

    cond do
      is_nil(desc) -> {:error, :unknown_source}
      not SourceDescriptors.editable?(desc) -> {:error, :read_only}
      true ->
        with :ok <- Schemas.validate(desc.record_kind, new_record) do
          case read_source(source_id) do
            {:ok, records} when is_list(records) and index < length(records) ->
              write_source(source_id, List.replace_at(records, index, new_record))

            {:ok, _} ->
              {:error, :index_out_of_range}

            err ->
              err
          end
        end
    end
  end

  # ── Private ─────────────────────────────────────────────────────────

  defp read_file(%{path: path, record_kind: :csv_row}) do
    case File.read(path) do
      {:ok, content} ->
        lines = String.split(content, "\n", trim: true)
        {:ok, Enum.map(lines, fn l -> %{"line" => l} end)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp read_file(%{path: path}) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} -> {:ok, data}
          {:error, reason} -> {:error, {:json_decode, reason}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp do_write(desc, records) do
    path = desc.path
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    tmp_path = path <> ".tmp.#{System.unique_integer([:positive])}"
    json = Jason.encode!(records, pretty: true)

    case File.write(tmp_path, json) do
      :ok ->
        File.rename!(tmp_path, path)

        Phoenix.PubSub.broadcast(
          Brain.PubSub,
          "training_data:updates",
          {:training_data, :source_updated, desc.id}
        )

        :ok

      {:error, reason} ->
        File.rm(tmp_path)
        {:error, reason}
    end
  end

  defp file_stats(%{path: path}) do
    case File.stat(path) do
      {:ok, %File.Stat{size: size, mtime: mtime}} ->
        %{
          exists: true,
          size_bytes: size,
          mtime: mtime,
          record_count: lazy_record_count(path)
        }

      {:error, _} ->
        %{exists: false, size_bytes: 0, mtime: nil, record_count: 0}
    end
  end

  defp lazy_record_count(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, list} when is_list(list) -> length(list)
          {:ok, map} when is_map(map) -> map_size(map)
          _ -> 0
        end

      _ ->
        0
    end
  end

  defp label_for_record(rec, %{record_kind: :intent_example}) do
    Map.get(rec, "intent") || Map.get(rec, "speech_act") || Map.get(rec, "sentiment") || "unknown"
  end

  defp label_for_record(rec, %{record_kind: kind}) when kind in [:text_classifier_row, :fv_classifier_row] do
    Map.get(rec, "label") || "unknown"
  end

  defp label_for_record(rec, %{record_kind: :kg_negative}) do
    Map.get(rec, "relation") || "unknown"
  end

  defp label_for_record(_rec, _desc), do: "entry"

  defp record_matches?(rec, filter) when is_map(rec) do
    rec
    |> Map.values()
    |> Enum.any?(fn
      v when is_binary(v) -> String.contains?(String.downcase(v), filter)
      _ -> false
    end)
  end

  defp record_matches?(_, _), do: false
end
