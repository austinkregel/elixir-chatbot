defmodule Atlas.Metrics do
  @moduledoc """
  Context module for metrics kept so they survive a restart
  (`Atlas.Schemas.MetricWindow`, `Atlas.Schemas.MetricValue`,
  `Atlas.Schemas.MetricRecord`).

  Metrics are written in measurement windows. A window belongs to a family of
  metrics in one scope: `{:world, world_id}` for what is measured per world,
  or `:node` for what the node measures as a whole. A family's values and
  records go to the open window of their scope, which the first write opens.
  `start_window!/2` ends the open window, keeping what it holds, and opens a
  new one, so one window can be compared with the next. `windows/3` reads the
  latest windows with their values and records.

  A value is combined with what its window holds by its aggregate: `:sum` adds
  it, `:min` keeps the smaller, `:max` the larger. A record replaces the
  window's record of the same name. The names, and what they mean, are the
  writer's (`Brain.Metrics.Aggregator`); this module stores whatever it is
  given.
  """

  import Ecto.Query

  alias Atlas.Repo
  alias Atlas.Schemas.{MetricRecord, MetricValue, MetricWindow}

  @type scope :: {:world, String.t()} | :node
  @type aggregate :: :sum | :min | :max
  @type value :: {String.t(), aggregate(), integer()}
  @type write :: %{
          required(:scope) => scope(),
          required(:family) => String.t(),
          optional(:values) => [value()],
          optional(:records) => [{String.t(), map()}]
        }
  @type window_contents :: %{
          window: MetricWindow.t(),
          values: %{String.t() => integer()},
          records: %{String.t() => %{record: map(), recorded_at: DateTime.t()}}
        }

  @aggregates [:sum, :min, :max]

  @doc """
  Writes each of `writes` to the open window of its family and scope, opening
  a window first where there is none: its `values` combined with the window's
  by their aggregates, its `records` replacing the window's of the same name.

  Every write happens in the database, in one transaction, so concurrent calls
  lose nothing. Raises, leaving nothing written, on no writes, a write with
  neither values nor records, a malformed scope or family, a blank name, a
  name given twice in one write, an unknown aggregate, a value that is not an
  integer, a `:sum` that is not positive, a record that is not a map, a name
  written with another aggregate than its window holds it by, and any
  database failure.
  """
  @spec write!([write()]) :: :ok
  def write!([_ | _] = writes) do
    Enum.each(writes, &validate_write!/1)

    {:ok, :ok} =
      Repo.transaction(fn ->
        now = DateTime.utc_now()
        Enum.each(writes, &write_one!(&1, now))
        :ok
      end)

    :ok
  end

  def write!(other) do
    raise ArgumentError, "Atlas.Metrics.write!/1 needs at least one write, got: #{inspect(other)}"
  end

  @doc """
  Ends the open window of `family` in `scope`, keeping what it holds, and
  opens a new one, which is returned. Raises on a malformed scope or family,
  or when another window was opened for them at the same time.
  """
  @spec start_window!(scope(), String.t()) :: MetricWindow.t()
  def start_window!(scope, family) do
    validate_scope!(scope)
    validate_family!(family)

    {:ok, window} =
      Repo.transaction(fn ->
        now = DateTime.utc_now()

        MetricWindow
        |> in_scope(scope)
        |> where([w], w.family == ^family and is_nil(w.ended_at))
        |> Repo.update_all(set: [ended_at: now])

        %MetricWindow{}
        |> MetricWindow.changeset(%{world_id: world_id(scope), family: family, started_at: now})
        |> Repo.insert!()
      end)

    window
  end

  @doc """
  The latest `limit` windows of `family` in `scope`, newest first, each with
  its values as `%{name => value}` and its records as
  `%{name => %{record: map, recorded_at: DateTime.t()}}`. A family nothing was
  written to and no window was started for in the scope has none.
  """
  @spec windows(scope(), String.t(), pos_integer()) :: [window_contents()]
  def windows(scope, family, limit) when is_integer(limit) and limit > 0 do
    validate_scope!(scope)
    validate_family!(family)

    windows =
      MetricWindow
      |> in_scope(scope)
      |> where([w], w.family == ^family)
      |> order_by([w], desc: w.started_at)
      |> limit(^limit)
      |> Repo.all()

    window_ids = Enum.map(windows, & &1.id)

    values =
      MetricValue
      |> where([v], v.window_id in ^window_ids)
      |> select([v], {v.window_id, v.name, v.value})
      |> Repo.all()
      |> Enum.group_by(&elem(&1, 0), fn {_window_id, name, value} -> {name, value} end)

    records =
      MetricRecord
      |> where([r], r.window_id in ^window_ids)
      |> select([r], {r.window_id, r.name, r.record, r.recorded_at})
      |> Repo.all()
      |> Enum.group_by(&elem(&1, 0), fn {_window_id, name, record, recorded_at} ->
        {name, %{record: record, recorded_at: recorded_at}}
      end)

    # A window with no rows has had nothing written to it yet: one just started.
    Enum.map(windows, fn window ->
      %{
        window: window,
        values: values |> Map.get(window.id, []) |> Map.new(),
        records: records |> Map.get(window.id, []) |> Map.new()
      }
    end)
  end

  defp write_one!(%{scope: scope, family: family} = write, now) do
    window = open_window!(scope, family)
    world_id = world_id(scope)

    case Map.get(write, :values, []) do
      [] ->
        :ok

      values ->
        rows =
          Enum.map(values, fn {name, aggregate, value} ->
            %{
              id: Ecto.UUID.generate(),
              world_id: world_id,
              window_id: window.id,
              name: name,
              aggregate: Atom.to_string(aggregate),
              value: value,
              inserted_at: now,
              updated_at: now
            }
          end)

        # A name held by another aggregate than the one written becomes NULL,
        # which the NOT NULL column refuses, failing the write.
        Repo.insert_all(MetricValue, rows,
          on_conflict:
            from(v in MetricValue,
              update: [
                set: [
                  value:
                    fragment(
                      """
                      CASE WHEN ? <> EXCLUDED.aggregate THEN NULL
                           WHEN ? = 'sum' THEN ? + EXCLUDED.value
                           WHEN ? = 'min' THEN LEAST(?, EXCLUDED.value)
                           ELSE GREATEST(?, EXCLUDED.value) END
                      """,
                      v.aggregate,
                      v.aggregate,
                      v.value,
                      v.aggregate,
                      v.value,
                      v.value
                    ),
                  updated_at: fragment("EXCLUDED.updated_at")
                ]
              ]
            ),
          conflict_target: [:window_id, :name]
        )
    end

    case Map.get(write, :records, []) do
      [] ->
        :ok

      records ->
        rows =
          Enum.map(records, fn {name, record} ->
            %{
              id: Ecto.UUID.generate(),
              world_id: world_id,
              window_id: window.id,
              name: name,
              record: record,
              recorded_at: now
            }
          end)

        Repo.insert_all(MetricRecord, rows,
          on_conflict: {:replace, [:record, :recorded_at]},
          conflict_target: [:window_id, :name]
        )
    end

    :ok
  end

  # The open window of the family in the scope, opened now when there is
  # none. The insert does nothing when a window is open, including one opened
  # by a concurrent transaction, which it waits for; the read then sees it.
  defp open_window!(scope, family) do
    Repo.insert_all(
      MetricWindow,
      [%{id: Ecto.UUID.generate(), world_id: world_id(scope), family: family, started_at: DateTime.utc_now()}],
      on_conflict: :nothing,
      conflict_target: {:unsafe_fragment, "(family, COALESCE(world_id, '')) WHERE ended_at IS NULL"}
    )

    MetricWindow
    |> in_scope(scope)
    |> where([w], w.family == ^family and is_nil(w.ended_at))
    |> Repo.one!()
  end

  defp in_scope(query, :node), do: where(query, [w], is_nil(w.world_id))
  defp in_scope(query, {:world, world_id}), do: where(query, [w], w.world_id == ^world_id)

  defp world_id(:node), do: nil
  defp world_id({:world, world_id}), do: world_id

  defp validate_write!(%{scope: scope, family: family} = write) do
    unknown = write |> Map.keys() |> Enum.reject(&(&1 in [:scope, :family, :values, :records]))

    if unknown != [] do
      raise ArgumentError, "Atlas.Metrics.write!/1 does not know #{inspect(unknown)} in #{inspect(write)}"
    end

    validate_scope!(scope)
    validate_family!(family)
    values = Map.get(write, :values, [])
    records = Map.get(write, :records, [])

    unless is_list(values) and is_list(records) and (values != [] or records != []) do
      raise ArgumentError,
            "Atlas.Metrics.write!/1 needs values or records to write to #{family} in #{inspect(scope)}, got: #{inspect(write)}"
    end

    Enum.each(values, fn
      {name, aggregate, value} when aggregate in @aggregates and is_integer(value) ->
        validate_name!(name)

        if aggregate == :sum and value <= 0 do
          raise ArgumentError,
                "Atlas.Metrics.write!/1 adds a positive integer to a sum, got #{value} for #{inspect(name)}"
        end

      other ->
        raise ArgumentError,
              "Atlas.Metrics.write!/1 writes {name, :sum | :min | :max, integer}, got: #{inspect(other)}"
    end)

    Enum.each(records, fn
      {name, record} when is_map(record) ->
        validate_name!(name)

      other ->
        raise ArgumentError, "Atlas.Metrics.write!/1 writes a record as {name, map}, got: #{inspect(other)}"
    end)

    Enum.each([{Enum.map(values, &elem(&1, 0)), "value"}, {Enum.map(records, &elem(&1, 0)), "record"}], fn
      {names, kind} ->
        case names -- Enum.uniq(names) do
          [] ->
            :ok

          duplicates ->
            raise ArgumentError,
                  "Atlas.Metrics.write!/1 writes each #{kind} name once per write, got #{inspect(Enum.uniq(duplicates))} more than once"
        end
    end)

    :ok
  end

  defp validate_write!(other) do
    raise ArgumentError, "Atlas.Metrics.write!/1 needs %{scope:, family:, values:, records:}, got: #{inspect(other)}"
  end

  defp validate_scope!(:node), do: :ok

  defp validate_scope!({:world, world_id} = scope) when is_binary(world_id) do
    if String.trim(world_id) == "" do
      raise ArgumentError, "Atlas.Metrics needs a world to scope to, got #{inspect(scope)}"
    end
  end

  defp validate_scope!(other) do
    raise ArgumentError, "Atlas.Metrics scopes to {:world, world_id} or :node, got: #{inspect(other)}"
  end

  defp validate_family!(family) when is_binary(family) do
    if String.trim(family) == "" do
      raise ArgumentError, "Atlas.Metrics needs a family of metrics, got #{inspect(family)}"
    end
  end

  defp validate_family!(other) do
    raise ArgumentError, "Atlas.Metrics needs a family of metrics, got: #{inspect(other)}"
  end

  defp validate_name!(name) when is_binary(name) do
    if String.trim(name) == "" do
      raise ArgumentError, "Atlas.Metrics.write!/1 needs a name, got #{inspect(name)}"
    end
  end

  defp validate_name!(other) do
    raise ArgumentError, "Atlas.Metrics.write!/1 needs a name, got: #{inspect(other)}"
  end
end
