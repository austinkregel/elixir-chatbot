defmodule Atlas.SemanticFactMetrics do
  @moduledoc """
  Context module for each world's semantic-fact metric counts, kept so they
  survive a restart (`Atlas.Schemas.SemanticFactMetricWindow`,
  `Atlas.Schemas.SemanticFactMetricCount`).

  Counts are made in measurement windows. A world's counts go to its open
  window, which the first count opens. `start_window!/1` ends the open window,
  keeping its counts, and opens a new one, so one window's responses can be
  compared with the next. `windows/2` reads the latest windows with their
  counts.

  The counter names are the caller's (`Brain.Metrics.Aggregator`); this
  module adds to whatever counters it is given.
  """

  import Ecto.Query

  alias Atlas.Repo
  alias Atlas.Schemas.{SemanticFactMetricCount, SemanticFactMetricWindow}

  @type window_counts :: %{window: SemanticFactMetricWindow.t(), counts: %{String.t() => pos_integer()}}

  @doc """
  Adds each amount in `counts` (`%{counter => amount}`) to that counter of the
  world's open window, opening a window first when the world has none.

  Every addition happens in the database, in one transaction, so concurrent
  calls lose no count. Raises on a blank world, an empty `counts`, a blank
  counter, or an amount that is not a positive integer, and on any database
  failure, leaving nothing counted.
  """
  @spec increment!(String.t(), %{String.t() => pos_integer()}) :: :ok
  def increment!(world_id, counts) when is_binary(world_id) and is_map(counts) do
    validate_world!(world_id, :increment!)

    if counts == %{} do
      raise ArgumentError, "Atlas.SemanticFactMetrics.increment!/2 needs a count for world #{inspect(world_id)}, got none"
    end

    Enum.each(counts, fn
      {counter, amount} when is_binary(counter) and is_integer(amount) and amount > 0 ->
        if String.trim(counter) == "" do
          raise ArgumentError, "Atlas.SemanticFactMetrics.increment!/2 needs a counter name, got #{inspect(counter)}"
        end

      other ->
        raise ArgumentError,
              "Atlas.SemanticFactMetrics.increment!/2 adds a positive integer to a named counter, got: #{inspect(other)}"
    end)

    {:ok, :ok} =
      Repo.transaction(fn ->
        window = open_window!(world_id)
        now = DateTime.utc_now()

        rows =
          Enum.map(counts, fn {counter, amount} ->
            %{
              id: Ecto.UUID.generate(),
              world_id: world_id,
              window_id: window.id,
              counter: counter,
              count: amount,
              inserted_at: now,
              updated_at: now
            }
          end)

        Repo.insert_all(SemanticFactMetricCount, rows,
          on_conflict:
            from(c in SemanticFactMetricCount,
              update: [
                set: [
                  count: fragment("? + EXCLUDED.count", c.count),
                  updated_at: fragment("EXCLUDED.updated_at")
                ]
              ]
            ),
          conflict_target: [:window_id, :counter]
        )

        :ok
      end)

    :ok
  end

  @doc """
  Ends the world's open window, keeping its counts, and opens a new one, which
  is returned. Raises on a blank world, or when another window was opened for
  the world at the same time.
  """
  @spec start_window!(String.t()) :: SemanticFactMetricWindow.t()
  def start_window!(world_id) when is_binary(world_id) do
    validate_world!(world_id, :start_window!)

    {:ok, window} =
      Repo.transaction(fn ->
        now = DateTime.utc_now()

        SemanticFactMetricWindow
        |> where([w], w.world_id == ^world_id and is_nil(w.ended_at))
        |> Repo.update_all(set: [ended_at: now])

        %SemanticFactMetricWindow{}
        |> SemanticFactMetricWindow.changeset(%{world_id: world_id, started_at: now})
        |> Repo.insert!()
      end)

    window
  end

  @doc """
  The world's latest `limit` windows, newest first, each with its counts as
  `%{counter => count}`. A world nothing was counted for and no window was
  started for has none.
  """
  @spec windows(String.t(), pos_integer()) :: [window_counts()]
  def windows(world_id, limit) when is_binary(world_id) and is_integer(limit) and limit > 0 do
    windows =
      SemanticFactMetricWindow
      |> where([w], w.world_id == ^world_id)
      |> order_by([w], desc: w.started_at)
      |> limit(^limit)
      |> Repo.all()

    window_ids = Enum.map(windows, & &1.id)

    counts =
      SemanticFactMetricCount
      |> where([c], c.window_id in ^window_ids)
      |> select([c], {c.window_id, c.counter, c.count})
      |> Repo.all()
      |> Enum.group_by(&elem(&1, 0), fn {_window_id, counter, count} -> {counter, count} end)

    # A window with no rows has counted nothing yet: one just started.
    Enum.map(windows, &%{window: &1, counts: counts |> Map.get(&1.id, []) |> Map.new()})
  end

  # The world's open window, opened now when it has none. The insert does
  # nothing when a window is open, including one opened by a concurrent
  # transaction, which it waits for; the read then sees it.
  defp open_window!(world_id) do
    Repo.insert_all(
      SemanticFactMetricWindow,
      [%{id: Ecto.UUID.generate(), world_id: world_id, started_at: DateTime.utc_now()}],
      on_conflict: :nothing,
      conflict_target: {:unsafe_fragment, "(world_id) WHERE ended_at IS NULL"}
    )

    SemanticFactMetricWindow
    |> where([w], w.world_id == ^world_id and is_nil(w.ended_at))
    |> Repo.one!()
  end

  defp validate_world!(world_id, function) do
    if String.trim(world_id) == "" do
      raise ArgumentError, "Atlas.SemanticFactMetrics.#{function} needs a world, got #{inspect(world_id)}"
    end
  end
end
