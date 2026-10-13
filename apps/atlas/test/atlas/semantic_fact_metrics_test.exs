defmodule Atlas.SemanticFactMetricsTest do
  @moduledoc """
  The store of each world's semantic-fact metric counts: counts add up in the
  world's open window, which the first count opens; starting a new window
  keeps the ended window's counts; one world's counts and windows leave
  another's; concurrent counting loses no count and opens one window; purging
  the world deletes them; a malformed count is refused.

  The store adds to whatever counter names it is given; which names are
  counters is `Brain.Metrics.Aggregator`'s to say.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.{SemanticFactMetricCount, SemanticFactMetricWindow}
  alias Atlas.SemanticFactMetrics

  defp test_world(name), do: "test_semantic_fact_metrics_#{name}_#{System.unique_integer([:positive])}"

  describe "increment!/2 and windows/2" do
    test "the first count opens the world's window, and later counts add to it" do
      world = test_world("adds")

      assert SemanticFactMetrics.windows(world, 5) == []

      :ok = SemanticFactMetrics.increment!(world, %{"retrievals" => 1, "facts_retrieved" => 3})
      :ok = SemanticFactMetrics.increment!(world, %{"retrievals" => 1, "patterns_found" => 1})

      assert [%{window: window, counts: counts}] = SemanticFactMetrics.windows(world, 5)
      assert counts == %{"retrievals" => 2, "facts_retrieved" => 3, "patterns_found" => 1}
      assert window.world_id == world
      assert %DateTime{} = window.started_at
      assert window.ended_at == nil
    end

    test "one world's counts and windows leave another world's" do
      world_a = test_world("a")
      world_b = test_world("b")

      :ok = SemanticFactMetrics.increment!(world_a, %{"retrievals" => 2})
      :ok = SemanticFactMetrics.increment!(world_b, %{"retrievals" => 5})
      SemanticFactMetrics.start_window!(world_a)
      :ok = SemanticFactMetrics.increment!(world_a, %{"retrievals" => 1})

      assert [%{counts: %{"retrievals" => 1}}, %{counts: %{"retrievals" => 2}}] =
               SemanticFactMetrics.windows(world_a, 5)

      assert [%{counts: %{"retrievals" => 5}, window: %{ended_at: nil}}] = SemanticFactMetrics.windows(world_b, 5)
    end

    test "refuses a blank world, no counts, a blank counter, or an amount that is not a positive integer" do
      world = test_world("refused")

      for {world_id, counts} <- [
            {"", %{"retrievals" => 1}},
            {"  ", %{"retrievals" => 1}},
            {world, %{}},
            {world, %{" " => 1}},
            {world, %{"retrievals" => 0}},
            {world, %{"retrievals" => -1}},
            {world, %{"retrievals" => 1.0}},
            {world, %{retrievals: 1}}
          ] do
        assert_raise ArgumentError, fn -> SemanticFactMetrics.increment!(world_id, counts) end
      end

      assert SemanticFactMetrics.windows(world, 5) == []
    end
  end

  describe "start_window!/1" do
    test "ends the open window, keeping its counts, and counts go to the new one" do
      world = test_world("window")
      :ok = SemanticFactMetrics.increment!(world, %{"retrievals" => 4, "used:synthesizer" => 2})
      [%{window: first}] = SemanticFactMetrics.windows(world, 5)

      second = SemanticFactMetrics.start_window!(world)

      assert second.world_id == world
      assert second.ended_at == nil
      assert DateTime.compare(second.started_at, first.started_at) == :gt

      assert [%{window: ^second, counts: new_counts}, %{window: ended, counts: kept}] =
               SemanticFactMetrics.windows(world, 5)

      assert new_counts == %{}
      assert kept == %{"retrievals" => 4, "used:synthesizer" => 2}
      assert ended.id == first.id
      assert ended.started_at == first.started_at
      assert ended.ended_at == second.started_at

      :ok = SemanticFactMetrics.increment!(world, %{"retrievals" => 1})

      assert [%{counts: %{"retrievals" => 1}}, %{counts: ^kept}] = SemanticFactMetrics.windows(world, 5)
      assert [%{window: ^second}] = SemanticFactMetrics.windows(world, 1)
    end

    test "opens a window for a world nothing was counted for" do
      world = test_world("fresh")

      window = SemanticFactMetrics.start_window!(world)

      assert [%{window: ^window, counts: counts}] = SemanticFactMetrics.windows(world, 5)
      assert counts == %{}
    end

    test "refuses a blank world" do
      assert_raise ArgumentError, fn -> SemanticFactMetrics.start_window!("") end
      assert_raise ArgumentError, fn -> SemanticFactMetrics.start_window!(" ") end
    end
  end

  describe "the open window index" do
    test "refuses a second open window for the same world" do
      world = test_world("two_open")
      SemanticFactMetrics.start_window!(world)

      assert {:error, changeset} =
               %SemanticFactMetricWindow{}
               |> SemanticFactMetricWindow.changeset(%{world_id: world, started_at: DateTime.utc_now()})
               |> Repo.insert()

      assert {"has already been taken", _} = changeset.errors[:world_id]
    end
  end

  describe "concurrent increment!/2" do
    # Every task counts on a connection of its own, outside the test's sandbox
    # transaction, so the transactions really run at the same time. What they
    # write is committed, so the world's rows are deleted when the test ends.
    setup do
      world = test_world("concurrent")

      on_exit(fn ->
        :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo, sandbox: false)
        Repo.delete_all(from(c in SemanticFactMetricCount, where: c.world_id == ^world))
        Repo.delete_all(from(w in SemanticFactMetricWindow, where: w.world_id == ^world))
        Ecto.Adapters.SQL.Sandbox.checkin(Repo)
      end)

      %{world: world}
    end

    test "loses no count and opens one window", %{world: world} do
      tasks = 40

      1..tasks
      |> Task.async_stream(
        fn task ->
          :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo, sandbox: false)

          try do
            SemanticFactMetrics.increment!(world, %{"retrievals" => 1, "facts_retrieved" => task})
          after
            Ecto.Adapters.SQL.Sandbox.checkin(Repo)
          end
        end,
        max_concurrency: tasks,
        timeout: 30_000
      )
      |> Enum.each(fn result -> assert result == {:ok, :ok} end)

      assert [%{counts: counts}] = SemanticFactMetrics.windows(world, 5)
      assert counts == %{"retrievals" => tasks, "facts_retrieved" => Enum.sum(1..tasks)}
    end
  end

  describe "Atlas.Worlds.purge!/1" do
    test "deletes the world's windows and counts, and leaves another world's" do
      world = test_world("purged")
      other = test_world("kept")

      for world_id <- [world, other] do
        :ok = SemanticFactMetrics.increment!(world_id, %{"retrievals" => 1})
        SemanticFactMetrics.start_window!(world_id)
        :ok = SemanticFactMetrics.increment!(world_id, %{"retrievals" => 2})
      end

      %{rows: rows} = Atlas.Worlds.purge!(world)

      assert rows[SemanticFactMetricCount] == 2
      assert rows[SemanticFactMetricWindow] == 2
      assert SemanticFactMetrics.windows(world, 5) == []

      assert [%{counts: %{"retrievals" => 2}}, %{counts: %{"retrievals" => 1}}] =
               SemanticFactMetrics.windows(other, 5)
    end
  end
end
