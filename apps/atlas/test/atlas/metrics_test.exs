defmodule Atlas.MetricsTest do
  @moduledoc """
  The store of metrics kept across restarts: values combine by their
  aggregate in the open window of their family and scope, which the first
  write opens; records replace the window's record of their name; starting a
  new window keeps the ended window's values and records; one world's, the
  node's and another family's windows leave each other's; concurrent writes
  lose nothing and open one window; purging a world deletes its windows and
  leaves the node's; malformed writes are refused and write nothing.

  The store keeps whatever names it is given; which names are metrics is
  `Brain.Metrics.Aggregator`'s to say.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Metrics
  alias Atlas.Schemas.{MetricRecord, MetricValue, MetricWindow}

  defp test_world(name), do: "test_metrics_#{name}_#{System.unique_integer([:positive])}"

  # A family no other test or application code writes to, so the node's
  # windows a test reads are its own.
  defp test_family(name), do: "test_family_#{name}_#{System.unique_integer([:positive])}"

  defp write!(scope, family, values, records \\ []) do
    :ok = Metrics.write!([%{scope: scope, family: family, values: values, records: records}])
  end

  describe "write!/1 and windows/3" do
    test "the first write opens the window, sums add up, and min and max keep the smaller and the larger" do
      world = {:world, test_world("adds")}

      assert Metrics.windows(world, "semantic_facts", 5) == []

      write!(world, "semantic_facts", [{"retrievals", :sum, 1}, {"facts_retrieved", :sum, 3}])
      write!(world, "semantic_facts", [{"retrievals", :sum, 1}, {"duration_min_us", :min, 40}, {"duration_max_us", :max, 40}])
      write!(world, "semantic_facts", [{"duration_min_us", :min, 7}, {"duration_max_us", :max, 900}])
      write!(world, "semantic_facts", [{"duration_min_us", :min, 12}, {"duration_max_us", :max, 300}])

      assert [%{window: window, values: values, records: records}] = Metrics.windows(world, "semantic_facts", 5)

      assert values == %{
               "retrievals" => 2,
               "facts_retrieved" => 3,
               "duration_min_us" => 7,
               "duration_max_us" => 900
             }

      assert records == %{}
      assert window.world_id == elem(world, 1)
      assert window.family == "semantic_facts"
      assert %DateTime{} = window.started_at
      assert window.ended_at == nil
    end

    test "a min or max of zero or below is kept" do
      family = test_family("zero")
      write!(:node, family, [{"min", :min, 5}, {"max", :max, -3}])
      write!(:node, family, [{"min", :min, 0}, {"max", :max, -9}])

      assert [%{values: %{"min" => 0, "max" => -3}}] = Metrics.windows(:node, family, 5)
    end

    test "a record replaces the window's record of its name, and is read back with when it was written" do
      family = test_family("records")
      write!(:node, family, [], [{"training:pos_tagger", %{"status" => "in_progress"}}])
      write!(:node, family, [], [{"training:pos_tagger", %{"status" => "completed", "duration_ms" => 12}}])
      write!(:node, family, [], [{"training:entity_trainer", %{"status" => "failed"}}])

      assert [%{values: values, records: records}] = Metrics.windows(:node, family, 5)
      assert values == %{}

      assert %{
               "training:pos_tagger" => %{record: %{"status" => "completed", "duration_ms" => 12}, recorded_at: %DateTime{}},
               "training:entity_trainer" => %{record: %{"status" => "failed"}}
             } = records
    end

    test "one world's, the node's and another family's windows leave each other's" do
      world_a = {:world, test_world("a")}
      world_b = {:world, test_world("b")}
      family = test_family("isolation")
      other_family = test_family("isolation_other")

      write!(world_a, family, [{"retrievals", :sum, 2}])
      write!(world_b, family, [{"retrievals", :sum, 5}])
      write!(:node, family, [{"retrievals", :sum, 7}])
      write!(world_a, other_family, [{"retrievals", :sum, 11}])
      Metrics.start_window!(world_a, family)
      write!(world_a, family, [{"retrievals", :sum, 1}])

      assert [%{values: %{"retrievals" => 1}}, %{values: %{"retrievals" => 2}}] = Metrics.windows(world_a, family, 5)
      assert [%{values: %{"retrievals" => 5}, window: %{ended_at: nil}}] = Metrics.windows(world_b, family, 5)
      assert [%{values: %{"retrievals" => 7}, window: %{ended_at: nil, world_id: nil}}] = Metrics.windows(:node, family, 5)
      assert [%{values: %{"retrievals" => 11}, window: %{ended_at: nil}}] = Metrics.windows(world_a, other_family, 5)
    end

    test "several writes in one call are written together, to each one's own window" do
      world = {:world, test_world("several")}
      family = test_family("several")

      :ok =
        Metrics.write!([
          %{scope: :node, family: family, values: [{"learning:document_processed", :sum, 1}]},
          %{scope: world, family: family, values: [{"learning:document_processed", :sum, 1}]}
        ])

      assert [%{values: %{"learning:document_processed" => 1}}] = Metrics.windows(:node, family, 5)
      assert [%{values: %{"learning:document_processed" => 1}}] = Metrics.windows(world, family, 5)
    end

    test "a name written with another aggregate than its window holds it by fails, and nothing of the call is written" do
      family = test_family("aggregate_clash")
      write!(:node, family, [{"duration", :min, 5}])

      assert_raise Postgrex.Error, fn ->
        Metrics.write!([
          %{scope: :node, family: family, values: [{"count", :sum, 1}]},
          %{scope: :node, family: family, values: [{"duration", :sum, 3}]}
        ])
      end

      assert [%{values: %{"duration" => 5}}] = Metrics.windows(:node, family, 5)
    end

    test "refuses malformed writes and writes nothing" do
      world = test_world("refused")
      family = test_family("refused")

      for writes <- [
            [],
            nil,
            [%{scope: {:world, ""}, family: family, values: [{"n", :sum, 1}]}],
            [%{scope: {:world, "  "}, family: family, values: [{"n", :sum, 1}]}],
            [%{scope: world, family: family, values: [{"n", :sum, 1}]}],
            [%{scope: nil, family: family, values: [{"n", :sum, 1}]}],
            [%{scope: {:world, world}, family: "", values: [{"n", :sum, 1}]}],
            [%{scope: {:world, world}, family: :semantic_facts, values: [{"n", :sum, 1}]}],
            [%{scope: {:world, world}, family: family}],
            [%{scope: {:world, world}, family: family, values: [], records: []}],
            [%{scope: {:world, world}, family: family, values: [{" ", :sum, 1}]}],
            [%{scope: {:world, world}, family: family, values: [{:n, :sum, 1}]}],
            [%{scope: {:world, world}, family: family, values: [{"n", :sum, 0}]}],
            [%{scope: {:world, world}, family: family, values: [{"n", :sum, -1}]}],
            [%{scope: {:world, world}, family: family, values: [{"n", :sum, 1.0}]}],
            [%{scope: {:world, world}, family: family, values: [{"n", :avg, 1}]}],
            [%{scope: {:world, world}, family: family, values: [{"n", :sum, 1}, {"n", :sum, 2}]}],
            [%{scope: {:world, world}, family: family, records: [{"r", "not a map"}]}],
            [%{scope: {:world, world}, family: family, records: [{"r", %{}}, {"r", %{}}]}],
            [%{scope: {:world, world}, family: family, values: [{"n", :sum, 1}], counts: %{}}],
            # The first write is well formed; the call is refused as a whole.
            [
              %{scope: {:world, world}, family: family, values: [{"n", :sum, 1}]},
              %{scope: {:world, world}, family: family, values: [{"n", :sum, 0}]}
            ]
          ] do
        assert_raise ArgumentError, fn -> Metrics.write!(writes) end
      end

      assert Metrics.windows({:world, world}, family, 5) == []
    end
  end

  describe "start_window!/2" do
    test "ends the open window, keeping its values and records, and writes go to the new one" do
      world = {:world, test_world("window")}
      write!(world, "semantic_facts", [{"retrievals", :sum, 4}, {"used:synthesizer", :sum, 2}], [{"r", %{"n" => 1}}])
      [%{window: first}] = Metrics.windows(world, "semantic_facts", 5)

      second = Metrics.start_window!(world, "semantic_facts")

      assert second.world_id == elem(world, 1)
      assert second.family == "semantic_facts"
      assert second.ended_at == nil
      assert DateTime.compare(second.started_at, first.started_at) == :gt

      assert [%{window: ^second, values: new_values, records: new_records}, %{window: ended, values: kept, records: kept_records}] =
               Metrics.windows(world, "semantic_facts", 5)

      assert new_values == %{}
      assert new_records == %{}
      assert kept == %{"retrievals" => 4, "used:synthesizer" => 2}
      assert %{"r" => %{record: %{"n" => 1}}} = kept_records
      assert ended.id == first.id
      assert ended.started_at == first.started_at
      assert ended.ended_at == second.started_at

      write!(world, "semantic_facts", [{"retrievals", :sum, 1}])

      assert [%{values: %{"retrievals" => 1}}, %{values: ^kept}] = Metrics.windows(world, "semantic_facts", 5)
      assert [%{window: ^second}] = Metrics.windows(world, "semantic_facts", 1)
    end

    test "ends the node's open window of the family, and leaves a world's of the same family" do
      world = {:world, test_world("node_window")}
      family = test_family("node_window")
      write!(:node, family, [{"n", :sum, 3}])
      write!(world, family, [{"n", :sum, 5}])

      second = Metrics.start_window!(:node, family)
      assert second.world_id == nil

      assert [%{window: ^second, values: %{}}, %{values: %{"n" => 3}}] = Metrics.windows(:node, family, 5)
      assert [%{values: %{"n" => 5}, window: %{ended_at: nil}}] = Metrics.windows(world, family, 5)
    end

    test "opens a window for a family and scope nothing was written to" do
      world = {:world, test_world("fresh")}

      window = Metrics.start_window!(world, "semantic_facts")

      assert [%{window: ^window, values: values, records: records}] = Metrics.windows(world, "semantic_facts", 5)
      assert values == %{}
      assert records == %{}
    end

    test "refuses a malformed scope or family" do
      assert_raise ArgumentError, fn -> Metrics.start_window!({:world, ""}, "semantic_facts") end
      assert_raise ArgumentError, fn -> Metrics.start_window!({:world, " "}, "semantic_facts") end
      assert_raise ArgumentError, fn -> Metrics.start_window!("world", "semantic_facts") end
      assert_raise ArgumentError, fn -> Metrics.start_window!(:node, "") end
    end
  end

  describe "the open window index" do
    test "refuses a second open window of a family in a world, and in the node" do
      world = test_world("two_open")
      family = test_family("two_open")

      for scope <- [{:world, world}, :node] do
        Metrics.start_window!(scope, family)

        world_id = if scope == :node, do: nil, else: world

        assert {:error, changeset} =
                 %MetricWindow{}
                 |> MetricWindow.changeset(%{world_id: world_id, family: family, started_at: DateTime.utc_now()})
                 |> Repo.insert()

        assert {"has already been taken", _} = changeset.errors[:family]
      end
    end
  end

  describe "concurrent write!/1" do
    # Every task writes on a connection of its own, outside the test's sandbox
    # transaction, so the transactions really run at the same time. What they
    # write is committed, so the test's rows are deleted when it ends.
    setup do
      world = test_world("concurrent")
      family = test_family("concurrent")

      on_exit(fn ->
        :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo, sandbox: false)
        windows = from(w in MetricWindow, where: w.family == ^family, select: w.id)
        Repo.delete_all(from(v in MetricValue, where: v.window_id in subquery(windows)))
        Repo.delete_all(from(r in MetricRecord, where: r.window_id in subquery(windows)))
        Repo.delete_all(from(w in MetricWindow, where: w.family == ^family))
        Ecto.Adapters.SQL.Sandbox.checkin(Repo)
      end)

      %{world: world, family: family}
    end

    test "loses nothing and opens one window per scope", %{world: world, family: family} do
      tasks = 40

      1..tasks
      |> Task.async_stream(
        fn task ->
          :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo, sandbox: false)

          try do
            values = [{"count", :sum, 1}, {"sum", :sum, task}, {"min", :min, task}, {"max", :max, task}]

            Metrics.write!([
              %{scope: {:world, world}, family: family, values: values},
              %{scope: :node, family: family, values: values, records: [{"last", %{"task" => task}}]}
            ])
          after
            Ecto.Adapters.SQL.Sandbox.checkin(Repo)
          end
        end,
        max_concurrency: tasks,
        timeout: 30_000
      )
      |> Enum.each(fn result -> assert result == {:ok, :ok} end)

      expected = %{"count" => tasks, "sum" => Enum.sum(1..tasks), "min" => 1, "max" => tasks}

      assert [%{values: ^expected}] = Metrics.windows({:world, world}, family, 5)
      assert [%{values: ^expected, records: %{"last" => %{record: %{"task" => task}}}}] = Metrics.windows(:node, family, 5)
      assert task in 1..tasks
    end
  end

  describe "Atlas.Worlds.purge!/1" do
    test "deletes the world's windows, values and records, and leaves another world's and the node's" do
      world = test_world("purged")
      other = test_world("kept")
      family = test_family("purge")

      for scope <- [{:world, world}, {:world, other}, :node] do
        write!(scope, family, [{"retrievals", :sum, 1}], [{"r", %{"n" => 1}}])
        Metrics.start_window!(scope, family)
        write!(scope, family, [{"retrievals", :sum, 2}])
      end

      %{rows: rows} = Atlas.Worlds.purge!(world)

      assert rows[MetricValue] == 2
      assert rows[MetricRecord] == 1
      assert rows[MetricWindow] == 2
      assert Metrics.windows({:world, world}, family, 5) == []

      for scope <- [{:world, other}, :node] do
        assert [%{values: %{"retrievals" => 2}}, %{values: %{"retrievals" => 1}, records: %{"r" => _}}] =
                 Metrics.windows(scope, family, 5)
      end
    end
  end
end
