defmodule Atlas.StatsTest do
  use Atlas.DataCase, async: false

  describe "Atlas.Stats" do
    test "get_overview returns valid structure" do
      overview = Atlas.Stats.get_overview()

      assert is_boolean(overview.connected)
      assert is_map(overview.repo)
      assert is_map(overview.graphs)
      assert is_map(overview.connection_pool)
      assert is_list(overview.migrations)
    end

    test "connected? returns true when repo is running" do
      assert Atlas.Stats.connected?() == true
    end

    test "repo_stats returns counts for all tables" do
      stats = Atlas.Stats.repo_stats()

      assert Map.has_key?(stats, :credentials)
      assert Map.has_key?(stats, :beliefs)
      assert Map.has_key?(stats, :episodes)
      assert Map.has_key?(stats, :semantic_facts)
      assert Map.has_key?(stats, :review_candidates)
      assert Map.has_key?(stats, :learned_facts)
    end

    test "migration_status returns applied migrations" do
      migrations = Atlas.Stats.migration_status()
      assert is_list(migrations)
      assert length(migrations) >= 2
    end
  end

  describe "Atlas.Stats.Collector" do
    test "get_metrics returns initial state" do
      metrics = Atlas.Stats.Collector.get_metrics()

      assert is_map(metrics)
      assert metrics.total_queries >= 0
      assert metrics.total_graph_queries >= 0
    end
  end
end
