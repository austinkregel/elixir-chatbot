defmodule World.EntityPromoterTest do
  use ExUnit.Case, async: false

  alias World.EntityPromoter

  setup do
    name = :"entity_promoter_test_#{:rand.uniform(100_000)}"
    {:ok, pid} = EntityPromoter.start_link(name: name)
    on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)
    {:ok, name: name, pid: pid}
  end

  describe "start_link/1" do
    test "starts successfully", %{pid: pid} do
      assert Process.alive?(pid)
    end
  end

  describe "ready?/1" do
    test "returns true when started", %{name: name} do
      assert EntityPromoter.ready?(name)
    end

    test "returns false for unregistered name" do
      refute EntityPromoter.ready?(:nonexistent_promoter)
    end
  end

  describe "stats/1" do
    test "returns initial stats", %{name: name} do
      stats = EntityPromoter.stats(name)

      assert is_map(stats)
      assert stats.total_promoted == 0
      assert stats.last_scan == nil
      assert stats.promoted_entities == []
    end
  end

  describe "scan_now/1" do
    test "triggers a scan without crashing", %{name: name} do
      EntityPromoter.scan_now(name)
      Process.sleep(100)
      stats = EntityPromoter.stats(name)
      assert is_map(stats)
    end
  end

  describe "handle_info :scan" do
    test "handles periodic scan", %{pid: pid} do
      send(pid, :scan)
      Process.sleep(100)
      assert Process.alive?(pid)
    end
  end
end
