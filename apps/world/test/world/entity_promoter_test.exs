defmodule World.EntityPromoterTest do
  @moduledoc """
  The promoter suggests entities for human review; it never teaches the
  gazetteer itself. Runs in an Atlas sandbox so the review candidates it
  writes roll back.
  """
  use ExUnit.Case, async: false

  alias Brain.Knowledge.ReviewQueue
  alias Brain.ML.Gazetteer
  alias World.EntityPromoter

  setup do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(%{async: false})

    on_exit(fn ->
      Brain.Test.AtlasSandbox.drain_and_stop_owner(owner)
      # Other World tests run without a sandbox owner.
      Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :auto)
    end)

    ReviewQueue.clear()

    # The knowledge-graph gate scores entities with a trained model; these
    # tests are about the review boundary, so the gate is switched off.
    previous = Application.get_env(:brain, :kg_signals)
    Application.put_env(:brain, :kg_signals, Keyword.put(previous || [], :entity_promoter_kg_gate, false))
    on_exit(fn -> Application.put_env(:brain, :kg_signals, previous) end)

    world_id = "promoter_test_#{System.unique_integer([:positive])}"
    name = "Quibbleton#{System.unique_integer([:positive])}"
    {:ok, world_id: world_id, name: name}
  end

  defp observations(name, count, opts \\ []) do
    for i <- 1..count do
      %{
        value: name,
        inferred_type: Keyword.get(opts, :type, "location"),
        confidence: Keyword.get(opts, :confidence, 0.8),
        context: "I went to #{name} (#{i})"
      }
    end
  end

  defp config, do: Application.fetch_env!(:world, EntityPromoter)

  describe "suggest/3" do
    test "suggests an entity seen often enough as a pending review candidate", %{
      world_id: world_id,
      name: name
    } do
      [suggested] = EntityPromoter.suggest(world_id, observations(name, config()[:min_occurrences]))

      assert suggested.value == name
      assert [candidate] = ReviewQueue.get_pending()
      assert candidate.finding.entity == name
      assert candidate.finding.entity_type == "location"
      assert candidate.finding.world_id == world_id
      assert candidate.status == :pending
    end

    test "never adds to the gazetteer itself", %{world_id: world_id, name: name} do
      EntityPromoter.suggest(world_id, observations(name, config()[:min_occurrences]))

      assert Gazetteer.lookup(name) == :not_found
      assert Gazetteer.lookup(name, world_id) == :not_found
    end

    test "waits for the minimum sample before suggesting", %{world_id: world_id, name: name} do
      min = config()[:min_occurrences]

      assert EntityPromoter.suggest(world_id, observations(name, min - 1)) == []
      assert ReviewQueue.get_pending() == []
    end

    test "the minimum sample is the configured one, not a constant", %{world_id: world_id, name: name} do
      stricter = Keyword.put(config(), :min_occurrences, 10)
      seen = observations(name, 5)

      assert EntityPromoter.suggest(world_id, seen, config: stricter) == []
      assert [_] = EntityPromoter.suggest(world_id, seen, config: Keyword.put(config(), :min_occurrences, 5))
    end

    test "skips low confidence and undetermined types", %{world_id: world_id, name: name} do
      min = config()[:min_occurrences]
      below = config()[:min_confidence] / 2

      assert EntityPromoter.suggest(world_id, observations(name, min, confidence: below)) == []
      assert EntityPromoter.suggest(world_id, observations(name, min, type: "unknown")) == []
      assert ReviewQueue.get_pending() == []
    end

    test "does not ask again about an entity already in the queue", %{world_id: world_id, name: name} do
      seen = observations(name, config()[:min_occurrences])

      assert [_] = EntityPromoter.suggest(world_id, seen)
      assert EntityPromoter.suggest(world_id, seen) == []
      assert length(ReviewQueue.get_pending()) == 1
    end

    test "does not ask again after a rejection", %{world_id: world_id, name: name} do
      seen = observations(name, config()[:min_occurrences])
      [_] = EntityPromoter.suggest(world_id, seen)
      [candidate] = ReviewQueue.get_pending()
      {:ok, _} = ReviewQueue.reject(candidate.id)

      assert EntityPromoter.suggest(world_id, seen) == []
    end

    test "the same entity in another world is its own suggestion", %{world_id: world_id, name: name} do
      seen = observations(name, config()[:min_occurrences])

      assert [_] = EntityPromoter.suggest(world_id, seen)
      assert [_] = EntityPromoter.suggest(world_id <> "_other", seen)
    end
  end

  describe "stats/1" do
    test "starts with nothing suggested" do
      name = :"entity_promoter_test_#{System.unique_integer([:positive])}"
      {:ok, pid} = EntityPromoter.start_link(name: name)
      on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)

      assert EntityPromoter.stats(name) == %{total_suggested: 0, last_scan: nil, suggested_entities: []}
    end
  end
end
