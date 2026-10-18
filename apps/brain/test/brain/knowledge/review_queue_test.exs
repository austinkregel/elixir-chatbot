defmodule Brain.Knowledge.ReviewQueueTest do
  alias Brain.Knowledge.Types
  alias Brain.Knowledge
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Knowledge.{ReviewQueue, SourceReliability}
  alias Brain.ML.Gazetteer
  alias Types.{Finding, SourceInfo, ReviewCandidate}

  # Every candidate belongs to a world: the test's own, `w`, the one whose
  # graphs `GraphCase` created. Its gazetteer overlay is dropped when the test
  # ends.
  setup %{graph_world_id: world} do
    ensure_pubsub_started()
    ensure_started(SourceReliability)
    ensure_started(ReviewQueue)
    ReviewQueue.clear()
    on_exit(fn -> Gazetteer.destroy_world_overlay(world) end)

    %{w: world}
  end

  describe "add/1" do
    test "adds candidate to queue", %{w: w} do
      candidate = build_test_candidate(w, "France", "Paris is the capital")

      {:ok, id} = ReviewQueue.add(candidate)

      assert is_binary(id)
      assert id == candidate.id
    end

    test "increments pending count", %{w: w} do
      initial_stats = ReviewQueue.stats()

      candidate = build_test_candidate(w, "Test", "Test claim")
      ReviewQueue.add(candidate)

      final_stats = ReviewQueue.stats()
      assert final_stats.pending == initial_stats.pending + 1
    end

    test "a candidate whose finding names no world raises and is not queued", %{w: w} do
      candidate = build_test_candidate(w, "Nowhere", "A claim about no world")
      worldless = %{candidate | finding: %{candidate.finding | world_id: nil}}

      assert_raise ArgumentError, ~r/names no world/, fn -> ReviewQueue.add(worldless) end
      assert {:error, :not_found} = ReviewQueue.get(candidate.id)
    end
  end

  describe "get_pending/1" do
    test "returns pending candidates", %{w: w} do
      candidate1 = build_test_candidate(w, "Entity1", "Claim 1")
      candidate2 = build_test_candidate(w, "Entity2", "Claim 2")

      ReviewQueue.add(candidate1)
      ReviewQueue.add(candidate2)

      pending = ReviewQueue.get_pending()

      assert length(pending) == 2
    end

    test "respects limit option", %{w: w} do
      for i <- 1..10 do
        ReviewQueue.add(build_test_candidate(w, "Entity#{i}", "Claim #{i}"))
      end

      pending = ReviewQueue.get_pending(limit: 5)

      assert length(pending) == 5
    end

    test "sorts by confidence by default", %{w: w} do
      low_conf = build_test_candidate(w, "Low", "Low confidence", confidence: 0.3)
      high_conf = build_test_candidate(w, "High", "High confidence", confidence: 0.9)

      ReviewQueue.add(low_conf)
      ReviewQueue.add(high_conf)

      pending = ReviewQueue.get_pending()
      assert hd(pending).aggregate_confidence == 0.9
    end
  end

  describe "approve/2" do
    test "updates status to approved", %{w: w} do
      candidate = build_test_candidate(w, "France", "Paris is the capital")
      ReviewQueue.add(candidate)

      {:ok, approved} = ReviewQueue.approve(candidate.id, "Verified correct")

      assert approved.status == :approved
      assert approved.reviewer_notes == "Verified correct"
      assert approved.reviewed_at != nil
    end

    test "removes from pending list", %{w: w} do
      candidate = build_test_candidate(w, "Test", "Test claim")
      ReviewQueue.add(candidate)

      initial_pending = length(ReviewQueue.get_pending())

      ReviewQueue.approve(candidate.id)

      final_pending = length(ReviewQueue.get_pending())
      assert final_pending == initial_pending - 1
    end

    test "returns error for non-existent ID" do
      {:error, :not_found} = ReviewQueue.approve("nonexistent-id")
    end

    test "integrating a candidate whose finding names no world raises", %{w: w} do
      candidate = build_test_candidate(w, "Nowhere", "A claim about no world")
      worldless = %{candidate | finding: %{candidate.finding | world_id: nil}}

      # Past `add/1`'s check, as a candidate held before worlds were required
      # would be.
      :ets.insert(:knowledge_review_queue, {worldless.id, worldless})

      Process.flag(:trap_exit, true)

      assert {{%ArgumentError{message: message}, _stack}, _call} =
               catch_exit(ReviewQueue.approve(worldless.id))

      assert message =~ "names no world"
    end
  end

  describe "reject/2" do
    test "updates status to rejected", %{w: w} do
      candidate = build_test_candidate(w, "Test", "Test claim")
      ReviewQueue.add(candidate)

      {:ok, rejected} = ReviewQueue.reject(candidate.id, "Factually incorrect")

      assert rejected.status == :rejected
      assert rejected.reviewer_notes == "Factually incorrect"
    end

    test "updates daily rejection count", %{w: w} do
      candidate = build_test_candidate(w, "Test", "Test claim")
      ReviewQueue.add(candidate)

      initial_stats = ReviewQueue.stats()

      ReviewQueue.reject(candidate.id)

      final_stats = ReviewQueue.stats()
      assert final_stats.rejected_today == initial_stats.rejected_today + 1
    end
  end

  describe "bulk_approve/1" do
    test "approves multiple candidates", %{w: w} do
      candidates =
        for i <- 1..5 do
          build_test_candidate(w, "Entity#{i}", "Claim #{i}")
        end

      ids =
        Enum.map(candidates, fn c ->
          ReviewQueue.add(c)
          c.id
        end)

      {:ok, count} = ReviewQueue.bulk_approve(ids)

      assert count == 5
      assert ReviewQueue.get_pending() == []
    end
  end

  describe "bulk_reject/1" do
    test "rejects multiple candidates", %{w: w} do
      candidates =
        for i <- 1..3 do
          build_test_candidate(w, "Entity#{i}", "Claim #{i}")
        end

      ids =
        Enum.map(candidates, fn c ->
          ReviewQueue.add(c)
          c.id
        end)

      {:ok, count} = ReviewQueue.bulk_reject(ids)

      assert count == 3
    end
  end

  describe "stats/0" do
    test "returns queue statistics" do
      stats = ReviewQueue.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :pending)
      assert Map.has_key?(stats, :approved)
      assert Map.has_key?(stats, :rejected)
      assert Map.has_key?(stats, :approved_today)
      assert Map.has_key?(stats, :rejected_today)
    end
  end

  describe "add_contradiction/3" do
    test "adds a contradiction for review, in the world it was found in", %{w: w} do
      new_fact = %{entity: "France", claim: "Lyon is the capital"}
      existing_belief = %{object: "Paris is the capital", confidence: 0.9}

      {:ok, id} = ReviewQueue.add_contradiction(w, new_fact, existing_belief)

      assert is_binary(id)

      pending = ReviewQueue.get_pending()
      contradiction = Enum.find(pending, &String.contains?(&1.finding.claim, "CONTRADICTION"))
      assert contradiction != nil
      assert contradiction.finding.world_id == w
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/needs the world/, fn ->
        ReviewQueue.add_contradiction(nil, %{entity: "France"}, %{})
      end
    end
  end

  describe "persistence" do
    test "persists across restarts, with its world", %{w: w} do
      candidate = build_test_candidate(w, "Persist", "Persistence test claim")
      ReviewQueue.add(candidate)
      ReviewQueue.persist()
      GenServer.stop(ReviewQueue)
      Process.sleep(100)

      ensure_started(ReviewQueue)

      pending = ReviewQueue.get_pending()
      found = Enum.find(pending, &(&1.id == candidate.id))
      assert found != nil
      assert found.finding.world_id == w
    end
  end

  describe "teaching the gazetteer" do
    # Names no source knows, so every entry a test finds was put there by the
    # approval under test.
    setup do
      name = "Zorblax#{System.unique_integer([:positive])}"
      on_exit(fn -> Gazetteer.remove_entry(name) end)
      {:ok, name: name}
    end

    test "a human approval teaches whatever type the reviewer approved, in the finding's world only",
         %{w: w, name: name} do
      # No allow-list: a type outside the old five (person, location, ...) is
      # learned like any other.
      candidate = entity_candidate(w, name, "music_album")
      {:ok, _} = ReviewQueue.add(candidate)
      assert world_types_of(name, w) == []

      {:ok, _} = ReviewQueue.approve(candidate.id, "checked")

      assert world_types_of(name, w) == ["music_album"]
      # The base gazetteer learns nothing from an approval.
      assert Gazetteer.lookup(name) == :not_found
    end

    test "approving a known name under a new type adds a reading", %{w: w, name: name} do
      {:ok, _} = Gazetteer.add_to_world(w, name, "location", %{reviewed: true})

      candidate = entity_candidate(w, name, "person")
      {:ok, _} = ReviewQueue.add(candidate)
      {:ok, _} = ReviewQueue.approve(candidate.id)

      assert Enum.sort(world_types_of(name, w)) == ["location", "person"]
    end

    test "a bulk approval teaches too, and is persisted with its world", %{w: w, name: name} do
      candidate = entity_candidate(w, name, "song")
      {:ok, _} = ReviewQueue.add(candidate)

      {:ok, 1} = ReviewQueue.bulk_approve([candidate.id])
      ReviewQueue.persist()

      assert world_types_of(name, w) == ["song"]
      {:ok, reviewed} = Brain.AtlasIntegration.load_reviewed_entities()
      assert {name, "song", w} in reviewed
    end

    test "rejecting, deferring or leaving pending teaches nothing", %{w: w, name: name} do
      pending = entity_candidate(w, name, "person")
      rejected = entity_candidate(w, name, "location")
      deferred = entity_candidate(w, name, "song")

      for c <- [pending, rejected, deferred], do: {:ok, _} = ReviewQueue.add(c)
      {:ok, _} = ReviewQueue.reject(rejected.id)
      {:ok, _} = ReviewQueue.defer(deferred.id)
      ReviewQueue.persist()

      assert world_types_of(name, w) == []
      {:ok, reviewed} = Brain.AtlasIntegration.load_reviewed_entities()
      refute Enum.any?(reviewed, fn {entity, _t, _w} -> entity == name end)
    end

    test "an auto-approval is recorded as such and teaches nothing", %{w: w, name: name} do
      ensure_started(Brain.Analysis.ComprehensionAssessor)
      profile_id = "review_test_profile_#{System.unique_integer([:positive])}"
      :ets.insert(:comprehension_profiles, {profile_id, %{verdict: :comprehended}, 0})
      on_exit(fn -> :ets.delete(:comprehension_profiles, profile_id) end)

      previous = Application.get_env(:brain, :auto_approval_enabled)
      Application.put_env(:brain, :auto_approval_enabled, true)
      on_exit(fn -> Application.put_env(:brain, :auto_approval_enabled, previous) end)

      sources =
        for i <- 1..3 do
          SourceInfo.new("https://corroborator#{i}.example/a", reliability_score: 0.9)
        end

      candidate =
        entity_candidate(w, name, "person",
          confidence: 0.95,
          corroborating_sources: sources,
          comprehension_profile_id: profile_id
        )

      {:ok, _} = ReviewQueue.add(candidate)
      ReviewQueue.persist()

      assert {:ok, %{status: :auto_approved}} = ReviewQueue.get(candidate.id)
      assert world_types_of(name, w) == []

      {:ok, reviewed} = Brain.AtlasIntegration.load_reviewed_entities()
      refute Enum.any?(reviewed, fn {entity, _t, _w} -> entity == name end)
    end
  end

  describe "approving an entity finding writes its instance to the knowledge graph" do
    setup do
      name = "Zorblax#{System.unique_integer([:positive])}"
      {:ok, name: name}
    end

    test "the instance records the finding's world", %{w: w, name: name} do
      candidate = entity_candidate(w, name, "person")
      {:ok, _} = ReviewQueue.add(candidate)
      {:ok, _} = ReviewQueue.approve(candidate.id)

      graph = Atlas.Graph.world_graph("knowledge_graph", w)
      {:ok, instance} = Brain.AtlasIntegration.find_node(graph, "EntityInstance", name)
      assert instance.properties["world_id"] == w
      assert instance.properties["type"] == "person"
      assert instance.properties["source"] == "knowledge_expansion"
    end

    test "the instance is written to the finding's world's graph only", %{w: w, name: name} do
      other = "test_review_other_#{System.unique_integer([:positive])}"
      Atlas.Graph.ensure_world_graphs!(other)

      candidate = entity_candidate(w, name, "person")
      {:ok, _} = ReviewQueue.add(candidate)
      {:ok, _} = ReviewQueue.approve(candidate.id)

      other_graph = Atlas.Graph.world_graph("knowledge_graph", other)
      assert :not_found = Brain.AtlasIntegration.find_node(other_graph, "EntityInstance", name)
    end
  end

  describe "has_entity_candidate?/3" do
    test "matches entity case-insensitively, by type and world, in any status", %{w: w} do
      source = SourceInfo.new("https://review-test.example/x")

      finding =
        Finding.new("Qux is a person", "Qux", source, entity_type: "person", world_id: w)

      candidate = ReviewCandidate.new(finding)
      {:ok, _} = ReviewQueue.add(candidate)
      {:ok, _} = ReviewQueue.reject(candidate.id)

      assert ReviewQueue.has_entity_candidate?("qux", "person", w)
      refute ReviewQueue.has_entity_candidate?("qux", "location", w)
      refute ReviewQueue.has_entity_candidate?("qux", "person", "#{w}_other")
    end

    test "asking without a world raises" do
      assert_raise FunctionClauseError, fn ->
        ReviewQueue.has_entity_candidate?("qux", "person", nil)
      end
    end
  end

  defp entity_candidate(world_id, entity, type, opts \\ []) do
    source = SourceInfo.new("https://review-test.example/#{entity}", reliability_score: 0.9)

    finding =
      Finding.new("#{entity} is a #{type}", entity, source,
        entity_type: type,
        confidence: Keyword.get(opts, :confidence, 0.7),
        world_id: world_id,
        comprehension_profile_id: Keyword.get(opts, :comprehension_profile_id)
      )

    ReviewCandidate.new(finding,
      aggregate_confidence: Keyword.get(opts, :confidence, 0.7),
      corroborating_sources: Keyword.get(opts, :corroborating_sources, [])
    )
  end

  # The types the world's gazetteer holds for a name. The names are ones the
  # base gazetteer does not know, so these are the overlay's.
  defp world_types_of(name, world_id) do
    case Gazetteer.lookup(name, world_id) do
      {:ok, infos} -> infos |> List.wrap() |> Enum.map(& &1[:entity_type])
      :not_found -> []
    end
  end

  defp build_test_candidate(world_id, entity, claim, opts \\ []) do
    confidence = Keyword.get(opts, :confidence, 0.7)
    domain = Keyword.get(opts, :domain, "test-source.com")

    source =
      SourceInfo.new("https://#{domain}/article", reliability_score: 0.8, trust_tier: :verified)

    finding =
      Finding.new(claim, entity, source, entity_type: "location", confidence: confidence, world_id: world_id)

    ReviewCandidate.new(finding, aggregate_confidence: confidence)
  end
end
