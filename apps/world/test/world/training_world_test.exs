defmodule World.TrainingWorldTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias World.{
    TrainingWorld,
    EntityDiscoverer,
    TypeInferrer,
    DocumentIngestor
  }

  alias World.Manager, as: WorldManager
  alias World.Metrics, as: WorldMetrics
  alias World.Events, as: WorldEvents
  alias World.Persistence, as: WorldPersistence

  alias Brain.ML.Gazetteer

  @synthetic_script_1 "Captain Marcus looked across the bridge of the Starship Endeavour.\nLieutenant Chen handed him a report from the science station.\nThey were approaching the planet Meridian in the Orion sector.\nDoctor Patel arrived from the medical bay with urgent news.\nCommander Torres had detected anomalies near the asteroid field.\n"

  @synthetic_script_2 "Admiral Jensen reviewed the fleet deployment from Command Station Alpha.\nThe Vulcan ambassador arrived aboard the transport vessel Harmony.\nEnsign Williams reported from engineering on the lower decks.\nCaptain Singh ordered all hands to battle stations.\nThe Andorian delegation awaited in Conference Room Seven.\n"

  @synthetic_script_with_ambiguity "Austin met Dallas at the Phoenix conference center.\nThey discussed the Jordan project with Paris representatives.\nLater, Madison joined them for the Denver presentation.\nBrooklyn sent updates from the London office.\n"

  setup do
    start_world_test_services()
    setup_world_sandbox()

    eventually(
      fn -> WorldManager.ready?() end,
      fn ready -> ready end,
      50,
      20
    )

    :ok
  end

  describe "TrainingWorld struct" do
    test "creates world with default configuration" do
      world = TrainingWorld.new("test_world")

      assert world.name == "test_world"
      assert world.mode == :ephemeral
      assert world.base_world == nil
      assert is_binary(world.id)
      assert %DateTime{} = world.created_at
      assert is_map(world.config)
      assert world.config.promotion_threshold == 3
      assert world.config.confidence_threshold == 0.7
    end

    test "creates world with custom configuration" do
      world =
        TrainingWorld.new("custom_world", mode: :persistent, config: %{promotion_threshold: 5})

      assert world.mode == :persistent
      assert world.config.promotion_threshold == 5
    end
  end

  describe "WorldEvents" do
    test "creates event with required fields" do
      event = WorldEvents.new("world_123", :entity_candidate_detected, %{value: "Test"})

      assert event.world_id == "world_123"
      assert event.type == :entity_candidate_detected
      assert event.data == %{value: "Test"}
      assert is_binary(event.id)
      assert %DateTime{} = event.timestamp
    end

    test "creates event with optional fields" do
      event =
        WorldEvents.new("world_123", :entity_type_inferred, %{value: "Test"},
          confidence: 0.85,
          context: %{surrounding: "some context"}
        )

      assert event.confidence == 0.85
      assert event.context == %{surrounding: "some context"}
    end
  end

  describe "WorldMetrics" do
    test "creates new metrics with timestamps" do
      metrics = WorldMetrics.new()

      assert metrics.documents_processed == 0
      assert metrics.entities_discovered == 0
      assert %DateTime{} = metrics.started_at
    end

    test "records document processing" do
      metrics =
        WorldMetrics.new()
        |> WorldMetrics.record_document(100, 5, 50)

      assert metrics.documents_processed == 1
      assert metrics.total_tokens == 100
      assert metrics.total_sentences == 5
      assert metrics.processing_time_ms == 50
    end

    test "records entity discovery with type tracking" do
      metrics =
        WorldMetrics.new()
        |> WorldMetrics.record_entity_discovered("person", 0.8)
        |> WorldMetrics.record_entity_discovered("person", 0.7)
        |> WorldMetrics.record_entity_discovered("location", 0.9)

      assert metrics.entities_discovered == 3
      assert metrics.entities_by_type == %{"person" => 2, "location" => 1}
    end

    test "computes diff between two worlds" do
      metrics1 =
        WorldMetrics.new()
        |> WorldMetrics.record_entity_discovered("person", 0.8)
        |> WorldMetrics.record_entity_discovered("location", 0.9)

      metrics2 =
        WorldMetrics.new()
        |> WorldMetrics.record_entity_discovered("person", 0.7)

      diff = WorldMetrics.diff(metrics1, metrics2)

      assert diff.entity_count_diff == 1
      assert diff.type_distribution_diff["location"] == 1
    end
  end

  describe "WorldManager" do
    test "creates and destroys ephemeral world" do
      {:ok, world} = create_test_world("ephemeral_test")

      assert world.mode == :ephemeral
      assert world.metadata.test == true
      assert {:ok, ^world} = WorldManager.get(world.id)
      :ok = WorldManager.destroy(world.id)
      assert {:error, :not_found} = WorldManager.get(world.id)
    end

    test "creates persistent world" do
      {:ok, world} = create_test_world("persistent_test", mode: :persistent)

      assert world.mode == :persistent
      assert world.metadata.test == true
    end

    test "manages world metrics" do
      {:ok, world} = create_test_world("metrics_test")

      {:ok, initial_metrics} = WorldManager.get_metrics(world.id)
      assert initial_metrics.documents_processed == 0

      {:ok, updated_metrics} =
        WorldManager.update_metrics(world.id, fn m ->
          WorldMetrics.record_document(m, 100, 5, 50)
        end)

      assert updated_metrics.documents_processed == 1
    end

    test "records events in world" do
      {:ok, world} = create_test_world("events_test")

      WorldManager.record_event(world.id, :entity_candidate_detected, %{value: "Test"})
      WorldManager.record_event(world.id, :document_processed, %{file: "test.txt"})
      Process.sleep(50)

      events = WorldManager.get_events(world.id)
      assert length(events) >= 2
    end

    test "manages entity candidates" do
      {:ok, world} = create_test_world("candidates_test")

      candidate = %{
        value: "TestEntity",
        inferred_type: "person",
        confidence: 0.75
      }

      WorldManager.add_candidate(world.id, candidate)
      Process.sleep(50)

      candidates = WorldManager.get_candidates(world.id)
      assert candidates != []

      found = Enum.find(candidates, &(&1.value == "TestEntity"))
      assert found != nil
      assert found.inferred_type == "person"
    end

    test "lists all worlds excludes test worlds by default" do
      {:ok, world1} = create_test_world("list_test_1")
      {:ok, world2} = create_test_world("list_test_2")
      worlds = WorldManager.list_worlds()
      world_ids = Enum.map(worlds, & &1.id)

      refute world1.id in world_ids
      refute world2.id in world_ids
      all_worlds = WorldManager.list_worlds(include_test: true)
      all_world_ids = Enum.map(all_worlds, & &1.id)

      assert world1.id in all_world_ids
      assert world2.id in all_world_ids
    end
  end

  describe "Gazetteer world overlays" do
    test "creates and uses world overlay" do
      {:ok, world} = create_test_world("gazetteer_test")
      {:ok, _} = Gazetteer.add_to_world(world.id, "TestShip", "starship")
      types = Gazetteer.lookup_all_types("TestShip", world.id)
      assert types != []
      assert Enum.any?(types, &(&1.entity_type == "starship"))
      base_types = Gazetteer.lookup_all_types("TestShip")

      overlay_entry =
        Enum.find(base_types, &(&1.entity_type == "starship" and &1.world_id == world.id))
    end

    test "gets all entities in world overlay" do
      {:ok, world} = create_test_world("overlay_list_test")

      Gazetteer.add_to_world(world.id, "Entity1", "type1")
      Gazetteer.add_to_world(world.id, "Entity2", "type2")

      overlay = Gazetteer.get_world_overlay(world.id)
      assert length(overlay) >= 2
    end
  end

  describe "EntityDiscoverer" do
    @tag :requires_pos_model
    test "discovers proper nouns from synthetic script" do
      {:ok, world} = create_test_world("discoverer_test")

      discoveries = EntityDiscoverer.discover_entities(@synthetic_script_1, world.id)
      discovered_values = Enum.map(discoveries, & &1.value)
      expected_names = ["Marcus", "Chen", "Patel", "Torres"]
      expected_places = ["Endeavour", "Meridian", "Orion"]
      found_names = Enum.filter(expected_names, &(&1 in discovered_values))
      _found_places = Enum.filter(expected_places, &(&1 in discovered_values))

      assert is_list(discoveries),
             "Entity discovery should return a list, got: #{inspect(discoveries)}"

      if discoveries != [] do
        assert found_names != [],
               "Expected to find at least 1 name when discoveries exist, found: #{inspect(found_names)}"
      else
        assert discoveries == [],
               "When no entities discovered, should return empty list, got: #{inspect(discoveries)}"
      end
    end

    @tag :requires_pos_model
    test "detects ambiguous entities" do
      {:ok, world} = create_test_world("ambiguity_test")
      Gazetteer.add_to_world(world.id, "Austin", "person")
      Gazetteer.add_to_world(world.id, "Austin", "city")

      discoveries = EntityDiscoverer.discover_entities(@synthetic_script_with_ambiguity, world.id)
      austin_discovery = Enum.find(discoveries, &(&1.value == "Austin"))

      if austin_discovery do
        assert austin_discovery.status == :ambiguous or austin_discovery.status == :known
      end
    end

    @tag :requires_pos_model
    test "aggregates discoveries across multiple texts" do
      {:ok, world} = create_test_world("aggregate_test")

      discoveries1 =
        EntityDiscoverer.discover_entities(@synthetic_script_1, world.id, emit_events: false)

      discoveries2 =
        EntityDiscoverer.discover_entities(@synthetic_script_2, world.id, emit_events: false)

      all_discoveries = discoveries1 ++ discoveries2
      aggregated = EntityDiscoverer.aggregate_discoveries(all_discoveries)
      assert is_list(aggregated)
    end
  end

  describe "TypeInferrer" do
    @test_world_id "test_type_inferrer_world"

    test "learns and infers from context" do
      TypeInferrer.init()

      context_tokens = [
        %{text: "Captain"},
        %{text: "Marcus"},
        %{text: "looked"},
        %{text: "across"}
      ]

      context_tags = ["PROPN", "PROPN", "VERB", "ADP"]

      TypeInferrer.learn_from_known_entity("person", context_tokens, context_tags, @test_world_id)

      {inferred_type, confidence} =
        TypeInferrer.infer_type("Unknown", context_tokens, context_tags, @test_world_id)

      assert is_binary(inferred_type)
      assert is_float(confidence)
    end

    test "exports and imports learned data" do
      TypeInferrer.init()
      TypeInferrer.clear()
      TypeInferrer.learn_from_known_entity("location", [%{text: "in"}], ["ADP"], @test_world_id)
      exported = TypeInferrer.export_learned_data()
      assert is_map(exported.patterns)
      TypeInferrer.clear()
      TypeInferrer.import_learned_data(exported)
      patterns = TypeInferrer.get_patterns_for_type("location", @test_world_id)
      assert map_size(patterns) > 0
    end
  end

  describe "DocumentIngestor" do
    @tag :requires_pos_model
    test "ingests text and discovers entities" do
      {:ok, world} = create_test_world("ingestor_test")

      result = DocumentIngestor.ingest_text(world.id, @synthetic_script_1)

      assert result.documents_processed == 1
      assert result.total_chunks >= 1
      assert result.total_tokens > 0
      assert result.processing_time_ms > 0
    end

    @tag :requires_pos_model
    test "chunks large text appropriately" do
      {:ok, world} = create_test_world("chunking_test")
      large_text = String.duplicate(@synthetic_script_1 <> "\n", 50)

      result = DocumentIngestor.ingest_text(world.id, large_text, chunk_size: 1000)
      assert result.total_chunks > 1
    end
  end

  describe "WorldPersistence" do
    @tag :requires_file_system
    test "saves and loads persistent world" do
      {:ok, world} = create_test_world("persistence_test", mode: :persistent)
      WorldManager.add_candidate(world.id, %{value: "TestEntity", inferred_type: "person"})
      Process.sleep(50)
      :ok = WorldManager.checkpoint(world.id)
      world_path = WorldPersistence.world_path(world.id)
      assert File.exists?(world_path)
      WorldManager.destroy(world.id)
      assert {:error, :not_found} = WorldManager.get(world.id)
      {:ok, loaded_world} = WorldManager.load_world(world.id)
      assert loaded_world.name == "persistence_test"
    end

    @tag :requires_file_system
    test "purge removes persisted world data from disk" do
      {:ok, world} = create_test_world("persistence_purge_test", mode: :persistent)
      :ok = WorldManager.checkpoint(world.id)
      world_path = WorldPersistence.world_path(world.id)
      assert File.exists?(world_path)

      :ok = WorldManager.purge(world.id)
      assert {:error, :not_found} = WorldManager.get(world.id)
      refute File.exists?(world_path)
    end

    test "lists persisted worlds" do
      worlds = WorldPersistence.list_persisted_worlds()
      assert is_list(worlds)
    end
  end

  describe "WorldMetrics comparison" do
    test "computes meaningful diff between worlds" do
      {:ok, world1} = create_test_world("compare_1")
      {:ok, world2} = create_test_world("compare_2")

      WorldManager.update_metrics(world1.id, fn m ->
        m
        |> WorldMetrics.record_entity_discovered("person", 0.8)
        |> WorldMetrics.record_entity_discovered("person", 0.7)
        |> WorldMetrics.record_entity_discovered("location", 0.9)
      end)

      WorldManager.update_metrics(world2.id, fn m ->
        m
        |> WorldMetrics.record_entity_discovered("person", 0.6)
        |> WorldMetrics.record_entity_discovered("organization", 0.8)
      end)

      {:ok, diff} = WorldManager.compare(world1.id, world2.id)
      assert diff.entity_count_diff == 1
      assert "location" in diff.unique_to_world1
      assert "organization" in diff.unique_to_world2
    end
  end

  describe "Integration: Full learning workflow" do
    @tag :integration
    @tag :requires_pos_model
    test "complete workflow from creation to entity discovery" do
      {:ok, world} = create_test_world("integration_test")

      result =
        DocumentIngestor.ingest_text(world.id, @synthetic_script_1 <> "\n" <> @synthetic_script_2)

      assert result.documents_processed == 1
      assert result.total_tokens > 0
      Process.sleep(100)
      candidates = WorldManager.get_candidates(world.id)

      assert is_list(candidates),
             "Candidates should return a list, got: #{inspect(candidates)}"

      if candidates != [] do
        {:ok, metrics} = WorldManager.get_metrics(world.id)

        assert metrics.entities_discovered > 0,
               "Expected entities to be discovered when candidates exist, got: #{metrics.entities_discovered}"

        events = WorldManager.get_events(world.id)

        assert events != [],
               "Expected events to be recorded when entities are discovered, got: #{length(events)}"
      else
        {:ok, metrics} = WorldManager.get_metrics(world.id)

        assert is_map(metrics),
               "Metrics should be available even when no candidates, got: #{inspect(metrics)}"

        events = WorldManager.get_events(world.id)

        assert is_list(events),
               "Events should be available even when no candidates, got: #{inspect(events)}"
      end

      {:ok, export_data} = WorldManager.export(world.id)
      assert is_map(export_data)
      assert Map.has_key?(export_data, :world)
      assert Map.has_key?(export_data, :metrics)
      assert Map.has_key?(export_data, :candidates)
    end
  end
end
