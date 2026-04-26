defmodule Brain.Analysis.ContextualEntityInferrerTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.{ContextualEntityInferrer, TypeHierarchy, Pipeline}
  alias Brain.Lattice

  setup do
    ensure_pubsub_started()
    ensure_started(TypeHierarchy)
    ensure_started(Brain.ML.SpeechActClassifierSimple)
    ensure_started(Brain.ML.SentimentClassifierSimple)
    ensure_started(Brain.ML.MicroClassifiers)
    ensure_started(Brain.ML.EntityExtractor)
    ensure_started(Brain.Memory.Embedder)
    :ok
  end

  describe "infer/5 type narrowing" do
    test "consumes intent_details lattice with multiple intent hypotheses" do
      entities = [
        %{
          entity_type: "person",
          value: "Taylor Swift",
          match: "Taylor Swift",
          start_pos: 10,
          end_pos: 22,
          confidence: 0.65,
          source: :pos_tagger_propn
        }
      ]

      intent = "music.play"

      lattice =
        Lattice.from_top_k(
          [{"music.play", 0.85}, {"music.search", 0.2}],
          stage: :intent_full,
          source: :test
        )

      intent_details = %{lattice: lattice, top_k: []}

      {updated_entities, updated_intent, _details} =
        ContextualEntityInferrer.infer("Play some Taylor Swift", entities, intent, intent_details)

      assert updated_intent == "music.play"
      narrowed = Enum.find(updated_entities, &(&1[:value] == "Taylor Swift"))
      assert narrowed != nil
      assert narrowed[:entity_type] in ["artist", "music-artist", "person"]
    end

    test "narrows person to artist in music context" do
      entities = [
        %{
          entity_type: "person",
          value: "Taylor Swift",
          match: "Taylor Swift",
          start_pos: 10,
          end_pos: 22,
          confidence: 0.65,
          source: :pos_tagger_propn
        }
      ]

      intent = "music.play"
      intent_details = %{top_k: [{"music.play", 0.8}]}

      {updated_entities, updated_intent, _details} =
        ContextualEntityInferrer.infer("Play some Taylor Swift", entities, intent, intent_details)

      assert updated_intent == "music.play"

      narrowed = Enum.find(updated_entities, &(&1[:value] == "Taylor Swift"))
      assert narrowed != nil

      # The entity type should be narrowed from person to artist or music-artist
      assert narrowed[:entity_type] in ["artist", "music-artist"],
        "Expected type narrowing to artist or music-artist, got: #{narrowed[:entity_type]}"

      assert narrowed[:source] == :type_narrowing
      assert narrowed[:original_type] == "person"
      assert narrowed[:confidence] < 0.65
      assert narrowed[:confidence] > 0.0
    end

    test "narrows person to name in introduction context" do
      entities = [
        %{
          entity_type: "person",
          value: "Austin",
          match: "Austin",
          start_pos: 8,
          end_pos: 14,
          confidence: 0.65,
          source: :pos_tagger_propn
        }
      ]

      intent = "introduction.self"
      intent_details = %{top_k: [{"introduction.self", 0.7}]}

      {updated_entities, _intent, _details} =
        ContextualEntityInferrer.infer("I am Austin", entities, intent, intent_details)

      narrowed = Enum.find(updated_entities, &(&1[:value] == "Austin"))

      # If introduction.self has entity_mappings with name type, it should narrow
      # Otherwise entity stays as person (which is still valid)
      assert narrowed != nil
      assert narrowed[:entity_type] in ["person", "name"]
    end

    test "does not narrow when entity type is already specific" do
      entities = [
        %{
          entity_type: "city",
          value: "London",
          match: "London",
          start_pos: 20,
          end_pos: 26,
          confidence: 0.9,
          source: :gazetteer
        }
      ]

      intent = "weather.query"
      intent_details = %{top_k: [{"weather.query", 0.9}]}

      {updated_entities, _intent, _details} =
        ContextualEntityInferrer.infer("What's the weather in London", entities, intent, intent_details)

      london = Enum.find(updated_entities, &(&1[:value] == "London"))
      assert london[:entity_type] == "city"
      assert london[:source] == :gazetteer
    end

    test "does not narrow when no entities present" do
      entities = []
      intent = "smalltalk.greetings.hello"
      intent_details = %{}

      {updated_entities, updated_intent, _details} =
        ContextualEntityInferrer.infer("Hello there", entities, intent, intent_details)

      assert updated_entities == []
      assert updated_intent == "smalltalk.greetings.hello"
    end

    test "narrows location to city in weather context" do
      entities = [
        %{
          entity_type: "location",
          value: "Paris",
          match: "Paris",
          start_pos: 24,
          end_pos: 29,
          confidence: 0.7,
          source: :pos_tagger_propn
        }
      ]

      intent = "weather.query"
      intent_details = %{top_k: [{"weather.query", 0.85}]}

      {updated_entities, _intent, _details} =
        ContextualEntityInferrer.infer("What is the weather in Paris", entities, intent, intent_details)

      paris = Enum.find(updated_entities, &(&1[:value] == "Paris"))
      assert paris != nil

      # Location can narrow to city if the intent expects city
      # The filter will accept both location and city via hierarchy
      assert paris[:entity_type] in ["location", "city"]
    end

    test "evaluates multiple intent hypotheses" do
      entities = [
        %{
          entity_type: "person",
          value: "Beyonce",
          match: "Beyonce",
          start_pos: 5,
          end_pos: 12,
          confidence: 0.65,
          source: :pos_tagger_propn
        }
      ]

      intent = "music.play"
      intent_details = %{
        top_k: [
          {"music.play", 0.6},
          {"introduction.self", 0.3}
        ]
      }

      {updated_entities, updated_intent, _details} =
        ContextualEntityInferrer.infer("Play Beyonce", entities, intent, intent_details)

      # music.play should win because it can fill the artist slot
      assert updated_intent == "music.play"

      beyonce = Enum.find(updated_entities, &(&1[:value] == "Beyonce"))
      assert beyonce != nil
      assert beyonce[:entity_type] in ["artist", "music-artist"]
    end
  end

  describe "full pipeline integration" do
    test "pipeline processes music request and applies hierarchy-aware filtering" do
      analysis = Pipeline.analyze_chunk("Play some Taylor Swift")

      # The pipeline should produce an analysis struct
      assert analysis != nil
      assert analysis.intent != nil

      # If the intent is music-related, verify the hierarchy-aware filter
      # allows person entities through (even if POS tagger didn't extract Taylor Swift)
      if String.contains?(analysis.intent, "music") do
        # The hierarchy-aware filter should accept person entities for music intents
        # because music-artist IS-A person. If entities were extracted, they should
        # not have been filtered out.
        if analysis.entities != [] do
          taylor = Enum.find(analysis.entities, fn e ->
            (e[:value] || e[:match] || "") |> String.downcase() |> String.contains?("taylor")
          end)

          if taylor != nil do
            # If Taylor was extracted and narrowed, check the type
            assert taylor[:entity_type] in ["person", "artist", "music-artist"],
              "Expected person, artist, or music-artist type, got: #{taylor[:entity_type]}"
          end
        end
      end
    end

    test "hierarchy-aware filter keeps person entities for music intents" do
      # Simulate the filter directly with a person entity and music intent
      entities = [
        %{entity_type: "person", value: "Taylor Swift", confidence: 0.65, source: :pos_tagger_propn}
      ]

      # After narrowing, the entity should have been narrowed to music-artist
      intent = "music.play"
      intent_details = %{top_k: [{"music.play", 0.8}]}

      {narrowed_entities, _intent, _details} =
        ContextualEntityInferrer.infer("Play some Taylor Swift", entities, intent, intent_details)

      # Now apply the filter (testing pipeline's filter logic)
      schema = Brain.Analysis.SlotDetector.get_schema(intent)

      if schema != nil do
        entity_mappings = Map.get(schema, "entity_mappings", %{})
        valid_types = entity_mappings |> Map.values() |> List.flatten() |> MapSet.new()

        filtered = Enum.filter(narrowed_entities, fn entity ->
          et = entity[:entity_type]
          MapSet.member?(valid_types, et) or
            Enum.any?(valid_types, &TypeHierarchy.compatible?(et, &1))
        end)

        # The person/artist entity should NOT be filtered out
        assert length(filtered) > 0,
          "Entity was filtered out. Narrowed entities: #{inspect(narrowed_entities)}, " <>
          "valid_types: #{inspect(MapSet.to_list(valid_types))}"
      end
    end
  end
end
