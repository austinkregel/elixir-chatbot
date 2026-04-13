defmodule Brain.ML.NovelEntityExtractionTest do
  @moduledoc """
  TDD tests for novel entity extraction from context.

  These tests define the desired behavior for extracting entities that
  are NOT in the Gazetteer, using POS tagging and multi-word merging.

  The pipeline we're testing:
    Raw text → POS tagger (PROPN detection) → multi-word merging →
    ContextualEntityInferrer (type narrowing) → slot filling
  """
  use Brain.Test.GraphCase, async: false

  alias Brain.ML.{POSTagger, EntityExtractor}

  @moduletag :requires_models

  # ============================================================================
  # Layer 1: POS Tagger — capitalized words in non-initial position → PROPN
  # ============================================================================

  describe "POS tagger proper noun detection" do
    setup do
      case POSTagger.load_model() do
        {:ok, model} -> {:ok, model: model}
        {:error, _} -> {:ok, model: nil}
      end
    end

    test "unknown capitalized word after verb is tagged as PROPN", %{model: model} do
      if model == nil do
        flunk("POS model not loaded -- cannot test PROPN detection")
      end

      # "Korvo" is NOT in training data at all — truly novel
      predictions = POSTagger.predict(["Play", "Korvo"], model)

      {_, tag} = Enum.at(predictions, 1)

      assert tag == "PROPN",
        "Expected 'Korvo' to be tagged as PROPN, got: #{tag}. " <>
        "Full tags: #{inspect(predictions)}"
    end

    test "consecutive unknown capitalized words are tagged as PROPN", %{model: model} do
      if model == nil do
        flunk("POS model not loaded -- cannot test PROPN detection")
      end

      # Neither "Korvo" nor "Mitski" appear in training data
      predictions = POSTagger.predict(["Play", "some", "Korvo", "Mitski"], model)

      tags = Enum.map(predictions, fn {_word, tag} -> tag end)

      assert Enum.at(tags, 2) == "PROPN",
        "Expected 'Korvo' at position 2 to be PROPN, got: #{Enum.at(tags, 2)}. " <>
        "Full: #{inspect(predictions)}"

      assert Enum.at(tags, 3) == "PROPN",
        "Expected 'Mitski' at position 3 to be PROPN, got: #{Enum.at(tags, 3)}. " <>
        "Full: #{inspect(predictions)}"
    end

    test "unknown capitalized words after preposition are PROPN", %{model: model} do
      if model == nil do
        flunk("POS model not loaded -- cannot test PROPN detection")
      end

      predictions = POSTagger.predict(["Listen", "to", "Xiomara", "Yiruma"], model)
      tags = Enum.map(predictions, fn {_word, tag} -> tag end)

      assert Enum.at(tags, 2) == "PROPN",
        "Expected 'Xiomara' to be PROPN, got: #{Enum.at(tags, 2)}"

      assert Enum.at(tags, 3) == "PROPN",
        "Expected 'Yiruma' to be PROPN, got: #{Enum.at(tags, 3)}"
    end
  end

  # ============================================================================
  # Layer 2: Entity Extractor — consecutive PROPNs merge into one entity
  # ============================================================================

  describe "EntityExtractor multi-word proper noun merging" do
    setup do
      ensure_started(EntityExtractor)
      :ok
    end

    test "consecutive novel PROPN tokens merge into a single entity" do
      # "Korvo Mitski" — neither word is in the training data or Gazetteer
      entities = EntityExtractor.extract_entities("Play some Korvo Mitski")

      merged = Enum.find(entities, fn e ->
        value = e[:value] || e[:match] || ""
        String.downcase(value) =~ "korvo" and String.downcase(value) =~ "mitski"
      end)

      assert merged != nil,
        "Expected 'Korvo Mitski' as a merged entity. " <>
        "Got entities: #{inspect(Enum.map(entities, & &1[:value]))}"

      assert merged[:value] == "Korvo Mitski",
        "Expected merged value 'Korvo Mitski', got: #{merged[:value]}"
    end

    test "single novel PROPN token stays as single entity" do
      entities = EntityExtractor.extract_entities("Hello Xiomara")

      xiomara = Enum.find(entities, fn e ->
        value = e[:value] || e[:match] || ""
        String.downcase(value) == "xiomara"
      end)

      assert xiomara != nil,
        "Expected 'Xiomara' to be extracted as an entity. " <>
        "Got entities: #{inspect(Enum.map(entities, & &1[:value]))}"

      assert xiomara[:value] == "Xiomara"
    end

    test "three consecutive novel PROPNs merge into one entity" do
      entities = EntityExtractor.extract_entities("Play some Korvo Xiomara Mitski")

      merged = Enum.find(entities, fn e ->
        value = e[:value] || e[:match] || ""
        String.downcase(value) =~ "korvo" and String.downcase(value) =~ "mitski"
      end)

      assert merged != nil,
        "Expected 'Korvo Xiomara Mitski' as a merged entity. " <>
        "Got entities: #{inspect(Enum.map(entities, & &1[:value]))}"
    end

    test "PROPN separated by non-PROPN are separate entities" do
      entities = EntityExtractor.extract_entities("Korvo loves Xiomara")

      merged = Enum.find(entities, fn e ->
        value = e[:value] || e[:match] || ""
        String.downcase(value) =~ "korvo" and String.downcase(value) =~ "xiomara"
      end)

      assert merged == nil,
        "Expected 'Korvo' and 'Xiomara' as separate entities, not merged"
    end
  end

  # ============================================================================
  # Layer 3: Full extraction with type narrowing
  # ============================================================================

  describe "novel entity extraction with contextual type narrowing" do
    setup do
      Brain.TestHelpers.require_services!(:brain)
      :ok
    end

    test "novel artist name is extracted, narrowed, and fills music slot" do
      alias Brain.Analysis.Pipeline

      # "Korvo Mitski" — completely absent from training data and Gazetteer
      analysis = Pipeline.analyze_chunk("Play some Korvo Mitski")

      entity = Enum.find(analysis.entities, fn e ->
        value = e[:value] || e[:match] || ""
        String.downcase(value) =~ "korvo"
      end)

      assert entity != nil,
        "Expected 'Korvo Mitski' entity to be extracted from analysis. " <>
        "Entities: #{inspect(Enum.map(analysis.entities, &{&1[:value], &1[:entity_type]}))}"

      assert entity[:entity_type] in ["artist", "music-artist", "person"],
        "Expected entity type artist/music-artist/person, got: #{entity[:entity_type]}"

      assert analysis.intent =~ "music",
        "Expected music intent, got: #{analysis.intent}"
    end

    test "novel person name is extracted in introduction context" do
      alias Brain.Analysis.Pipeline

      # "Xiomara" — not in training data or Gazetteer
      analysis = Pipeline.analyze_chunk("My name is Xiomara")

      entity = Enum.find(analysis.entities, fn e ->
        value = e[:value] || e[:match] || ""
        String.downcase(value) =~ "xiomara"
      end)

      assert entity != nil,
        "Expected 'Xiomara' to be extracted as an entity. " <>
        "Entities: #{inspect(Enum.map(analysis.entities, &{&1[:value], &1[:entity_type]}))}"

      assert entity[:entity_type] in ["person", "name"],
        "Expected person/name type, got: #{entity[:entity_type]}"
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp ensure_started(child_spec) do
    case start_supervised(child_spec) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
      {:error, _reason} -> :ok
    end
  end
end
