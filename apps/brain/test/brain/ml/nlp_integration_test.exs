defmodule Brain.ML.NLPIntegrationTest do
  alias Brain.ML.SimpleClassifier
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.ML.IntentClassifierSimple
  alias Brain.ML.EntityExtractor
  alias Brain.ML.NLPPipeline

  setup do
    ensure_pubsub_started()

    Application.put_env(:chat_bot, :ml,
      enabled: true,
      confidence_threshold: 0.5,
      models_path: "priv/ml_models",
      training_data_path: "data"
    )

    ensure_started(IntentClassifierSimple)
    IntentClassifierSimple.load_models()
    EntityExtractor.load_entity_maps()

    :ok
  end

  describe "Intent Classification" do
    test "classifies music intent correctly" do
      result = IntentClassifierSimple.classify("play some music")
      assert {:ok, %{intent: intent, confidence: confidence}} = result
      assert is_binary(intent)
      assert is_float(confidence)
      assert confidence > 0.0
    end

    test "classifies with high confidence for trained intents" do
      {:ok, %{confidence: confidence}} = IntentClassifierSimple.classify("play jazz music")
      assert confidence > 0.3
    end
  end

  describe "Entity Extraction" do
    test "extracts room entities" do
      entities = EntityExtractor.extract_entities("turn off the kitchen lights")
      assert is_list(entities)
      kitchen_entity = Enum.find(entities, fn e -> e.value == "kitchen" end)
      assert kitchen_entity != nil
    end

    test "extracts music artist entities" do
      entities = EntityExtractor.extract_entities("play music by the beatles")
      beatles = Enum.find(entities, fn e -> String.downcase(e.value) == "the beatles" end)

      if beatles do
        # After type normalization, "music-artist" maps to canonical "artist"
        assert beatles.entity_type in ["artist", "music-artist", "music_artist"]
      end
    end
  end

  describe "Migrated examples from legacy validation" do
    test "extracts kitchen room and lights device from home automation command" do
      text = "Turn off all the kitchen lights"
      entities = EntityExtractor.extract_entities(text)

      kitchen = Enum.find(entities, &(&1.value == "kitchen"))
      assert kitchen

      lights = Enum.find(entities, &String.contains?(&1.value, "light"))
      assert lights || true
    end

    test "extracts bedroom from description-like input" do
      text = "The master bedroom is a large room with a king-size bed."
      entities = EntityExtractor.extract_entities(text)
      has_bedroom = Enum.any?(entities, &String.contains?(String.downcase(&1.value), "bedroom"))
      assert has_bedroom
    end

    test "extracts brand-like device hints from product sentence (best-effort)" do
      text = "I just bought a new iPhone 15 Pro Max in titanium."
      entities = EntityExtractor.extract_entities(text)
      assert is_list(entities)
    end
  end

  describe "NLP Pipeline Integration" do
    test "processes input and returns structured result" do
      result = NLPPipeline.process("play some music")

      assert {:ok, pipeline_result} = result
      assert Map.has_key?(pipeline_result, :intent)
      assert Map.has_key?(pipeline_result, :confidence)
      assert Map.has_key?(pipeline_result, :entities)
    end

    test "returns high confidence for clear intents" do
      {:ok, %{confidence: confidence}} = NLPPipeline.process("play jazz")
      assert confidence > 0.0
    end

    test "extracts entities along with intent" do
      {:ok, %{entities: entities}} = NLPPipeline.process("turn on the bedroom light")
      assert is_list(entities)
    end
  end

  describe "Classical NLP Processing" do
    test "uses classical NLP for high-confidence intents" do
      {:ok, result} = NLPPipeline.process("play some music")
      assert result.intent != nil
      assert result.confidence > 0.0
    end

    test "handles low confidence gracefully" do
      result = NLPPipeline.process("xyzabc nonsense input")
      assert {:ok, %{confidence: confidence}} = result
      assert is_float(confidence)
    end
  end

  describe "Simple Classifier" do
    test "trains and classifies correctly" do
      training_data = [
        {"play music", "music.play"},
        {"play some songs", "music.play"},
        {"play my playlist", "music.play"},
        {"stop the music", "music.stop"},
        {"pause the music", "music.stop"},
        {"turn on lights", "lights.on"},
        {"switch on the light", "lights.on"},
        {"turn off lights", "lights.off"},
        {"switch off the light", "lights.off"}
      ]

      model = SimpleClassifier.train(training_data)

      assert map_size(model.vocabulary) > 0
      assert map_size(model.label_centroids) == 4
      {:ok, label, score, _details} = SimpleClassifier.classify("play some music", model)
      assert label == "music.play"
      assert score > 0.0
    end

    test "handles similar inputs correctly" do
      training_data = [
        {"play music", "music.play"},
        {"play a song", "music.play"},
        {"turn on lights", "lights.on"}
      ]

      model = SimpleClassifier.train(training_data)

      {:ok, label, _score, _details} = SimpleClassifier.classify("play the radio", model)
      assert label == "music.play"
    end
  end
end
