defmodule Brain.ML.NLPPipeline do
  @moduledoc "NLP Pipeline orchestrator that coordinates entity extraction and intent classification.\n\nThis module provides the main entry point for classical NLP processing,\ncombining:\n- Gazetteer-based entity lookup for known entities\n- BIO-tagged entity recognition for unknown entities\n- TF-IDF based intent classification\n- Unicode-aware tokenization\n\nThe pipeline prioritizes speed and accuracy by using:\n1. Pre-built gazetteer lookups (O(1) average)\n2. Token-based pattern matching (no regex)\n3. Centroid-based intent classification\n"

  alias Brain.ML
  require Logger

  alias ML.{EntityExtractor, IntentClassifierSimple, Gazetteer, Tokenizer}

  @type pipeline_result :: %{
          intent: String.t(),
          confidence: float(),
          entities: list(EntityExtractor.entity_match()),
          context: String.t(),
          processing_method: :classical
        }

  @doc "Initialize the NLP pipeline by loading all required models and data.\nShould be called at application startup.\n"
  def init do
    Logger.info("Initializing NLP pipeline...")

    case Gazetteer.start_link() do
      {:ok, _pid} ->
        Logger.info("Gazetteer GenServer started")

      {:error, {:already_started, _pid}} ->
        Logger.debug("Gazetteer already running")

      {:error, reason} ->
        Logger.warning("Failed to start Gazetteer GenServer", %{reason: reason})
    end

    case Gazetteer.load_all() do
      {:ok, stats} ->
        Logger.info("Gazetteer loaded", stats)

      {:error, reason} ->
        Logger.warning("Gazetteer loading failed, will use fallback", %{reason: reason})
    end

    case EntityExtractor.load_entity_maps() do
      {:ok, maps} ->
        Logger.info("Entity maps loaded", %{count: map_size(maps)})

      {:error, reason} ->
        Logger.warning("Entity maps loading failed", %{reason: reason})
    end

    case IntentClassifierSimple.load_models() do
      {:ok, _model} ->
        Logger.info("Intent classifier loaded")

      {:error, reason} ->
        Logger.warning("Intent classifier loading failed", %{reason: reason})
    end

    Logger.info("NLP pipeline initialization complete")
    :ok
  end

  @doc "Main entry point for text processing using classical NLP.\nReturns {:ok, result} or {:error, reason}.\n\n## Options\n\n- `:discourse` - Discourse analysis result for entity disambiguation\n- `:speech_act` - Speech act classification for entity disambiguation\n"
  def process(text, opts \\ []) do
    Logger.debug("Processing text with classical NLP", %{text: text})

    try do
      tokens = Tokenizer.tokenize(text)
      Logger.debug("Tokenized input", %{token_count: length(tokens)})
      entities = EntityExtractor.extract_entities(text, opts)
      Logger.debug("Extracted entities", %{count: length(entities)})

      case IntentClassifierSimple.classify(text) do
        {:ok, %{intent: intent, confidence: confidence}} ->
          Logger.debug("Classified intent", %{intent: intent, confidence: confidence})

          if should_use_classical_result?(confidence) do
            result = build_result(text, intent, confidence, entities, tokens)
            {:ok, result}
          else
            Logger.debug("Low confidence, marked for fallback", %{
              confidence: confidence,
              threshold: get_confidence_threshold()
            })

            {:ok,
             %{
               confidence: confidence,
               intent: intent,
               entities: entities,
               fallback: true,
               tokens: tokens
             }}
          end

        {:error, reason} ->
          Logger.warning("Intent classification failed", %{reason: reason})

          {:ok,
           %{
             confidence: 0.0,
             intent: "unknown",
             entities: entities,
             fallback: true,
             tokens: tokens
           }}
      end
    rescue
      error ->
        stacktrace = __STACKTRACE__
        Logger.error("NLP pipeline failed: #{Exception.message(error)}")
        Logger.error("Stacktrace: #{Exception.format_stacktrace(stacktrace)}")
        {:error, "Pipeline processing failed: #{inspect(error)}"}
    end
  end

  @doc "Process text with enhanced entity extraction using the BIO model.\nUse this for more thorough entity detection at the cost of speed.\n"
  def process_enhanced(text) do
    Logger.debug("Processing text with enhanced NLP", %{text: text})

    try do
      tokens = Tokenizer.tokenize(text)
      entities = EntityExtractor.extract_entities_with_model(text)
      Logger.debug("Enhanced entity extraction", %{count: length(entities)})

      case IntentClassifierSimple.classify(text) do
        {:ok, %{intent: intent, confidence: confidence}} ->
          result = build_result(text, intent, confidence, entities, tokens)
          {:ok, result}

        {:error, _reason} ->
          {:ok,
           %{
             confidence: 0.0,
             intent: "unknown",
             entities: entities,
             fallback: true
           }}
      end
    rescue
      error ->
        Logger.error("Enhanced NLP pipeline failed", %{error: inspect(error)})
        {:error, "Pipeline processing failed: #{inspect(error)}"}
    end
  end

  @doc "Extract features from text (entities + intent) for learning.\n\nOptions:\n- `:discourse` - Discourse analysis result for entity disambiguation\n- `:speech_act` - Speech act classification result for entity disambiguation\n"
  def extract_features(text, opts \\ []) do
    tokens = Tokenizer.tokenize(text)
    entities = EntityExtractor.extract_entities(text, opts)

    case IntentClassifierSimple.classify(text) do
      {:ok, %{intent: intent, confidence: confidence}} ->
        %{
          intent: intent,
          confidence: confidence,
          entities: entities,
          tokens: Enum.map(tokens, & &1.normalized),
          text: text
        }

      {:error, _reason} ->
        %{
          intent: "unknown",
          confidence: 0.0,
          entities: entities,
          tokens: Enum.map(tokens, & &1.normalized),
          text: text
        }
    end
  end

  @doc "Tokenize text using the pipeline's tokenizer.\nExposed for external use.\n"
  def tokenize(text) do
    Tokenizer.tokenize(text)
  end

  @doc "Normalize text for comparison.\n"
  def normalize(text) do
    Tokenizer.normalize(text)
  end

  @doc "Check if classical result should be used based on confidence threshold.\n"
  def should_use_classical_result?(confidence) do
    threshold = get_confidence_threshold()
    confidence >= threshold
  end

  @doc "Check if the pipeline is ready (models loaded).\n"
  def ready? do
    Gazetteer.loaded?() or EntityExtractor.get_entity_maps() != %{}
  end

  defp get_confidence_threshold do
    Application.get_env(:brain, :ml)[:confidence_threshold] || 0.75
  end

  defp build_result(text, intent, confidence, entities, tokens) do
    context = build_context(text, intent, entities)
    formatted_entities = format_entities_for_learner(entities)
    relationships = build_relationships(entities)
    facts = build_facts(entities, intent)

    token_info = %{
      count: length(tokens),
      words: Enum.filter(tokens, &(&1.type == :word)) |> length(),
      numbers: Enum.filter(tokens, &(&1.type == :number)) |> length()
    }

    %{
      intent: intent,
      confidence: confidence,
      entities: formatted_entities,
      relationships: relationships,
      facts: facts,
      context: context,
      processing_method: :classical,
      token_info: token_info
    }
  end

  defp build_context(_text, intent, entities) do
    entity_summary =
      entities
      |> Enum.take(5)
      |> Enum.map_join(
        ", ",
        fn entity ->
          "#{Map.get(entity, :entity_type, "unknown")}: #{Map.get(entity, :value, "")}"
        end
      )

    if String.length(entity_summary) > 0 do
      "#{intent} with #{entity_summary}"
    else
      intent
    end
  end

  defp format_entities_for_learner(entities) do
    Enum.map(entities, fn entity ->
      %{
        entity_type: Map.get(entity, :entity_type, "unknown"),
        value: Map.get(entity, :value, ""),
        match: Map.get(entity, :match, ""),
        start_pos: Map.get(entity, :start_pos, 0),
        end_pos: Map.get(entity, :end_pos, 0),
        confidence: Map.get(entity, :confidence, 0.5)
      }
    end)
  end

  defp build_relationships(entities) do
    relationships = []
    devices = Enum.filter(entities, fn e -> Map.get(e, :entity_type) == "device" end)
    rooms = Enum.filter(entities, fn e -> Map.get(e, :entity_type) == "room" end)

    device_room_relationships =
      for device <- devices, room <- rooms do
        %{
          "subject" => Map.get(device, :value),
          "relation" => "located_in",
          "object" => Map.get(room, :value),
          "confidence" =>
            min(
              Map.get(device, :confidence, 0.5),
              Map.get(room, :confidence, 0.5)
            )
        }
      end

    locations = Enum.filter(entities, fn e -> Map.get(e, :entity_type) == "location" end)

    times =
      Enum.filter(entities, fn e ->
        Map.get(e, :entity_type) in ["relative_date", "date", "day_name"]
      end)

    location_time_relationships =
      for location <- locations, time <- times do
        %{
          "subject" => "query",
          "relation" => "for_location_at_time",
          "location" => Map.get(location, :value),
          "time" => Map.get(time, :value),
          "confidence" =>
            min(
              Map.get(location, :confidence, 0.5),
              Map.get(time, :confidence, 0.5)
            )
        }
      end

    relationships ++ device_room_relationships ++ location_time_relationships
  end

  defp build_facts(entities, intent) do
    intent_fact = %{
      "type" => "intent",
      "value" => intent,
      "confidence" => 0.9
    }

    entity_facts =
      entities
      |> Enum.map(fn entity ->
        %{
          "type" => "entity",
          "entity_type" => Map.get(entity, :entity_type, "unknown"),
          "value" => Map.get(entity, :value, ""),
          "confidence" => Map.get(entity, :confidence, 0.5)
        }
      end)

    [intent_fact | entity_facts]
  end
end