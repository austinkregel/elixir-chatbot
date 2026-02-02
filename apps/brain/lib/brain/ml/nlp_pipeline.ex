defmodule Brain.ML.NLPPipeline do
  @moduledoc """
  NLP Pipeline orchestrator that coordinates entity extraction and intent classification.

  This module provides the main entry point for classical NLP processing,
  combining:
  - Gazetteer-based entity lookup for known entities
  - BIO-tagged entity recognition for unknown entities
  - TF-IDF based intent classification
  - Unicode-aware tokenization

  The pipeline prioritizes speed and accuracy by using:
  1. Pre-built gazetteer lookups (O(1) average)
  2. Token-based pattern matching (no regex)
  3. Centroid-based intent classification
  """

  require Logger

  alias Brain.ML.{EntityExtractor, IntentClassifierSimple, Gazetteer, Tokenizer}

  @type pipeline_result :: %{
          intent: String.t(),
          confidence: float(),
          entities: list(EntityExtractor.entity_match()),
          context: String.t(),
          processing_method: :classical
        }

  # ============================================================================
  # Initialization
  # ============================================================================

  @doc """
  Initialize the NLP pipeline by loading all required models and data.
  Should be called at application startup.
  """
  def init do
    Logger.info("Initializing NLP pipeline...")

    # Start the Gazetteer if not already running
    case Gazetteer.start_link() do
      {:ok, _pid} ->
        Logger.info("Gazetteer GenServer started")

      {:error, {:already_started, _pid}} ->
        Logger.debug("Gazetteer already running")

      {:error, reason} ->
        Logger.warning("Failed to start Gazetteer GenServer", %{reason: reason})
    end

    # Load gazetteer data
    case Gazetteer.load_all() do
      {:ok, stats} ->
        Logger.info("Gazetteer loaded", stats)

      {:error, reason} ->
        Logger.warning("Gazetteer loading failed, will use fallback", %{reason: reason})
    end

    # Load entity maps as fallback
    case EntityExtractor.load_entity_maps() do
      {:ok, maps} ->
        Logger.info("Entity maps loaded", %{count: map_size(maps)})

      {:error, reason} ->
        Logger.warning("Entity maps loading failed", %{reason: reason})
    end

    # Load intent classifier
    case IntentClassifierSimple.load_models() do
      {:ok, _model} ->
        Logger.info("Intent classifier loaded")

      {:error, reason} ->
        Logger.warning("Intent classifier loading failed", %{reason: reason})
    end

    Logger.info("NLP pipeline initialization complete")
    :ok
  end

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Main entry point for text processing using classical NLP.
  Returns {:ok, result} or {:error, reason}.

  ## Options

  - `:discourse` - Discourse analysis result for entity disambiguation
  - `:speech_act` - Speech act classification for entity disambiguation
  """
  def process(text, opts \\ []) do
    Logger.debug("Processing text with classical NLP", %{text: text})

    try do
      # Tokenize input for analysis
      tokens = Tokenizer.tokenize(text)
      Logger.debug("Tokenized input", %{token_count: length(tokens)})

      # Extract entities using gazetteer and patterns
      # Pass context for disambiguation if available
      entities = EntityExtractor.extract_entities(text, opts)
      Logger.debug("Extracted entities", %{count: length(entities)})

      # Classify intent
      case IntentClassifierSimple.classify(text) do
        {:ok, %{intent: intent, confidence: confidence}} ->
          Logger.debug("Classified intent", %{intent: intent, confidence: confidence})

          # Check if we should use this result
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
          # Return entities even if intent fails
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

  @doc """
  Process text with enhanced entity extraction using the BIO model.
  Use this for more thorough entity detection at the cost of speed.
  """
  def process_enhanced(text) do
    Logger.debug("Processing text with enhanced NLP", %{text: text})

    try do
      tokens = Tokenizer.tokenize(text)

      # Use enhanced entity extraction with BIO model
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

  @doc """
  Extract features from text (entities + intent) for learning.

  Options:
  - `:discourse` - Discourse analysis result for entity disambiguation
  - `:speech_act` - Speech act classification result for entity disambiguation
  """
  def extract_features(text, opts \\ []) do
    tokens = Tokenizer.tokenize(text)

    # Extract entities with disambiguation context if available
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

  @doc """
  Tokenize text using the pipeline's tokenizer.
  Exposed for external use.
  """
  def tokenize(text) do
    Tokenizer.tokenize(text)
  end

  @doc """
  Normalize text for comparison.
  """
  def normalize(text) do
    Tokenizer.normalize(text)
  end

  @doc """
  Check if classical result should be used based on confidence threshold.
  """
  def should_use_classical_result?(confidence) do
    threshold = get_confidence_threshold()
    confidence >= threshold
  end

  @doc """
  Check if the pipeline is ready (models loaded).
  """
  def ready? do
    Gazetteer.loaded?() or EntityExtractor.get_entity_maps() != %{}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_confidence_threshold do
    Application.get_env(:brain, :ml)[:confidence_threshold] || 0.75
  end

  defp build_result(text, intent, confidence, entities, tokens) do
    # Build context from entities and intent
    context = build_context(text, intent, entities)

    # Format entities for compatibility with existing Learner module
    formatted_entities = format_entities_for_learner(entities)

    # Build relationships from entities
    relationships = build_relationships(entities)

    # Build facts from entities
    facts = build_facts(entities, intent)

    # Build token info
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
      # Limit summary length
      |> Enum.take(5)
      |> Enum.map(fn entity ->
        "#{Map.get(entity, :entity_type, "unknown")}: #{Map.get(entity, :value, "")}"
      end)
      |> Enum.join(", ")

    if String.length(entity_summary) > 0 do
      "#{intent} with #{entity_summary}"
    else
      intent
    end
  end

  defp format_entities_for_learner(entities) do
    # Standardized format with :entity_type key
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
    # Build simple relationships between entities
    relationships = []

    # Look for device-room relationships
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

    # Look for location-time relationships
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
    # Add intent-based facts
    intent_fact = %{
      "type" => "intent",
      "value" => intent,
      "confidence" => 0.9
    }

    # Add entity-based facts
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
