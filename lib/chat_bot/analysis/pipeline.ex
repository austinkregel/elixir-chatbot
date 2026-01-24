defmodule ChatBot.Analysis.Pipeline do
  @moduledoc """
  Orchestrates the text analysis pipeline.

  The pipeline processes user input through multiple stages:
  1. Semantic chunking (break input into utterances)
  2. Parallel analysis (discourse + speech act classification)
  3. Sequential analysis (slot detection + context resolution)
  4. Internal model building (combine all results)

  Each stage builds on the previous, creating a comprehensive
  understanding of the user's input.
  """

  alias ChatBot.Analysis.{
    InternalModel,
    Chunk,
    ChunkAnalysis,
    SemanticChunker,
    DiscourseAnalyzer,
    SpeechActClassifier,
    SlotDetector,
    ContextResolver,
    AnaphoraResolver,
    LearningStore,
    Progress
  }

  alias ChatBot.ML.EntityExtractor

  require Logger

  @doc """
  Processes user input through the complete analysis pipeline.

  Options:
  - :participants - conversation participants (default: [:user, :bot])
  - :bot_names - additional names the bot responds to
  - :conversation_history - list of previous context snapshots for slot resolution
  - :user_profile - map of user preferences (location, timezone, etc.)
  - :skip_entity_extraction - if true, skips entity extraction (for testing)
  - :entities - pre-extracted entities to use instead of extracting

  Returns an InternalModel struct with complete analysis.
  """
  def process(text, opts \\ []) when is_binary(text) do
    # Wrap with telemetry span for async, non-blocking metrics
    ChatBot.Telemetry.span(:pipeline_process, %{text_length: String.length(text)}, fn ->
      do_process(text, opts)
    end)
  end

  defp do_process(text, opts) do
    Logger.debug("Starting analysis pipeline", %{text_length: String.length(text)})

    start_time = System.monotonic_time(:millisecond)

    Progress.report(opts, :pipeline_start, %{text_length: String.length(text)})

    # Create the initial model
    model = InternalModel.new(text)

    # Stage 1: Chunk the input
    chunks = SemanticChunker.chunk(text)
    model = InternalModel.with_chunks(model, chunks)

    Logger.debug("Chunking complete", %{chunk_count: length(chunks)})
    Progress.report(opts, :chunking_complete, %{chunk_count: length(chunks)})

    # Stage 2 & 3: Analyze each chunk
    analyses = analyze_chunks(chunks, opts)
    model = InternalModel.with_analyses(model, analyses)

    # Stage 4: Determine overall strategy
    model = InternalModel.determine_strategy(model)

    # Build strategy reasoning for debug inspector
    chunk_strategies = Enum.map(analyses, & &1.response_strategy)
    has_expressives = Enum.any?(analyses, &(&1.speech_act.category == :expressive))
    has_substantive = Enum.any?(analyses, fn a ->
      a.speech_act.category in [:directive, :assertive] or a.speech_act.is_question
    end)
    all_missing = Enum.flat_map(analyses, & &1.missing_context)

    decision_reason = cond do
      Enum.all?(chunk_strategies, &(&1 == :can_respond)) ->
        "All #{length(chunk_strategies)} chunk(s) can respond"
      Enum.all?(chunk_strategies, &(&1 == :cannot_respond)) ->
        "No chunks can respond"
      Enum.all?(chunk_strategies, &(&1 == :defer_to_user)) ->
        "Bot was not addressed in any chunk"
      length(all_missing) > 0 and Enum.any?(chunk_strategies, &(&1 == :can_respond)) ->
        "Partial: can respond to some, missing slots: #{Enum.join(all_missing, ", ")}"
      length(all_missing) > 0 ->
        "Missing required slots: #{Enum.join(all_missing, ", ")}"
      true ->
        "Default strategy applied"
    end

    Progress.report(opts, :strategy_determined, %{
      overall_strategy: model.overall_strategy,
      chunk_strategies: chunk_strategies,
      has_expressives: has_expressives,
      has_substantive: has_substantive,
      missing_slots_count: length(all_missing),
      missing_slots: all_missing,
      decision_reason: decision_reason,
      suggested_prompts: model.suggested_prompts
    })

    # Record timing
    elapsed = System.monotonic_time(:millisecond) - start_time

    Logger.debug("Pipeline complete", %{
      chunk_count: length(chunks),
      strategy: model.overall_strategy,
      elapsed_ms: elapsed
    })

    # Record feedback for learning
    record_pipeline_result(model)

    Progress.report(opts, :pipeline_complete, %{elapsed_ms: elapsed})

    model
  end

  @doc """
  Processes a single chunk through the analysis pipeline.

  Useful for testing or when you already have chunks.
  """
  def analyze_chunk(chunk_text, opts \\ []) when is_binary(chunk_text) do
    chunk = Chunk.new(chunk_text, 0, 0, String.length(chunk_text) - 1)
    analyze_single_chunk(chunk, opts)
  end

  @doc """
  Returns a summary of the analysis for debugging/logging.
  """
  def summarize(%InternalModel{} = model) do
    %{
      input: String.slice(model.raw_input, 0, 50) <> "...",
      chunks: length(model.chunks),
      analyses:
        Enum.map(model.analyses, fn a ->
          %{
            text: String.slice(a.text, 0, 30),
            addressee: a.discourse.addressee,
            speech_act: {a.speech_act.category, a.speech_act.sub_type},
            intent: a.intent,
            strategy: a.response_strategy
          }
        end),
      overall_strategy: model.overall_strategy,
      prompts: model.suggested_prompts
    }
  end

  # Private functions

  defp analyze_chunks(chunks, opts) do
    # Process chunks - could be parallelized with Task.async_stream
    # but keeping simple for now
    Enum.map(chunks, fn chunk ->
      analyze_single_chunk(chunk, opts)
    end)
  end

  defp analyze_single_chunk(chunk, opts) do
    participants = Keyword.get(opts, :participants, [:user, :bot])
    bot_names = Keyword.get(opts, :bot_names, [])
    history = Keyword.get(opts, :conversation_history, [])
    profile = Keyword.get(opts, :user_profile, %{})

    Progress.report(opts, :chunk_start, %{
      chunk_index: chunk.index,
      chunk_text: chunk.text,
      chunk_length: String.length(chunk.text)
    })

    # Stage 2a: Discourse analysis (who is being addressed)
    discourse_task =
      Task.async(fn ->
        DiscourseAnalyzer.analyze(chunk.text,
          participants: participants,
          bot_names: bot_names
        )
      end)

    # Stage 2b: Speech act classification (what type of utterance)
    speech_act_task =
      Task.async(fn ->
        SpeechActClassifier.classify(chunk.text)
      end)

    # Wait for parallel tasks with timeout
    discourse_result =
      try do
        Task.await(discourse_task, 3000)
      catch
        :exit, {:timeout, _} ->
          Task.shutdown(discourse_task, :brutal_kill)
          DiscourseAnalyzer.analyze("")
      end

    speech_act_result =
      try do
        Task.await(speech_act_task, 3000)
      catch
        :exit, {:timeout, _} ->
          Task.shutdown(speech_act_task, :brutal_kill)
          SpeechActClassifier.classify("")
      end

    Progress.report(opts, :discourse_complete, %{
      chunk_index: chunk.index,
      addressee: Map.get(discourse_result, :addressee),
      confidence: Map.get(discourse_result, :confidence)
    })

    Progress.report(opts, :speech_act_complete, %{
      chunk_index: chunk.index,
      category: Map.get(speech_act_result, :category),
      sub_type: Map.get(speech_act_result, :sub_type),
      confidence: Map.get(speech_act_result, :confidence),
      is_question: Map.get(speech_act_result, :is_question)
    })

    # Stage 2c: Anaphora resolution (resolve pronouns/references from history)
    {resolved_text, anaphora_entities} =
      resolve_anaphora(chunk.text, history, chunk.index, opts)

    # Stage 3a: Entity extraction (use resolved text for better extraction)
    entities = extract_entities(resolved_text, opts)

    # Merge anaphora-resolved entities with extracted entities
    entities = merge_anaphora_entities(entities, anaphora_entities)

    Progress.report(opts, :entities_extracted, %{
      chunk_index: chunk.index,
      entity_count: length(entities),
      entities: entities |> Enum.take(25) |> Enum.map(&entity_to_dev_map/1)
    })

    # Stage 3b: Intent determination
    {intent, intent_method, intent_confidence} =
      determine_intent(speech_act_result, entities, chunk.text)

    Progress.report(opts, :intent_determined, %{
      chunk_index: chunk.index,
      intent: intent,
      intent_method: intent_method,
      intent_confidence: intent_confidence
    })

    # Stage 3b.5: Filter entities to only those relevant to the intent's slot schema
    # This prevents entities from being used for the wrong intent
    # (e.g., "Austin" as location when intent is smalltalk.greeting)
    relevant_entities = filter_entities_by_intent(entities, intent)

    Progress.report(opts, :entities_filtered, %{
      chunk_index: chunk.index,
      original_count: length(entities),
      filtered_count: length(relevant_entities),
      excluded_types:
        (Enum.map(entities, & &1[:entity]) -- Enum.map(relevant_entities, & &1[:entity]))
        |> Enum.uniq()
    })

    # Stage 3c: Slot detection (use only relevant entities)
    slot_result = SlotDetector.detect(intent, relevant_entities)

    Progress.report(opts, :slots_detected, %{
      chunk_index: chunk.index,
      missing_required: Map.get(slot_result, :missing_required, []),
      filled_count: map_size(Map.get(slot_result, :filled_slots, %{})),
      filled_slots: Map.get(slot_result, :filled_slots, %{})
    })

    # Stage 3d: Context resolution
    user_id = Keyword.get(opts, :user_id)

    resolved_slots =
      ContextResolver.resolve(slot_result,
        conversation_history: history,
        user_profile: profile,
        user_id: user_id
      )

    Progress.report(opts, :context_resolved, %{
      chunk_index: chunk.index,
      all_required_filled: Map.get(resolved_slots, :all_required_filled),
      missing_required: Map.get(resolved_slots, :missing_required, []),
      filled_slots: Map.get(resolved_slots, :filled_slots, %{})
    })

    # Build the chunk analysis
    # Store only slot-relevant entities to prevent cross-intent contamination
    analysis =
      ChunkAnalysis.new(chunk.index, chunk.text)
      |> Map.put(:discourse, discourse_result)
      |> Map.put(:speech_act, speech_act_result)
      |> Map.put(:intent, intent)
      |> Map.put(:entities, relevant_entities)
      |> Map.put(:slots, resolved_slots)
      |> Map.put(:missing_context, resolved_slots.missing_required)
      |> calculate_confidence()
      |> ChunkAnalysis.determine_response_strategy()

    Progress.report(opts, :chunk_complete, %{
      chunk_index: chunk.index,
      response_strategy: analysis.response_strategy,
      confidence: analysis.confidence
    })

    analysis
  end

  defp extract_entities(text, opts) do
    cond do
      Keyword.get(opts, :skip_entity_extraction, false) ->
        []

      Keyword.has_key?(opts, :entities) ->
        Keyword.get(opts, :entities, [])

      true ->
        # Use the existing entity extractor from ML module
        try do
          EntityExtractor.extract_entities(text)
        rescue
          _ -> []
        catch
          :exit, _ -> []
        end
    end
  end

  defp entity_to_dev_map(entity) when is_map(entity) do
    type =
      Map.get(entity, :entity) || Map.get(entity, "entity") || Map.get(entity, :type) ||
        Map.get(entity, "type")

    value =
      Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :name) ||
        Map.get(entity, "name")

    conf = Map.get(entity, :confidence) || Map.get(entity, "confidence")

    %{
      type: type,
      value: value,
      confidence: conf
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp entity_to_dev_map(other), do: %{value: inspect(other)}

  defp determine_intent(speech_act, entities, text) do
    # For expressive speech acts (greetings, farewells, thanks), use speech act directly
    # This prevents entity-based overrides (e.g., "Hello" matching song "Hello")
    if speech_act.category == :expressive do
      {infer_intent_from_speech_act(speech_act, text), :speech_act_expressive, nil}
    else
      # For non-expressive speech acts, try multiple strategies

      # 1. First, check if entities suggest an intent
      case SlotDetector.suggest_intent_from_entities(entities) do
        {:ok, intent, score} ->
          {intent, :entity_based, score}

        {:error, :no_match} ->
          # 2. Try keyword-based heuristic for substantive intents
          case SlotDetector.suggest_intent_from_keywords(text) do
            {:ok, intent, confidence} ->
              {intent, :keyword_heuristic, confidence}

            {:error, :no_match} ->
              # 3. Fall back to speech act based intent
              {infer_intent_from_speech_act(speech_act, text), :speech_act_fallback, nil}
          end
      end
    end
  end

  defp infer_intent_from_speech_act(speech_act, text) do
    lower_text = String.downcase(text)

    cond do
      # Greetings
      speech_act.sub_type == :greeting ->
        "smalltalk.greeting"

      # Farewells
      speech_act.sub_type == :farewell ->
        "smalltalk.farewell"

      # Thanks
      speech_act.sub_type == :thanks ->
        "smalltalk.thanks"

      # Weather related
      String.contains?(lower_text, "weather") ->
        "weather.query"

      # News related
      String.contains?(lower_text, "news") ->
        "news.query"

      # Music related
      String.contains?(lower_text, ["play", "music", "song", "album"]) ->
        "music.play"

      # Device control
      String.contains?(lower_text, ["turn on", "turn off", "switch", "light"]) ->
        "device.control"

      # Search related
      String.contains?(lower_text, ["search", "find", "look up", "google"]) ->
        "search.web"

      # Question (factual or opinion)
      speech_act.is_question ->
        if String.contains?(lower_text, ["think", "opinion", "feel"]) do
          "question.opinion"
        else
          "question.factual"
        end

      # Request for action
      speech_act.sub_type == :request_action ->
        "action.request"

      # Request for information
      speech_act.sub_type == :request_information ->
        "information.request"

      # General smalltalk
      speech_act.category == :expressive ->
        "smalltalk.general"

      # Unknown
      true ->
        "unknown"
    end
  end

  defp calculate_confidence(analysis) do
    # Calculate overall confidence as weighted average
    # Safely extract confidence values with defaults
    discourse_conf =
      case analysis.discourse do
        %{confidence: c} when is_number(c) -> c
        _ -> 0.5
      end

    speech_act_conf =
      case analysis.speech_act do
        %{confidence: c} when is_number(c) -> c
        _ -> 0.5
      end

    slot_conf =
      case analysis.slots do
        nil ->
          0.5

        %{all_required_filled: true} ->
          1.0

        %{filled_slots: filled, missing_required: missing} ->
          filled_count = map_size(filled || %{})
          missing_count = length(missing || [])
          total = missing_count + filled_count
          if total == 0, do: 1.0, else: filled_count / total

        _ ->
          0.5
      end

    # Weighted average - ensure all values are floats
    confidence =
      (discourse_conf * 0.3 + speech_act_conf * 0.4 + slot_conf * 0.3)
      |> Float.round(3)

    %{analysis | confidence: confidence}
  end

  defp record_pipeline_result(%InternalModel{} = model) do
    feedback_type =
      case model.overall_strategy do
        :can_respond -> :successful_response
        :needs_clarification -> :clarification_needed
        :partial_response_with_clarification -> :clarification_needed
        _ -> nil
      end

    if feedback_type do
      LearningStore.record_feedback(feedback_type, %{
        chunk_count: length(model.chunks),
        strategy: model.overall_strategy
      })
    end
  end

  # Anaphora resolution helpers

  defp resolve_anaphora(text, history, chunk_index, opts) do
    case AnaphoraResolver.resolve_and_substitute(text, history) do
      {:ok, resolved_text, resolved_entities} ->
        if length(resolved_entities) > 0 do
          Progress.report(opts, :anaphora_resolved, %{
            chunk_index: chunk_index,
            resolved_count: length(resolved_entities),
            entities:
              Enum.map(resolved_entities, fn e ->
                %{
                  entity: e[:entity] || e["entity"],
                  value: e[:value] || e["value"]
                }
              end)
          })
        end

        {resolved_text, resolved_entities}

      _ ->
        {text, []}
    end
  rescue
    e ->
      Logger.warning("Anaphora resolution failed", %{error: Exception.message(e)})
      {text, []}
  end

  defp merge_anaphora_entities(entities, anaphora_entities) when is_list(anaphora_entities) do
    # Convert anaphora entities to the expected format
    converted =
      Enum.map(anaphora_entities, fn e ->
        %{
          entity: e[:entity] || e["entity"],
          value: e[:value] || e["value"],
          confidence: 0.75,
          source: :anaphora_resolution
        }
      end)

    # Merge, avoiding duplicates (prefer extracted over resolved)
    extracted_types =
      entities
      |> Enum.map(&(&1[:entity] || &1["entity"]))
      |> MapSet.new()

    unique_anaphora =
      Enum.reject(converted, fn e ->
        MapSet.member?(extracted_types, e[:entity])
      end)

    entities ++ unique_anaphora
  end

  defp merge_anaphora_entities(entities, _), do: entities

  # ============================================================================
  # Intent-Based Entity Filtering
  # ============================================================================

  @doc false
  # Filter entities to only include those relevant to the intent's slot schema.
  # This prevents entities from being used for the wrong intent.
  # For example, if intent is "smalltalk.greeting" (no slots), all entities are filtered out.
  # If intent is "weather.query", only location/date/time entities are kept.
  defp filter_entities_by_intent(entities, intent) when is_list(entities) do
    # Get the slot schema for this intent
    schema = SlotDetector.get_schema(intent)

    if schema == nil do
      # No schema - keep all entities (conservative fallback)
      entities
    else
      # Get all entity types that can map to slots for this intent
      entity_mappings = Map.get(schema, "entity_mappings", %{})

      # Build a set of all valid entity types for this intent
      valid_types =
        entity_mappings
        |> Map.values()
        |> List.flatten()
        |> MapSet.new()

      if MapSet.size(valid_types) == 0 do
        # Intent has no slots (e.g., smalltalk.greeting) - filter out all entities
        # This prevents entities like "Austin" in "I'm Austin" from leaking
        []
      else
        # Keep only entities whose type matches a valid slot type
        Enum.filter(entities, fn entity ->
          entity_type = entity[:entity] || entity["entity"]
          MapSet.member?(valid_types, entity_type)
        end)
      end
    end
  end

  defp filter_entities_by_intent(entities, _), do: entities
end
