defmodule Brain.Analysis.Pipeline do
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

  alias Brain.Analysis.{
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
    Progress,
    IntentRegistry,
    NoveltyDetector,
    IntentReviewQueue,
    EventExtractor
  }
  alias Brain.Analysis.Types.IntentReviewCandidate

  alias Brain.ML.EntityExtractor
  alias Brain.ML.LSTM.MultiTaskModel
  alias Brain.ML.LSTM.UnifiedModel

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
    Brain.Telemetry.span(:pipeline_process, %{text_length: String.length(text)}, fn ->
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

    has_substantive =
      Enum.any?(analyses, fn a ->
        a.speech_act.category in [:directive, :assertive] or a.speech_act.is_question
      end)

    all_missing = Enum.flat_map(analyses, & &1.missing_context)

    decision_reason =
      cond do
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

    # Stage 2b.5: Sentiment analysis (parallel, via UnifiedModel)
    sentiment_task =
      Task.async(fn ->
        if UnifiedModel.ready?() do
          case UnifiedModel.classify_sentiment(chunk.text) do
            {:ok, result} -> result
            _ -> %{label: :neutral, confidence: 0.5}
          end
        else
          %{label: :neutral, confidence: 0.5}
        end
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

    sentiment_result =
      try do
        Task.await(sentiment_task, 3000)
      catch
        :exit, {:timeout, _} ->
          Task.shutdown(sentiment_task, :brutal_kill)
          %{label: :neutral, confidence: 0.5}
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

    Progress.report(opts, :sentiment_complete, %{
      chunk_index: chunk.index,
      label: Map.get(sentiment_result, :label),
      confidence: Map.get(sentiment_result, :confidence)
    })

    # Stage 2c: Anaphora resolution (resolve pronouns/references from history)
    {resolved_text, anaphora_entities} =
      resolve_anaphora(chunk.text, history, chunk.index, opts)

    # Stage 3a: Entity extraction (use resolved text for better extraction)
    # Pass discourse and speech_act context for disambiguation
    entity_opts =
      opts ++
        [
          discourse: discourse_result,
          speech_act: speech_act_result
        ]

    entities = extract_entities(resolved_text, entity_opts)

    # Merge anaphora-resolved entities with extracted entities
    entities = merge_anaphora_entities(entities, anaphora_entities)

    Progress.report(opts, :entities_extracted, %{
      chunk_index: chunk.index,
      entity_count: length(entities),
      entities: entities |> Enum.take(25) |> Enum.map(&entity_to_dev_map/1)
    })

    # Stage 3a.5: Event extraction (extract actor-verb-object structures)
    events = extract_events(resolved_text, entities, opts)

    Progress.report(opts, :events_extracted, %{
      chunk_index: chunk.index,
      event_count: length(events),
      events: events |> Enum.take(5) |> Enum.map(&event_to_dev_map/1)
    })

    # Stage 3b: Intent determination
    {intent, intent_method, intent_confidence, intent_details} =
      determine_intent(speech_act_result, entities, chunk.text)

    Progress.report(opts, :intent_determined, %{
      chunk_index: chunk.index,
      intent: intent,
      intent_method: intent_method,
      intent_confidence: intent_confidence,
      margin: Map.get(intent_details, :margin, 0.0)
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
        (Enum.map(entities, & &1[:entity_type]) -- Enum.map(relevant_entities, & &1[:entity_type]))
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

    # Check for novel intent candidates (after slots are resolved)
    # Only check if confidence is available (classifier-based)
    if intent_confidence != nil do
      maybe_record_novel_candidate(chunk.text, intent, intent_confidence, intent_details, speech_act_result, entities, resolved_slots, opts)
    end

    # Build the chunk analysis
    # Store only slot-relevant entities to prevent cross-intent contamination
    analysis =
      ChunkAnalysis.new(chunk.index, chunk.text)
      |> Map.put(:discourse, discourse_result)
      |> Map.put(:speech_act, speech_act_result)
      |> Map.put(:sentiment, sentiment_result)
      |> Map.put(:intent, intent)
      |> Map.put(:entities, relevant_entities)
      |> Map.put(:slots, resolved_slots)
      |> Map.put(:missing_context, resolved_slots.missing_required)
      |> ChunkAnalysis.with_events(events)
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
        # Pass opts to include discourse and speech_act context for disambiguation
        try do
          EntityExtractor.extract_entities(text, opts)
        rescue
          _ -> []
        catch
          :exit, _ -> []
        end
    end
  end

  defp entity_to_dev_map(entity) when is_map(entity) do
    type = Map.get(entity, :entity_type)
    value = Map.get(entity, :value)
    conf = Map.get(entity, :confidence)
    source = Map.get(entity, :source)

    %{
      type: type,
      value: value,
      confidence: conf,
      source: source
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp entity_to_dev_map(other), do: %{value: inspect(other)}

  # Extract events from text using LSTM POS tags and entities
  defp extract_events(text, entities, opts) do
    if Keyword.get(opts, :skip_event_extraction, false) do
      []
    else
      # Get POS tags from MultiTaskModel if available
      case get_pos_tags(text) do
        {:ok, pos_tags, tokens} ->
          analysis_input = %{
            pos_tags: pos_tags,
            entities: entities,
            tokens: tokens
          }

          case EventExtractor.extract(analysis_input, opts) do
            {:ok, events} -> events
            {:error, _reason} -> []
          end

        {:error, _reason} ->
          []
      end
    end
  end

  defp get_pos_tags(text) do
    if MultiTaskModel.ready?() do
      case MultiTaskModel.analyze(text) do
        {:ok, %{pos_tags: pos_tags, tokens: tokens}} ->
          {:ok, pos_tags, tokens}

        {:ok, result} when is_map(result) ->
          # Handle different response formats
          pos_tags = Map.get(result, :pos_tags, [])
          tokens = Map.get(result, :tokens, [])
          {:ok, pos_tags, tokens}

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:error, :model_not_ready}
    end
  end

  defp event_to_dev_map(%{action: action, actor: actor, object: object, confidence: confidence}) do
    %{
      action: Map.get(action, :lemma, Map.get(action, :verb)),
      actor: if(actor, do: Map.get(actor, :text)),
      object: if(object, do: Map.get(object, :text)),
      confidence: confidence
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp event_to_dev_map(event) when is_struct(event) do
    event_to_dev_map(Map.from_struct(event))
  end

  defp event_to_dev_map(other), do: %{value: inspect(other)}

  defp determine_intent(speech_act, _entities, text) do
    # Trust the trained intent classifier - it was trained on actual user intents
    # No entity-based overrides - these caused consistent misclassification:
    # - "Play some music" misclassified as news.query
    # - "Turn on lights" misclassified as weather.query
    # - "Hello, I'm Austin" had Austin misclassified as location
    #
    # The classifier is the source of truth for intent detection.

    classifier_intent = extract_classifier_intent(speech_act)
    best_score = speech_act.confidence
    second_score = Map.get(speech_act, :second_score, 0.0)
    margin = Map.get(speech_act, :margin, 0.0)
    top_k = Map.get(speech_act, :top_k, [])

    cond do
      # Use the trained classifier's intent if available
      classifier_intent != nil ->
        {classifier_intent, :classifier, best_score, %{second_score: second_score, margin: margin, top_k: top_k}}

      # For expressive speech acts without explicit classifier intent,
      # infer from speech act category (greeting, farewell, thanks, etc.)
      speech_act.category == :expressive ->
        inferred = infer_intent_from_speech_act(speech_act, text)
        {inferred, :speech_act, nil, %{second_score: 0.0, margin: 0.0, top_k: []}}

      # Default: use speech act inference
      true ->
        inferred = infer_intent_from_speech_act(speech_act, text)
        {inferred, :speech_act, nil, %{second_score: 0.0, margin: 0.0, top_k: []}}
    end
  end

  # Extract the classifier's intent from speech act indicators
  defp extract_classifier_intent(speech_act) do
    speech_act.indicators
    |> Enum.find_value(fn indicator ->
      case String.split(indicator, ":", parts: 2) do
        ["intent", intent] -> intent
        _ -> nil
      end
    end)
  end

  defp infer_intent_from_speech_act(speech_act, _text) do
    # Use IntentRegistry mapping for canonical intent names
    # This ensures consistency with TemplateStore
    case IntentRegistry.intent_for_speech_act(speech_act.sub_type) do
      canonical_intent when is_binary(canonical_intent) ->
        # Found a canonical intent in the registry
        canonical_intent

      nil ->
        # Fallback for unmapped speech acts
        cond do
          # Question
          speech_act.is_question ->
            "question.factual"

          # Command/directive
          speech_act.sub_type == :command ->
            "action.request"

          # Request for action
          speech_act.sub_type == :request_action ->
            "action.request"

          # Request for information
          speech_act.sub_type == :request_information ->
            "information.request"

          # General expressive
          speech_act.category == :expressive ->
            "smalltalk.general"

          # Assertive statement
          speech_act.category == :assertive ->
            "unknown"

          # Unknown
          true ->
            "unknown"
        end
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
                  entity_type: e[:entity_type],
                  value: e[:value]
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
          entity_type: e[:entity_type],
          value: e[:value],
          confidence: 0.75,
          source: :anaphora_resolution
        }
      end)

    # Merge, avoiding duplicates (prefer extracted over resolved)
    extracted_types =
      entities
      |> Enum.map(& &1[:entity_type])
      |> MapSet.new()

    unique_anaphora =
      Enum.reject(converted, fn e ->
        MapSet.member?(extracted_types, e[:entity_type])
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
          entity_type = entity[:entity_type]
          MapSet.member?(valid_types, entity_type)
        end)
      end
    end
  end

  defp filter_entities_by_intent(entities, _), do: entities

  # ============================================================================
  # Novel Intent Detection
  # ============================================================================

  defp maybe_record_novel_candidate(text, intent, confidence, details, speech_act, entities, slot_result, opts) do
    # Only check if enabled and queue is ready
    enabled = Application.get_env(:brain, :intent_promotion_enabled, false)

    if enabled and IntentReviewQueue.ready?() do
      best_score = confidence || 0.0
      margin = Map.get(details, :margin, 0.0)

      case NoveltyDetector.is_novel?(best_score, margin) do
        {:novel, novelty_score} ->
          # Only record if substantive
          if NoveltyDetector.is_substantive?(speech_act, intent) do
            record_novel_candidate(text, intent, best_score, details, speech_act, entities, slot_result, novelty_score, opts)
          end

        :not_novel ->
          :ok
      end
    else
      :ok
    end
  end

  defp record_novel_candidate(text, intent, best_score, details, _speech_act, entities, slot_result, novelty_score, opts) do
    conversation_id = Keyword.get(opts, :conversation_id)
    world_id = Keyword.get(opts, :world_id)

    slot_fill_summary = %{
      filled_slots: Map.get(slot_result, :filled_slots, %{}),
      missing_required: Map.get(slot_result, :missing_required, []),
      missing_optional: Map.get(slot_result, :missing_optional, [])
    }

    candidate =
      IntentReviewCandidate.new(text, intent, best_score,
        conversation_id: conversation_id,
        world_id: world_id,
        second_score: Map.get(details, :second_score, 0.0),
        margin: Map.get(details, :margin, 0.0),
        top_k: Map.get(details, :top_k, []),
        extracted_entities: entities,
        slot_fill_summary: slot_fill_summary
      )

    case IntentReviewQueue.add(candidate) do
      {:ok, _id} ->
        Logger.debug("Recorded novel intent candidate",
          intent: intent,
          score: best_score,
          margin: Map.get(details, :margin, 0.0),
          novelty_score: novelty_score
        )

      {:error, reason} ->
        Logger.warning("Failed to record novel intent candidate", reason: inspect(reason))
    end
  end
end
