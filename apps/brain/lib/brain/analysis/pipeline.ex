defmodule Brain.Analysis.Pipeline do
  @moduledoc "Orchestrates the text analysis pipeline.\n\nThe pipeline processes user input through multiple stages:\n1. Semantic chunking (break input into utterances)\n2. Parallel analysis (discourse + speech act classification)\n3. Sequential analysis (slot detection + context resolution)\n4. Internal model building (combine all results)\n\nEach stage builds on the previous, creating a comprehensive\nunderstanding of the user's input.\n"

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

  alias Brain.FactDatabase.Integration, as: FactIntegration
  alias Brain.Epistemic.BeliefStore

  alias Brain.Telemetry
  require Logger

  @doc "Processes user input through the complete analysis pipeline.\n\nOptions:\n- :participants - conversation participants (default: [:user, :bot])\n- :bot_names - additional names the bot responds to\n- :conversation_history - list of previous context snapshots for slot resolution\n- :user_profile - map of user preferences (location, timezone, etc.)\n- :skip_entity_extraction - if true, skips entity extraction (for testing)\n- :entities - pre-extracted entities to use instead of extracting\n\nReturns an InternalModel struct with complete analysis.\n"
  def process(text, opts \\ []) when is_binary(text) do
    Telemetry.span(:pipeline_process, %{text_length: String.length(text)}, fn ->
      do_process(text, opts)
    end)
  end

  defp do_process(text, opts) do
    Logger.debug("Starting analysis pipeline", %{text_length: String.length(text)})

    start_time = System.monotonic_time(:millisecond)

    Progress.report(opts, :pipeline_start, %{text_length: String.length(text)})
    model = InternalModel.new(text)
    chunks = SemanticChunker.chunk(text)
    model = InternalModel.with_chunks(model, chunks)

    Logger.debug("Chunking complete", %{chunk_count: length(chunks)})
    Progress.report(opts, :chunking_complete, %{chunk_count: length(chunks)})
    analyses = analyze_chunks(chunks, opts)
    model = InternalModel.with_analyses(model, analyses)
    model = InternalModel.determine_strategy(model)
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

        all_missing != [] and Enum.any?(chunk_strategies, &(&1 == :can_respond)) ->
          "Partial: can respond to some, missing slots: #{Enum.join(all_missing, ", ")}"

        all_missing != [] ->
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

    elapsed = System.monotonic_time(:millisecond) - start_time

    Logger.debug("Pipeline complete", %{
      chunk_count: length(chunks),
      strategy: model.overall_strategy,
      elapsed_ms: elapsed
    })

    record_pipeline_result(model)

    Progress.report(opts, :pipeline_complete, %{elapsed_ms: elapsed})

    model
  end

  @doc "Processes a single chunk through the analysis pipeline.\n\nUseful for testing or when you already have chunks.\n"
  def analyze_chunk(chunk_text, opts \\ []) when is_binary(chunk_text) do
    chunk = Chunk.new(chunk_text, 0, 0, String.length(chunk_text) - 1)
    analyze_single_chunk(chunk, opts)
  end

  @doc "Returns a summary of the analysis for debugging/logging.\n"
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

  defp analyze_chunks(chunks, opts) do
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

    discourse_task =
      Task.async(fn ->
        DiscourseAnalyzer.analyze(chunk.text, participants: participants, bot_names: bot_names)
      end)

    speech_act_task =
      Task.async(fn ->
        SpeechActClassifier.classify(chunk.text)
      end)

    sentiment_task =
      Task.async(fn ->
        case Brain.ML.LSTM.Integration.classify_sentiment(chunk.text) do
          {:ok, result} ->
            result

          {:error, reason} ->
            raise "Sentiment classification failed: #{inspect(reason)}. " <>
                    "Run `mix train` to train the sentiment classifier."
        end
      end)

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

          raise "Sentiment classification timed out. " <>
                  "Ensure the sentiment classifier is trained and responsive."
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

    {resolved_text, anaphora_entities} =
      resolve_anaphora(chunk.text, history, chunk.index, opts)

    entity_opts =
      opts ++
        [discourse: discourse_result, speech_act: speech_act_result]

    entities = extract_entities(resolved_text, entity_opts)
    entities = merge_anaphora_entities(entities, anaphora_entities)

    Progress.report(opts, :entities_extracted, %{
      chunk_index: chunk.index,
      entity_count: length(entities),
      entities: entities |> Enum.take(25) |> Enum.map(&entity_to_dev_map/1)
    })

    events = extract_events(resolved_text, entities, opts)

    Progress.report(opts, :events_extracted, %{
      chunk_index: chunk.index,
      event_count: length(events),
      events: events |> Enum.take(5) |> Enum.map(&event_to_dev_map/1)
    })

    {intent, intent_method, intent_confidence, intent_details} =
      determine_intent(speech_act_result, entities, chunk.text)

    Progress.report(opts, :intent_determined, %{
      chunk_index: chunk.index,
      intent: intent,
      intent_method: intent_method,
      intent_confidence: intent_confidence,
      margin: Map.get(intent_details, :margin, 0.0)
    })

    relevant_entities = filter_entities_by_intent(entities, intent)

    Progress.report(opts, :entities_filtered, %{
      chunk_index: chunk.index,
      original_count: length(entities),
      filtered_count: length(relevant_entities),
      excluded_types:
        (Enum.map(entities, & &1[:entity_type]) -- Enum.map(relevant_entities, & &1[:entity_type]))
        |> Enum.uniq()
    })

    # Epistemic fact verification - check claims against existing beliefs
    fact_result = verify_facts_in_chunk(chunk.text, relevant_entities, speech_act_result, opts)

    Progress.report(opts, :fact_verification, %{
      chunk_index: chunk.index,
      epistemic_status: fact_result.status,
      fact_verification: fact_result.verification,
      related_beliefs_count: length(fact_result.beliefs),
      related_beliefs:
        Enum.map(fact_result.beliefs, fn belief ->
          %{
            id: Map.get(belief, :id),
            subject: Map.get(belief, :subject),
            predicate: Map.get(belief, :predicate),
            object: Map.get(belief, :object),
            confidence: Map.get(belief, :confidence),
            source: Map.get(belief, :source),
            node_id: Map.get(belief, :node_id)
          }
        end)
    })

    slot_result = SlotDetector.detect(intent, relevant_entities)

    Progress.report(opts, :slots_detected, %{
      chunk_index: chunk.index,
      missing_required: Map.get(slot_result, :missing_required, []),
      filled_count: map_size(Map.get(slot_result, :filled_slots, %{})),
      filled_slots: Map.get(slot_result, :filled_slots, %{})
    })

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

    if intent_confidence != nil do
      maybe_record_novel_candidate(
        chunk.text,
        intent,
        intent_confidence,
        intent_details,
        speech_act_result,
        entities,
        resolved_slots,
        opts
      )
    end

    analysis =
      ChunkAnalysis.new(chunk.index, chunk.text)
      |> Map.put(:discourse, discourse_result)
      |> Map.put(:speech_act, speech_act_result)
      |> Map.put(:sentiment, sentiment_result)
      |> Map.put(:intent, intent)
      |> Map.put(:entities, relevant_entities)
      |> Map.put(:slots, resolved_slots)
      |> Map.put(:missing_context, resolved_slots.missing_required)
      |> Map.put(:fact_verification, fact_result.verification)
      |> Map.put(:related_beliefs, fact_result.beliefs)
      |> Map.put(:epistemic_status, fact_result.status)
      |> ChunkAnalysis.with_events(events)
      |> calculate_confidence()
      |> ChunkAnalysis.determine_response_strategy()

    # Async belief extraction from events (non-blocking)
    maybe_extract_beliefs_from_events(events, opts)

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

  defp entity_to_dev_map(other) do
    %{value: inspect(other)}
  end

  defp extract_events(text, entities, opts) do
    if Keyword.get(opts, :skip_event_extraction, false) do
      []
    else
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
      actor:
        if(actor) do
          Map.get(actor, :text)
        end,
      object:
        if(object) do
          Map.get(object, :text)
        end,
      confidence: confidence
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp event_to_dev_map(event) when is_struct(event) do
    event_to_dev_map(Map.from_struct(event))
  end

  defp event_to_dev_map(other) do
    %{value: inspect(other)}
  end

  defp determine_intent(speech_act, _entities, text) do
    classifier_intent = extract_classifier_intent(speech_act)
    best_score = speech_act.confidence
    second_score = Map.get(speech_act, :second_score, 0.0)
    margin = Map.get(speech_act, :margin, 0.0)
    top_k = Map.get(speech_act, :top_k, [])

    cond do
      classifier_intent != nil ->
        {classifier_intent, :classifier, best_score,
         %{second_score: second_score, margin: margin, top_k: top_k}}

      speech_act.category == :expressive ->
        inferred = infer_intent_from_speech_act(speech_act, text)
        {inferred, :speech_act, nil, %{second_score: 0.0, margin: 0.0, top_k: []}}

      true ->
        inferred = infer_intent_from_speech_act(speech_act, text)
        {inferred, :speech_act, nil, %{second_score: 0.0, margin: 0.0, top_k: []}}
    end
  end

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
    case IntentRegistry.intent_for_speech_act(speech_act.sub_type) do
      canonical_intent when is_binary(canonical_intent) ->
        canonical_intent

      nil ->
        cond do
          speech_act.is_question ->
            "question.factual"

          speech_act.sub_type == :command ->
            "action.request"

          speech_act.sub_type == :request_action ->
            "action.request"

          speech_act.sub_type == :request_information ->
            "information.request"

          speech_act.category == :expressive ->
            "smalltalk.acknowledgment"

          speech_act.category == :assertive ->
            "unknown"

          true ->
            "unknown"
        end
    end
  end

  defp maybe_extract_beliefs_from_events([], _opts), do: :ok

  defp maybe_extract_beliefs_from_events(events, opts) do
    user_id = Keyword.get(opts, :user_id)

    if user_id && Brain.Epistemic.Types.Config.auto_extraction_enabled?() do
      Task.Supervisor.start_child(
        Brain.Knowledge.AgentSupervisor,
        fn ->
          try do
            Brain.Epistemic.BeliefStore.extract_beliefs_from_events(events, user_id)
          rescue
            e ->
              require Logger
              Logger.debug("Belief extraction from events failed: #{Exception.message(e)}")
          end
        end
      )
    end

    :ok
  rescue
    _ -> :ok
  end

  defp calculate_confidence(analysis) do
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

          if total == 0 do
            1.0
          else
            filled_count / total
          end

        _ ->
          0.5
      end

    # Reduce slot weight when classifier confidence is low to prevent
    # slot filling from inflating confidence for misclassified intents.
    # When speech_act_conf < 0.5, slot filling shouldn't boost confidence.
    slot_weight = if speech_act_conf < 0.5, do: 0.1, else: 0.3
    discourse_weight = 0.3
    speech_act_weight = 0.6 - slot_weight

    confidence =
      (discourse_conf * discourse_weight + speech_act_conf * speech_act_weight + slot_conf * slot_weight)
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

  defp resolve_anaphora(text, history, chunk_index, opts) do
    case AnaphoraResolver.resolve_and_substitute(text, history) do
      {:ok, resolved_text, resolved_entities} ->
        if resolved_entities != [] do
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
    converted =
      Enum.map(anaphora_entities, fn e ->
        %{
          entity_type: e[:entity_type],
          value: e[:value],
          confidence: 0.75,
          source: :anaphora_resolution
        }
      end)

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

  defp merge_anaphora_entities(entities, _) do
    entities
  end

  @doc false
  defp filter_entities_by_intent(entities, intent) when is_list(entities) do
    schema = SlotDetector.get_schema(intent)

    if schema == nil do
      entities
    else
      entity_mappings = Map.get(schema, "entity_mappings", %{})

      valid_types =
        entity_mappings
        |> Map.values()
        |> List.flatten()
        |> MapSet.new()

      if MapSet.size(valid_types) == 0 do
        []
      else
        Enum.filter(entities, fn entity ->
          entity_type = entity[:entity_type]
          MapSet.member?(valid_types, entity_type)
        end)
      end
    end
  end

  defp filter_entities_by_intent(entities, _) do
    entities
  end

  defp maybe_record_novel_candidate(
         text,
         intent,
         confidence,
         details,
         speech_act,
         entities,
         slot_result,
         opts
       ) do
    enabled = Application.get_env(:brain, :intent_promotion_enabled, false)

    if enabled and IntentReviewQueue.ready?() do
      best_score = confidence || 0.0
      margin = Map.get(details, :margin, 0.0)

      case NoveltyDetector.is_novel?(best_score, margin) do
        {:novel, novelty_score} ->
          if NoveltyDetector.is_substantive?(speech_act, intent) do
            record_novel_candidate(
              text,
              intent,
              best_score,
              details,
              speech_act,
              entities,
              slot_result,
              novelty_score,
              opts
            )

            # Publish novel input for LearningTriggers
            inferred_domain =
              case IntentRegistry.domain(intent) do
                nil -> :unknown
                d -> d
              end

            broadcast_novel_input(text, novelty_score, inferred_domain)
          end

        :not_novel ->
          :ok
      end
    else
      :ok
    end
  end

  defp broadcast_novel_input(text, novelty_score, domain) do
    if Process.whereis(Brain.PubSub) do
      Phoenix.PubSub.broadcast(
        Brain.PubSub,
        "learning:novel_input",
        {:novel_input, text, novelty_score, domain}
      )
    end
  rescue
    _ -> :ok
  end

  defp record_novel_candidate(
         text,
         intent,
         best_score,
         details,
         _speech_act,
         entities,
         slot_result,
         novelty_score,
         opts
       ) do
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

  # ============================================================================
  # Epistemic Fact Verification
  # ============================================================================

  # Verifies factual claims in a chunk against existing beliefs.
  # Only runs for assertive statements (observations, claims). Returns
  # epistemic context that can influence response generation.
  # Returns: %{verification: result, beliefs: list, status: atom}
  defp verify_facts_in_chunk(text, entities, speech_act, _opts) do
    # Only verify for assertive statements (observations, claims)
    if speech_act.category == :assertive do
      subject = extract_subject_from_entities(entities, text)

      case subject do
        nil ->
          %{verification: nil, beliefs: [], status: :unchecked}

        subject_text ->
          start_time = System.monotonic_time(:millisecond)

          # Query related beliefs from the epistemic store
          beliefs = query_related_beliefs(subject_text)

          # Verify the claim against existing beliefs
          verification = safe_verify_fact(subject_text, text)

          status =
            case verification do
              {:verified, _} -> :verified
              {:contradicted, _} -> :contradicted
              {:uncertain, _} -> :uncertain
              _ -> :unchecked
            end

          duration_ms = System.monotonic_time(:millisecond) - start_time

          # Emit telemetry for fact verification
          Telemetry.emit_fact_verification(status, subject_text, duration_ms, %{
            beliefs_count: length(beliefs),
            verification_result: verification
          })

          %{verification: verification, beliefs: beliefs, status: status}
      end
    else
      %{verification: nil, beliefs: [], status: :unchecked}
    end
  end

  defp extract_subject_from_entities(entities, text) do
    # Try to find a subject entity (e.g., "sky", "grass", "car")
    subject_entity =
      Enum.find(entities, fn entity ->
        entity_type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type")
        # Look for common subject types
        entity_type in ["object", "thing", "location", "person", "noun"]
      end)

    case subject_entity do
      nil ->
        # Fall back to extracting subject from text using simple heuristics
        extract_subject_from_text(text)

      entity ->
        Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :match)
    end
  end

  defp extract_subject_from_text(text) do
    # Simple extraction: look for "The X is Y" or "X is Y" patterns
    # Using tokenization (no regex per .cursorrules)
    # Tokenizer returns maps with :normalized and :text fields
    tokens = Brain.ML.Tokenizer.tokenize(text)

    # Extract normalized words from token maps
    words =
      tokens
      |> Enum.filter(fn token -> Map.get(token, :type) == :word end)
      |> Enum.map(fn token -> Map.get(token, :normalized) || Map.get(token, :text) end)
      |> Enum.map(&String.downcase/1)

    case words do
      ["the", subject | _rest] -> subject
      [subject, "is" | _rest] when subject not in ["it", "this", "that"] -> subject
      _ -> nil
    end
  end

  defp query_related_beliefs(subject_text) do
    predicate = normalize_predicate(subject_text)

    case BeliefStore.query_beliefs(subject: :world, predicate: predicate, min_confidence: 0.5) do
      {:ok, beliefs} -> beliefs
      _ -> []
    end
  rescue
    _ -> []
  end

  defp normalize_predicate(text) when is_binary(text) do
    text
    |> String.downcase()
    |> String.trim()
    |> String.to_atom()
  rescue
    _ -> :unknown
  end

  defp normalize_predicate(_), do: :unknown

  defp safe_verify_fact(subject, text) do
    FactIntegration.verify_fact(subject, text)
  rescue
    error ->
      Logger.warning("Fact verification failed", error: inspect(error))
      {:uncertain, :verification_error}
  end
end