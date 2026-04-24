defmodule Brain.Analysis.Pipeline do
  @moduledoc "Orchestrates the text analysis pipeline.\n\nThe pipeline processes user input through multiple stages:\n1. Semantic chunking (break input into utterances)\n2. Parallel analysis (discourse + speech act classification)\n3. Sequential analysis (slot detection + context resolution)\n4. Internal model building (combine all results)\n\nEach stage builds on the previous, creating a comprehensive\nunderstanding of the user's input.\n"

  alias Brain.Analysis.{
    InternalModel,
    Chunk,
    ChunkAnalysis,
    ChunkPriority,
    ChunkProfile,
    FeatureExtractor,
    SemanticChunker,
    DiscourseAnalyzer,
    SpeechActClassifier,
    SlotDetector,
    ContextResolver,
    AnaphoraResolver,
    LearningStore,
    Progress,
    NoveltyDetector,
    EventExtractor,
    EventLinker,
    SemanticRoleLabeler,
    TypeHierarchy,
    ContextAccumulator,
    EntityGraphEnricher
  }

  alias Brain.ML.{EntityExtractor, MicroClassifiers, POSTagger, Tokenizer}

  alias Brain.Memory.Embedder

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

    Brain.Graph.Writer.write_analysis(model)

    Progress.report(opts, :pipeline_complete, %{elapsed_ms: elapsed})

    model
  end

  @doc "Processes a single chunk through the analysis pipeline.\n\nUseful for testing or when you already have chunks. Internally runs the\nfull two-pass analysis (pass 1 + cross-chunk aggregation + pass 2) on a\nlist of one chunk so the public contract is identical to the multi-chunk\npath.\n"
  def analyze_chunk(chunk_text, opts \\ []) when is_binary(chunk_text) do
    chunk = Chunk.new(chunk_text, 0, 0, String.length(chunk_text) - 1)
    [analysis] = analyze_chunks([chunk], opts)
    analysis
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

  # Two-pass per-chunk analysis with cross-chunk enrichment between passes.
  #
  # Pass 1 (`analyze_chunk_pass1/2`): cheap per-chunk analyses (discourse,
  # entities, speech act, sentiment, POS, events, SRL, preliminary intent).
  # No `:intent_full` classification, no slot detection, no fact verification.
  #
  # Cross-chunk aggregation (`build_cross_chunk_context/2`): selects the
  # primary / question chunk via `ChunkPriority`, builds the entity union,
  # and runs `ContextAccumulator` over the per-chunk signals so pass 2 has
  # turn-level context that no individual chunk could see on its own.
  #
  # Pass 2 (`refine_chunk_pass2/3`): only runs for "substantive" chunks
  # (primary, question, or assertive/directive with at least one entity).
  # It builds the full feature vector with cross-chunk context fused into
  # `:accumulated_context`, classifies `:intent_full`, refines speech act,
  # then runs entity inference + slot detection + fact verification +
  # novelty detection. Non-substantive chunks (filler like "thanks") keep
  # their pass-1 analysis and skip the expensive pass 2 work entirely.
  defp analyze_chunks(chunks, opts) do
    pass1_analyses = Enum.map(chunks, &analyze_chunk_pass1(&1, opts))

    cross_context = build_cross_chunk_context(chunks, pass1_analyses, opts)

    pass1_analyses
    |> Enum.map(fn pass1 ->
      if substantive?(pass1, cross_context) do
        refine_chunk_pass2(pass1, cross_context, opts)
      else
        finalize_pass1(pass1, opts)
      end
    end)
    |> Enum.sort_by(& &1.chunk_index)
  end

  defp analyze_chunk_pass1(chunk, opts) do
    participants = Keyword.get(opts, :participants, [:user, :bot])
    bot_names = Keyword.get(opts, :bot_names, [])
    history = Keyword.get(opts, :conversation_history, [])

    Progress.report(opts, :chunk_start, %{
      chunk_index: chunk.index,
      chunk_text: chunk.text,
      chunk_length: String.length(chunk.text)
    })

    discourse_task =
      Task.async(fn ->
        DiscourseAnalyzer.analyze(chunk.text, participants: participants, bot_names: bot_names)
      end)

    basic_entities =
      try do
        EntityExtractor.extract_entities(chunk.text)
      rescue
        _ -> []
      catch
        :exit, _ -> []
      end

    speech_act_task =
      Task.async(fn ->
        SpeechActClassifier.classify(chunk.text, entities: basic_entities)
      end)

    sentiment_task =
      Task.async(fn ->
        case Brain.ML.SentimentClassifierSimple.classify(chunk.text) do
          {:ok, result} ->
            result

          _ ->
            %{label: :neutral, score: 0.5}
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

          Logger.warning("Sentiment classification timed out. Using neutral fallback.")
          %{label: :neutral, score: 0.5}
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

    entities = EntityGraphEnricher.enrich(entities)

    Progress.report(opts, :entities_graph_enriched, %{
      chunk_index: chunk.index,
      graph_known_count: Enum.count(entities, &Map.get(&1, :graph_known, false)),
      total_neighbors: entities |> Enum.map(&Map.get(&1, :graph_neighbor_count, 0)) |> Enum.sum()
    })

    pos_result = get_pos_tags(resolved_text)

    events = extract_events(pos_result, entities, opts)

    Progress.report(opts, :events_extracted, %{
      chunk_index: chunk.index,
      event_count: length(events),
      events: events |> Enum.take(5) |> Enum.map(&event_to_dev_map/1)
    })

    {event_frames, _pos_tags_for_linking} = link_events(events, entities, pos_result, opts)

    Progress.report(opts, :events_linked, %{
      chunk_index: chunk.index,
      event_frame_count: length(event_frames)
    })

    srl_frames = run_srl(pos_result, entities, opts)

    Progress.report(opts, :srl_complete, %{
      chunk_index: chunk.index,
      srl_frame_count: length(srl_frames)
    })

    {prelim_intent, intent_method, prelim_intent_conf, prelim_intent_details} =
      determine_intent(speech_act_result, entities, chunk.text, opts)

    Progress.report(opts, :intent_determined, %{
      chunk_index: chunk.index,
      intent: prelim_intent,
      intent_method: intent_method,
      intent_confidence: prelim_intent_conf,
      margin: Map.get(prelim_intent_details, :margin, 0.0),
      pass: 1
    })

    ChunkAnalysis.new(chunk.index, chunk.text)
    |> Map.put(:discourse, discourse_result)
    |> Map.put(:speech_act, speech_act_result)
    |> Map.put(:sentiment, sentiment_result)
    |> Map.put(:intent, prelim_intent)
    |> Map.put(:entities, entities)
    |> ChunkAnalysis.with_events(events)
    |> Map.put(:event_frames, event_frames)
    |> Map.put(:srl_frames, srl_frames)
    |> Map.put(:pos_tags, pos_result_to_tags(pos_result))
    |> Map.put(:pass, 1)
    |> Map.put(
      :_prelim_intent,
      %{
        intent: prelim_intent,
        method: intent_method,
        confidence: prelim_intent_conf,
        details: prelim_intent_details
      }
    )
  end

  # Builds the cross-chunk context that pass 2 uses to enrich each
  # substantive chunk. Cheap signals only — no ML calls here.
  defp build_cross_chunk_context(chunks, pass1_analyses, opts) do
    raw_text = chunks |> Enum.map(& &1.text) |> Enum.join(" ")

    primary = ChunkPriority.select_primary(pass1_analyses)
    question = ChunkPriority.question_chunk(pass1_analyses)

    entity_union = union_chunk_entities(pass1_analyses)
    familiarity = entity_familiarity_signal(entity_union)

    accumulator =
      ContextAccumulator.new(raw_text, opts)
      |> ContextAccumulator.add_signal(
        :speech_act,
        Map.get(primary.speech_act || %{}, :category, :unknown),
        Map.get(primary.speech_act || %{}, :confidence, 0.0)
      )
      |> ContextAccumulator.add_signal(
        :discourse,
        Map.get(primary.discourse || %{}, :addressee, :unknown),
        Map.get(primary.discourse || %{}, :confidence, 0.0)
      )
      |> ContextAccumulator.add_signal(:entity_familiarity, length(entity_union) > 0, familiarity)
      |> ContextAccumulator.accumulate()

    %{
      pass1_analyses: pass1_analyses,
      primary_chunk_index: primary.chunk_index,
      question_chunk_index: question && question.chunk_index,
      entity_union: entity_union,
      accumulator: accumulator,
      raw_text: raw_text
    }
  end

  defp union_chunk_entities(pass1_analyses) do
    pass1_analyses
    |> Enum.flat_map(fn analysis -> analysis.entities || [] end)
    |> Enum.uniq_by(fn entity ->
      {Map.get(entity, :entity_type) || Map.get(entity, "entity_type"),
       Map.get(entity, :value) || Map.get(entity, "value")}
    end)
  end

  defp entity_familiarity_signal([]), do: 0.5
  defp entity_familiarity_signal(entities) do
    EntityGraphEnricher.familiarity_score(entities)
  rescue
    _ -> 0.5
  end

  # A chunk needs pass 2 if it is the primary or question chunk for the
  # turn, OR if it carries substantive content on its own (an assertive or
  # directive with at least one extracted entity), OR if it is the only
  # chunk in the turn (degenerate case — there's no other chunk to defer
  # to).
  defp substantive?(%ChunkAnalysis{} = analysis, %{
         pass1_analyses: pass1_analyses,
         primary_chunk_index: primary_index,
         question_chunk_index: question_index
       }) do
    cond do
      length(pass1_analyses) == 1 -> true
      analysis.chunk_index == primary_index -> true
      analysis.chunk_index == question_index -> true
      true -> assertive_or_directive_with_entity?(analysis)
    end
  end

  defp assertive_or_directive_with_entity?(%ChunkAnalysis{speech_act: nil}), do: false

  defp assertive_or_directive_with_entity?(%ChunkAnalysis{speech_act: speech_act, entities: ents}) do
    cat = Map.get(speech_act, :category)
    cat in [:directive, :assertive] and length(ents || []) > 0
  end

  # For non-substantive chunks: don't run pass 2, but do compute confidence
  # + response_strategy from the pass-1 data so downstream components see a
  # complete `ChunkAnalysis` regardless of pass.
  defp finalize_pass1(%ChunkAnalysis{} = pass1, opts) do
    Progress.report(opts, :chunk_pass2_skipped, %{
      chunk_index: pass1.chunk_index,
      reason: :non_substantive,
      speech_act: Map.get(pass1.speech_act || %{}, :category)
    })

    pass1
    |> Map.delete(:_prelim_intent)
    |> Map.put(:slots, default_empty_slots())
    |> calculate_confidence()
    |> ChunkAnalysis.determine_response_strategy()
  end

  defp default_empty_slots do
    %Brain.Analysis.SlotResult{
      schema_name: nil,
      filled_slots: %{},
      missing_required: [],
      missing_optional: [],
      all_required_filled: true
    }
  end

  defp refine_chunk_pass2(%ChunkAnalysis{} = pass1, cross_context, opts) do
    history = Keyword.get(opts, :conversation_history, [])
    profile = Keyword.get(opts, :user_profile, %{})

    pass1_with_context = Map.put(pass1, :accumulated_context, cross_context.accumulator)

    {refined_speech_act, intent, intent_method, intent_confidence, intent_details} =
      classify_intent_full_and_refine(pass1_with_context, opts)

    Progress.report(opts, :intent_refined, %{
      chunk_index: pass1.chunk_index,
      intent: intent,
      intent_method: intent_method,
      intent_confidence: intent_confidence,
      margin: Map.get(intent_details, :margin, 0.0),
      pass: 2
    })

    {entities_after_inference, intent, intent_details} =
      Brain.Analysis.ContextualEntityInferrer.infer(
        pass1.text,
        pass1.entities,
        intent,
        intent_details,
        world_id: Keyword.get(opts, :world_id, "default")
      )

    relevant_entities =
      entities_after_inference
      |> filter_entities_by_intent(intent)
      |> maybe_retype_pos_music_artists(pass1.text)

    Progress.report(opts, :entities_filtered, %{
      chunk_index: pass1.chunk_index,
      original_count: length(entities_after_inference),
      filtered_count: length(relevant_entities),
      excluded_types:
        (Enum.map(entities_after_inference, & &1[:entity_type]) --
           Enum.map(relevant_entities, & &1[:entity_type]))
        |> Enum.uniq()
    })

    fact_result = verify_facts_in_chunk(pass1.text, relevant_entities, refined_speech_act, opts)

    Progress.report(opts, :fact_verification, %{
      chunk_index: pass1.chunk_index,
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
      chunk_index: pass1.chunk_index,
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
      chunk_index: pass1.chunk_index,
      all_required_filled: Map.get(resolved_slots, :all_required_filled),
      missing_required: Map.get(resolved_slots, :missing_required, []),
      filled_slots: Map.get(resolved_slots, :filled_slots, %{})
    })

    if intent_confidence != nil do
      maybe_record_novel_candidate(
        pass1.text,
        intent,
        intent_confidence,
        intent_details,
        refined_speech_act,
        entities_after_inference,
        resolved_slots,
        opts
      )
    end

    analysis =
      pass1
      |> Map.delete(:_prelim_intent)
      |> Map.put(:speech_act, refined_speech_act)
      |> Map.put(:intent, intent)
      |> Map.put(:entities, relevant_entities)
      |> Map.put(:slots, resolved_slots)
      |> Map.put(:missing_context, resolved_slots.missing_required)
      |> Map.put(:fact_verification, fact_result.verification)
      |> Map.put(:related_beliefs, fact_result.beliefs)
      |> Map.put(:epistemic_status, fact_result.status)
      |> Map.put(:accumulated_context, cross_context.accumulator)
      |> Map.put(:pass, 2)
      |> calculate_confidence()
      |> ChunkAnalysis.determine_response_strategy()

    maybe_extract_beliefs_from_events(pass1.events, opts)

    Progress.report(opts, :chunk_complete, %{
      chunk_index: pass1.chunk_index,
      response_strategy: analysis.response_strategy,
      confidence: analysis.confidence,
      pass: 2
    })

    analysis
  end

  # Runs the full-features `:intent_full` classification on the pass-1
  # analysis (with cross-chunk context fused into `:accumulated_context`),
  # then splices that intent into the speech-act voter via
  # `SpeechActClassifier.refine_with_intent/3` and re-runs `determine_intent`
  # on the refined speech act.
  defp classify_intent_full_and_refine(pass1_with_context, opts) do
    {fv_intent, fv_conf} = classify_intent_full(pass1_with_context)

    refined_speech_act =
      SpeechActClassifier.refine_with_intent(pass1_with_context.speech_act, fv_intent, fv_conf)

    {intent, intent_method, intent_confidence, intent_details} =
      determine_intent(refined_speech_act, pass1_with_context.entities, pass1_with_context.text, opts)

    {refined_speech_act, intent, intent_method, intent_confidence, intent_details}
  end

  defp classify_intent_full(pass1_with_context) do
    if MicroClassifiers.ready?() do
      try do
        {feature_vector, _word_feats} = FeatureExtractor.extract(pass1_with_context)

        case MicroClassifiers.classify_vector(:intent_full, feature_vector) do
          {:ok, intent, confidence} when intent != "unknown" and intent != "" ->
            {intent, confidence}

          _ ->
            {nil, 0.0}
        end
      rescue
        e ->
          Logger.debug("Pipeline pass 2 intent_full classification failed: #{Exception.message(e)}")
          {nil, 0.0}
      end
    else
      {nil, 0.0}
    end
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

  defp extract_events(pos_result, entities, opts) do
    if Keyword.get(opts, :skip_event_extraction, false) do
      []
    else
      case pos_result do
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
    tokens = Tokenizer.tokenize_words(text)

    case POSTagger.get_model() do
      {:ok, model} ->
        pos_tags = POSTagger.predict(tokens, model)
        {:ok, pos_tags, tokens}

      {:error, reason} ->
        {:error, reason}
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

  defp link_events(events, entities, pos_result, _opts) do
    case pos_result do
      {:ok, pos_tags_tuples, tokens} ->
        tag_strings = extract_tag_strings(pos_tags_tuples)

        token_maps = Enum.map(tokens, fn t ->
          %{text: t, normalized: String.downcase(t)}
        end)

        frames = EventLinker.link(events, entities, token_maps, tag_strings)
        {frames, tag_strings}

      {:error, _} ->
        {[], []}
    end
  rescue
    e ->
      Logger.warning("EventLinker failed: #{Exception.message(e)}")
      {[], []}
  end

  defp run_srl(pos_result, entities, _opts) do
    case pos_result do
      {:ok, pos_tags_tuples, tokens} ->
        tag_strings = extract_tag_strings(pos_tags_tuples)
        bio_tags = generate_srl_bio_tags(tokens, tag_strings, entities)
        SemanticRoleLabeler.label(tokens, bio_tags, entities)

      {:error, _} ->
        []
    end
  rescue
    e ->
      Logger.warning("SemanticRoleLabeler failed: #{Exception.message(e)}")
      []
  end

  defp pos_result_to_tags({:ok, pos_tags, _tokens}), do: pos_tags
  defp pos_result_to_tags(_), do: []

  defp extract_tag_strings(pos_tags) do
    Enum.map(pos_tags, fn
      {_token, tag} when is_binary(tag) -> tag
      tag when is_binary(tag) -> tag
      _ -> "NN"
    end)
  end

  defp generate_srl_bio_tags(tokens, pos_tags, entities) do
    entity_spans = build_entity_spans(tokens, entities)

    tokens
    |> Enum.with_index()
    |> Enum.map(fn {_token, idx} ->
      pos = Enum.at(pos_tags, idx, "NN")
      entity_role = Map.get(entity_spans, idx)

      cond do
        pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] ->
          "B-V"

        entity_role != nil ->
          entity_role

        idx == 0 and pos in ["NN", "NNP", "NNS", "NNPS", "PRP"] ->
          "B-ARG0"

        pos in ["NN", "NNP", "NNS", "NNPS"] ->
          "B-ARG1"

        true ->
          "O"
      end
    end)
  end

  defp build_entity_spans(tokens, entities) do
    downcased = Enum.map(tokens, &String.downcase/1)

    entities
    |> Enum.reduce(%{}, fn entity, acc ->
      value = Map.get(entity, :value) || Map.get(entity, "value") || ""
      entity_type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type") || ""
      entity_words = value |> String.downcase() |> String.split()

      role = entity_type_to_srl_role(entity_type)

      case find_span_start(downcased, entity_words) do
        nil -> acc
        start_idx ->
          entity_words
          |> Enum.with_index()
          |> Enum.reduce(acc, fn {_word, i}, inner_acc ->
            prefix = if i == 0, do: "B-", else: "I-"
            Map.put(inner_acc, start_idx + i, "#{prefix}#{role}")
          end)
      end
    end)
  end

  defp find_span_start(tokens, entity_words) do
    len = length(entity_words)

    tokens
    |> Enum.with_index()
    |> Enum.find_value(fn {_tok, idx} ->
      window = Enum.slice(tokens, idx, len)

      if window == entity_words do
        idx
      end
    end)
  end

  defp entity_type_to_srl_role(type) when is_atom(type), do: entity_type_to_srl_role(to_string(type))
  defp entity_type_to_srl_role(type) do
    downcased = String.downcase(type)
    cond do
      downcased in ["person", "user", "agent"] -> "ARG0"
      downcased in ["location", "place", "city", "country", "geo"] -> "ARGM-LOC"
      downcased in ["temporal", "date", "time", "datetime"] -> "ARGM-TMP"
      true -> "ARG1"
    end
  end

  @disambiguation_threshold 0.5
  @low_confidence_floor 0.3

  defp determine_intent(speech_act, entities, text, opts) do
    classifier_intent = extract_classifier_intent(speech_act)
    best_score = speech_act.intent_confidence || speech_act.confidence
    second_score = Map.get(speech_act, :second_score, 0.0)
    margin = Map.get(speech_act, :margin, 0.0)
    top_k = Map.get(speech_act, :top_k, [])

    cond do
      classifier_intent != nil and best_score >= @disambiguation_threshold ->
        {classifier_intent, :classifier, best_score,
         %{second_score: second_score, margin: margin, top_k: top_k}}

      classifier_intent != nil ->
        case disambiguate_with_atlas(classifier_intent, top_k, entities, text, opts) do
          {:ok, refined_intent, refined_score, method} ->
            {refined_intent, method, refined_score,
             %{second_score: second_score, margin: margin, top_k: top_k,
               original_intent: classifier_intent, original_score: best_score}}

          :no_opinion when best_score < @low_confidence_floor ->
            inferred = infer_intent_from_speech_act(speech_act, text)
            {inferred, :speech_act_fallback, nil,
             %{second_score: second_score, margin: margin, top_k: top_k,
               original_intent: classifier_intent, original_score: best_score}}

          :no_opinion ->
            {classifier_intent, :classifier, best_score,
             %{second_score: second_score, margin: margin, top_k: top_k}}
        end

      speech_act.category == :expressive ->
        inferred = infer_intent_from_speech_act(speech_act, text)
        {inferred, :speech_act, nil, %{second_score: 0.0, margin: 0.0, top_k: []}}

      true ->
        inferred = infer_intent_from_speech_act(speech_act, text)
        {inferred, :speech_act, nil, %{second_score: 0.0, margin: 0.0, top_k: []}}
    end
  end

  defp disambiguate_with_atlas(classifier_intent, top_k, entities, _text, opts) do
    conversation_id = Keyword.get(opts, :conversation_id)

    # Query the conversation graph for recent topics to provide context
    recent_topics = graph_recent_topics(conversation_id)

    # Query the knowledge graph for entity neighborhoods to understand context
    entity_neighborhoods = graph_entity_neighborhoods(entities)

    # Use the graph-derived entity types to score each candidate intent
    candidates = normalize_top_k(top_k)

    scored =
      Enum.map(candidates, fn {intent, score} ->
        atlas_score = score_intent_with_graph_context(
          intent, entity_neighborhoods, recent_topics
        )
        {intent, score + atlas_score}
      end)
      |> Enum.sort_by(fn {_, s} -> -s end)

    case scored do
      [{best_intent, best_score} | _] when best_intent != classifier_intent ->
        {:ok, best_intent, best_score, :atlas_disambiguation}

      _ ->
        :no_opinion
    end
  rescue
    _ -> :no_opinion
  end

  defp graph_recent_topics(nil), do: []

  defp graph_recent_topics(conversation_id) do
    Brain.Graph.Reader.conversation_topics(conversation_id)
  rescue
    _ -> []
  end

  defp graph_entity_neighborhoods(entities) when is_list(entities) do
    Brain.Graph.Reader.entity_context(entities, depth: 1)
  rescue
    _ -> []
  end

  defp graph_entity_neighborhoods(_), do: []

  defp normalize_top_k(top_k) when is_list(top_k) do
    Enum.flat_map(top_k, fn
      %{intent: intent, score: score} -> [{intent, score}]
      {intent, score} when is_binary(intent) -> [{intent, score}]
      _ -> []
    end)
  end

  defp normalize_top_k(_), do: []

  @doc false
  defp score_intent_with_graph_context(intent, entity_neighborhoods, recent_topics, profile \\ nil) do
    expected_types =
      case profile do
        %ChunkProfile{domain: domain} when domain != :unknown ->
          expected_entity_types_from_domain(domain)

        _ ->
          expected_entity_types_from_domain(extract_domain_from_intent(intent))
      end

    graph_entity_types =
      entity_neighborhoods
      |> Enum.flat_map(fn
        %{node: %{properties: props}} ->
          type = Map.get(props, "type", "")
          if type != "", do: [type], else: []

        %{neighbors: neighbors} when is_list(neighbors) ->
          Enum.flat_map(neighbors, fn v ->
            label = Map.get(v, :label, "")
            if label != "", do: [label], else: []
          end)

        _ ->
          []
      end)
      |> Enum.uniq()

    entity_type_overlap =
      if expected_types != [] and graph_entity_types != [] do
        matches =
          Enum.count(graph_entity_types, fn graph_type ->
            Enum.any?(expected_types, fn expected ->
              Tokenizer.tokens_overlap?(graph_type, expected)
            end)
          end)

        matches * 0.1
      else
        0.0
      end

    topic_overlap =
      if recent_topics != [] do
        intent_domain =
          case profile do
            %ChunkProfile{domain: domain} when domain != :unknown ->
              to_string(domain)

            _ ->
              extract_domain_from_intent(intent)
          end

        if intent_domain != "" do
          topic_similarity = compute_topic_similarity(intent_domain, recent_topics)
          topic_similarity * 0.15
        else
          0.0
        end
      else
        0.0
      end

    min(entity_type_overlap + topic_overlap, 0.3)
  end

  defp compute_topic_similarity(domain, topics) do
    if Embedder.ready?() do
      domain_vec = Embedder.embed(domain)

      similarities =
        Enum.map(topics, fn topic ->
          topic_vec = Embedder.embed(topic)
          FourthWall.Math.cosine_similarity(domain_vec, topic_vec)
        end)

      case similarities do
        [] -> 0.0
        sims -> Enum.max(sims)
      end
    else
      0.0
    end
  rescue
    _ -> 0.0
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

  @external_resource Path.join(:code.priv_dir(:brain), "analysis/speech_act_intent_map.json")
  @speech_act_intent_map Path.join(:code.priv_dir(:brain), "analysis/speech_act_intent_map.json")
                         |> File.read!()
                         |> Jason.decode!()

  defp infer_intent_from_speech_act(speech_act, _text, profile \\ nil) do
    profile_label =
      case profile do
        %ChunkProfile{derived_label: label} when is_binary(label) and label != "" -> label
        _ -> nil
      end

    if profile_label do
      profile_label
    else
      case Map.get(@speech_act_intent_map, to_string(speech_act.sub_type)) do
        canonical_intent when is_binary(canonical_intent) ->
          canonical_intent

        nil ->
          sub_type_str = to_string(speech_act.sub_type)
          category_str = to_string(speech_act.category)

          cond do
            speech_act.is_question ->
              Map.get(@speech_act_intent_map, "question", "question.factual")

            Map.has_key?(@speech_act_intent_map, sub_type_str) ->
              Map.get(@speech_act_intent_map, sub_type_str)

            Map.has_key?(@speech_act_intent_map, category_str) ->
              Map.get(@speech_act_intent_map, category_str)

            true ->
              Map.get(@speech_act_intent_map, "default", "unknown")
          end
      end
    end
  end

  defp maybe_extract_beliefs_from_events([], _opts), do: :ok

  defp maybe_extract_beliefs_from_events(events, opts) do
    user_id = Keyword.get(opts, :user_id) || "anonymous"

    if Brain.Epistemic.Types.Config.auto_extraction_enabled?() do
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

    sentiment_conf =
      case analysis.sentiment do
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

    entity_fam = EntityGraphEnricher.familiarity_score(analysis.entities)

    intent_conf =
      case analysis.speech_act do
        %{intent_confidence: c} when is_number(c) and c > 0 -> c
        _ -> 0.5
      end

    acc =
      %ContextAccumulator{}
      |> ContextAccumulator.add_signal(:discourse, Map.get(analysis.discourse, :addressee), discourse_conf)
      |> ContextAccumulator.add_signal(:speech_act, Map.get(analysis.speech_act, :category), speech_act_conf)
      |> ContextAccumulator.add_signal(:slot_fill, Map.get(analysis.slots || %{}, :all_required_filled, false), slot_conf)
      |> ContextAccumulator.add_signal(:sentiment, Map.get(analysis.sentiment || %{}, :label, :neutral), sentiment_conf)
      |> ContextAccumulator.add_signal(:entity_familiarity, entity_fam > 0.5, entity_fam)
      |> ContextAccumulator.add_signal(:intent, analysis.intent, intent_conf)
      |> ContextAccumulator.accumulate()

    confidence = ContextAccumulator.effective_confidence(acc) |> Float.round(3)

    analysis
    |> Map.put(:confidence, confidence)
    |> Map.put(:accumulated_context, acc)
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
        # Schema exists but defines no entity type filter — keep extracted entities
        # instead of dropping them (empty mappings used to wipe PROPN hints, etc.).
        entities
      else
        Enum.filter(entities, fn entity ->
          entity_type = entity[:entity_type]

          # Keep POS-derived proper-noun spans even when typed `unknown`/`person`
          # so music (and similar) intents can still slot OOV artist/title names.
          Map.get(entity, :source) == :pos_tagger_propn or
            MapSet.member?(valid_types, entity_type) or
            Enum.any?(valid_types, &Brain.Analysis.TypeHierarchy.compatible?(entity_type, &1))
        end)
      end
    end
  end

  defp filter_entities_by_intent(entities, _) do
    entities
  end

  # OOV proper nouns from the POS tagger are often typed `unknown` while the
  # utterance is clearly a music command ("Play some …"). Downstream slot and
  # feature tests expect `artist` (or `person`) for those spans.
  defp maybe_retype_pos_music_artists(entities, text)
       when is_list(entities) and is_binary(text) do
    t = text |> String.trim() |> String.downcase()

    musicish? =
      String.match?(t, ~r/^play\b/) or
        String.match?(t, ~r/^find me\b.*\b(music|jazz|songs?|tracks?)\b/)

    if musicish? do
      Enum.map(entities, fn e ->
        if Map.get(e, :source) == :pos_tagger_propn and retypeable_music_unknown?(e) do
          Map.put(e, :entity_type, "artist")
        else
          e
        end
      end)
    else
      entities
    end
  end

  defp maybe_retype_pos_music_artists(entities, _), do: entities

  defp retypeable_music_unknown?(e) do
    t = e[:entity_type]

    t in [nil, :unknown, "unknown", "thing"]
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
    side_effects = Keyword.get(opts, :side_effects, true)

    if enabled and side_effects do
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

            if NoveltyDetector.is_researchable?(text, entities, speech_act) do
              profile = Keyword.get(opts, :profile)

              inferred_domain =
                case profile do
                  %ChunkProfile{domain: domain} when domain != :unknown -> domain
                  _ ->
                    case extract_domain_from_intent(intent) do
                      "" -> :unknown
                      d -> String.to_atom(d)
                    end
                end

              broadcast_novel_input(text, novelty_score, inferred_domain, entities)
            end
          end

        :not_novel ->
          :ok
      end
    else
      :ok
    end
  end

  defp broadcast_novel_input(text, novelty_score, domain, entities) do
    if Process.whereis(Brain.PubSub) do
      Phoenix.PubSub.broadcast(
        Brain.PubSub,
        "learning:novel_input",
        {:novel_input, text, novelty_score, domain, entities}
      )
    end
  rescue
    _ -> :ok
  end

  defp record_novel_candidate(
         _text,
         intent,
         best_score,
         details,
         _speech_act,
         _entities,
         _slot_result,
         novelty_score,
         _opts
       ) do
    Logger.debug("Novel intent candidate detected (review queue removed)",
      intent: intent,
      score: best_score,
      margin: Map.get(details, :margin, 0.0),
      novelty_score: novelty_score
    )
  end

  # ============================================================================
  # Epistemic Fact Verification
  # ============================================================================

  # Verifies factual claims in a chunk against existing beliefs.
  # Only runs for assertive statements (observations, claims). Returns
  # epistemic context that can influence response generation.
  # Returns: %{verification: result, beliefs: list, status: atom}
  defp verify_facts_in_chunk(text, entities, speech_act, _opts) do
    if speech_act.category == :assertive do
      subject = extract_subject_from_entities(entities, text)

      case subject do
        nil ->
          record_verification_miss(:no_subject)
          %{verification: nil, beliefs: [], status: :unchecked}

        subject_text ->
          start_time = System.monotonic_time(:millisecond)

          beliefs = query_related_beliefs(subject_text)
          verification = safe_verify_fact(subject_text, text)

          status =
            case verification do
              {:verified, _} -> :verified
              {:contradicted, _} -> :contradicted
              {:uncertain, _} -> :uncertain
              _ -> :unchecked
            end

          miss_reason =
            cond do
              status in [:verified, :contradicted] -> nil
              beliefs == [] -> :no_beliefs
              status == :uncertain -> :no_facts
              true -> nil
            end

          if miss_reason, do: record_verification_miss(miss_reason)

          duration_ms = System.monotonic_time(:millisecond) - start_time

          Telemetry.emit_fact_verification(status, subject_text, duration_ms, %{
            beliefs_count: length(beliefs),
            verification_result: verification,
            miss_reason: miss_reason
          })

          %{verification: verification, beliefs: beliefs, status: status}
      end
    else
      %{verification: nil, beliefs: [], status: :unchecked}
    end
  end

  defp record_verification_miss(reason) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(Brain.Metrics.Aggregator, {:record_verification_miss_reason, reason})
    end
  end

  defp extract_subject_from_entities(entities, text) do
    subject_types = TypeHierarchy.config("subject_capable_types", [])

    subject_entity =
      Enum.find(entities, fn entity ->
        entity_type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type")
        entity_type in subject_types
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
    normalized = text |> String.downcase() |> String.trim()
    String.to_existing_atom(normalized)
  rescue
    ArgumentError -> :unknown
  end

  defp normalize_predicate(_), do: :unknown

  defp safe_verify_fact(subject, text) do
    FactIntegration.verify_fact(subject, text)
  rescue
    error ->
      Logger.warning("Fact verification failed", error: inspect(error))
      {:uncertain, :verification_error}
  end

  defp extract_domain_from_intent(nil), do: ""
  defp extract_domain_from_intent(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [domain, _] -> domain
      _ -> intent
    end
  end
  defp extract_domain_from_intent(_), do: ""

  @doc false
  def expected_entity_types_from_domain(domain) do
    case domain do
      :weather -> ["location", "date", "time"]
      :music -> ["artist", "song", "genre", "album"]
      :smarthome -> ["device", "room", "setting"]
      :reminder -> ["date", "time", "description"]
      :navigation -> ["location", "address"]
      :search -> ["query", "topic"]
      :meta -> []
      :smalltalk -> ["person", "name"]
      _ -> []
    end
  end
end
