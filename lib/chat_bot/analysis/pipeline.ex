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
    Progress.report(opts, :strategy_determined, %{overall_strategy: model.overall_strategy})

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

    Progress.report(opts, :chunk_start, %{chunk_index: chunk.index, chunk_length: String.length(chunk.text)})

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

    # Stage 3a: Entity extraction
    entities = extract_entities(chunk.text, opts)
    Progress.report(opts, :entities_extracted, %{
      chunk_index: chunk.index,
      entity_count: length(entities),
      entities: entities |> Enum.take(25) |> Enum.map(&entity_to_dev_map/1)
    })

    # Stage 3b: Intent determination
    intent = determine_intent(speech_act_result, entities, chunk.text)
    Progress.report(opts, :intent_determined, %{chunk_index: chunk.index, intent: intent})

    # Stage 3c: Slot detection
    slot_result = SlotDetector.detect(intent, entities)
    Progress.report(opts, :slots_detected, %{
      chunk_index: chunk.index,
      missing_required: Map.get(slot_result, :missing_required, []),
      filled_count: map_size(Map.get(slot_result, :filled_slots, %{})),
      filled_slots: Map.get(slot_result, :filled_slots, %{})
    })

    # Stage 3d: Context resolution
    resolved_slots =
      ContextResolver.resolve(slot_result,
        conversation_history: history,
        user_profile: profile
      )

    Progress.report(opts, :context_resolved, %{
      chunk_index: chunk.index,
      all_required_filled: Map.get(resolved_slots, :all_required_filled),
      missing_required: Map.get(resolved_slots, :missing_required, []),
      filled_slots: Map.get(resolved_slots, :filled_slots, %{})
    })

    # Build the chunk analysis
    analysis =
      ChunkAnalysis.new(chunk.index, chunk.text)
      |> Map.put(:discourse, discourse_result)
      |> Map.put(:speech_act, speech_act_result)
      |> Map.put(:intent, intent)
      |> Map.put(:entities, entities)
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
    type = Map.get(entity, :entity) || Map.get(entity, "entity") || Map.get(entity, :type) || Map.get(entity, "type")
    value = Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :name) || Map.get(entity, "name")
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
      infer_intent_from_speech_act(speech_act, text)
    else
      # For non-expressive speech acts, try multiple strategies

      # 1. First, check if entities suggest an intent
      case SlotDetector.suggest_intent_from_entities(entities) do
        {:ok, intent, _score} ->
          intent

        {:error, :no_match} ->
          # 2. Try keyword-based heuristic for substantive intents
          case SlotDetector.suggest_intent_from_keywords(text) do
            {:ok, intent, _confidence} ->
              intent

            {:error, :no_match} ->
              # 3. Fall back to speech act based intent
              infer_intent_from_speech_act(speech_act, text)
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
end
