defmodule Brain.Response.Generator do
  @moduledoc """
  Generative response synthesis entry point.

  This module orchestrates response generation using a generative pipeline:

  1. **Retrieve Context** - Query memory for similar episodes, get semantic facts
  2. **Synthesize** - Compose response from primitives and domain knowledge
  3. **Compose** - Weave parts together using speech act analysis
  4. **Refine** - Score/improve using LSTM if available

  The system generates novel responses by combining:
  - Domain knowledge (from priv/knowledge/domains/*.json)
  - Similar past episodes (memory-augmented)
  - Response primitives (hedges, acknowledgments, connectors)
  - Entity slot filling

  Brain should use this module instead of implementing response logic directly.
  """

  require Logger

  alias Brain.Response.{TemplateStore, MemoryAugmented, FactRetriever, Composer, TemplateBlender}
  alias Brain.Response.{LSTMResponse, ResponseQuality, Synthesizer}
  alias Brain.Analysis.IntentRegistry
  alias Brain.Memory.{Store, Think}
  alias Brain.Code.QueryHandler

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Generate a response for the given intent and entities.

  Uses a generative pipeline:
  1. Retrieve similar episodes from memory
  2. Synthesize response from domain knowledge and primitives
  3. Fall back to templates if synthesis doesn't produce a result
  4. Apply LSTM scoring if available

  Returns:
  - {:ok, response, :synthesized} for generated responses
  - {:ok, response, :memory_adapted} for memory-adapted responses
  - {:ok, response, :template} for template-based responses
  - {:ok, response, :lstm_selected} for LSTM-scored best response
  - {:ok, response, :fallback} for fallback responses
  """
  def generate(intent, entities, query_text \\ nil) do
    generate_with_events(intent, entities, query_text, [])
  end

  @doc """
  Generate a response with event context for better slot filling.

  When events are provided, they are used to:
  - Provide action/actor/object slots for template filling
  - Enhance context retrieval from memory
  - Improve response relevance based on user intent structure

  ## Examples

      events = [%Event{action: %{lemma: "play"}, object: %{text: "jazz"}}]
      generate_with_events("music.play", entities, "Play some jazz", events)
  """
  def generate_with_events(intent, entities, query_text, events) when is_list(events) do
    # Build context for generation, including event-based context
    context = build_generation_context_with_events(intent, entities, query_text, events)

    # Try generative pipeline first
    result = run_generative_pipeline(intent, entities, query_text, context)

    # Try LSTM refinement if available
    result = maybe_refine_with_lstm(result, query_text, intent, entities)

    # Quality check
    maybe_improve_response(result, query_text, intent, entities)
  end

  # ============================================================================
  # Context Building
  # ============================================================================

  defp build_generation_context(intent, entities, query_text) do
    build_generation_context_with_events(intent, entities, query_text, [])
  end

  defp build_generation_context_with_events(intent, entities, query_text, events) do
    # Retrieve similar episodes from memory
    similar_episodes = retrieve_similar_episodes(intent, entities, query_text)

    # Retrieve event-related episodes if events present
    event_episodes = retrieve_event_episodes(events)

    # Build event-based slots for template filling
    event_slots = build_context_from_events(events, entities)

    # Get confidence from classification (default to medium)
    confidence = 0.7

    %{
      similar_episodes: similar_episodes ++ event_episodes,
      confidence: confidence,
      intent: intent,
      entities: entities,
      query_text: query_text,
      events: events,
      event_slots: event_slots
    }
  end

  defp retrieve_event_episodes(events) when is_list(events) and length(events) > 0 do
    # Query memory for episodes related to the primary event action
    case get_primary_action(events) do
      nil ->
        []

      action_lemma ->
        if Process.whereis(Store) do
          case Store.query_events_by_action(action_lemma, 3) do
            {:ok, episodes} -> episodes
            _ -> []
          end
        else
          []
        end
    end
  end

  defp retrieve_event_episodes(_), do: []

  @doc """
  Build template context slots from extracted events.

  This provides action/actor/object slots for more relevant response generation.

  ## Examples

      events = [%Event{action: %{lemma: "play"}, object: %{text: "jazz"}}]
      build_context_from_events(events, entities)
      # => %{action: "play", object: "jazz", location: "London", ...}
  """
  def build_context_from_events(events, entities) when is_list(events) do
    # Start with entity-based slots
    base_slots = build_entity_slots(entities)

    # Add event-based slots (override entity slots if more specific)
    event_slots = extract_event_slots(events)

    Map.merge(base_slots, event_slots)
  end

  defp build_entity_slots(entities) when is_list(entities) do
    # Extract slots from entities by type
    Enum.reduce(entities, %{}, fn entity, acc ->
      type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type") || Map.get(entity, :type)
      value = Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :text)

      case type do
        t when t in ["location", "city", "place"] -> Map.put(acc, :location, value)
        t when t in ["person", "name"] -> Map.put(acc, :person, value)
        t when t in ["song", "music-artist", "artist"] -> Map.put(acc, :music, value)
        t when t in ["device", "lights", "heating"] -> Map.put(acc, :device, value)
        t when t in ["date", "time"] -> Map.put(acc, :when, value)
        _ -> acc
      end
    end)
  end

  defp build_entity_slots(_), do: %{}

  defp extract_event_slots(events) when is_list(events) and length(events) > 0 do
    # Get slots from the primary (highest confidence) event
    primary = get_primary_event(events)

    if primary do
      slots = %{}

      # Add action
      slots =
        case primary do
          %{action: %{lemma: lemma}} when is_binary(lemma) ->
            Map.put(slots, :action, lemma)

          %{action: %{verb: verb}} when is_binary(verb) ->
            Map.put(slots, :action, String.downcase(verb))

          _ ->
            slots
        end

      # Add actor
      slots =
        case primary do
          %{actor: %{text: text}} when is_binary(text) ->
            Map.put(slots, :actor, text)

          _ ->
            slots
        end

      # Add object
      slots =
        case primary do
          %{object: %{text: text}} when is_binary(text) ->
            Map.put(slots, :object, text)

          _ ->
            slots
        end

      slots
    else
      %{}
    end
  end

  defp extract_event_slots(_), do: %{}

  defp get_primary_event(events) when is_list(events) and length(events) > 0 do
    Enum.max_by(events, fn e -> Map.get(e, :confidence, 0.0) end, fn -> nil end)
  end

  defp get_primary_event(_), do: nil

  defp get_primary_action(events) do
    case get_primary_event(events) do
      %{action: %{lemma: lemma}} when is_binary(lemma) -> lemma
      %{action: %{verb: verb}} when is_binary(verb) -> String.downcase(verb)
      _ -> nil
    end
  end

  defp retrieve_similar_episodes(intent, entities, query_text) do
    # Build query from intent + entities + query text
    query =
      [
        intent || "",
        query_text || "",
        entities |> Enum.map(fn e -> e[:value] || e["value"] || "" end) |> Enum.join(" ")
      ]
      |> Enum.filter(&(&1 != ""))
      |> Enum.join(" ")

    if query != "" and Process.whereis(Store) do
      case Store.query_similar(query, 5) do
        {:ok, episodes} -> episodes
        _ -> []
      end
    else
      []
    end
  rescue
    _ -> []
  end

  # ============================================================================
  # Generative Pipeline
  # ============================================================================

  defp run_generative_pipeline(intent, entities, query_text, context) do
    # Step 1: Try to synthesize from domain knowledge + memory
    case Synthesizer.synthesize(intent, entities,
           confidence: context.confidence,
           similar_episodes: context.similar_episodes
         ) do
      {:ok, response} ->
        {:ok, response, :synthesized}

      :not_synthesized ->
        # Step 2: Try memory-augmented generation
        case try_memory_augmented(intent, entities) do
          {:ok, response} ->
            {:ok, response, :memory_adapted}

          :not_handled ->
            # Step 3: Try template-based response
            case try_template_response(intent, entities) do
              {:ok, response} ->
                {:ok, response, :template}

              :not_handled ->
                # Step 4: Handle special cases (code, factual)
                case try_special_handlers(intent, entities, query_text) do
                  {:ok, response} ->
                    {:ok, response, :special_handler}

                  :not_handled ->
                    # Step 5: Fallback
                    response = Synthesizer.get_fallback_response()
                    {:ok, response, :fallback}
                end
            end
        end
    end
  end

  defp maybe_refine_with_lstm({:ok, response, type}, query_text, intent, entities) do
    if query_text && LSTMResponse.ready?() && type not in [:lstm_selected, :special_handler] do
      case LSTMResponse.generate(query_text, intent, entities) do
        {:ok, lstm_response, score} when score > 0.7 ->
          # Only use LSTM response if significantly better
          {:ok, lstm_response, :lstm_selected}

        _ ->
          {:ok, response, type}
      end
    else
      {:ok, response, type}
    end
  end

  # Check response quality and try to improve if needed
  defp maybe_improve_response({:ok, response, type}, query_text, intent, entities) do
    # Skip quality check for synthesized/special responses
    if type in [:synthesized, :special_handler, :lstm_selected] or is_nil(query_text) do
      {:ok, response, type}
    else
      case ResponseQuality.quick_check(query_text, response) do
        :ok ->
          {:ok, response, type}

        :warning ->
          Logger.debug("Response quality warning for intent #{intent}")
          {:ok, response, type}

        :poor ->
          Logger.debug("Poor response quality detected, attempting improvement")

          case ResponseQuality.improve(query_text, response, intent: intent, entities: entities) do
            {:improved, better_response, _analysis} ->
              {:ok, better_response, :quality_improved}

            _ ->
              {:ok, response, type}
          end
      end
    end
  end

  # ============================================================================
  # Special Handlers (Code, Factual queries)
  # ============================================================================

  defp try_special_handlers(intent, entities, query_text) do
    # Use IntentRegistry to determine domain instead of string matching
    domain = IntentRegistry.domain(intent)

    cond do
      # Code-related intents
      domain == :code ->
        handle_code_intent(intent, entities, query_text)

      # Factual questions (domain is :question in intent_registry.json)
      domain == :question or intent == "knowledge.query" ->
        handle_factual_query(entities, query_text)

      true ->
        :not_handled
    end
  end

  defp handle_code_intent(intent, entities, query_text) do
    world_id = get_code_world_id()

    case QueryHandler.handle(intent, entities, world_id: world_id, query_text: query_text) do
      {:ok, response} -> {:ok, response}
      :not_handled -> :not_handled
    end
  end

  defp handle_factual_query(entities, query_text) do
    if FactRetriever.ready?() do
      case FactRetriever.retrieve(query_text, entities) do
        {:ok, facts} when is_list(facts) and length(facts) > 0 ->
          response = format_factual_response(query_text, facts)
          {:ok, response}

        _ ->
          :not_handled
      end
    else
      :not_handled
    end
  end

  defp get_code_world_id do
    case Process.get(:current_world_id) do
      nil -> "default"
      world_id -> world_id
    end
  end
  
  defp maybe_improve_response(other, _query, _intent, _entities), do: other

  @doc """
  Generate a response with full path tracking for debugging/inspection.

  Returns:
  - {:ok, response, response_type, path} where path is a list of steps taken

  The path shows exactly how the response was reached:
  - Which handlers were tried
  - Why each was skipped or selected
  - What data stores were accessed
  """
  def generate_with_path(intent, entities, query_text \\ nil) do
    path = []
    context = build_generation_context(intent, entities, query_text)

    # Step 1: Try synthesis from domain knowledge
    path = path ++ [%{step: :try, handler: :synthesizer, intent: intent}]

    case Synthesizer.synthesize(intent, entities,
           confidence: context.confidence,
           similar_episodes: context.similar_episodes
         ) do
      {:ok, response} ->
        path = path ++ [%{step: :selected, handler: :synthesizer, reason: "synthesized from domain knowledge"}]
        {:ok, response, :synthesized, path}

      :not_synthesized ->
        path = path ++ [%{step: :skip, handler: :synthesizer, reason: "no domain knowledge for intent"}]

        # Step 2: Try memory-augmented
        {path, memory_result} = try_memory_with_path(intent, entities, path)

        case memory_result do
          {:ok, response} ->
            path = path ++ [%{step: :selected, handler: :memory_augmented, reason: "similar episodes found"}]
            {:ok, response, :memory_adapted, path}

          :not_handled ->
            # Step 3: Try template-based
            {path, template_result} = try_template_with_path(intent, entities, path)

            case template_result do
              {:ok, response} ->
                path = path ++ [%{step: :selected, handler: :template, reason: "template found for intent"}]
                {:ok, response, :template, path}

              :not_handled ->
                # Step 4: Try special handlers
                path = path ++ [%{step: :try, handler: :special, intent: intent}]

                case try_special_handlers(intent, entities, query_text) do
                  {:ok, response} ->
                    path = path ++ [%{step: :selected, handler: :special, reason: "special handler matched"}]
                    {:ok, response, :special_handler, path}

                  :not_handled ->
                    # Step 5: Fallback
                    response = Synthesizer.get_fallback_response()
                    path = path ++ [%{step: :selected, handler: :fallback, reason: "no handlers matched"}]
                    {:ok, response, :fallback, path}
                end
            end
        end
    end
  end

  defp try_memory_with_path(intent, entities, path) do
    path = path ++ [%{step: :try, handler: :memory_augmented, store: "Memory.Store"}]

    case try_memory_augmented(intent, entities) do
      {:ok, response} ->
        {path, {:ok, response}}

      :not_handled ->
        path = path ++ [%{step: :skip, handler: :memory_augmented, reason: "no similar episodes or embedder not ready"}]
        {path, :not_handled}
    end
  end

  defp try_template_with_path(intent, entities, path) do
    path = path ++ [%{step: :try, handler: :template, store: "TemplateStore", intent: intent}]

    case try_template_response(intent, entities) do
      {:ok, response} ->
        {path, {:ok, response}}

      :not_handled ->
        path = path ++ [%{step: :skip, handler: :template, reason: "no template for intent"}]
        {path, :not_handled}
    end
  end

  @doc """
  Generate a response using context-aware template selection.

  This uses conditional template matching and semantic ranking:
  1. Filter templates by conditions that match the context
  2. Rank matching templates by similarity to the query
  3. Fall back to cross-intent semantic search if needed

  ## Parameters
  - `intent` - The classified intent name
  - `entities` - List of extracted entities
  - `query_text` - The original user query
  - `context` - Additional context (filled_slots, missing_slots, confidence, speech_act)

  ## Returns
  - {:ok, response, :conditional_template} for condition-matched templates
  - {:ok, response, :semantic_fallback} for cross-intent semantic match
  - Falls back to regular generate/3 if conditional selection fails
  """
  def generate_with_context(intent, entities, query_text, context \\ %{}) do
    # Build full context with entities
    full_context = build_template_context(entities, context)
    confidence = Map.get(context, :confidence, 0.7)

    # Try synthesis from domain knowledge first
    case Synthesizer.synthesize(intent, entities, confidence: confidence) do
      {:ok, response} ->
        {:ok, response, :synthesized}

      :not_synthesized ->
        # Try context-aware template selection
        case try_conditional_template(intent, query_text, entities, full_context) do
          {:ok, response, type} ->
            {:ok, response, type}

          :not_handled ->
            # Try template blending for novel responses
            case try_blended_response(query_text, full_context) do
              {:ok, response} ->
                {:ok, response, :blended}

              :not_handled ->
                # Fall back to memory-augmented
                case try_memory_augmented(intent, entities) do
                  {:ok, response} ->
                    {:ok, response, :memory_augmented}

                  :not_handled ->
                    # Fall back to regular template
                    case try_template_response(intent, entities) do
                      {:ok, response} ->
                        {:ok, response, :template}

                      :not_handled ->
                        response = Synthesizer.get_fallback_response()
                        {:ok, response, :fallback}
                    end
                end
            end
        end
    end
  end

  @doc """
  Builds the context map for conditional template selection from entities and analysis.
  """
  def build_template_context(entities, additional_context \\ %{}) do
    entity_types = Enum.map(entities, fn e -> e[:entity_type] || e["entity_type"] end)

    %{
      entities: entities,
      entity_types: entity_types,
      filled_slots: Map.get(additional_context, :filled_slots, []),
      missing_slots: Map.get(additional_context, :missing_slots, []),
      confidence: Map.get(additional_context, :confidence, 0.5),
      speech_act: Map.get(additional_context, :speech_act, %{})
    }
  end

  @doc """
  Generate a response for an expressive speech act.
  Used for greetings, farewells, thanks, apologies, etc.
  """
  def generate_expressive(speech_act) when is_map(speech_act) do
    sub_type = Map.get(speech_act, :sub_type)
    TemplateStore.get_expressive_response(sub_type)
  end

  def generate_expressive(_), do: nil

  @doc """
  Generate a combined response for an analysis model with multiple speech acts.

  This handles multi-chunk inputs where we may have expressives (greetings)
  combined with directives (questions/commands).
  """
  def generate_from_analysis(analysis_model, intent, entities, query_text) do
    speech_acts =
      analysis_model.analyses
      |> Enum.map(& &1.speech_act)

    # Aggregate sentiment across chunks (use strongest non-neutral signal)
    overall_sentiment = aggregate_sentiment(analysis_model.analyses)

    expressives =
      speech_acts
      |> Enum.filter(&(&1.category == :expressive))
      |> Enum.uniq_by(& &1.sub_type)

    directives = Enum.filter(speech_acts, &(&1.category == :directive))

    has_substantive_content =
      length(directives) > 0 or
        (intent != nil and intent != "" and
           not IntentRegistry.greeting?(intent))

    response_parts = []
    response_types = []

    # Add expressive acknowledgment if present
    {response_parts, response_types} =
      if length(expressives) > 0 do
        expressive = List.first(expressives)
        expressive_response = generate_expressive(expressive)

        if expressive_response do
          {[expressive_response | response_parts], [:expressive | response_types]}
        else
          {response_parts, response_types}
        end
      else
        {response_parts, response_types}
      end

    # Add substantive response for directives/questions/commands
    {response_parts, response_types} =
      if has_substantive_content do
        {:ok, substantive_response, response_type} = generate(intent, entities, query_text)
        {[substantive_response | response_parts], [response_type | response_types]}
      else
        {response_parts, response_types}
      end

    valid_parts =
      response_parts
      |> Enum.reverse()
      |> Enum.filter(&(&1 != nil and &1 != ""))

    # Primary type is the last substantive response type (the main content handler)
    # Expressive responses (greetings, etc.) are secondary to substantive content
    primary_type = 
      response_types
      |> Enum.reject(&(&1 == :expressive))
      |> List.first(:fallback)

    response =
      case valid_parts do
        [] ->
          {:ok, resp, _} = generate(intent, entities, query_text)
          resp

        [single] ->
          single

        parts ->
          # Combine parts intelligently
          weave_response_parts(parts, Enum.reverse(response_types))
      end

    # Prepend empathetic acknowledgment for negative sentiment
    response = maybe_add_sentiment_prefix(response, overall_sentiment)

    {response, primary_type}
  end

  # NOTE: Domain-specific handlers have been replaced by the Synthesizer module
  # which loads response frames from priv/knowledge/domains/*.json files.
  # This enables generative responses without hardcoded strings.

  @doc """
  Formats a code snippet for display in a response.

  ## Options
    - `:language` - The programming language for syntax highlighting
    - `:start_line` - Starting line number
    - `:max_lines` - Maximum lines to show (default: 20)
  """
  def format_code_snippet(code, opts \\ []) do
    language = Keyword.get(opts, :language, "")
    max_lines = Keyword.get(opts, :max_lines, 20)

    lines = String.split(code, "\n")
    
    truncated = if length(lines) > max_lines do
      shown = Enum.take(lines, max_lines)
      remaining = length(lines) - max_lines
      shown ++ ["# ... #{remaining} more lines"]
    else
      lines
    end

    code_block = Enum.join(truncated, "\n")
    
    """
    ```#{language}
    #{code_block}
    ```
    """
  end

  # ============================================================================
  # Semantic Fact Retrieval Helpers
  # ============================================================================

  defp generate_factual_with_semantic_search(entities, query_text) do
    alias Brain.Response.SemanticFactRetriever

    query_str = query_text || ""

    # Try semantic search first (data-driven approach)
    if SemanticFactRetriever.ready?() and query_str != "" do
      results = SemanticFactRetriever.search(query_str, limit: 3, threshold: 0.25)

      if results != [] do
        response = format_semantic_results(query_str, results)
        {:ok, response}
      else
        # Fall back to old method if no semantic matches
        try_keyword_fact_retrieval(entities, query_str)
      end
    else
      # Fall back to keyword search if semantic retriever not ready
      try_keyword_fact_retrieval(entities, query_str)
    end
  end

  defp try_keyword_fact_retrieval(entities, query_str) do
    if FactRetriever.available?() do
      entity_names = extract_entity_names_for_facts(entities)
      facts = FactRetriever.get_facts_for_query(query_str, entity_names)

      if facts != [] do
        response = format_factual_response(query_str, facts)
        {:ok, response}
      else
        :not_handled
      end
    else
      :not_handled
    end
  end

  defp format_semantic_results(_query_text, results) do
    # Get the best matching fact
    best = List.first(results)
    fact = best.fact
    similarity = best.similarity

    # Format based on confidence
    if similarity > 0.6 do
      # High confidence - present as knowledge
      "#{fact.fact}"
    else
      # Lower confidence - hedged response
      "Based on what I know: #{fact.fact}"
    end
  end

  # ============================================================================
  # Template and Memory-Based Response Helpers
  # ============================================================================

  defp try_memory_augmented(intent, entities) do
    case MemoryAugmented.generate(intent, entities) do
      {:ok, response, _metadata} ->
        {:ok, response}

      _ ->
        :not_handled
    end
  end

  defp try_template_response(intent, entities) do
    if TemplateStore.ready?() do
      case TemplateStore.get_random_template(intent) do
        nil ->
          :not_handled

        template ->
          response = TemplateStore.substitute_slots(template, entities)
          {:ok, response}
      end
    else
      :not_handled
    end
  end

  defp try_conditional_template(intent, query_text, entities, context) do
    if TemplateStore.ready?() do
      case TemplateStore.get_best_template(intent, query_text, context) do
        {:ok, template} ->
          response = TemplateStore.substitute_slots(template, entities)
          {:ok, response, :conditional_template}

        {:ok, template, :fallback} ->
          response = TemplateStore.substitute_slots(template, entities)
          {:ok, response, :semantic_fallback}

        {:error, _reason} ->
          :not_handled
      end
    else
      :not_handled
    end
  end

  defp try_blended_response(query_text, context) do
    if TemplateBlender.ready?() do
      case TemplateBlender.blend(query_text, context) do
        {:ok, response} when is_binary(response) and response != "" ->
          {:ok, response}

        _ ->
          :not_handled
      end
    else
      :not_handled
    end
  end

  defp try_template_with_slots(intent, entities) do
    if TemplateStore.ready?() do
      case TemplateStore.get_random_template(intent) do
        nil -> :not_handled
        template -> {:ok, TemplateStore.substitute_slots(template, entities)}
      end
    else
      :not_handled
    end
  end

  # ============================================================================
  # Factual Response Formatting
  # ============================================================================

  defp format_factual_response(query_text, facts)
       when is_binary(query_text) and query_text != "" do
    alias Brain.ML.Tokenizer
    alias Brain.ML.POSTagger

    query_tokens = Tokenizer.tokenize(query_text)
    speech_act = Brain.Analysis.SpeechActClassifier.classify(query_text)
    is_question = Map.get(speech_act, :is_question, false)

    query_numbers =
      query_tokens
      |> Enum.filter(fn t -> t.type == :number end)
      |> Enum.map(fn t -> parse_number(t.text) end)
      |> Enum.reject(&is_nil/1)

    content_words = extract_content_words_with_pos(query_tokens)

    if is_question and length(query_numbers) > 0 and length(content_words) > 0 do
      stated_number = List.first(query_numbers)
      relevant_fact = find_matching_fact(facts, content_words)

      if relevant_fact do
        fact_text = relevant_fact.fact
        fact_tokens = Tokenizer.tokenize(fact_text)

        fact_numbers =
          fact_tokens
          |> Enum.filter(fn t -> t.type == :number end)
          |> Enum.map(fn t -> parse_number(t.text) end)
          |> Enum.reject(&is_nil/1)

        case find_matching_number(stated_number, fact_numbers, content_words, fact_tokens) do
          {:match, _} ->
            "Yes, #{fact_text}."

          {:mismatch, _actual} ->
            "No, #{fact_text}."

          :no_comparison ->
            formatted = FactRetriever.format_facts([relevant_fact], 1)
            "Here's what I know: #{Enum.join(formatted, ". ")}."
        end
      else
        formatted = FactRetriever.format_facts(facts, 2)
        "Here's what I know: #{Enum.join(formatted, ". ")}."
      end
    else
      formatted = FactRetriever.format_facts(facts, 2)
      "Here's what I know: #{Enum.join(formatted, ". ")}."
    end
  end

  defp format_factual_response(_query_text, facts) do
    formatted = FactRetriever.format_facts(facts, 2)
    "Here's what I know: #{Enum.join(formatted, ". ")}."
  end

  defp extract_content_words_with_pos(tokens) do
    alias Brain.ML.POSTagger

    content_tags = ~w(NOUN PROPN VERB ADJ ADV NUM)
    token_texts = Enum.map(tokens, fn t -> t.text end)

    case POSTagger.load_model() do
      {:ok, model} ->
        POSTagger.predict(token_texts, model)
        |> Enum.filter(fn {_word, tag} -> tag in content_tags end)
        |> Enum.map(fn {word, _tag} -> String.downcase(word) end)
        |> Enum.filter(fn w -> String.length(w) > 2 end)

      {:error, _} ->
        tokens
        |> Enum.filter(fn t -> t.type == :word and String.length(t.text) > 2 end)
        |> Enum.map(fn t -> String.downcase(t.text) end)
    end
  end

  defp parse_number(text) do
    cleaned = String.replace(text, ",", "")

    case Integer.parse(cleaned) do
      {num, ""} -> num
      {num, "." <> _} -> num
      _ -> nil
    end
  end

  defp find_matching_fact(facts, content_words) do
    Enum.find(facts, fn fact ->
      fact_text = String.downcase(fact.fact)
      entity_text = String.downcase(fact.entity)

      Enum.any?(content_words, fn word ->
        String.contains?(fact_text, word) or String.contains?(entity_text, word)
      end)
    end)
  end

  defp find_matching_number(stated_number, fact_numbers, content_words, fact_tokens) do
    if fact_numbers == [] do
      :no_comparison
    else
      fact_words = Enum.map(fact_tokens, fn t -> String.downcase(t.text) end)

      relevant_numbers =
        fact_tokens
        |> Enum.with_index()
        |> Enum.filter(fn {t, _idx} -> t.type == :number end)
        |> Enum.filter(fn {_t, idx} ->
          nearby_words = Enum.slice(fact_words, max(0, idx - 3), 7)
          Enum.any?(content_words, fn cw -> cw in nearby_words end)
        end)
        |> Enum.map(fn {t, _idx} -> parse_number(t.text) end)
        |> Enum.reject(&is_nil/1)

      numbers_to_check = if relevant_numbers != [], do: relevant_numbers, else: fact_numbers

      if stated_number in numbers_to_check do
        {:match, stated_number}
      else
        {:mismatch, List.first(numbers_to_check)}
      end
    end
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  defp find_entity_value(entities, entity_type) when is_list(entities) do
    entity =
      Enum.find(entities, fn e ->
        e[:entity_type] == entity_type
      end)

    if entity do
      entity[:value] || entity["value"]
    else
      nil
    end
  end

  defp find_entity_value(_, _), do: nil

  defp extract_entity_names_for_facts(entities) when is_list(entities) do
    entities
    |> Enum.map(fn e ->
      e[:value] || e["value"] || ""
    end)
    |> Enum.filter(&(&1 != ""))
  end

  defp extract_entity_names_for_facts(_), do: []

  defp weave_response_parts(parts, types) do
    # Use Composer for sophisticated multi-part response weaving
    valid_parts = Enum.filter(parts, &(&1 != nil and &1 != ""))

    if length(valid_parts) <= 1 do
      # No need for weaving with single part
      Enum.join(valid_parts, " ")
    else
      # Build analyses for Composer
      analyses = build_analyses_for_composer(valid_parts, types)
      Composer.weave_multi_chunk_response(valid_parts, analyses)
    end
  end

  defp build_analyses_for_composer(parts, types) do
    # Pad types to match parts length if needed
    padded_types =
      if length(types) < length(parts) do
        types ++ List.duplicate(:assertive, length(parts) - length(types))
      else
        types
      end

    Enum.zip(parts, padded_types)
    |> Enum.map(fn {_part, type} ->
      %{
        speech_act: %{
          category: type_to_category(type),
          is_question: false,
          sub_type: nil
        }
      }
    end)
  end

  defp type_to_category(:expressive), do: :expressive
  defp type_to_category(:domain), do: :directive
  defp type_to_category(:template), do: :assertive
  defp type_to_category(:memory_augmented), do: :assertive
  defp type_to_category(_), do: :assertive

  # Aggregate sentiment across multiple chunk analyses.
  # Returns the strongest non-neutral sentiment, or :neutral.
  defp aggregate_sentiment(analyses) do
    sentiments =
      analyses
      |> Enum.map(&Map.get(&1, :sentiment))
      |> Enum.reject(&is_nil/1)

    case sentiments do
      [] ->
        %{label: :neutral, confidence: 0.5}

      sentiments ->
        # Pick the sentiment with highest confidence that isn't neutral
        non_neutral =
          Enum.filter(sentiments, fn s ->
            label = Map.get(s, :label, :neutral)
            label != :neutral and label != "neutral"
          end)

        case non_neutral do
          [] -> List.first(sentiments)
          found -> Enum.max_by(found, &Map.get(&1, :confidence, 0.0))
        end
    end
  end

  # Add empathetic prefix for negative sentiment with high confidence
  defp maybe_add_sentiment_prefix(response, %{label: label, confidence: confidence})
       when label in [:negative, "negative"] and confidence >= 0.7 do
    prefix =
      Enum.random([
        "I understand.",
        "I hear you.",
        "I can see that's frustrating."
      ])

    "#{prefix} #{response}"
  end

  defp maybe_add_sentiment_prefix(response, _sentiment), do: response
end
