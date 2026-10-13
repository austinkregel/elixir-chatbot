defmodule Brain.Response.Generator do
  @moduledoc "Generative response synthesis entry point.\n\nThis module orchestrates response generation using a generative pipeline:\n\n1. **Retrieve Context** - Query memory for similar episodes, get semantic facts\n2. **Synthesize** - Compose response from primitives and domain knowledge\n3. **Compose** - Weave parts together using speech act analysis\n4. **Refine / improve** - Quality-check and optionally rewrite via the\n   feature-vector / template pipeline (the historical LSTM rescorer has\n   been removed; the hook is preserved as a no-op pass-through).\n\nThe system generates novel responses by combining:\n- Domain knowledge (from priv/knowledge/domains/*.json)\n- Recurring patterns: semantic facts consolidated from episodes (`Brain.Memory.RecurringPattern`), ahead of single episodes\n- Similar past episodes (memory-augmented)\n- Response primitives (hedges, acknowledgments, connectors)\n- Entity slot filling\n\nBrain should use this module instead of implementing response logic directly.\n"

  alias Brain.Analysis.SpeechActClassifier
  alias Brain.Memory
  alias Brain.Response
  require Logger

  alias Response.{TemplateStore, MemoryAugmented, FactRetriever, TemplateBlender}
  alias Response.{ResponseQuality, Synthesizer, Enricher}
  alias Response.RefinementLoop
  alias Memory.{RecurringPattern, Store}
  alias Brain.Code.QueryHandler

  @doc "Generate a response for the given intent and entities.\n\n`opts` must carry `:world_id`, the world whose memory and code index the response draws on; absent raises. It may carry `:side_effects`; with `false`, a service call that would change state (a device action) is refused during enrichment.\n\nUses a generative pipeline:\n1. Retrieve similar episodes from memory\n2. Synthesize response from domain knowledge and primitives\n3. Fall back to templates if synthesis doesn't produce a result\n4. Run the (now no-op) refinement hook and the quality improver\n\nReturns:\n- {:ok, response, :synthesized} for generated responses\n- {:ok, response, :memory_adapted} for memory-adapted responses\n- {:ok, response, :template} for template-based responses\n- {:ok, response, :fallback} for fallback responses\n"
  def generate(intent, entities, query_text, opts) do
    generate_with_events(intent, entities, query_text, [], opts)
  end

  @doc "Generate a response with event context for better slot filling.\n\nWhen events are provided, they are used to:\n- Provide action/actor/object slots for template filling\n- Enhance context retrieval from memory\n- Improve response relevance based on user intent structure\n\n`opts` takes the same options as `generate/4`; `:world_id` is required.\n\n## Examples\n\n    events = [%Event{action: %{lemma: \"play\"}, object: %{text: \"jazz\"}}]\n    generate_with_events(\"music.play\", entities, \"Play some jazz\", events, world_id: \"default\")\n"
  def generate_with_events(intent, entities, query_text, events, opts) when is_list(events) do
    world_id = world_id!(opts, :generate_with_events)

    Brain.Telemetry.span(:response_generate, %{intent: intent}, fn ->
      context =
        intent
        |> build_generation_context_with_events(entities, query_text, events, %{world_id: world_id})
        |> Map.put(:side_effects, Keyword.get(opts, :side_effects, true))

      # Build slot map for enrichment from entity list
      slots = build_slot_map_for_enrichment(entities)
      slots = if query_text, do: Map.put(slots, :_query_text, query_text), else: slots
      filled_slots = slots |> Map.keys() |> Enum.reject(fn k -> k in [:_query_text, :_ha_entity_id] end) |> Enum.map(&to_string/1)
      context = Map.put(context, :filled_slots, filled_slots)

      # Prepare context with enrichment data (before pipeline, for template conditions)
      context = Enricher.prepare_context(intent, slots, context)

      {result, recurring_pattern_use} = run_generative_pipeline(intent, entities, query_text, context)
      result = maybe_enrich_response(result, context)
      result = maybe_improve_response(result, query_text, intent, entities)

      # The response is settled: a synthesized one is never rewritten by the
      # improver, so the pattern it was adapted from is reported once, here.
      :ok = RecurringPattern.report_use(recurring_pattern_use, opts)
      result
    end)
  end

  defp build_slot_map_for_enrichment(entities) when is_list(entities) do
    Enum.reduce(entities, %{}, fn entity, acc ->
      type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type")
      value = Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :text)

      if type && value do
        slot_name = safe_slot_atom(type)
        metadata = Map.get(entity, :metadata) || Map.get(entity, "metadata")
        ha_entity_id = get_ha_entity_id(entity, metadata)

        acc = Map.put(acc, slot_name, value)

        if ha_entity_id do
          Map.put(acc, :_ha_entity_id, ha_entity_id)
        else
          acc
        end
      else
        acc
      end
    end)
  end

  defp build_slot_map_for_enrichment(_), do: %{}

  defp get_ha_entity_id(entity, metadata) do
    Map.get(entity, :ha_entity_id) ||
      Map.get(entity, "ha_entity_id") ||
      (is_map(metadata) && Map.get(metadata, :ha_entity_id)) ||
      nil
  end

  defp safe_slot_atom(type) when is_atom(type), do: type
  defp safe_slot_atom(type) when is_binary(type) do
    normalized = type |> to_string() |> String.downcase()
    String.to_existing_atom(normalized)
  rescue
    ArgumentError -> :entity
  end

  defp maybe_enrich_response({:ok, response, type}, context) do
    {:ok, enriched_response} = Enricher.enrich_response(response, context)
    {:ok, enriched_response, type}
  end

  defp maybe_enrich_response(other, _context), do: other

  defp world_id!(opts, function) do
    case opts[:world_id] do
      world_id when is_binary(world_id) and world_id != "" ->
        world_id

      other ->
        raise ArgumentError,
              "Brain.Response.Generator.#{function} requires :world_id, the world whose memory and " <>
                "code index the response draws on; got #{inspect(other)}"
    end
  end

  defp build_generation_context(intent, entities, query_text, world_id) do
    build_generation_context_with_events(intent, entities, query_text, [], %{world_id: world_id})
  end

  # `opts` must carry `:world_id`; it is also kept on the returned context so the
  # later strategies (memory, special handlers) read the same world.
  defp build_generation_context_with_events(intent, entities, query_text, events, opts) do
    world_id = Map.fetch!(opts, :world_id)
    memory_query = memory_query(intent, entities, query_text)
    similar_episodes = retrieve_similar_episodes(memory_query, world_id)
    semantic_facts = retrieve_semantic_facts(memory_query, world_id)
    recurring_pattern = RecurringPattern.find(semantic_facts, world_id)
    :ok = RecurringPattern.retrieved(semantic_facts, recurring_pattern, world_id, [])
    event_episodes = retrieve_event_episodes(events, world_id)
    event_slots = build_context_from_events(events, entities)

    epistemic_context = Map.get(opts, :epistemic_context, %{})
    user_id = Map.get(opts, :user_id)
    conversation_id = Map.get(opts, :conversation_id)

    user_prefs = if user_id, do: Brain.Graph.Reader.user_preferences(user_id, world_id: world_id), else: []

    conversation_topics =
      if conversation_id,
        do: Brain.Graph.Reader.conversation_topics(to_string(conversation_id), world_id: world_id),
        else: []

    recent_context =
      if conversation_id,
        do: Brain.Graph.Reader.recent_context(to_string(conversation_id), world_id: world_id),
        else: []

    event_frames = Map.get(opts, :event_frames, [])
    srl_frames = Map.get(opts, :srl_frames, [])

    entity_familiarity = Brain.Analysis.EntityGraphEnricher.familiarity_score(entities)

    fallback_confidence = 0.5 + entity_familiarity * 0.4
    confidence = Map.get(opts, :analysis_confidence) || fallback_confidence
    should_hedge = Map.get(opts, :should_hedge, false)

    %{
      similar_episodes: similar_episodes ++ event_episodes,
      semantic_facts: semantic_facts,
      recurring_pattern: recurring_pattern,
      confidence: confidence,
      should_hedge: should_hedge,
      intent: intent,
      entities: entities,
      query_text: query_text,
      events: events,
      event_slots: event_slots,
      epistemic_context: epistemic_context,
      user_prefs: user_prefs,
      conversation_topics: conversation_topics,
      recent_context: recent_context,
      event_frames: event_frames,
      srl_frames: srl_frames,
      entity_familiarity: entity_familiarity,
      user_id: user_id,
      conversation_id: conversation_id,
      world_id: world_id
    }
  end


  defp retrieve_event_episodes(events, world_id) when is_list(events) and events != [] do
    case get_primary_action(events) do
      nil ->
        []

      action_lemma ->
        if Process.whereis(Store) do
          case Store.query_events_by_action(action_lemma, 3, world_id: world_id) do
            {:ok, episodes} -> episodes
            _ -> []
          end
        else
          []
        end
    end
  end

  defp retrieve_event_episodes(_, _world_id) do
    []
  end

  @doc "Build template context slots from extracted events.\n\nThis provides action/actor/object slots for more relevant response generation.\n\n## Examples\n\n    events = [%Event{action: %{lemma: \"play\"}, object: %{text: \"jazz\"}}]\n    build_context_from_events(events, entities)\n    # => %{action: \"play\", object: \"jazz\", location: \"London\", ...}\n"
  def build_context_from_events(events, entities) when is_list(events) do
    base_slots = build_entity_slots(entities)
    event_slots = extract_event_slots(events)

    Map.merge(base_slots, event_slots)
  end

  @external_resource Path.join(:code.priv_dir(:brain), "knowledge/entity_slot_mappings.json")
  @entity_slot_config Path.join(:code.priv_dir(:brain), "knowledge/entity_slot_mappings.json")
                      |> File.read!()
                      |> Jason.decode!()
  @entity_type_to_slot Map.get(@entity_slot_config, "entity_type_to_slot", %{})

  defp build_entity_slots(entities) when is_list(entities) do
    Enum.reduce(entities, %{}, fn entity, acc ->
      type =
        Map.get(entity, :entity_type) || Map.get(entity, "entity_type") || Map.get(entity, :type)

      value = Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :text)

      case Map.get(@entity_type_to_slot, type) do
        nil -> acc
        slot_name -> Map.put(acc, String.to_atom(slot_name), value)
      end
    end)
  end

  defp build_entity_slots(_) do
    %{}
  end

  defp extract_event_slots(events) when is_list(events) and events != [] do
    primary = get_primary_event(events)

    if primary do
      slots = %{}

      slots =
        case primary do
          %{action: %{lemma: lemma}} when is_binary(lemma) ->
            Map.put(slots, :action, lemma)

          %{action: %{verb: verb}} when is_binary(verb) ->
            Map.put(slots, :action, String.downcase(verb))

          _ ->
            slots
        end

      slots =
        case primary do
          %{actor: %{text: text}} when is_binary(text) ->
            Map.put(slots, :actor, text)

          _ ->
            slots
        end

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

  defp extract_event_slots(_) do
    %{}
  end

  defp get_primary_event(events) when is_list(events) and events != [] do
    Enum.max_by(events, fn e -> Map.get(e, :confidence, 0.0) end, fn -> nil end)
  end

  defp get_primary_event(_) do
    nil
  end

  defp get_primary_action(events) do
    case get_primary_event(events) do
      %{action: %{lemma: lemma}} when is_binary(lemma) -> lemma
      %{action: %{verb: verb}} when is_binary(verb) -> String.downcase(verb)
      _ -> nil
    end
  end

  # The text memory is searched with: the intent, the user's words and the
  # entities' values.
  defp memory_query(intent, entities, query_text) do
    [
      intent || "",
      query_text || "",
      entities |> Enum.map_join(" ", fn e -> e[:value] || e["value"] || "" end)
    ]
    |> Enum.filter(&(&1 != ""))
    |> Enum.join(" ")
  end

  # The world's semantic facts most similar to the turn, searched with the
  # same text as its similar episodes.
  defp retrieve_semantic_facts("", _world_id), do: []

  defp retrieve_semantic_facts(query, world_id) do
    case Store.query_semantic(query, 5, world_id: world_id) do
      {:ok, semantic_facts} ->
        semantic_facts

      {:error, reason} ->
        raise "Brain.Response.Generator: the semantic facts of world #{inspect(world_id)} could not " <>
                "be searched: #{inspect(reason)}"
    end
  end

  defp retrieve_similar_episodes(query, world_id) do
    if query != "" and Process.whereis(Store) do
      case Store.query_similar(query, 5, world_id: world_id) do
        {:ok, episodes} -> episodes
        _ -> []
      end
    else
      []
    end
  rescue
    _ -> []
  end

  # The response and how it drew on the turn's recurring pattern
  # (`Brain.Memory.RecurringPattern.pattern_use/0`, `nil` for none).
  defp run_generative_pipeline(intent, entities, query_text, context) do
    case Synthesizer.synthesize(intent, entities,
           confidence: context.confidence,
           similar_episodes: context.similar_episodes,
           recurring_pattern: context.recurring_pattern,
           context: context
         ) do
      {:ok, response, recurring_pattern} ->
        {{:ok, response, :synthesized}, synthesizer_use(recurring_pattern)}

      :not_synthesized ->
        case try_memory_augmented(intent, entities, context.world_id) do
          {:ok, response} ->
            {{:ok, response, :memory_adapted}, nil}

          :not_handled ->
            case try_template_response(intent, entities, context) do
              {:ok, response} ->
                {{:ok, response, :template}, nil}

              :not_handled ->
                case try_special_handlers(intent, entities, query_text, context.world_id) do
                  {:ok, response} ->
                    {{:ok, response, :special_handler}, nil}

                  :not_handled ->
                    response = Synthesizer.get_fallback_response()
                    {{:ok, response, :fallback}, nil}
                end
            end
        end
    end
  end

  defp synthesizer_use(nil), do: nil
  defp synthesizer_use(%RecurringPattern{} = pattern), do: {pattern, :synthesizer}

  defp maybe_improve_response({:ok, response, type}, query_text, intent, entities) do
    if type in [:synthesized, :special_handler] or is_nil(query_text) do
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

  defp maybe_improve_response(other, _query, _intent, _entities) do
    other
  end

  defp try_special_handlers(intent, entities, query_text, world_id) do
    domain = infer_domain_from_label(intent)

    cond do
      domain == :code ->
        handle_code_intent(intent, entities, query_text, world_id)

      domain == :question or intent == "knowledge.query" ->
        handle_factual_query(entities, query_text, world_id)

      true ->
        :not_handled
    end
  end

  defp infer_domain_from_label(nil), do: nil
  defp infer_domain_from_label(intent) when is_binary(intent) do
    intent |> String.split(".", parts: 2) |> List.first() |> safe_to_domain_atom()
  end
  defp infer_domain_from_label(_), do: nil

  defp safe_to_domain_atom(str) when is_binary(str) do
    String.to_existing_atom(str)
  rescue
    ArgumentError -> String.to_atom(str)
  end

  defp handle_code_intent(intent, entities, query_text, world_id) do
    case QueryHandler.handle(intent, entities, world_id: world_id, query_text: query_text) do
      {:ok, response} -> {:ok, response}
      :not_handled -> :not_handled
    end
  end

  # Handle factual queries using semantic search with keyword fallback.
  # This uses the TF-IDF semantic retriever for best results, falling back
  # to keyword-based FactRetriever when semantic search isn't available.
  defp handle_factual_query(entities, query_text, world_id) do
    generate_factual_with_semantic_search(entities, query_text, world_id)
  end

  @doc "Generate a response with full path tracking for debugging/inspection.\n\n`opts` must carry `:world_id`, the world whose memory and code index the response draws on; absent raises.\n\nReturns:\n- {:ok, response, response_type, path} where path is a list of steps taken\n\nThe path shows exactly how the response was reached:\n- Which handlers were tried\n- Why each was skipped or selected\n- What data stores were accessed\n"
  def generate_with_path(intent, entities, query_text, opts) do
    world_id = world_id!(opts, :generate_with_path)
    path = []
    context = build_generation_context(intent, entities, query_text, world_id)
    path = path ++ [%{step: :try, handler: :synthesizer, intent: intent}]

    case Synthesizer.synthesize(intent, entities,
           confidence: context.confidence,
           similar_episodes: context.similar_episodes,
           recurring_pattern: context.recurring_pattern,
           context: context
         ) do
      {:ok, response, recurring_pattern} ->
        :ok = RecurringPattern.report_use(synthesizer_use(recurring_pattern), opts)

        path =
          path ++
            [
              %{
                step: :selected,
                handler: :synthesizer,
                reason: "synthesized from domain knowledge"
              }
            ]

        {:ok, response, :synthesized, path}

      :not_synthesized ->
        path =
          path ++
            [%{step: :skip, handler: :synthesizer, reason: "no domain knowledge for intent"}]

        {path, memory_result} = try_memory_with_path(intent, entities, path, world_id)

        case memory_result do
          {:ok, response} ->
            path =
              path ++
                [%{step: :selected, handler: :memory_augmented, reason: "similar episodes found"}]

            {:ok, response, :memory_adapted, path}

          :not_handled ->
            {path, template_result} = try_template_with_path(intent, entities, path)

            case template_result do
              {:ok, response} ->
                path =
                  path ++
                    [%{step: :selected, handler: :template, reason: "template found for intent"}]

                {:ok, response, :template, path}

              :not_handled ->
                path = path ++ [%{step: :try, handler: :special, intent: intent}]

                case try_special_handlers(intent, entities, query_text, world_id) do
                  {:ok, response} ->
                    path =
                      path ++
                        [%{step: :selected, handler: :special, reason: "special handler matched"}]

                    {:ok, response, :special_handler, path}

                  :not_handled ->
                    response = Synthesizer.get_fallback_response()

                    path =
                      path ++
                        [%{step: :selected, handler: :fallback, reason: "no handlers matched"}]

                    {:ok, response, :fallback, path}
                end
            end
        end
    end
  end

  defp try_memory_with_path(intent, entities, path, world_id) do
    path = path ++ [%{step: :try, handler: :memory_augmented, store: "Memory.Store"}]

    case try_memory_augmented(intent, entities, world_id) do
      {:ok, response} ->
        {path, {:ok, response}}

      :not_handled ->
        path =
          path ++
            [
              %{
                step: :skip,
                handler: :memory_augmented,
                reason: "no similar episodes or embedder not ready"
              }
            ]

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

  @doc "Generate a response using context-aware template selection.\n\nThis uses conditional template matching and semantic ranking:\n1. Filter this intent's templates by conditions that match the context\n2. Rank matching templates by similarity to the query\n\n## Parameters\n- `intent` - The classified intent name\n- `entities` - List of extracted entities\n- `query_text` - The original user query\n- `context` - Additional context (filled_slots, missing_slots, confidence, speech_act); must carry `:world_id`, the world whose memory the response draws on (absent raises)\n\n## Returns\n- {:ok, response, :conditional_template} for condition-matched templates\n- Falls back to regular generate/4 if conditional selection fails\n"
  def generate_with_context(intent, entities, query_text, context) do
    world_id = world_id!(context, :generate_with_context)
    full_context = entities |> build_template_context(context) |> Map.put(:world_id, world_id)
    confidence = Map.get(context, :confidence, 0.7)

    # No recurring pattern is passed, so none is adapted from.
    case Synthesizer.synthesize(intent, entities, confidence: confidence, context: context) do
      {:ok, response, nil} ->
        {:ok, response, :synthesized}

      :not_synthesized ->
        case try_conditional_template(intent, query_text, entities, full_context) do
          {:ok, response, type} ->
            {:ok, response, type}

          :not_handled ->
            case try_blended_response(world_id, query_text, full_context) do
              {:ok, response} ->
                {:ok, response, :blended}

              :not_handled ->
                case try_memory_augmented(intent, entities, world_id) do
                  {:ok, response} ->
                    {:ok, response, :memory_augmented}

                  :not_handled ->
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

  @doc "Builds the context map for conditional template selection from entities and analysis.\n"
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
  Generates a response using the synthesis pipeline (progressive refinement).

  This is the primary response generation path. It uses the DiscoursePlanner,
  ContentSpecifier, SurfaceRealizer, and RefinementLoop to produce a response
  grounded in the analysis context.

  Returns `{response, method}` on success or `{:error, reason}` on failure.
  """
  def generate_via_synthesis(analysis_model, _intent, _entities, _query_text, opts \\ %{}) do
    loop_opts = Keyword.new(opts)

    case RefinementLoop.generate(analysis_model, loop_opts) do
      {:ok, {:ouro_dry_run, _messages} = response, %{method: :ouro_dry_run}} ->
        {response, :ouro_dry_run}

      {:ok, response, metadata} when is_binary(response) and response != "" ->
        method = Map.get(metadata, :method, :synthesis_pipeline)
        enriched_response = maybe_enrich_synthesis_response(response, opts)
        {enriched_response, method}

      {:ok, nil, %{method: :silence_preferred} = _metadata} ->
        {nil, :silence_preferred}

      {:error, reason} ->
        Logger.error("Synthesis pipeline failed: #{inspect(reason)}")
        {:error, {:generation_failed, reason}}
    end
  end

  defp maybe_enrich_synthesis_response(response, opts) do
    unified_context = Map.get(opts, :unified_context, %{})
    enrichment = if is_map(unified_context), do: Map.get(unified_context, :enrichment, %{}), else: %{}
    enriched_data = if is_map(enrichment), do: Map.get(enrichment, :enriched_data, %{}), else: %{}

    enriched =
      if enriched_data != %{} do
        case Enricher.enrich_response(response, %{enriched_data: enriched_data}) do
          {:ok, result} -> result
          _ -> response
        end
      else
        response
      end

    validate_and_strip_placeholders(enriched, enriched_data)
  end

  @placeholder_pattern ~r/\$([a-z][a-z0-9_]*)/
  defp validate_and_strip_placeholders(text, enriched_data) do
    remaining = Regex.scan(@placeholder_pattern, text)

    if remaining == [] do
      text
    else
      valid_keys =
        enriched_data
        |> Map.keys()
        |> Enum.map(&to_string/1)
        |> MapSet.new()

      Enum.reduce(remaining, text, fn [full_match, key], acc ->
        if MapSet.member?(valid_keys, key) do
          Logger.warning("Placeholder $#{key} was not substituted despite being in enriched_data")
          acc
        else
          Logger.warning("Stripping unknown placeholder $#{key} from response")
          String.replace(acc, full_match, "")
        end
      end)
      |> String.replace(~r/\s{2,}/, " ")
      |> String.trim()
    end
  end

  @doc "Formats a code snippet for display in a response.\n\n## Options\n  - `:language` - The programming language for syntax highlighting\n  - `:start_line` - Starting line number\n  - `:max_lines` - Maximum lines to show (default: 20)\n"
  def format_code_snippet(code, opts \\ []) do
    language = Keyword.get(opts, :language, "")
    max_lines = Keyword.get(opts, :max_lines, 20)

    lines = String.split(code, "\n")

    truncated =
      if length(lines) > max_lines do
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

  # Semantic fact retrieval with fallback to keyword-based retrieval.
  # Uses SemanticFactRetriever for TF-IDF similarity matching, falling back
  # to FactRetriever keyword search if semantic search yields no results.
  defp generate_factual_with_semantic_search(entities, query_text, world_id) do
    alias Brain.Response.SemanticFactRetriever

    query_str = query_text || ""

    # Expand query with graph-derived related concepts
    {_original, related_terms} = Brain.Graph.Reader.expand_query(query_str, entities, world_id: world_id)
    expanded_query = if related_terms != [], do: query_str <> " " <> Enum.join(Enum.take(related_terms, 3), " "), else: query_str

    if SemanticFactRetriever.ready?() and query_str != "" do
      results = SemanticFactRetriever.search(world_id, expanded_query, limit: 3, threshold: 0.25)

      if results != [] do
        response = format_semantic_results(query_str, results)
        {:ok, response}
      else
        try_keyword_fact_retrieval(entities, query_str, world_id)
      end
    else
      try_keyword_fact_retrieval(entities, query_str, world_id)
    end
  end

  defp try_keyword_fact_retrieval(entities, query_str, world_id) do
    if FactRetriever.available?() do
      entity_names = extract_entity_names_for_facts(entities)
      facts = FactRetriever.get_facts_for_query(query_str, entity_names)

      if facts != [] do
        response = format_factual_response(query_str, facts, entities, world_id)
        {:ok, response}
      else
        :not_handled
      end
    else
      :not_handled
    end
  end

  defp format_semantic_results(_query_text, results) do
    best = List.first(results)
    fact = best.fact
    similarity = best.similarity

    if similarity > 0.6 do
      "#{fact.fact}"
    else
      "Based on what I know: #{fact.fact}"
    end
  end

  defp try_memory_augmented(intent, entities, world_id) do
    case MemoryAugmented.generate(intent, entities, world_id: world_id) do
      {:ok, response, _metadata} ->
        {:ok, response}

      _ ->
        :not_handled
    end
  end

  defp try_template_response(intent, entities, context \\ nil) do
    if TemplateStore.ready?() do
      template =
        if is_map(context) and Map.has_key?(context, :query_text) do
          case TemplateStore.get_best_template(Map.fetch!(context, :world_id), intent, context.query_text, context) do
            {:ok, t} -> t
            # This intent has no template that matches the context. Asking for a
            # random one cannot help -- get_random_template/1 reads the same
            # per-intent list -- so fall through to the caller's next strategy.
            {:error, _reason} -> nil
          end
        else
          TemplateStore.get_random_template(intent)
        end

      case template do
        nil ->
          :not_handled

        t ->
          response = TemplateStore.substitute_slots(t, entities)
          {:ok, response}
      end
    else
      :not_handled
    end
  end

  defp try_conditional_template(intent, query_text, entities, context) do
    if TemplateStore.ready?() do
      case TemplateStore.get_best_template(Map.fetch!(context, :world_id), intent, query_text, context) do
        {:ok, template} ->
          response = TemplateStore.substitute_slots(template, entities)
          {:ok, response, :conditional_template}

        {:error, _reason} ->
          :not_handled
      end
    else
      :not_handled
    end
  end

  defp try_blended_response(world_id, query_text, context) do
    if TemplateBlender.ready?() do
      case TemplateBlender.blend(world_id, query_text, context) do
        {:ok, response} when is_binary(response) and response != "" ->
          {:ok, response}

        _ ->
          :not_handled
      end
    else
      :not_handled
    end
  end

  defp format_factual_response(query_text, facts, entities, world_id)

  defp format_factual_response(query_text, facts, entities, world_id)
       when is_binary(query_text) and query_text != "" do
    alias Brain.ML.Tokenizer
    alias Brain.ML.POSTagger

    query_tokens = Tokenizer.tokenize(query_text)
    speech_act = SpeechActClassifier.classify(query_text, entities: entities, world_id: world_id)
    is_question = Map.get(speech_act, :is_question, false)

    query_numbers =
      query_tokens
      |> Enum.filter(fn t -> t.type == :number end)
      |> Enum.map(fn t -> parse_number(t.text) end)
      |> Enum.reject(&is_nil/1)

    content_words = extract_content_words_with_pos(query_tokens)

    if is_question and query_numbers != [] and content_words != [] do
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

  defp format_factual_response(_query_text, facts, _entities, _world_id) do
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
    content_set = MapSet.new(content_words)

    Enum.find(facts, fn fact ->
      fact_tokens = Brain.ML.Tokenizer.tokenize_normalized(fact.fact) |> MapSet.new()
      entity_tokens = Brain.ML.Tokenizer.tokenize_normalized(fact.entity) |> MapSet.new()
      all_fact_tokens = MapSet.union(fact_tokens, entity_tokens)

      not MapSet.disjoint?(content_set, all_fact_tokens)
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

      numbers_to_check =
        if relevant_numbers != [] do
          relevant_numbers
        else
          fact_numbers
        end

      if stated_number in numbers_to_check do
        {:match, stated_number}
      else
        {:mismatch, List.first(numbers_to_check)}
      end
    end
  end

  defp extract_entity_names_for_facts(entities) when is_list(entities) do
    entities
    |> Enum.map(fn e ->
      e[:value] || e["value"] || ""
    end)
    |> Enum.filter(&(&1 != ""))
  end

  defp extract_entity_names_for_facts(_) do
    []
  end

end
