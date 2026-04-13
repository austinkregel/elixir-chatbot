defmodule Brain.Response.Synthesizer do
  @moduledoc "Generative response composition from primitives and domain knowledge.\n\nInstead of template-based responses, this module composes responses\nfrom primitives based on:\n- Domain knowledge (loaded from priv/knowledge/domains/*.json)\n- Confidence levels of the knowledge being shared\n- Speech act analysis\n- Entity slot filling\n- Disclosure policy decisions\n\nResponse primitives:\n- Acknowledgments - \"Sure,\", \"Of course,\"\n- Hedges - confidence-based language adjustments\n- Entity verbalizations - \"in $location\", \"by $artist\"\n- Response frames - domain-specific sentence structures\n- Uncertainty markers - \"but that's just my understanding\"\n\nThis enables novel, appropriate responses without massive training data.\n"

  alias Brain.Epistemic.Types.SelfKnowledgeAssessment
  alias Brain.Epistemic.DisclosurePolicy
  alias Brain.Analysis.IntentRegistry

  require Logger

  @domains_path "priv/knowledge/domains"
  @primitives_path "priv/knowledge/domains/primitives.json"
  @external_resource @primitives_path

  @primitives (case File.read(@primitives_path) do
                 {:ok, content} ->
                   case Jason.decode(content) do
                     {:ok, data} -> data
                     {:error, _} -> %{}
                   end

                 {:error, _} ->
                   %{}
               end)
  @domain_files Path.wildcard(Path.join(@domains_path, "*.json"))
  @external_resource @domains_path

  for file <- @domain_files do
    @external_resource file
  end

  @domain_knowledge @domain_files
                    |> Enum.reject(&String.ends_with?(&1, "primitives.json"))
                    |> Enum.reduce(%{}, fn file, acc ->
                      case File.read(file) do
                        {:ok, content} ->
                          case Jason.decode(content) do
                            {:ok, data} ->
                              domain = Map.get(data, "domain", Path.basename(file, ".json"))
                              Map.put(acc, domain, data)

                            {:error, _} ->
                              acc
                          end

                        {:error, _} ->
                          acc
                      end
                    end)
  @soft_prefaces Map.get(@primitives, "hedges", %{})
                 |> Map.get("low_confidence", [
                   "From what I remember",
                   "Based on our conversations",
                   "If I recall correctly",
                   "From what you've shared"
                 ])

  @evidence_clauses [
    "based on what you've mentioned",
    "from our previous chats",
    "according to what you've told me"
  ]

  @uncertainty_markers Map.get(@primitives, "uncertainty_markers", [
                         "but I could be off",
                         "though I might be misremembering",
                         "but that's just my impression",
                         "though I'm not entirely certain"
                       ])

  @correction_invites Map.get(@primitives, "correction_invites", [
                        "Feel free to correct me if I'm wrong.",
                        "Let me know if I've got anything mixed up.",
                        "Please tell me if that's not quite right.",
                        "I'm happy to be corrected on any of this."
                      ])
  @empty_knowledge_responses Map.get(@primitives, "empty_knowledge", [
                               "I don't have much information about you yet. We're just getting to know each other.",
                               "I haven't learned much about you so far. Is there anything you'd like to share?",
                               "We haven't really talked much yet, so I don't have much to go on.",
                               "I'm still getting to know you. We haven't shared much yet."
                             ])

  @external_resource Path.join(:code.priv_dir(:brain), "response/frame_key_mappings.json")
  @frame_key_config Path.join(:code.priv_dir(:brain), "response/frame_key_mappings.json")
                    |> File.read!()
                    |> Jason.decode!()
  @frame_key_mappings Map.get(@frame_key_config, "slot_to_frame_key", %{})
  @frame_key_default Map.get(@frame_key_config, "default", "general")

  @doc "Synthesizes a response for any intent using domain knowledge and primitives.\n\nThis is the primary entry point for generative response creation.\n\n## Parameters\n- `intent` - The classified intent (e.g., \"weather.query\")\n- `entities` - List of extracted entities\n- `opts` - Options including:\n  - `:confidence` - Classification confidence (0.0-1.0)\n  - `:speech_act` - Speech act analysis result\n  - `:similar_episodes` - Similar past interactions from memory\n  - `:semantic_facts` - Retrieved semantic facts\n\n## Returns\n`{:ok, response}` or `:not_synthesized`\n"
  def synthesize(intent, entities, opts \\ []) do
    domain = IntentRegistry.domain(intent) || infer_domain(intent)
    confidence = Keyword.get(opts, :confidence, 0.7)
    similar_episodes = Keyword.get(opts, :similar_episodes, [])
    context = Keyword.get(opts, :context, %{})

    # Check epistemic context FIRST — contradictions and uncertainty should
    # take priority over memory-based or domain-based responses. This ensures
    # contradictions are surfaced even when the intent has no domain mapping
    # or when memory episodes would otherwise short-circuit the pipeline.
    epistemic_context = Map.get(context, :epistemic_context, %{})

    case handle_epistemic_context(epistemic_context, entities, confidence) do
      {:ok, response} ->
        {:ok, response}

      :continue ->
        case adapt_from_episodes(similar_episodes, entities) do
          {:ok, response} ->
            {:ok, response}

          :no_adaptation ->
            synthesize_from_domain(domain, intent, entities, confidence, opts)
        end
    end
  end

  @doc "Synthesizes a response using domain knowledge and entity slots.\n"
  def synthesize_from_domain(domain, intent, entities, confidence, opts \\ [])

  def synthesize_from_domain(nil, _intent, _entities, _confidence, _opts) do
    :not_synthesized
  end

  def synthesize_from_domain(domain, _intent, entities, confidence, opts) do
    domain_str = to_string(domain)
    domain_config = Map.get(@domain_knowledge, domain_str, %{})
    context = Keyword.get(opts, :context, %{})

    # Check for epistemic contradictions before normal synthesis
    epistemic_context = Map.get(context, :epistemic_context, %{})

    case handle_epistemic_context(epistemic_context, entities, confidence) do
      {:ok, response} ->
        {:ok, response}

      :continue ->
        if map_size(domain_config) == 0 do
          :not_synthesized
        else
          # Check if we have enriched data - if so, prefer enriched templates
          case try_enriched_response(domain_config, entities, context, confidence) do
            {:ok, response} ->
              response = maybe_add_graph_context(response, context)
              {:ok, response}

            :not_enriched ->
              # Fall back to standard response frames
              frame_key = determine_frame_key(domain_config, entities)
              frames = get_in(domain_config, ["response_frames", frame_key]) || []

              if frames == [] do
                :not_synthesized
              else
                frame = Enum.random(frames)
                filled_response = fill_entity_slots(frame, entities)
                final_response = maybe_add_acknowledgment(filled_response, confidence, domain_config)
                final_response = maybe_add_graph_context(final_response, context)

                {:ok, final_response}
              end
          end
        end
    end
  end

  defp maybe_add_graph_context(response, context) do
    entity_familiarity = Map.get(context, :entity_familiarity, 0.5)
    user_prefs = Map.get(context, :user_prefs, [])

    response = apply_interlocutor_adaptations(response, user_prefs)

    if entity_familiarity < 0.3 do
      add_hedging_prefix(response)
    else
      response
    end
  end

  defp apply_interlocutor_adaptations(response, []), do: response

  defp apply_interlocutor_adaptations(response, prefs) when is_list(prefs) do
    Enum.reduce(prefs, response, fn
      %{rel_type: "WANTS", topic: topic}, acc ->
        case String.downcase(topic) do
          "fahrenheit" -> String.replace(acc, ~r/(\d+)\s*°?\s*C\b/, "\\1°F")
          "celsius" -> String.replace(acc, ~r/(\d+)\s*°?\s*F\b/, "\\1°C")
          _ -> acc
        end

      _, acc ->
        acc
    end)
  end

  defp apply_interlocutor_adaptations(response, _), do: response

  defp add_hedging_prefix(response) do
    hedges = [
      "Based on what I know, ",
      "From what I understand, ",
      "As far as I can tell, ",
      "I think "
    ]

    Enum.random(hedges) <> String.downcase(String.first(response)) <> String.slice(response, 1..-1//1)
  end

  # Try to build a response using enriched_response_frames if enrichment succeeded
  defp try_enriched_response(domain_config, entities, context, confidence) do
    enriched_data = Map.get(context, :enriched_data, %{})
    enrichment_status = Map.get(context, :enrichment_status)

    enriched_frames = Map.get(domain_config, "enriched_response_frames", %{})

    cond do
      # Check for enrichment failure - use error templates
      enrichment_status == :failed and Map.has_key?(enriched_frames, "service_error") ->
        frames = get_in(enriched_frames, ["service_error", "templates"]) || []
        select_and_fill_enriched_frame(frames, entities, enriched_data, domain_config, confidence)

      # Service not configured - use honest "not available" templates instead of "Let me check..."
      Map.has_key?(enriched_frames, "service_not_configured") ->
        frame_config = enriched_frames["service_not_configured"]
        condition = Map.get(frame_config, "condition")

        if condition && Brain.Response.ConditionEvaluator.evaluate(condition, context) do
          frames = Map.get(frame_config, "templates", [])
          select_and_fill_enriched_frame(frames, entities, enriched_data, domain_config, confidence)
        else
          :not_enriched
        end

      # No enriched data available
      enriched_data == %{} or enriched_data == nil ->
        :not_enriched

      # Try to find a matching enriched frame
      map_size(enriched_frames) > 0 ->
        case find_matching_enriched_frame(enriched_frames, entities, enriched_data) do
          nil ->
            :not_enriched

          {_frame_key, frame_config} ->
            frames = Map.get(frame_config, "templates", [])
            select_and_fill_enriched_frame(frames, entities, enriched_data, domain_config, confidence)
        end

      true ->
        :not_enriched
    end
  end

  defp find_matching_enriched_frame(enriched_frames, entities, enriched_data) do
    # Sort frames by specificity (more required fields = more specific)
    enriched_frames
    |> Enum.reject(fn {key, _} -> key == "service_error" or key == "service_not_configured" end)
    |> Enum.sort_by(fn {_, config} ->
      required = Map.get(config, "requires_enrichment", [])
      -length(required)  # Negative for descending order (most specific first)
    end)
    |> Enum.find(fn {_key, config} ->
      required = Map.get(config, "requires_enrichment", [])
      has_all_required_enrichment?(required, enriched_data) and has_required_entities?(config, entities)
    end)
  end

  defp has_all_required_enrichment?(required_fields, enriched_data) do
    Enum.all?(required_fields, fn field ->
      field_key = safe_field_atom(field)
      field_str = to_string(field)
      Map.has_key?(enriched_data, field_key) or Map.has_key?(enriched_data, field_str)
    end)
  end

  defp safe_field_atom(field) when is_atom(field), do: field
  defp safe_field_atom(field) when is_binary(field) do
    String.to_existing_atom(field)
  rescue
    ArgumentError -> field
  end

  defp has_required_entities?(_config, _entities) do
    # For now, assume entities are available if we got this far
    # Could be enhanced to check specific entity requirements
    true
  end

  defp select_and_fill_enriched_frame([], _entities, _enriched_data, _domain_config, _confidence) do
    :not_enriched
  end

  defp select_and_fill_enriched_frame(frames, entities, enriched_data, domain_config, confidence) do
    frame = Enum.random(frames)

    # Fill both entity slots and enrichment placeholders
    filled_response =
      frame
      |> fill_entity_slots(entities)
      |> fill_enrichment_slots(enriched_data)

    final_response = maybe_add_acknowledgment(filled_response, confidence, domain_config)
    {:ok, final_response}
  end

  defp fill_enrichment_slots(text, enriched_data) when is_binary(text) do
    Brain.Response.Formatting.substitute_placeholders(text, enriched_data)
  end

  @doc "Synthesizes a clarification request when required slots are missing.\n"
  def synthesize_clarification(intent, entities, missing_slots, opts \\ []) do
    domain = IntentRegistry.domain(intent) || infer_domain(intent)
    domain_str = to_string(domain)
    domain_config = Map.get(@domain_knowledge, domain_str, %{})

    clarification_prefix =
      @primitives
      |> Map.get("clarification_requests", %{})
      |> Map.get("missing_required", ["I need a bit more information."])
      |> Enum.random()

    slot_clarification = get_slot_clarification(intent, missing_slots, domain_config)
    partial_ack = build_partial_acknowledgment(entities, domain_config, opts)

    response =
      [partial_ack, clarification_prefix, slot_clarification]
      |> Enum.filter(&(&1 != nil and &1 != ""))
      |> Enum.join(" ")

    {:ok, response}
  end

  @doc "Synthesizes an expressive response (greeting, farewell, thanks, etc.).\n"
  def synthesize_expressive(sub_type, _opts \\ []) do
    smalltalk_config = Map.get(@domain_knowledge, "smalltalk", %{})
    frames = get_in(smalltalk_config, ["response_frames", to_string(sub_type)]) || []

    if frames != [] do
      {:ok, Enum.random(frames)}
    else
      :not_synthesized
    end
  end

  @doc "Gets a fallback response when nothing else works.\nRaises if no fallback data is available in primitives.json.\n"
  def get_fallback_response do
    select_from_pool("fallback_responses")
  end

  @doc "Gets a defer response when the bot wasn't directly addressed.\n"
  def get_defer_response do
    select_from_pool("defer_responses")
  end

  @doc "Gets a response when the NLP system cannot understand the input.\n"
  def get_cannot_respond_response do
    select_from_pool("cannot_respond")
  end

  @doc "Gets a generic clarification request when no specific prompts are available.\n"
  def get_generic_clarification do
    pool = @primitives
    |> Map.get("clarification_requests", %{})
    |> Map.get("generic", [])

    if pool == [], do: raise("No clarification data in primitives.json")
    Enum.random(pool)
  end

  @doc "Gets a transition phrase for adding additional information.\n"
  def get_transition_phrase(type \\ :additional_info) do
    type_str = to_string(type)

    pool = @primitives
    |> Map.get("transition_phrases", %{})
    |> Map.get(type_str, [])

    if pool == [], do: raise("No transition phrase data for #{type_str} in primitives.json")
    Enum.random(pool)
  end

  @doc "Gets a response when knowledge about the user is empty/minimal.\n"
  def get_empty_knowledge_response do
    Enum.random(@empty_knowledge_responses)
  end

  @doc "Gets a quality fallback response for response improvement.\n"
  def get_quality_fallback do
    select_from_pool("quality_fallback")
  end

  defp select_from_pool(key) do
    pool = Map.get(@primitives, key, [])
    if pool == [], do: raise("No #{key} data in primitives.json")
    Enum.random(pool)
  end

  defp adapt_from_episodes([], _entities) do
    :no_adaptation
  end

  defp adapt_from_episodes(episodes, entities) do
    best_episode =
      episodes
      |> Enum.filter(fn {episode, similarity} ->
        similarity >= 0.7 and episode.outcome != nil and episode.outcome != ""
      end)
      |> Enum.max_by(fn {_ep, sim} -> sim end, fn -> nil end)

    case best_episode do
      nil ->
        :no_adaptation

      {episode, _similarity} ->
        adapted = adapt_response_pattern(episode.outcome, entities)
        {:ok, adapted}
    end
  rescue
    _ -> :no_adaptation
  end

  defp adapt_response_pattern(outcome, entities) when is_binary(outcome) do
    Enum.reduce(entities, outcome, fn entity, acc ->
      entity_type = entity[:entity_type] || entity["entity_type"] || ""
      entity_value = entity[:value] || entity["value"] || ""

      if entity_type != "" and entity_value != "" do
        acc
        |> String.replace("$#{entity_type}", entity_value)
        |> String.replace("#{entity_type}", entity_value)
      else
        acc
      end
    end)
  end

  defp adapt_response_pattern(_, _entities) do
    ""
  end

  defp determine_frame_key(domain_config, entities) do
    slot_requirements = Map.get(domain_config, "slot_requirements", %{})
    required_slots = Map.get(slot_requirements, "required", [])

    filled_required =
      Enum.filter(required_slots, fn slot ->
        find_entity_value(entities, slot) != nil
      end)

    missing_required = required_slots -- filled_required

    graph_type_key = infer_frame_from_graph_types(entities, domain_config)

    cond do
      missing_required != [] ->
        case missing_required do
          [single] -> "missing_#{single}"
          _ -> "missing_both"
        end

      graph_type_key != nil ->
        graph_type_key

      filled_required == [] ->
        "general"

      true ->
        sorted_key = filled_required |> Enum.sort() |> Enum.join(",")
        Map.get(@frame_key_mappings, sorted_key, @frame_key_default)
    end
  end

  defp infer_frame_from_graph_types(entities, domain_config) do
    response_frames = Map.get(domain_config, "response_frames", %{})
    frame_keys = Map.keys(response_frames)

    graph_types =
      entities
      |> Enum.map(&Map.get(&1, :graph_type))
      |> Enum.reject(&is_nil/1)
      |> Enum.map(&String.downcase/1)

    Enum.find(frame_keys, fn key ->
      Enum.any?(graph_types, fn gt -> String.contains?(key, gt) end)
    end)
  end

  defp fill_entity_slots(frame, entities) do
    Enum.reduce(entities, frame, fn entity, acc ->
      entity_type = entity[:entity_type] || entity["entity_type"] || ""
      entity_value = entity[:value] || entity["value"] || ""

      if entity_type != "" and entity_value != "" do
        String.replace(acc, "$#{entity_type}", entity_value)
      else
        acc
      end
    end)
  end

  defp maybe_add_acknowledgment(response, confidence, domain_config) do
    if confidence >= 0.7 do
      ack_prefixes = Map.get(domain_config, "acknowledgment_prefixes", [])

      if ack_prefixes != [] and :rand.uniform() > 0.5 do
        "#{Enum.random(ack_prefixes)} #{String.downcase(String.first(response))}#{String.slice(response, 1..-1//1)}"
      else
        response
      end
    else
      response
    end
  end

  defp get_slot_clarification(intent, missing_slots, _domain_config) do
    meta = IntentRegistry.get(intent)
    templates = if meta, do: Map.get(meta, "clarification_templates", %{}), else: %{}

    case missing_slots do
      [slot] ->
        Map.get(templates, slot) || get_generic_clarification()

      _ ->
        get_generic_clarification()
    end
  end

  defp build_partial_acknowledgment(entities, _domain_config, _opts) do
    filled_values =
      entities
      |> Enum.map(fn e -> e[:value] || e["value"] end)
      |> Enum.filter(&(&1 != nil and &1 != ""))

    if filled_values != [] do
      "I understand you're interested in #{Enum.join(filled_values, " and ")}."
    else
      nil
    end
  end

  defp find_entity_value(entities, entity_type) do
    Enum.find_value(entities, fn entity ->
      type = entity[:entity_type] || entity["entity_type"]

      if type == entity_type do
        entity[:value] || entity["value"]
      else
        nil
      end
    end)
  end

  defp infer_domain(intent) when is_binary(intent) do
    Brain.Analysis.IntentRegistry.domain(intent)
  end

  defp infer_domain(_) do
    nil
  end

  @doc "Synthesizes a response for a meta-cognitive query.\n\nTakes a SelfKnowledgeAssessment and produces a natural, appropriate\nresponse with proper hedging.\n"
  def synthesize_self_knowledge_response(%SelfKnowledgeAssessment{} = assessment, opts \\ []) do
    context = Keyword.get(opts, :context, %{})
    rhetorical_strategy = determine_rhetorical_strategy(assessment, context)
    filtered = DisclosurePolicy.filter_discloseable(assessment, context)

    case rhetorical_strategy do
      :no_knowledge ->
        synthesize_no_knowledge_response(opts)

      :limited_knowledge ->
        synthesize_limited_knowledge_response(filtered, opts)

      :moderate_knowledge ->
        synthesize_moderate_knowledge_response(filtered, opts)

      :rich_knowledge ->
        synthesize_rich_knowledge_response(filtered, opts)
    end
  end

  @doc "Synthesizes a response for a single fact with appropriate hedging.\n"
  def synthesize_fact_mention(fact, hedging_level, _opts \\ []) do
    key_human = humanize_key(fact.key)
    value = format_value(fact.value)

    case hedging_level do
      :none ->
        "#{key_human}: #{value}"

      :light ->
        "I think #{key_human} is #{value}"

      :strong ->
        "I might be wrong, but I believe #{key_human} is #{value}"

      :do_not_disclose ->
        nil
    end
  end

  @doc "Determines the rhetorical strategy based on assessment content.\n"
  def determine_rhetorical_strategy(%SelfKnowledgeAssessment{} = assessment, _context) do
    discloseable_count = length(assessment.discloseable)
    uncertain_count = length(assessment.inferred_uncertain)
    total = discloseable_count + uncertain_count

    cond do
      total == 0 -> :no_knowledge
      total <= 2 -> :limited_knowledge
      total <= 5 -> :moderate_knowledge
      true -> :rich_knowledge
    end
  end

  @doc "Gets response primitives for building custom responses.\n"
  def get_primitives do
    %{
      soft_prefaces: expand_phrases(@soft_prefaces),
      evidence_clauses: expand_phrases(@evidence_clauses),
      uncertainty_markers: expand_phrases(@uncertainty_markers),
      correction_invites: @correction_invites
    }
  end

  defp expand_phrases(phrases) when is_list(phrases) do
    if Process.whereis(Brain.ML.Lexicon) do
      original_set = MapSet.new(phrases)

      expanded =
        phrases
        |> Enum.flat_map(fn phrase ->
          variant = paraphrase_phrase(phrase)
          if variant != phrase and not MapSet.member?(original_set, variant) do
            [phrase, variant]
          else
            [phrase]
          end
        end)
        |> Enum.uniq()

      expanded
    else
      phrases
    end
  end

  defp paraphrase_phrase(phrase) do
    words = String.split(phrase)

    paraphrased =
      Enum.map(words, fn word ->
        lower = String.downcase(word)

        if String.length(lower) >= 4 and Brain.ML.Lexicon.known_word?(lower) do
          case Brain.ML.Lexicon.synonyms(lower) do
            [] -> word
            [syn | _] ->
              if String.length(syn) >= 3 and not String.contains?(syn, "_") do
                syn
              else
                word
              end
          end
        else
          word
        end
      end)

    Enum.join(paraphrased, " ")
  end

  defp synthesize_no_knowledge_response(_opts) do
    Enum.random(@empty_knowledge_responses)
  end

  defp synthesize_limited_knowledge_response(assessment, opts) do
    facts = assessment.discloseable ++ assessment.inferred_uncertain

    if facts == [] do
      synthesize_no_knowledge_response(opts)
    else
      preface = Enum.random(@soft_prefaces)
      fact_mentions = build_fact_mentions(facts, 2)

      uncertainty =
        if assessment.inferred_uncertain != [] do
          ", " <> Enum.random(@uncertainty_markers)
        else
          ""
        end

      correction = Enum.random(@correction_invites)

      "#{preface}, #{fact_mentions}#{uncertainty}. #{correction}"
    end
  end

  defp synthesize_moderate_knowledge_response(assessment, opts) do
    high_conf = assessment.discloseable
    uncertain = assessment.inferred_uncertain

    if high_conf == [] and uncertain == [] do
      synthesize_no_knowledge_response(opts)
    else
      preface = Enum.random(@soft_prefaces)

      high_conf_part =
        if high_conf != [] do
          high_conf_text = build_fact_mentions(high_conf, 3)
          "you've mentioned #{high_conf_text}"
        else
          nil
        end

      uncertain_part =
        if uncertain != [] do
          uncertain_text = build_fact_mentions(uncertain, 2)
          "I also got the impression that #{uncertain_text}, but I'm not certain about that"
        else
          nil
        end

      evidence = Enum.random(@evidence_clauses)

      parts =
        [preface, high_conf_part, uncertain_part, evidence]
        |> Enum.filter(&(&1 != nil))

      closing = Enum.random(@correction_invites)

      build_flowing_response(parts, closing)
    end
  end

  defp synthesize_rich_knowledge_response(assessment, opts) do
    high_conf = Enum.take(assessment.discloseable, 4)
    uncertain = Enum.take(assessment.inferred_uncertain, 2)

    if high_conf == [] and uncertain == [] do
      synthesize_no_knowledge_response(opts)
    else
      intro = "#{Enum.random(@soft_prefaces)}, here's what I know:"

      high_conf_text =
        if high_conf != [] do
          build_fact_list(high_conf)
        else
          ""
        end

      uncertain_text =
        if uncertain != [] do
          "I'm less sure about: #{build_fact_mentions(uncertain, 3)}"
        else
          ""
        end

      disclaimer =
        "That said, this is all #{Enum.random(@evidence_clauses)}, " <>
          "so it's pretty limited. #{Enum.random(@correction_invites)}"

      [intro, high_conf_text, uncertain_text, disclaimer]
      |> Enum.filter(&(&1 != ""))
      |> Enum.join(" ")
    end
  end

  defp build_fact_mentions(facts, max_count) do
    facts
    |> Enum.take(max_count)
    |> Enum.map(&format_single_fact/1)
    |> join_with_and()
  end

  defp build_fact_list(facts) do
    facts
    |> Enum.map(&format_single_fact/1)
    |> Enum.map_join(
      "; ",
      &("- " <> &1)
    )
  end

  defp format_single_fact(fact) do
    key_human = humanize_key(fact.key)
    value = format_value(fact.value)

    templates = [
      "#{key_human} is #{value}",
      "your #{key_human} (#{value})",
      "you mentioned #{key_human}: #{value}",
      "#{value} (#{key_human})"
    ]

    Enum.random(templates)
  end

  defp humanize_key(key) when is_atom(key) do
    key
    |> Atom.to_string()
    |> humanize_key()
  end

  defp humanize_key(key) when is_binary(key) do
    key
    |> String.replace("_", " ")
    |> String.replace("-", " ")
  end

  defp humanize_key(_) do
    "something"
  end

  defp format_value(value), do: Brain.Response.Formatting.format_value(value)

  defp join_with_and([]) do
    ""
  end

  defp join_with_and([single]) do
    single
  end

  defp join_with_and([a, b]) do
    "#{a} and #{b}"
  end

  defp join_with_and(items) do
    {last, rest} = List.pop_at(items, -1)
    Enum.join(rest, ", ") <> ", and #{last}"
  end

  defp build_flowing_response(parts, closing) do
    main =
      parts
      |> Enum.filter(&(&1 != nil and &1 != ""))
      |> Enum.join(", ")

    "#{main}. #{closing}"
  end

  # ============================================================================
  # Epistemic Context Handling
  # ============================================================================

  # Handles epistemic context to generate appropriate responses for
  # contradicted or uncertain claims.
  # Returns: {:ok, response} or :continue
  defp handle_epistemic_context(%{status: :contradicted} = context, _entities, _confidence) do
    beliefs = Map.get(context, :beliefs, [])
    synthesize_contradiction_response(beliefs)
  end

  defp handle_epistemic_context(%{status: :uncertain}, _entities, confidence) when confidence < 0.6 do
    synthesize_uncertain_response()
  end

  defp handle_epistemic_context(_, _, _) do
    :continue
  end

  defp synthesize_contradiction_response([belief | _rest]) do
    stored_object = get_belief_object(belief)

    pool = Map.get(@primitives, "contradiction_responses", [])

    if pool != [] do
      base = Enum.random(pool)
      {:ok, String.replace(base, "$stored", stored_object)}
    else
      {:ok, select_from_pool("cannot_respond")}
    end
  end

  defp synthesize_contradiction_response([]) do
    {:ok, select_from_pool("cannot_respond")}
  end

  defp synthesize_uncertain_response do
    {:ok, get_generic_clarification()}
  end

  defp get_belief_object(%{object: object}) when is_binary(object), do: object
  defp get_belief_object(%{object: object}) when is_atom(object), do: to_string(object)
  defp get_belief_object(%{"object" => object}) when is_binary(object), do: object
  defp get_belief_object(_), do: "something different"
end
