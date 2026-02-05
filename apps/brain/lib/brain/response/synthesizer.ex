defmodule Brain.Response.Synthesizer do
  @moduledoc """
  Generative response composition from primitives and domain knowledge.

  Instead of template-based responses, this module composes responses
  from primitives based on:
  - Domain knowledge (loaded from priv/knowledge/domains/*.json)
  - Confidence levels of the knowledge being shared
  - Speech act analysis
  - Entity slot filling
  - Disclosure policy decisions

  Response primitives:
  - Acknowledgments - "Sure,", "Of course,"
  - Hedges - confidence-based language adjustments
  - Entity verbalizations - "in $location", "by $artist"
  - Response frames - domain-specific sentence structures
  - Uncertainty markers - "but that's just my understanding"

  This enables novel, appropriate responses without massive training data.
  """

  alias Brain.Epistemic.Types.SelfKnowledgeAssessment
  alias Brain.Epistemic.DisclosurePolicy
  alias Brain.Analysis.IntentRegistry

  require Logger

  # ============================================================================
  # Domain Knowledge Loading
  # ============================================================================

  @domains_path "priv/knowledge/domains"
  @primitives_path "priv/knowledge/domains/primitives.json"

  # Load primitives at compile time
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

  # Load all domain knowledge at compile time
  @domain_files Path.wildcard(Path.join(@domains_path, "*.json"))
  @external_resource @domains_path

  for file <- @domain_files do
    @external_resource file
  end

  @domain_knowledge (
                      @domain_files
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
                    )

  # Legacy primitives for epistemic responses (kept for backward compatibility)
  @soft_prefaces Map.get(@primitives, "hedges", %{}) |> Map.get("low_confidence", [
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

  # Load empty knowledge responses from primitives.json
  @empty_knowledge_responses Map.get(@primitives, "empty_knowledge", [
                               "I don't have much information about you yet. We're just getting to know each other.",
                               "I haven't learned much about you so far. Is there anything you'd like to share?",
                               "We haven't really talked much yet, so I don't have much to go on.",
                               "I'm still getting to know you. We haven't shared much yet."
                             ])

  # ============================================================================
  # General Response Synthesis (NEW)
  # ============================================================================

  @doc """
  Synthesizes a response for any intent using domain knowledge and primitives.

  This is the primary entry point for generative response creation.

  ## Parameters
  - `intent` - The classified intent (e.g., "weather.query")
  - `entities` - List of extracted entities
  - `opts` - Options including:
    - `:confidence` - Classification confidence (0.0-1.0)
    - `:speech_act` - Speech act analysis result
    - `:similar_episodes` - Similar past interactions from memory
    - `:semantic_facts` - Retrieved semantic facts

  ## Returns
  `{:ok, response}` or `:not_synthesized`
  """
  def synthesize(intent, entities, opts \\ []) do
    domain = IntentRegistry.domain(intent) || infer_domain(intent)
    confidence = Keyword.get(opts, :confidence, 0.7)
    similar_episodes = Keyword.get(opts, :similar_episodes, [])

    # Try to adapt from similar episodes first
    case adapt_from_episodes(similar_episodes, entities) do
      {:ok, response} ->
        {:ok, response}

      :no_adaptation ->
        # Fall back to domain-based synthesis
        synthesize_from_domain(domain, intent, entities, confidence, opts)
    end
  end

  @doc """
  Synthesizes a response using domain knowledge and entity slots.
  """
  def synthesize_from_domain(domain, intent, entities, confidence, opts \\ [])

  def synthesize_from_domain(nil, _intent, _entities, _confidence, _opts) do
    :not_synthesized
  end

  def synthesize_from_domain(domain, _intent, entities, confidence, _opts) do
    domain_str = to_string(domain)
    domain_config = Map.get(@domain_knowledge, domain_str, %{})

    if map_size(domain_config) == 0 do
      :not_synthesized
    else
      # Determine which response frame to use based on filled slots
      frame_key = determine_frame_key(domain_config, entities)
      frames = get_in(domain_config, ["response_frames", frame_key]) || []

      if length(frames) == 0 do
        :not_synthesized
      else
        # Select and fill a frame
        frame = Enum.random(frames)
        filled_response = fill_entity_slots(frame, entities)

        # Add acknowledgment prefix based on confidence
        final_response = maybe_add_acknowledgment(filled_response, confidence, domain_config)

        {:ok, final_response}
      end
    end
  end

  @doc """
  Synthesizes a clarification request when required slots are missing.
  """
  def synthesize_clarification(intent, entities, missing_slots, opts \\ []) do
    domain = IntentRegistry.domain(intent) || infer_domain(intent)
    domain_str = to_string(domain)
    domain_config = Map.get(@domain_knowledge, domain_str, %{})

    # Get clarification prefix
    clarification_prefix =
      @primitives
      |> Map.get("clarification_requests", %{})
      |> Map.get("missing_required", ["I need a bit more information."])
      |> Enum.random()

    # Get slot-specific clarification from intent registry or domain config
    slot_clarification = get_slot_clarification(intent, missing_slots, domain_config)

    # Build the response with any partial information we have
    partial_ack = build_partial_acknowledgment(entities, domain_config, opts)

    response =
      [partial_ack, clarification_prefix, slot_clarification]
      |> Enum.filter(&(&1 != nil and &1 != ""))
      |> Enum.join(" ")

    {:ok, response}
  end

  @doc """
  Synthesizes an expressive response (greeting, farewell, thanks, etc.).
  """
  def synthesize_expressive(sub_type, _opts \\ []) do
    smalltalk_config = Map.get(@domain_knowledge, "smalltalk", %{})
    frames = get_in(smalltalk_config, ["response_frames", to_string(sub_type)]) || []

    if length(frames) > 0 do
      {:ok, Enum.random(frames)}
    else
      :not_synthesized
    end
  end

  @doc """
  Gets a fallback response when nothing else works.
  """
  def get_fallback_response do
    fallbacks =
      Map.get(@primitives, "fallback_responses", [
        "I'm not sure how to respond to that. Could you try rephrasing?"
      ])

    Enum.random(fallbacks)
  end

  @doc """
  Gets a defer response when the bot wasn't directly addressed.
  """
  def get_defer_response do
    responses =
      Map.get(@primitives, "defer_responses", [
        "I'm here if you need me!"
      ])

    Enum.random(responses)
  end

  @doc """
  Gets a response when the NLP system cannot understand the input.
  """
  def get_cannot_respond_response do
    responses =
      Map.get(@primitives, "cannot_respond", [
        "I'm not sure what to make of that. Could you try rephrasing?"
      ])

    Enum.random(responses)
  end

  @doc """
  Gets a generic clarification request when no specific prompts are available.
  """
  def get_generic_clarification do
    clarifications =
      @primitives
      |> Map.get("clarification_requests", %{})
      |> Map.get("generic", ["Could you tell me more?"])

    Enum.random(clarifications)
  end

  @doc """
  Gets a transition phrase for adding additional information.
  """
  def get_transition_phrase(type \\ :additional_info) do
    type_str = to_string(type)

    phrases =
      @primitives
      |> Map.get("transition_phrases", %{})
      |> Map.get(type_str, ["Also,"])

    Enum.random(phrases)
  end

  @doc """
  Gets a response when knowledge about the user is empty/minimal.
  """
  def get_empty_knowledge_response do
    Enum.random(@empty_knowledge_responses)
  end

  @doc """
  Gets a quality fallback response for response improvement.
  """
  def get_quality_fallback do
    fallbacks =
      Map.get(@primitives, "quality_fallback", [
        "I'm here to help. Could you tell me more about what you're looking for?"
      ])

    Enum.random(fallbacks)
  end

  # ============================================================================
  # Episode Adaptation
  # ============================================================================

  defp adapt_from_episodes([], _entities), do: :no_adaptation

  defp adapt_from_episodes(episodes, entities) do
    # Find the best episode to adapt
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
        # Try to adapt the outcome as a response pattern
        adapted = adapt_response_pattern(episode.outcome, entities)
        {:ok, adapted}
    end
  rescue
    _ -> :no_adaptation
  end

  defp adapt_response_pattern(outcome, entities) when is_binary(outcome) do
    # Replace entity placeholders in the outcome with current entities
    Enum.reduce(entities, outcome, fn entity, acc ->
      entity_type = entity[:entity_type] || entity["entity_type"] || ""
      entity_value = entity[:value] || entity["value"] || ""

      if entity_type != "" and entity_value != "" do
        # Replace common placeholder patterns
        acc
        |> String.replace("$#{entity_type}", entity_value)
        |> String.replace("#{entity_type}", entity_value)
      else
        acc
      end
    end)
  end

  defp adapt_response_pattern(_, _entities), do: ""

  # ============================================================================
  # Frame Selection and Slot Filling
  # ============================================================================

  defp determine_frame_key(domain_config, entities) do
    slot_requirements = Map.get(domain_config, "slot_requirements", %{})
    required_slots = Map.get(slot_requirements, "required", [])

    # Check which required slots are filled
    filled_required =
      Enum.filter(required_slots, fn slot ->
        find_entity_value(entities, slot) != nil
      end)

    missing_required = required_slots -- filled_required

    # Determine frame key based on what's filled
    cond do
      length(missing_required) > 0 ->
        # Missing required slot - use clarification frame
        case missing_required do
          [single] -> "missing_#{single}"
          _ -> "missing_both"
        end

      length(filled_required) == 0 ->
        # No required slots defined or none filled
        "general"

      true ->
        # All required slots filled - use specific frame
        case Enum.sort(filled_required) do
          ["location"] -> "has_location"
          ["content", "date"] -> "create_with_date"
          ["content"] -> "create_content_only"
          ["device", "action"] -> "control_device"
          ["music-artist"] -> "play_artist"
          ["song"] -> "play_song"
          ["music-artist", "song"] -> "play_artist_song"
          ["topic"] -> "with_topic"
          ["symbol"] -> "explain_symbol"
          _ -> "general"
        end
    end
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
    # Only add acknowledgment for high-confidence, actionable responses
    if confidence >= 0.7 do
      ack_prefixes = Map.get(domain_config, "acknowledgment_prefixes", [])

      if length(ack_prefixes) > 0 and :rand.uniform() > 0.5 do
        "#{Enum.random(ack_prefixes)} #{String.downcase(String.first(response))}#{String.slice(response, 1..-1//1)}"
      else
        response
      end
    else
      response
    end
  end

  defp get_slot_clarification(intent, missing_slots, _domain_config) do
    # Try to get from intent registry first
    case IntentRegistry.get(intent) do
      nil ->
        # Generic clarification
        case missing_slots do
          [slot] -> "What #{humanize_key(slot)} would you like?"
          _ -> "Could you provide more details?"
        end

      meta ->
        templates = Map.get(meta, "clarification_templates", %{})

        case missing_slots do
          [slot] ->
            Map.get(templates, slot, "What #{humanize_key(slot)} would you like?")

          _ ->
            "Could you provide more details?"
        end
    end
  end

  defp build_partial_acknowledgment(entities, _domain_config, _opts) do
    # Acknowledge what we did understand
    filled_values =
      entities
      |> Enum.map(fn e -> e[:value] || e["value"] end)
      |> Enum.filter(&(&1 != nil and &1 != ""))

    if length(filled_values) > 0 do
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

  # Fallback domain inference when intent is not in IntentRegistry.
  # This uses simple prefix matching as a last resort. The primary domain
  # lookup through IntentRegistry.domain/1 is data-driven.
  # TODO: Consider loading prefix mappings from a config file if this grows.
  defp infer_domain(intent) when is_binary(intent) do
    cond do
      String.starts_with?(intent, "weather") -> :weather
      String.starts_with?(intent, "music") -> :music
      String.starts_with?(intent, "device") or String.starts_with?(intent, "smarthome") -> :device
      String.starts_with?(intent, "news") -> :news
      String.starts_with?(intent, "reminder") -> :reminder
      String.starts_with?(intent, "code") -> :code
      String.starts_with?(intent, "question") -> :question
      true -> nil
    end
  end

  defp infer_domain(_), do: nil

  @doc """
  Synthesizes a response for a meta-cognitive query.

  Takes a SelfKnowledgeAssessment and produces a natural, appropriate
  response with proper hedging.
  """
  def synthesize_self_knowledge_response(%SelfKnowledgeAssessment{} = assessment, opts \\ []) do
    context = Keyword.get(opts, :context, %{})
    rhetorical_strategy = determine_rhetorical_strategy(assessment, context)

    # Filter through disclosure policy
    filtered = DisclosurePolicy.filter_discloseable(assessment, context)

    # Build response based on what we can disclose
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

  @doc """
  Synthesizes a response for a single fact with appropriate hedging.
  """
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

  @doc """
  Determines the rhetorical strategy based on assessment content.
  """
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

  @doc """
  Gets response primitives for building custom responses.
  """
  def get_primitives do
    %{
      soft_prefaces: @soft_prefaces,
      evidence_clauses: @evidence_clauses,
      uncertainty_markers: @uncertainty_markers,
      correction_invites: @correction_invites
    }
  end

  # ============================================================================
  # Private Synthesis Functions
  # ============================================================================

  defp synthesize_no_knowledge_response(_opts) do
    Enum.random(@empty_knowledge_responses)
  end

  defp synthesize_limited_knowledge_response(assessment, opts) do
    facts = assessment.discloseable ++ assessment.inferred_uncertain

    if length(facts) == 0 do
      synthesize_no_knowledge_response(opts)
    else
      preface = Enum.random(@soft_prefaces)
      fact_mentions = build_fact_mentions(facts, 2)

      uncertainty =
        if length(assessment.inferred_uncertain) > 0 do
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

    if length(high_conf) == 0 and length(uncertain) == 0 do
      synthesize_no_knowledge_response(opts)
    else
      # Start with preface
      preface = Enum.random(@soft_prefaces)

      # Add high confidence facts
      high_conf_part =
        if length(high_conf) > 0 do
          high_conf_text = build_fact_mentions(high_conf, 3)
          "you've mentioned #{high_conf_text}"
        else
          nil
        end

      # Add uncertain facts with hedging
      uncertain_part =
        if length(uncertain) > 0 do
          uncertain_text = build_fact_mentions(uncertain, 2)
          "I also got the impression that #{uncertain_text}, but I'm not certain about that"
        else
          nil
        end

      # Add evidence clause
      evidence = Enum.random(@evidence_clauses)

      # Build parts list
      parts =
        [preface, high_conf_part, uncertain_part, evidence]
        |> Enum.filter(&(&1 != nil))

      # Build and add closing
      closing = Enum.random(@correction_invites)

      build_flowing_response(parts, closing)
    end
  end

  defp synthesize_rich_knowledge_response(assessment, opts) do
    high_conf = Enum.take(assessment.discloseable, 4)
    uncertain = Enum.take(assessment.inferred_uncertain, 2)

    if length(high_conf) == 0 and length(uncertain) == 0 do
      synthesize_no_knowledge_response(opts)
    else
      # For rich knowledge, we structure more carefully
      intro = "#{Enum.random(@soft_prefaces)}, here's what I know:"

      # Group facts by theme if possible
      high_conf_text =
        if length(high_conf) > 0 do
          build_fact_list(high_conf)
        else
          ""
        end

      uncertain_text =
        if length(uncertain) > 0 do
          "I'm less sure about: #{build_fact_mentions(uncertain, 3)}"
        else
          ""
        end

      # Epistemic disclaimer
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
    |> Enum.map(&("- " <> &1))
    |> Enum.join("; ")
  end

  defp format_single_fact(fact) do
    key_human = humanize_key(fact.key)
    value = format_value(fact.value)

    # Vary the phrasing
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

  defp humanize_key(_), do: "something"

  defp format_value(value) when is_binary(value), do: value
  defp format_value(value) when is_atom(value), do: Atom.to_string(value)
  defp format_value(value) when is_number(value), do: to_string(value)
  defp format_value(value) when is_list(value), do: Enum.join(value, ", ")
  defp format_value(value), do: inspect(value)

  defp join_with_and([]), do: ""
  defp join_with_and([single]), do: single

  defp join_with_and([a, b]), do: "#{a} and #{b}"

  defp join_with_and(items) do
    {last, rest} = List.pop_at(items, -1)
    Enum.join(rest, ", ") <> ", and #{last}"
  end

  defp build_flowing_response(parts, closing) do
    # Join parts into a flowing sentence
    main =
      parts
      |> Enum.filter(&(&1 != nil and &1 != ""))
      |> Enum.join(", ")

    "#{main}. #{closing}"
  end
end
