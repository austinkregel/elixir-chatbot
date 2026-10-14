defmodule Brain.Response.Synthesizer do
  @moduledoc "Generative response composition from primitives and domain knowledge.\n\nInstead of template-based responses, this module composes responses\nfrom primitives based on:\n- Domain knowledge (loaded from priv/knowledge/domains/*.json)\n- Confidence levels of the knowledge being shared\n- Speech act analysis\n- Entity slot filling\n- Disclosure policy decisions\n\nResponse primitives:\n- Acknowledgments - \"Sure,\", \"Of course,\"\n- Hedges - confidence-based language adjustments\n- Entity verbalizations - \"in $location\", \"by $artist\"\n- Response frames - domain-specific sentence structures\n- Uncertainty markers - \"but that's just my understanding\"\n\nThis enables novel, appropriate responses without massive training data.\n"

  alias Brain.Analysis.SlotDetector
  alias Brain.Memory.RecurringPattern
  alias Brain.Epistemic.Types.SelfKnowledgeAssessment
  alias Brain.Epistemic.DisclosurePolicy

  require Logger

  @domains_path "priv/knowledge/domains"
  @primitives_path "priv/knowledge/domains/primitives.json"
  @external_resource @primitives_path

  @primitives @primitives_path |> File.read!() |> Jason.decode!()
  @domain_files Path.wildcard(Path.join(@domains_path, "*.json"))
  @external_resource @domains_path

  for file <- @domain_files do
    @external_resource file
  end

  @domain_knowledge @domain_files
                    |> Enum.reject(&String.ends_with?(&1, "primitives.json"))
                    |> Enum.reduce(%{}, fn file, acc ->
                      data = file |> File.read!() |> Jason.decode!()
                      domain = Map.get(data, "domain", Path.basename(file, ".json"))
                      Map.put(acc, domain, data)
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

  @doc """
  Synthesizes a response for any intent using domain knowledge and primitives.

  This is the primary entry point for generative response creation.

  Unless the epistemic context answers first, a response is adapted from
  memory before domain knowledge is tried: from the turn's recurring pattern
  when there is one, its reply with the turn's entities filled in; otherwise
  from the most similar single episode at least
  `Brain.Memory.RecurringPattern.similarity_threshold/0` similar, its outcome
  filled in the same way. The pattern comes first because its reply was given
  across several matching messages, a single episode's only once.

  Synthesizing reports nothing: a response is synthesized on each pass of a
  refined turn, and only the one the turn settles on is its response. The
  pattern a response was adapted from is returned, for the caller that settles
  on the response to report (`Brain.Memory.RecurringPattern.used/3`).

  ## Parameters
  - `intent` - The classified intent (e.g., "weather.query")
  - `entities` - List of extracted entities
  - `opts` - Options including:
    - `:confidence` - Classification confidence (0.0-1.0)
    - `:speech_act` - Speech act analysis result
    - `:similar_episodes` - Similar past interactions from memory
    - `:recurring_pattern` - The turn's recurring pattern
      (`Brain.Memory.RecurringPattern.find/2`), or `nil`

  ## Returns
  `{:ok, response, recurring_pattern}`, where `recurring_pattern` is the
  `%Brain.Memory.RecurringPattern{}` the response was adapted from, or `nil`
  for a response built from anything else; or `:not_synthesized`
  """
  @spec synthesize(String.t() | nil, list(), keyword()) ::
          {:ok, String.t(), RecurringPattern.t() | nil} | :not_synthesized
  def synthesize(intent, entities, opts \\ []) do
    profile = Keyword.get(opts, :profile)
    domain = if profile, do: profile.domain, else: infer_domain(intent)
    confidence = Keyword.get(opts, :confidence, 0.7)
    similar_episodes = Keyword.get(opts, :similar_episodes, [])
    recurring_pattern = Keyword.get(opts, :recurring_pattern)
    context = Keyword.get(opts, :context, %{})

    # Check epistemic context FIRST — contradictions and uncertainty should
    # take priority over memory-based or domain-based responses. This ensures
    # contradictions are surfaced even when the intent has no domain mapping
    # or when memory episodes would otherwise short-circuit the pipeline.
    epistemic_context = Map.get(context, :epistemic_context, %{})

    case handle_epistemic_context(epistemic_context, entities, confidence) do
      {:ok, response} ->
        {:ok, response, nil}

      :continue ->
        case adapt_from_memory(recurring_pattern, similar_episodes, entities) do
          {:ok, response, adapted_from} ->
            {:ok, response, adapted_from}

          :no_adaptation ->
            case synthesize_from_domain(domain, intent, entities, confidence, opts) do
              {:ok, response} -> {:ok, response, nil}
              :not_synthesized -> :not_synthesized
            end
        end
    end
  end

  # A recurring pattern outranks any single episode.
  defp adapt_from_memory(%RecurringPattern{} = pattern, _similar_episodes, entities) do
    {:ok, adapt_response_pattern(pattern.reply, entities), pattern}
  end

  defp adapt_from_memory(nil, similar_episodes, entities) do
    case adapt_from_episodes(similar_episodes, entities) do
      {:ok, response} -> {:ok, response, nil}
      :no_adaptation -> :no_adaptation
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
              frames = if frame_key, do: get_in(domain_config, ["response_frames", frame_key]), else: []

              if frames == [] do
                :not_synthesized
              else
                slot_type_aliases = get_in(domain_config, ["slot_requirements", "slot_type_aliases"]) || %{}
                frame = Enum.random(frames)
                filled_response = fill_entity_slots(frame, entities, slot_type_aliases)
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

    hedges |> Enum.random() |> String.trim_trailing() |> join_after(response)
  end

  # Try to build a response using enriched_response_frames if enrichment
  # succeeded (`enriched_frame/3`). A failed enrichment, a service that is not
  # set up among them, is answered before synthesis
  # (`DiscoursePlanner.enrichment_error_primitive/2`).
  defp try_enriched_response(domain_config, entities, context, confidence) do
    enriched_data = Map.get(context, :enriched_data, %{})

    enriched_frames = Map.get(domain_config, "enriched_response_frames", %{})

    cond do
      # No enriched data available
      enriched_data == %{} or enriched_data == nil ->
        :not_enriched

      # Try to find a matching enriched frame
      map_size(enriched_frames) > 0 ->
        case enriched_frame(enriched_frames, entities, enriched_data) do
          nil ->
            :not_enriched

          {frame_key, frame_config} ->
            select_and_fill_enriched_frame(frame_key, frame_config, entities, enriched_data, domain_config, confidence)
        end

      true ->
        :not_enriched
    end
  end

  @doc """
  The enriched frame, `{key, config}` of `enriched_frames` (a domain's
  `enriched_response_frames`), that words a reply from `enriched_data` and
  `entities`, or nil when none fires.

  A frame fires only when the enrichment and the entities give it everything
  its templates say:

    * every field its "requires_enrichment" names carries a sayable value: a
      non-blank string, a number, or a non-empty list of those. A boolean is
      never sayable, since it would read as "true" or "false": a frame that
      requires one raises, and is to test it under "when" and word it in its
      templates instead;
    * every field its "when" names holds the value given there: a string,
      number or boolean equal to the field's value, {"min": n, "max": m}
      (either bound alone, both inclusive) around a number, or null for a
      field the enrichment does not carry or carries as nil. A field the
      enrichment does not carry meets no other condition. Null tells apart
      what different operations return: a reminder created carries no
      reminder_count, the reminders a check found do;
    * every other `$placeholder` its templates name is filled by an entity of
      that type with a non-blank value.

  So a frame for "nothing was found" is chosen by the value that says so (a
  count of 0), never by a field merely being present. Of the frames that
  fire, the most specific is chosen (`frame_specificity/1`). Two or more
  firing at the top specificity raise: nothing in the data says which of them
  is meant, and picking by the order the frames happen to be read in would be
  a silent guess.
  """
  @spec enriched_frame(map(), list(), map()) :: {String.t(), map()} | nil
  def enriched_frame(enriched_frames, entities, enriched_data)
      when is_map(enriched_frames) and is_list(entities) and is_map(enriched_data) do
    firing =
      enriched_frames
      |> Enum.filter(fn {frame_key, config} -> frame_fires?(frame_key, config, entities, enriched_data) end)
      |> Enum.group_by(fn {_key, config} -> frame_specificity(config) end)

    case Enum.max_by(firing, fn {specificity, _frames} -> specificity end, fn -> nil end) do
      nil ->
        nil

      {_specificity, [frame]} ->
        frame

      {specificity, tied} ->
        raise ArgumentError,
              "the enriched frames #{inspect(tied |> Enum.map(&elem(&1, 0)) |> Enum.sort())} all fire, each " <>
                "with specificity #{specificity}, for the enrichment #{inspect(enriched_data)}; one of them " <>
                "must require or test more, or exclude the others with a \"when\" condition"
    end
  end

  @doc """
  How specific an enriched frame is: the fields its `requires_enrichment`
  names and the conditions its `when` sets, together. Of the frames that fire
  for an enrichment, the most specific is chosen, and two at the top raise.
  """
  @spec frame_specificity(map()) :: non_neg_integer()
  def frame_specificity(config) when is_map(config) do
    length(required_enrichment(config)) + map_size(frame_conditions(config))
  end

  defp frame_fires?(frame_key, config, entities, enriched_data) do
    required = required_enrichment(config)

    Enum.all?(required, &required_field_sayable?(frame_key, &1, enriched_data)) and
      Enum.all?(frame_conditions(config), fn {field, expected} ->
        condition_holds?(frame_key, field, fetch_enriched(enriched_data, field), expected)
      end) and
      config
      |> frame_templates(frame_key)
      |> Enum.flat_map(&template_placeholders/1)
      |> Enum.reject(&(&1 in required))
      |> Enum.all?(&(entity_value(entities, &1) != nil))
  end

  defp required_enrichment(config) do
    case Map.get(config, "requires_enrichment", []) do
      fields when is_list(fields) -> Enum.map(fields, &to_string/1)
      other -> raise ArgumentError, "an enriched frame's requires_enrichment must be a list of fields, got: #{inspect(other)}"
    end
  end

  defp frame_conditions(config) do
    case Map.get(config, "when", %{}) do
      conditions when is_map(conditions) -> conditions
      other -> raise ArgumentError, "an enriched frame's \"when\" must map fields to values, got: #{inspect(other)}"
    end
  end

  defp frame_templates(config, frame_key) do
    case Map.get(config, "templates") do
      [_ | _] = templates -> templates
      other -> raise ArgumentError, "the enriched frame #{inspect(frame_key)} has no templates, got: #{inspect(other)}"
    end
  end

  defp required_field_sayable?(frame_key, field, enriched_data) do
    case fetch_enriched(enriched_data, field) do
      {:ok, value} when is_boolean(value) ->
        raise ArgumentError,
              "the enriched frame #{inspect(frame_key)} requires #{field}, which holds the boolean #{value}; " <>
                "a boolean is never said as text: test it under \"when\" and word it in the templates"

      {:ok, value} ->
        sayable?(value)

      :error ->
        false
    end
  end

  # A value worth saying: a non-blank string, a number, an atom other than
  # nil and the booleans, or a non-empty list of those. `nil`, `""`, `[]`, a
  # map, and a list holding any of them are not.
  defp sayable?(value) when is_list(value), do: value != [] and Enum.all?(value, &sayable_item?/1)
  defp sayable?(value), do: sayable_item?(value)

  defp sayable_item?(value) when is_binary(value), do: String.trim(value) != ""
  defp sayable_item?(value) when is_number(value), do: true
  defp sayable_item?(value) when is_boolean(value) or is_nil(value), do: false
  defp sayable_item?(value) when is_atom(value), do: true
  defp sayable_item?(_value), do: false

  defp condition_holds?(_frame_key, _field, :error, nil), do: true
  defp condition_holds?(_frame_key, _field, :error, _expected), do: false
  defp condition_holds?(_frame_key, _field, {:ok, value}, nil), do: value == nil

  defp condition_holds?(frame_key, field, {:ok, value}, %{} = bounds) do
    case {Map.keys(bounds) -- ["min", "max"], value} do
      {[], value} when bounds != %{} and is_number(value) ->
        Enum.all?(bounds, fn
          {"min", min} when is_number(min) -> value >= min
          {"max", max} when is_number(max) -> value <= max
          {bound, other} -> raise ArgumentError, "the enriched frame #{inspect(frame_key)} bounds #{field} with #{bound} #{inspect(other)}, which is not a number"
        end)

      _other ->
        raise ArgumentError,
              "the enriched frame #{inspect(frame_key)} bounds #{field} with #{inspect(bounds)}; bounds are " <>
                "\"min\" and/or \"max\" around a number, and #{field} holds #{inspect(value)}"
    end
  end

  defp condition_holds?(_frame_key, _field, {:ok, value}, expected)
       when is_binary(expected) or is_number(expected) or is_boolean(expected),
       do: value == expected

  defp condition_holds?(frame_key, field, _value, expected) do
    raise ArgumentError,
          "the enriched frame #{inspect(frame_key)} tests #{field} against #{inspect(expected)}; " <>
            "a condition is a string, number, boolean or null, or {\"min\": n, \"max\": m}"
  end

  # The enrichment's value for `field`, kept under its name as a string or an atom.
  defp fetch_enriched(enriched_data, field) do
    case Map.fetch(enriched_data, field) do
      {:ok, value} ->
        {:ok, value}

      :error ->
        Enum.find_value(enriched_data, :error, fn {key, value} ->
          if is_atom(key) and Atom.to_string(key) == field, do: {:ok, value}
        end)
    end
  end

  @template_placeholder ~r/\$([a-z_][a-z0-9_]*)/

  defp template_placeholders(template) do
    @template_placeholder |> Regex.scan(template) |> Enum.map(fn [_whole, name] -> name end)
  end

  # The non-blank value of the first entity of `type`, or nil.
  defp entity_value(entities, type) do
    Enum.find_value(entities, fn entity ->
      entity_type = to_string(entity[:entity_type] || entity["entity_type"] || "")
      value = entity[:value] || entity["value"]

      if entity_type == type and is_binary(value) and String.trim(value) != "", do: value
    end)
  end

  # Each `$placeholder` is filled whole, so `$alarm_time` is never read as
  # `$alarm` followed by "_time": a required field from the enrichment, any
  # other name from the entity of that type (`enriched_frame/3`
  # chose the frame because both are there). A template naming a field the
  # enrichment carries but the frame does not require raises, since that field
  # was never checked for a value worth saying.
  defp select_and_fill_enriched_frame(frame_key, frame_config, entities, enriched_data, domain_config, confidence) do
    required = required_enrichment(frame_config)
    template = frame_config |> frame_templates(frame_key) |> Enum.random()

    filled_response =
      Regex.replace(@template_placeholder, template, fn _whole, name ->
        cond do
          name in required ->
            {:ok, value} = fetch_enriched(enriched_data, name)
            format_value(value)

          fetch_enriched(enriched_data, name) != :error ->
            raise ArgumentError,
                  "the enriched frame #{inspect(frame_key)}'s template #{inspect(template)} names $#{name}, " <>
                    "which the frame does not require"

          true ->
            entity_value(entities, name)
        end
      end)

    final_response = maybe_add_acknowledgment(filled_response, confidence, domain_config)
    {:ok, final_response}
  end

  @doc "Synthesizes a clarification request when required slots are missing.\n"
  def synthesize_clarification(intent, entities, missing_slots, opts \\ []) do
    profile = Keyword.get(opts, :profile)
    domain = if profile, do: profile.domain, else: infer_domain(intent)
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

  @doc """
  The reply to a service's error, from the content of the primitive planned
  for it (`Brain.Response.DiscoursePlanner.enrichment_error_primitive/2`).

  A `:missing_slots` reply asks for the first missing slot with the intent's
  own clarification prompt (`SlotDetector.get_clarification_prompt/2`). Any
  other reply is one of primitives.json's `service_replies` templates for
  the content's `reply`, with its placeholders filled: `$service` is the
  content's `service_name`; for an `item_kind`, primitives.json's
  `service_items` entry for the kind gives `$item_noun` and `$item_plural`,
  and `$candidates` lists each of the content's `candidates` and
  `$description` describes the content's `criteria`, as the kind's
  `description` segments describe an item, each segment kept only when the
  item gives its field a non-blank value.

  Raises when there are no templates for the reply or no `service_items`
  entry for the kind, when a template names a placeholder the content does
  not fill, or when an item gives none of its kind's description fields a
  value, since it could not be told apart.
  """
  @spec synthesize_service_reply(map()) :: {:ok, String.t()}
  def synthesize_service_reply(%{reply: :missing_slots, missing_slots: [slot | _], intent: intent})
      when is_binary(intent) do
    {:ok, SlotDetector.get_clarification_prompt(slot, intent)}
  end

  def synthesize_service_reply(%{reply: reply} = content) when is_atom(reply) do
    templates =
      case get_in(@primitives, ["service_replies", Atom.to_string(reply)]) do
        [_ | _] = templates ->
          templates

        other ->
          raise ArgumentError,
                "primitives.json has no service_replies templates for #{inspect(reply)}, got: #{inspect(other)}"
      end

    values =
      %{}
      |> maybe_put_value("service", content, :service_name, & &1)
      |> put_item_values(content)

    {:ok, fill_service_template!(Enum.random(templates), values)}
  end

  defp put_item_values(values, %{item_kind: item_kind} = content) when is_atom(item_kind) do
    items = Map.fetch!(@primitives, "service_items")
    kind = service_item_kind!(items, item_kind)

    values
    |> Map.merge(%{"item_noun" => Map.fetch!(kind, "singular"), "item_plural" => Map.fetch!(kind, "plural")})
    |> maybe_put_value("candidates", content, :candidates, fn candidates ->
      candidates
      |> Enum.map(&describe_service_item!(&1, kind, item_kind))
      |> join_candidates(items)
    end)
    |> maybe_put_value("description", content, :criteria, &describe_service_item!(&1, kind, item_kind))
  end

  defp put_item_values(values, _content), do: values

  defp item_field(item, field), do: Enum.find_value(item, fn {key, value} -> if to_string(key) == field, do: value end)

  defp service_item_kind!(items, item_kind) do
    case get_in(items, ["kinds", Atom.to_string(item_kind)]) do
      %{"singular" => _, "plural" => _, "description" => [_ | _]} = kind ->
        kind

      other ->
        raise ArgumentError,
              "primitives.json's service_items has no complete entry for the item kind #{inspect(item_kind)}, got: #{inspect(other)}"
    end
  end

  defp maybe_put_value(values, name, content, key, describe) do
    case Map.fetch(content, key) do
      {:ok, value} -> Map.put(values, name, describe.(value))
      :error -> values
    end
  end

  # An item as its kind's description segments describe it; a segment whose
  # field the item leaves blank is left out.
  defp describe_service_item!(item, kind, item_kind) when is_map(item) do
    kind
    |> Map.fetch!("description")
    |> Enum.flat_map(fn %{"field" => field, "text" => text} ->
      case item_field(item, field) do
        value when is_binary(value) ->
          if String.trim(value) == "", do: [], else: [String.replace(text, "$value", value)]

        nil ->
          []

        other ->
          raise ArgumentError,
                "the #{item_kind} item's #{field} must be a string to be described, got: #{inspect(other)}"
      end
    end)
    |> case do
      [] ->
        raise ArgumentError,
              "the #{item_kind} item #{inspect(item)} gives none of the fields it is described by a value"

      segments ->
        Enum.join(segments, " ")
    end
  end

  defp join_candidates([only], _items), do: only

  defp join_candidates(descriptions, items) do
    {last, rest} = List.pop_at(descriptions, -1)

    Enum.join(rest, Map.fetch!(items, "candidate_separator")) <>
      Map.fetch!(items, "candidate_conjunction") <> last
  end

  # Every placeholder the template names must be filled.
  defp fill_service_template!(template, values) do
    Regex.replace(~r/\$([a-z_]+)/, template, fn placeholder, name ->
      case Map.fetch(values, name) do
        {:ok, value} ->
          value

        :error ->
          raise ArgumentError,
                "the service clarification #{inspect(template)} names #{placeholder}, which nothing fills; " <>
                  "filled: #{inspect(Map.keys(values))}"
      end
    end)
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
        similarity >= RecurringPattern.similarity_threshold() and episode.outcome != nil and
          episode.outcome != ""
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

  # The frame for the slots the entities fill, looked up in this order:
  #
  #   1. a missing required slot gives "missing_<slot>" ("missing_both" for more);
  #   2. the domain's own frame keyed by every filled slot, required and
  #      optional ("device,room");
  #   3. the domain's own frame keyed by the filled required slots ("device");
  #   4. a frame whose key names an entity's graph type;
  #   5. the shared slot-to-frame mapping, for either key;
  #   6. the domain's "general" frame, only when no slot is filled at all.
  #
  # A candidate counts only if the domain has frames under it. Nil when none
  # does: the caller answers :not_synthesized rather than a generic frame that
  # ignores the slots that were filled.
  defp determine_frame_key(domain_config, entities) do
    slot_requirements = Map.get(domain_config, "slot_requirements", %{})
    required_slots = Map.get(slot_requirements, "required", [])
    optional_slots = Map.get(slot_requirements, "optional", [])
    slot_type_aliases = Map.get(slot_requirements, "slot_type_aliases", %{})
    frames = Map.get(domain_config, "response_frames", %{})

    filled? = fn slot ->
      find_entity_value_by_types(entities, Map.get(slot_type_aliases, slot, [slot])) != nil
    end

    filled_required = Enum.filter(required_slots, filled?)
    filled_optional = Enum.filter(optional_slots, filled?)
    missing_required = required_slots -- filled_required

    if missing_required != [] do
      case missing_required do
        [single] -> "missing_#{single}"
        _ -> "missing_both"
      end
    else
      all_key = slot_key(filled_required ++ filled_optional)
      required_key = slot_key(filled_required)

      no_slot_filled =
        if filled_required == [] and filled_optional == [], do: [@frame_key_default], else: []

      [
        all_key,
        required_key,
        infer_frame_from_graph_types(entities, domain_config),
        Map.get(@frame_key_mappings, all_key),
        Map.get(@frame_key_mappings, required_key)
      ]
      |> Kernel.++(no_slot_filled)
      |> Enum.find(fn key -> is_binary(key) and Map.get(frames, key, []) != [] end)
    end
  end

  defp slot_key(slots), do: slots |> Enum.uniq() |> Enum.sort() |> Enum.join(",")

  defp infer_frame_from_graph_types(entities, domain_config) do
    response_frames = Map.get(domain_config, "response_frames", %{})
    frame_keys = Map.keys(response_frames)

    # An empty graph type (a node with no type or label) names nothing; kept,
    # it is a substring of every key and picked whichever frame came first.
    graph_types =
      entities
      |> Enum.map(&Map.get(&1, :graph_type))
      |> Enum.reject(&(&1 in [nil, ""]))
      |> Enum.map(&String.downcase/1)

    Enum.find(frame_keys, fn key ->
      Enum.any?(graph_types, fn gt -> String.contains?(key, gt) end)
    end)
  end

  # Every entity first fills the placeholder named by its own type, and only then
  # do aliases fill what is left. Filling both in one pass let an entity reach a
  # slot it is merely an alias for before the entity that is that slot's own type
  # got there: smarthome aliases "room" to $device, so "$device in the $room"
  # could come out as "the kitchen in the kitchen".
  defp fill_entity_slots(frame, entities, slot_type_aliases) do
    reverse_alias_map = build_reverse_alias_map(slot_type_aliases)

    typed =
      for entity <- entities,
          type = to_string(entity[:entity_type] || entity["entity_type"] || ""),
          value = entity[:value] || entity["value"] || "",
          type != "" and value != "",
          do: {type, value}

    exact = Enum.reduce(typed, frame, fn {type, value}, acc -> String.replace(acc, "$#{type}", value) end)

    Enum.reduce(typed, exact, fn {type, value}, acc ->
      reverse_alias_map
      |> Map.get(type, [])
      |> Enum.reduce(acc, fn slot_name, inner_acc -> String.replace(inner_acc, "$#{slot_name}", value) end)
    end)
  end

  defp build_reverse_alias_map(slot_type_aliases) do
    Enum.reduce(slot_type_aliases, %{}, fn {slot_name, type_list}, acc ->
      Enum.reduce(type_list, acc, fn type, inner_acc ->
        type_str = to_string(type)
        existing = Map.get(inner_acc, type_str, [])

        if slot_name in existing do
          inner_acc
        else
          Map.put(inner_acc, type_str, [slot_name | existing])
        end
      end)
    end)
  end

  defp maybe_add_acknowledgment(response, confidence, domain_config) do
    if confidence >= 0.7 do
      ack_prefixes = Map.get(domain_config, "acknowledgment_prefixes", [])

      if ack_prefixes != [] and :rand.uniform() > 0.5 do
        join_after(Enum.random(ack_prefixes), response)
      else
        response
      end
    else
      response
    end
  end

  # The UD tag the POS tagger gives a proper noun (`Brain.ML.POSTagger.valid_tags/0`).
  @proper_noun_tag "PROPN"

  @doc """
  `sentence` said after `prefix`, a phrase put before it ("Sure,", "Done.",
  "Also,"), separated by a space.

  A prefix that ends a sentence ("Done.") leaves `sentence` as written. One
  that continues it ("Sure,") lowercases the sentence's first letter, which
  was capitalized only for starting a sentence, unless its first word is
  written capitalized wherever it stands: a proper noun, as the POS tagger
  tags it in `sentence`, or a word the closed-class lexicon declares
  capitalized (`Brain.Lexicon.ClosedClass.capitalized?/1`), the pronoun "I".

  Raises for an empty sentence, and when the POS tagger cannot be loaded.
  """
  @spec join_after(String.t(), String.t()) :: String.t()
  def join_after(prefix, sentence) when is_binary(prefix) and is_binary(sentence) do
    "#{prefix} #{continue_after(prefix, sentence)}"
  end

  defp continue_after(prefix, sentence) do
    cond do
      Brain.ML.Tokenizer.ends_with_terminal_punctuation?(prefix) -> sentence
      keeps_capital?(sentence) -> sentence
      true -> String.downcase(String.first(sentence)) <> String.slice(sentence, 1..-1//1)
    end
  end

  defp keeps_capital?(sentence) do
    case Brain.ML.Tokenizer.tokenize_words(sentence) do
      [] ->
        raise ArgumentError, "Synthesizer: there is no sentence to put after a prefix, got: #{inspect(sentence)}"

      [first | _] = words ->
        Brain.Lexicon.ClosedClass.capitalized?(first) or proper_noun_first?(words)
    end
  end

  defp proper_noun_first?([first | _] = words) do
    case Brain.ML.POSTagger.load_model() do
      {:ok, model} ->
        match?([{_word, @proper_noun_tag} | _], Brain.ML.POSTagger.predict(words, model))

      {:error, reason} ->
        raise "Synthesizer: the POS tagger could not be loaded to tell whether #{inspect(first)} is a proper " <>
                "noun, which decides whether it keeps its capital after a prefix: #{inspect(reason)}"
    end
  end

  # Clarification prompts live in priv/analysis/intent_registry.json and are
  # owned by SlotDetector. This module used to load a parallel
  # priv/analysis/slot_schemas.json -- a file nothing ever generated -- which
  # silently compiled to %{}, so every missing slot got the generic prompt.
  defp get_slot_clarification(intent, missing_slots, _domain_config) do
    case {missing_slots, intent} do
      {[slot], intent} when is_binary(intent) ->
        SlotDetector.get_clarification_prompt(slot, intent)

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

  defp find_entity_value_by_types(entities, acceptable_types) do
    type_strings = Enum.map(acceptable_types, &to_string/1)

    Enum.find_value(entities, fn entity ->
      type = to_string(entity[:entity_type] || entity["entity_type"] || "")

      if type in type_strings do
        entity[:value] || entity["value"]
      else
        nil
      end
    end)
  end

  defp infer_domain(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [domain | _] when domain != "" ->
        String.to_existing_atom(domain)
      _ -> nil
    end
  rescue
    ArgumentError -> String.to_atom(intent |> String.split(".", parts: 2) |> List.first())
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
    original_set = MapSet.new(phrases)

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
  end

  defp paraphrase_phrase(phrase) do
    words = String.split(phrase)

    paraphrased =
      Enum.map(words, fn word ->
        lower = String.downcase(word)

        if String.length(lower) >= 4 and Brain.Lexicon.known_word?(lower) do
          case Brain.Lexicon.synonyms(lower) do
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
