defmodule Brain.Response.DiscoursePlanner do
  @moduledoc """
  Plans the discourse structure of a response as an ordered sequence of primitives.

  Takes the full analysis context and produces a list of `%Primitive{}` structs
  with `type` and `variant` set, representing the "shape" of the response.

  ## Three-layer planning

  1. **Backbone**: Speech act + intent determine the core primitive sequence
     from data-driven patterns in `data/response/discourse_patterns.json`.
  2. **Epistemic modification**: Confidence, novelty, and contradiction signals
     insert hedging, correction invites, or override the backbone.
  3. **Sentiment/discourse adjustment**: Emotional signals and discourse role
     prepend attunement or suppress response.

  ## A service's error

  When the service the turn's intent was dispatched to returned an error
  (the unified context's enrichment has `enrichment_status: :failed`), the
  turn's plan is the one primitive that answers that error, in place of every
  chunk's plan: what the user asked for was not carried out, and the reply
  says why or asks what is needed. The error's reason is switched on as the
  structured term the service returned (`enrichment_error_primitive/2`); a
  reason with no planned response raises, and so does a failed enrichment
  that carries no reason.
  """

  alias Brain.Response.Primitive
  alias Brain.Analysis.{InternalModel, ChunkAnalysis, ChunkProfile}

  require Logger

  @brain_priv :code.priv_dir(:brain) |> to_string()
  @patterns_path Path.join(@brain_priv, "response/discourse_patterns.json")
  @external_resource @patterns_path

  # discourse_patterns.json drives every response plan. An unreadable or
  # malformed file used to compile to %{}, leaving the planner with nothing but
  # its default single-primitive backbone.
  @patterns @patterns_path |> File.read!() |> Jason.decode!() |> Map.fetch!("patterns")

  @doc """
  Plans a discourse structure for the given analysis model.

  Returns an ordered list of `%Primitive{}` structs with `type` and `variant`
  set, and `content` partially seeded from analysis signals.
  """
  def plan(model, opts \\ [])

  def plan(%InternalModel{} = model, opts) do
    case enrichment_failure(opts) do
      {:failed, reason, enrichment} ->
        [
          enrichment_error_primitive(reason, %{
            service: Map.get(enrichment, :enrichment_service),
            intent: Map.get(enrichment, :intent)
          })
        ]

      :not_failed ->
        model.analyses
        |> Enum.flat_map(&plan_chunk(&1, model, opts))
        |> insert_transitions(model.analyses)
    end
  end

  def plan(_, _opts), do: [Primitive.new(:acknowledgment, :general), Primitive.new(:follow_up, :continuation)]

  @doc """
  The primitive a service's error is answered with. Its content names the
  reply (`reply`, the key of its wording in primitives.json's
  `service_replies`), keeps the error's reason as `enrichment_error`, and
  carries `origin`'s `intent` and `service` (the service's name) with the
  service's display name as `service_name`.

  A reply that asks the user something is a `:follow_up` `:clarification`
  (its `ambiguity_type` is the reply); one that reports what was not done is
  a `:content` `:action_result` (its `result` is the reply); one that says
  what cannot be done here is a `:framing` `:boundary` (its `capability` is
  the reply). A clarification that a named slot answers carries
  `missing_slots`, which the user's reply fills
  (`Brain.Analysis.FollowupDetector`).

  | Reason | Returned by | Primitive | Reply |
  | --- | --- | --- | --- |
  | `{:ambiguous, kind, items}` (two or more) | LocalScheduler `handle_call/3` (`:cancel_matching`) | clarification | `:ambiguous_item`, with `item_kind`, `candidates` |
  | `{:event_not_identified, events}` (one or more) | LocalScheduler `cancel_matching/4` | clarification | `:ambiguous_item`, with `item_kind: :calendar`, `candidates` |
  | `:missing_location` | Weather `enrich/4` | clarification | `:missing_slots`, `missing_slots: ["location"]` |
  | `:location_not_found` | Weather `geocode/2` | clarification | `:location_not_found`, `missing_slots: ["location"]` |
  | `{:no_matching_items, kind, %{}}` | LocalScheduler `handle_call/3` | action result | `:no_items` |
  | `{:no_matching_items, kind, criteria}` | LocalScheduler `handle_call/3` | action result | `:no_matching_items`, with `criteria` |
  | `:no_ringing_alarm` | LocalScheduler `snooze_alarm/3` | action result | `:no_ringing_alarm` |
  | `:side_effects_off` | Dispatcher `dispatch/3` | action result | `:side_effects_off` |
  | `:rate_limited` | Weather `geocode/2`, `fetch_current/2`, `fetch_forecast/2` | action result | `:rate_limited` |
  | `:service_unavailable` | Weather, the same three | action result | `:unavailable` |
  | `{:connection_error, reason}` | HomeAssistant `http_get/2`, `http_post/3` | action result | `:unavailable` |
  | `:service_error` | Dispatcher `dispatch_to_service/4` (the service raised) | action result | `:service_failed` |
  | `{:api_error, status}` | Weather, the same three | action result | `:service_failed` |
  | `{:http_error, status, body}`, status not 401 or 403 | HomeAssistant `http_get/2`, `http_post/3` | action result | `:service_failed` |
  | `:unexpected_response` | HomeAssistant `get_domain_states/3` | action result | `:service_failed` |
  | `%Jason.DecodeError{}` | HomeAssistant `http_get/2` (an unparseable body) | action result | `:service_failed` |
  | `:entity_not_found` | HomeAssistant `handle_query/5` | action result | `:device_not_found` |
  | `:missing_credentials` | Dispatcher `dispatch_to_service/4`; `Enricher.prepare_context/3` | boundary | `:not_configured` |
  | `:missing_api_key` | Weather `get_current_weather/2`, `get_forecast/2` | boundary | `:not_configured` |
  | `:invalid_api_key` | Weather, the same three as `:rate_limited` | boundary | `:invalid_credentials` |
  | `{:http_error, 401 or 403, body}` | HomeAssistant `http_get/2`, `http_post/3` | boundary | `:invalid_credentials` |
  | `:service_disabled` | Dispatcher `dispatch_to_service/4` | boundary | `:disabled` |
  | `{:unsupported_intent, intent}` | HomeAssistant `enrich/4` | boundary | `:unsupported` |
  | `:unsupported_intent` | LocalScheduler `enrich/4` | boundary | `:unsupported` |

  `Brain.Services.SystemStatus.enrich/4` returns no error. Any reason not in
  this table raises `ArgumentError`: it has no planned response.
  """
  @spec enrichment_error_primitive(term(), %{optional(:service) => atom() | nil, optional(:intent) => String.t() | nil}) ::
          Primitive.t()
  def enrichment_error_primitive(reason, origin) when is_map(origin) do
    {type, variant, content} = planned_reply!(reason)
    service = Map.get(origin, :service)

    Primitive.new(
      type,
      variant,
      Map.merge(content, %{
        enrichment_error: reason,
        intent: Map.get(origin, :intent),
        service: service,
        service_name: service_display_name!(service)
      })
    )
  end

  defp planned_reply!({:ambiguous, kind, [_, _ | _] = items}) when is_atom(kind),
    do: question(:ambiguous_item, %{item_kind: kind, candidates: items})

  defp planned_reply!({:event_not_identified, [_ | _] = events}),
    do: question(:ambiguous_item, %{item_kind: :calendar, candidates: events})

  defp planned_reply!(:missing_location), do: question(:missing_slots, %{missing_slots: ["location"]})
  defp planned_reply!(:location_not_found), do: question(:location_not_found, %{missing_slots: ["location"]})

  defp planned_reply!({:no_matching_items, kind, criteria}) when is_atom(kind) and criteria == %{},
    do: report(:no_items, %{item_kind: kind})

  defp planned_reply!({:no_matching_items, kind, criteria}) when is_atom(kind) and is_map(criteria),
    do: report(:no_matching_items, %{item_kind: kind, criteria: criteria})

  defp planned_reply!(:no_ringing_alarm), do: report(:no_ringing_alarm, %{item_kind: :alarm})
  defp planned_reply!(:side_effects_off), do: report(:side_effects_off)
  defp planned_reply!(:rate_limited), do: report(:rate_limited)
  defp planned_reply!(:service_unavailable), do: report(:unavailable)
  defp planned_reply!({:connection_error, _reason}), do: report(:unavailable)
  defp planned_reply!(:service_error), do: report(:service_failed)
  defp planned_reply!({:api_error, status}) when is_integer(status), do: report(:service_failed)

  defp planned_reply!({:http_error, status, _body}) when status in [401, 403],
    do: boundary(:invalid_credentials)

  defp planned_reply!({:http_error, status, _body}) when is_integer(status), do: report(:service_failed)
  defp planned_reply!(:unexpected_response), do: report(:service_failed)
  defp planned_reply!(%Jason.DecodeError{}), do: report(:service_failed)
  defp planned_reply!(:entity_not_found), do: report(:device_not_found)
  defp planned_reply!(:missing_credentials), do: boundary(:not_configured)
  defp planned_reply!(:missing_api_key), do: boundary(:not_configured)
  defp planned_reply!(:invalid_api_key), do: boundary(:invalid_credentials)
  defp planned_reply!(:service_disabled), do: boundary(:disabled)
  defp planned_reply!({:unsupported_intent, intent}) when is_binary(intent), do: boundary(:unsupported)
  defp planned_reply!(:unsupported_intent), do: boundary(:unsupported)

  defp planned_reply!(reason) do
    raise ArgumentError,
          "DiscoursePlanner has no planned response for the service error #{inspect(reason)}; " <>
            "a service error the turn's response cannot answer is not passed over"
  end

  defp question(reply, content), do: {:follow_up, :clarification, Map.merge(content, %{reply: reply, ambiguity_type: reply})}
  defp report(reply, content \\ %{}), do: {:content, :action_result, Map.merge(content, %{reply: reply, result: reply})}
  defp boundary(reply), do: {:framing, :boundary, %{reply: reply, capability: reply}}

  defp service_display_name!(nil), do: nil

  defp service_display_name!(name) when is_atom(name) do
    case Brain.Services.Dispatcher.find_service_by_name(name) do
      nil -> raise ArgumentError, "DiscoursePlanner: no registered service is named #{inspect(name)}"
      service -> service.display_name()
    end
  end

  # The error of the service the turn's intent was dispatched to, when it
  # returned one (`Brain.Response.Enricher.prepare_context/3`).
  defp enrichment_failure(opts) do
    enrichment =
      case Keyword.get(opts, :unified_context, %{}) do
        %{enrichment: enrichment} when is_map(enrichment) -> enrichment
        _ -> %{}
      end

    case Map.get(enrichment, :enrichment_status) do
      :failed ->
        case Map.fetch(enrichment, :enrichment_error) do
          {:ok, reason} when not is_nil(reason) ->
            {:failed, reason, enrichment}

          _ ->
            raise ArgumentError,
                  "DiscoursePlanner: the enrichment failed but carries no :enrichment_error, " <>
                    "so there is nothing to answer the failure with; got #{inspect(enrichment)}"
        end

      _ ->
        :not_failed
    end
  end

  @doc """
  Plans primitives for a single chunk analysis.
  """
  def plan_chunk(analysis, model, opts \\ [])

  def plan_chunk(%ChunkAnalysis{} = analysis, %InternalModel{} = _model, opts) do
    signals = extract_signals(analysis, opts)

    signals
    |> select_backbone()
    |> apply_epistemic_modifications(signals)
    |> apply_sentiment_adjustment(signals)
    |> seed_content(analysis)
  end

  def plan_chunk(_, _, _opts), do: [Primitive.new(:acknowledgment, :general)]

  defp extract_signals(%ChunkAnalysis{} = a, opts) do
    speech_act = a.speech_act || %{}
    sentiment = a.sentiment || %{}
    slots = a.slots
    profile = a.profile
    unified_context = Keyword.get(opts, :unified_context, %{})

    enrichment = if is_map(unified_context), do: Map.get(unified_context, :enrichment, %{}), else: %{}
    enrichment_status = if is_map(enrichment), do: Map.get(enrichment, :enrichment_status), else: nil
    enriched_data = if is_map(enrichment), do: Map.get(enrichment, :enriched_data, %{}), else: %{}

    intent_domain = if match?(%ChunkProfile{domain: d} when d not in [:unknown, nil], profile),
      do: safe_to_string(profile.domain),
      else: extract_domain(a.intent)

    %{
      speech_act_category: safe_to_string(Map.get(speech_act, :category)),
      sub_type: safe_to_string(Map.get(speech_act, :sub_type)),
      is_question: Map.get(speech_act, :is_question, false),
      is_imperative: Map.get(speech_act, :is_imperative, false),
      intent: a.intent,
      intent_domain: intent_domain,
      confidence: a.confidence || 0.0,
      sentiment: safe_to_string(Map.get(sentiment, :label, :neutral)),
      sentiment_confidence: Map.get(sentiment, :confidence, 0.0),
      response_strategy: safe_to_string(a.response_strategy),
      epistemic_status: safe_to_string(a.epistemic_status),
      entities: a.entities || [],
      has_missing_slots: has_missing_slots?(slots),
      missing_slots: get_missing_slots(slots),
      low_confidence: (a.confidence || 0.0) < 0.4,
      related_beliefs: a.related_beliefs || [],
      accumulated_context: a.accumulated_context,
      has_enrichment: enrichment_status == :success and enriched_data != %{},
      has_facts: (a.related_beliefs || []) != []
    }
  end

  defp select_backbone(signals) do
    pattern = find_matching_pattern(signals)
    backbone_to_primitives(pattern)
  end

  defp find_matching_pattern(signals) do
    @patterns
    |> Enum.sort_by(fn {_name, %{"match" => match}} -> -map_size(match) end)
    |> Enum.find(fn {_name, %{"match" => match}} ->
      matches_signals?(match, signals)
    end)
    |> case do
      {_name, pattern} -> pattern
      nil -> Map.get(@patterns, "default_fallback", %{"backbone" => [%{"type" => "acknowledgment", "variant" => "general"}, %{"type" => "follow_up", "variant" => "continuation"}]})
    end
  end

  defp matches_signals?(match, _signals) when map_size(match) == 0, do: true

  defp matches_signals?(match, signals) do
    Enum.all?(match, fn {key, expected} ->
      actual = safe_signal_lookup(signals, key)
      matches_value?(actual, expected)
    end)
  end

  defp safe_signal_lookup(signals, key) when is_binary(key) do
    Map.get(signals, String.to_atom(key))
  end

  defp matches_value?(actual, expected) when is_boolean(expected), do: actual == expected
  defp matches_value?(actual, expected) when is_binary(expected), do: safe_to_string(actual) == expected
  defp matches_value?(_, _), do: false

  defp backbone_to_primitives(%{"backbone" => backbone}) when is_list(backbone) do
    Enum.map(backbone, fn spec ->
      type = String.to_atom(spec["type"])
      variant = if spec["variant"], do: String.to_atom(spec["variant"]), else: nil
      Primitive.new(type, variant)
    end)
  end

  defp backbone_to_primitives(_), do: [Primitive.new(:acknowledgment, :general)]

  defp apply_epistemic_modifications(primitives, signals) do
    primitives
    |> maybe_insert_hedging(signals)
    |> maybe_append_correction_invite(signals)
    |> maybe_override_for_contradiction(signals)
  end

  defp maybe_insert_hedging(primitives, %{low_confidence: true, confidence: conf}) do
    hedging = Primitive.new(:hedging, nil, %{confidence_level: conf, confidence_source: :accumulated})

    case Enum.find_index(primitives, &(&1.type in [:content, :framing])) do
      nil -> [hedging | primitives]
      idx -> List.insert_at(primitives, idx, hedging)
    end
  end

  defp maybe_insert_hedging(primitives, %{accumulated_context: %{should_hedge: true} = ctx}) do
    conf = Map.get(ctx, :effective_confidence, 0.5)
    hedging = Primitive.new(:hedging, nil, %{confidence_level: conf, confidence_source: :accumulated})

    case Enum.find_index(primitives, &(&1.type in [:content, :framing])) do
      nil -> [hedging | primitives]
      idx -> List.insert_at(primitives, idx, hedging)
    end
  end

  defp maybe_insert_hedging(primitives, _signals), do: primitives

  defp maybe_append_correction_invite(primitives, %{low_confidence: true}) do
    if Enum.any?(primitives, &(&1.type == :follow_up and &1.variant == :correction_invite)) do
      primitives
    else
      primitives ++ [Primitive.new(:follow_up, :correction_invite)]
    end
  end

  defp maybe_append_correction_invite(primitives, _), do: primitives

  defp maybe_override_for_contradiction(primitives, %{epistemic_status: "contradicted"}) do
    if Enum.any?(primitives, &(&1.type == :contradiction_response)) do
      primitives
    else
      [Primitive.new(:contradiction_response), Primitive.new(:follow_up, :correction_invite)]
    end
  end

  defp maybe_override_for_contradiction(primitives, _), do: primitives

  @high_arousal_threshold 0.85
  @moderate_arousal_threshold 0.6

  defp apply_sentiment_adjustment(primitives, signals) do
    primitives
    |> apply_arousal_scaled_attunement(signals)
  end

  defp apply_arousal_scaled_attunement(_primitives, %{sentiment: "negative", sentiment_confidence: conf})
       when conf > @high_arousal_threshold do
    attunement = Primitive.new(:attunement, :empathy, %{
      arousal_level: :high,
      sentiment_confidence: conf,
      de_escalation: true
    })

    [attunement]
  end

  defp apply_arousal_scaled_attunement(primitives, %{sentiment: "negative", sentiment_confidence: conf})
       when conf > @moderate_arousal_threshold do
    if Enum.any?(primitives, &(&1.type == :attunement)) do
      primitives
    else
      attunement = Primitive.new(:attunement, :empathy, %{
        arousal_level: :moderate,
        sentiment_confidence: conf,
        de_escalation: false
      })
      [attunement | primitives]
    end
  end

  defp apply_arousal_scaled_attunement(primitives, _), do: primitives

  defp seed_content(primitives, %ChunkAnalysis{} = analysis) do
    Enum.map(primitives, &seed_primitive_content(&1, analysis))
  end

  defp seed_primitive_content(%Primitive{type: :acknowledgment, variant: :social} = p, analysis) do
    sub_type = get_in_safe(analysis, [:speech_act, :sub_type]) || :unknown
    Primitive.merge_content(p, %{speech_act_sub_type: sub_type})
  end

  defp seed_primitive_content(%Primitive{type: :acknowledgment, variant: :action} = p, analysis) do
    Primitive.merge_content(p, %{action: analysis.intent, capability: :unknown, status: :pending})
  end

  defp seed_primitive_content(%Primitive{type: :acknowledgment, variant: :learning} = p, analysis) do
    Primitive.merge_content(p, %{entities: analysis.entities})
  end

  defp seed_primitive_content(%Primitive{type: :framing} = p, analysis) do
    Primitive.merge_content(p, %{
      intent: analysis.intent,
      entities: analysis.entities,
      speech_act: analysis.speech_act
    })
  end

  defp seed_primitive_content(%Primitive{type: :hedging} = p, analysis) do
    conf = p.content[:confidence_level] || analysis.confidence || 0.5
    Primitive.merge_content(p, %{confidence_level: conf})
  end

  defp seed_primitive_content(%Primitive{type: :content} = p, analysis) do
    Primitive.merge_content(p, %{
      intent: analysis.intent,
      entities: analysis.entities,
      text: analysis.text,
      related_beliefs: analysis.related_beliefs || [],
      confidence: analysis.confidence
    })
  end

  defp seed_primitive_content(%Primitive{type: :attunement} = p, analysis) do
    sentiment = analysis.sentiment || %{}
    Primitive.merge_content(p, %{
      sentiment_label: Map.get(sentiment, :label, :neutral),
      sentiment_confidence: Map.get(sentiment, :confidence, 0.0),
      context: analysis.text,
      arousal_level: p.content[:arousal_level] || :low
    })
  end

  defp seed_primitive_content(%Primitive{type: :follow_up, variant: :clarification} = p, analysis) do
    slots = analysis.slots
    Primitive.merge_content(p, %{
      missing_slots: get_missing_slots(slots),
      intent: analysis.intent,
      partial_understanding: analysis.entities
    })
  end

  defp seed_primitive_content(%Primitive{type: :follow_up} = p, analysis) do
    Primitive.merge_content(p, %{
      intent: analysis.intent,
      entities: analysis.entities,
      text: analysis.text
    })
  end

  defp seed_primitive_content(%Primitive{type: :contradiction_response} = p, analysis) do
    Primitive.merge_content(p, %{
      related_beliefs: analysis.related_beliefs || [],
      text: analysis.text,
      entities: analysis.entities
    })
  end

  defp seed_primitive_content(%Primitive{type: :transition} = p, _analysis), do: p
  defp seed_primitive_content(p, _analysis), do: p

  defp insert_transitions(primitives, analyses) when length(analyses) <= 1, do: primitives

  defp insert_transitions(primitives, analyses) do
    chunk_count = length(analyses)
    if chunk_count <= 1 do
      primitives
    else
      chunks_with_primitives = Enum.chunk_by(primitives, & &1.content[:chunk_index])
      Enum.intersperse(chunks_with_primitives, [Primitive.new(:transition)])
      |> List.flatten()
    end
  end

  defp extract_domain(nil), do: nil
  defp extract_domain(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [domain | _] -> domain
      _ -> intent
    end
  end
  defp extract_domain(intent), do: safe_to_string(intent)

  defp has_missing_slots?(nil), do: false
  defp has_missing_slots?(%{missing_required: missing}) when is_list(missing), do: missing != []
  defp has_missing_slots?(%{all_required_filled: false}), do: true
  defp has_missing_slots?(_), do: false

  defp get_missing_slots(nil), do: []
  defp get_missing_slots(%{missing_required: missing}) when is_list(missing), do: missing
  defp get_missing_slots(_), do: []

  defp safe_to_string(nil), do: nil
  defp safe_to_string(val) when is_atom(val), do: Atom.to_string(val)
  defp safe_to_string(val) when is_binary(val), do: val
  defp safe_to_string(val), do: inspect(val)

  defp get_in_safe(struct, keys) when is_struct(struct) do
    get_in_safe(Map.from_struct(struct), keys)
  end

  defp get_in_safe(map, []) when is_map(map), do: map

  defp get_in_safe(map, [key | rest]) when is_map(map) do
    case Map.get(map, key) do
      nil -> nil
      val when is_struct(val) -> get_in_safe(Map.from_struct(val), rest)
      val when is_map(val) -> get_in_safe(val, rest)
      val when rest == [] -> val
      _ -> nil
    end
  end

  defp get_in_safe(_, _), do: nil
end
