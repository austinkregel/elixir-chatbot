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
  """

  alias Brain.Response.Primitive
  alias Brain.Analysis.{InternalModel, ChunkAnalysis}

  require Logger

  @brain_priv :code.priv_dir(:brain) |> to_string()
  @patterns_path Path.join(@brain_priv, "response/discourse_patterns.json")
  @external_resource @patterns_path

  @patterns (case File.read(@patterns_path) do
               {:ok, content} ->
                 case Jason.decode(content) do
                   {:ok, %{"patterns" => patterns}} -> patterns
                   _ -> %{}
                 end

               _ ->
                 %{}
             end)

  @doc """
  Plans a discourse structure for the given analysis model.

  Returns an ordered list of `%Primitive{}` structs with `type` and `variant`
  set, and `content` partially seeded from analysis signals.
  """
  def plan(%InternalModel{} = model) do
    model.analyses
    |> Enum.flat_map(&plan_chunk(&1, model))
    |> insert_transitions(model.analyses)
  end

  def plan(_), do: [Primitive.new(:acknowledgment, :general), Primitive.new(:follow_up, :continuation)]

  @doc """
  Plans primitives for a single chunk analysis.
  """
  def plan_chunk(%ChunkAnalysis{} = analysis, %InternalModel{} = _model) do
    signals = extract_signals(analysis)

    signals
    |> select_backbone()
    |> apply_epistemic_modifications(signals)
    |> apply_sentiment_adjustment(signals)
    |> seed_content(analysis)
  end

  def plan_chunk(_, _), do: [Primitive.new(:acknowledgment, :general)]

  defp extract_signals(%ChunkAnalysis{} = a) do
    speech_act = a.speech_act || %{}
    sentiment = a.sentiment || %{}
    slots = a.slots

    %{
      speech_act_category: safe_to_string(Map.get(speech_act, :category)),
      sub_type: safe_to_string(Map.get(speech_act, :sub_type)),
      is_question: Map.get(speech_act, :is_question, false),
      is_imperative: Map.get(speech_act, :is_imperative, false),
      intent: a.intent,
      intent_domain: extract_domain(a.intent),
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
      accumulated_context: a.accumulated_context
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
