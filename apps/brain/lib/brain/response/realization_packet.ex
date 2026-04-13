defmodule Brain.Response.RealizationPacket do
  @moduledoc """
  Assembles a structured realization packet for the Ouro model.

  Converts the primitive plan, analysis signals, and unified context into
  a JSON-serializable packet formatted as a ChatML conversation.
  The Ouro model receives this as a structured prompt and generates
  the realized response text.

  The packet directly serializes the actual analysis structs (ChunkAnalysis,
  SpeechActResult, DiscourseResult, SlotResult, Event, ContextAccumulator)
  populated by the pipeline -- no invented abstractions.
  """

  alias Brain.Response.Primitive
  alias Brain.Analysis.ChunkAnalysis

  @system_prompt """
  You are a response realization engine.

  You are given a structured response plan as JSON.

  Rules:
  1. Preserve all payload data exactly.
  2. Do not invent facts, tools, or capabilities.
  3. If uncertainty is specified, maintain it.
  4. If clarification is required, include it clearly.
  5. Combine adjacent units naturally into flowing prose.
  6. Do not mention the plan or internal structure.
  7. Output only the final candidate response text.
  8. Match the specified tone.
  9. Keep the response concise unless verbosity is set to high.
  """

  @doc """
  Builds a realization packet from primitives, analysis, and unified context.

  Returns a list of ChatML messages ready for tokenization.
  """
  def build(primitives, analysis \\ %ChunkAnalysis{}, opts \\ [])

  def build(primitives, analysis, opts) when is_list(opts) do
    unified_context = Keyword.get(opts, :unified_context, %{})

    packet = %{
      mode: "plan_realization",
      tone: extract_tone(analysis, opts),
      verbosity: Keyword.get(opts, :verbosity, "medium"),
      analysis: serialize_analysis(analysis),
      context: serialize_context(unified_context),
      plan: Enum.map(primitives, &serialize_primitive/1)
    }

    [
      %{role: "system", content: @system_prompt},
      %{role: "user", content: Jason.encode!(packet)}
    ]
  end

  @doc """
  Serializes a single primitive into a map suitable for JSON encoding.

  The content map is serialized exactly as ContentSpecifier populated it.
  The primitive's struct-level confidence is included alongside type/variant.
  """
  def serialize_primitive(%Primitive{} = p) do
    content =
      p.content
      |> Enum.map(fn {k, v} -> {to_string(k), serialize_value(v)} end)
      |> Map.new()

    base = %{
      "type" => to_string(p.type),
      "variant" => if(p.variant, do: to_string(p.variant)),
      "content" => content
    }

    if p.confidence do
      Map.put(base, "confidence", p.confidence)
    else
      base
    end
  end

  defp serialize_analysis(analysis) do
    %{
      "text" => safe_get(analysis, :text),
      "intent" => safe_get(analysis, :intent),
      "confidence" => safe_get(analysis, :confidence, 0.5),
      "response_strategy" => safe_get(analysis, :response_strategy) |> stringify_atom(),
      "speech_act" => serialize_speech_act(analysis),
      "discourse" => serialize_discourse(analysis),
      "sentiment" => serialize_sentiment(analysis),
      "entities" => serialize_entities(analysis),
      "slots" => serialize_slots(analysis),
      "epistemic_status" => safe_get(analysis, :epistemic_status) |> stringify_atom(),
      "related_beliefs" => serialize_beliefs(analysis),
      "events" => serialize_events(analysis),
      "fact_verification" => serialize_fact_verification(analysis)
    }
  end

  defp serialize_speech_act(analysis) do
    sa = safe_get(analysis, :speech_act)

    case sa do
      nil ->
        nil

      sa when is_struct(sa) ->
        %{
          "category" => Map.get(sa, :category) |> stringify_atom(),
          "sub_type" => Map.get(sa, :sub_type) |> stringify_atom(),
          "confidence" => Map.get(sa, :confidence),
          "is_question" => Map.get(sa, :is_question, false),
          "is_imperative" => Map.get(sa, :is_imperative, false)
        }

      sa when is_map(sa) ->
        serialize_value(sa)

      _ ->
        nil
    end
  end

  defp serialize_discourse(analysis) do
    d = safe_get(analysis, :discourse)

    case d do
      nil ->
        nil

      d when is_struct(d) ->
        %{
          "addressee" => Map.get(d, :addressee) |> stringify_atom(),
          "confidence" => Map.get(d, :confidence),
          "direct_address_detected" => Map.get(d, :direct_address_detected, false)
        }

      d when is_map(d) ->
        serialize_value(d)

      _ ->
        nil
    end
  end

  defp serialize_sentiment(analysis) do
    s = safe_get(analysis, :sentiment)

    case s do
      nil -> nil
      %{label: label, confidence: conf} -> %{"label" => stringify_atom(label), "confidence" => conf}
      s when is_map(s) -> serialize_value(s)
      _ -> nil
    end
  end

  defp serialize_entities(analysis) do
    entities = safe_get(analysis, :entities, [])

    Enum.map(entities, fn entity ->
      entity
      |> Enum.map(fn {k, v} -> {to_string(k), serialize_value(v)} end)
      |> Map.new()
    end)
  end

  defp serialize_slots(analysis) do
    slots = safe_get(analysis, :slots)

    case slots do
      nil ->
        nil

      s when is_struct(s) ->
        %{
          "filled" => serialize_value(Map.get(s, :filled_slots, %{})),
          "missing_required" => (Map.get(s, :missing_required, []) || []) |> Enum.map(&to_string/1),
          "missing_optional" => (Map.get(s, :missing_optional, []) || []) |> Enum.map(&to_string/1)
        }

      s when is_map(s) ->
        serialize_value(s)

      _ ->
        nil
    end
  end

  defp serialize_beliefs(analysis) do
    beliefs = safe_get(analysis, :related_beliefs, [])

    Enum.map(beliefs || [], fn belief ->
      belief
      |> Enum.map(fn {k, v} -> {to_string(k), serialize_value(v)} end)
      |> Map.new()
    end)
  end

  defp serialize_events(analysis) do
    events = safe_get(analysis, :events, [])

    Enum.map(events || [], fn event ->
      case event do
        e when is_struct(e) ->
          e |> Map.from_struct() |> serialize_value()

        e when is_map(e) ->
          serialize_value(e)

        _ ->
          nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp serialize_fact_verification(analysis) do
    fv = safe_get(analysis, :fact_verification)

    case fv do
      nil -> nil
      fv when is_struct(fv) -> fv |> Map.from_struct() |> serialize_value()
      fv when is_map(fv) -> serialize_value(fv)
      _ -> nil
    end
  end

  defp serialize_context(unified_context) when unified_context == %{}, do: nil
  defp serialize_context(nil), do: nil

  defp serialize_context(unified_context) when is_map(unified_context) do
    acc = Map.get(unified_context, :accumulator, %{})
    enrichment = Map.get(unified_context, :enrichment, %{})
    graph = Map.get(unified_context, :graph, %{})
    memory = Map.get(unified_context, :memory, %{})

    ctx = %{}

    ctx =
      if is_map(acc) and acc != %{} do
        ctx
        |> Map.put("accumulated_confidence", Map.get(acc, :combined_confidence))
        |> Map.put("conflict_measure", Map.get(acc, :conflict_measure))
        |> Map.put("should_hedge", Map.get(acc, :should_hedge, false))
        |> Map.put("entity_familiarity", Map.get(acc, :entity_familiarity))
        |> Map.put("conversation_topics", Map.get(acc, :conversation_topics, []))
      else
        ctx
      end

    ctx =
      if is_map(enrichment) do
        enriched_data = Map.get(enrichment, :enriched_data, %{})

        if enriched_data != %{} do
          Map.put(ctx, "enriched_data", serialize_value(enriched_data))
        else
          ctx
        end
      else
        ctx
      end

    ctx =
      if is_map(graph) do
        user_prefs = Map.get(graph, :user_preferences, [])

        if user_prefs != [] do
          Map.put(ctx, "user_preferences", serialize_value(user_prefs))
        else
          ctx
        end
      else
        ctx
      end

    ctx =
      if is_map(memory) do
        episodes = Map.get(memory, :similar_episodes, [])

        if episodes != [] do
          Map.put(ctx, "similar_episodes", serialize_value(Enum.take(episodes, 3)))
        else
          ctx
        end
      else
        ctx
      end

    if ctx == %{}, do: nil, else: ctx
  end

  # Value serialization helpers

  @doc false
  def serialize_value(v) when is_atom(v) and not is_nil(v) and not is_boolean(v), do: to_string(v)
  def serialize_value(v) when is_binary(v), do: v
  def serialize_value(v) when is_number(v), do: v
  def serialize_value(v) when is_boolean(v), do: v
  def serialize_value(nil), do: nil
  def serialize_value(v) when is_list(v), do: Enum.map(v, &serialize_value/1)

  def serialize_value(v) when is_struct(v) do
    v |> Map.from_struct() |> serialize_value()
  end

  def serialize_value(v) when is_map(v) do
    v
    |> Enum.map(fn {k, val} -> {to_string(k), serialize_value(val)} end)
    |> Map.new()
  end

  def serialize_value(v) when is_tuple(v), do: Tuple.to_list(v) |> serialize_value()
  def serialize_value(v), do: inspect(v)

  defp extract_tone(analysis, opts) do
    explicit = Keyword.get(opts, :tone)

    cond do
      explicit -> explicit
      negative_sentiment?(analysis) -> "gentle"
      high_confidence?(analysis) -> "confident"
      true -> "neutral"
    end
  end

  defp negative_sentiment?(%{sentiment: %{label: label}}) when label in [:negative, "negative"],
    do: true

  defp negative_sentiment?(_), do: false

  defp high_confidence?(analysis), do: analysis_confidence(analysis) >= 0.8

  defp analysis_confidence(%{confidence: c}) when is_number(c), do: c
  defp analysis_confidence(_), do: 0.5

  defp stringify_atom(nil), do: nil
  defp stringify_atom(v) when is_atom(v), do: to_string(v)
  defp stringify_atom(v) when is_binary(v), do: v
  defp stringify_atom(v), do: inspect(v)

  defp safe_get(struct, key, default \\ nil) when is_map(struct) do
    Map.get(struct, key, default)
  rescue
    _ -> default
  end
end
