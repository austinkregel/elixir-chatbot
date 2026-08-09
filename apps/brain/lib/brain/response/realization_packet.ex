defmodule Brain.Response.RealizationPacket do
  @moduledoc """
  Assembles a natural-language realization prompt for the Ouro LoopLM model.

  Converts the primitive plan, analysis signals, and unified context into
  syntactically correct English sentences formatted as a ChatML conversation.
  The Ouro model was trained on natural language (SFT data: math, code,
  science, chat) in ChatML format -- never on structured JSON -- so the
  prompt must be readable prose, not a JSON blob.

  Debug JSON dumps are still written to tmp/realization_packets/ for
  diagnostics, but the actual prompt sent to Ouro is natural language.
  """

  alias Brain.Directive.Assessment
  alias Brain.Response.Primitive
  alias Brain.Analysis.{ChunkAnalysis, ChunkProfile}

  require Logger

  # The mechanical output instructions — how to respond, independent of *who* is
  # responding. These stay constant across every soul.
  @base_instructions """
  Generate a natural response based on the context below.
  Be concise and helpful. Do not mention internal systems, plans, or analysis.
  Output only the final response text.
  If $placeholder tokens appear in the instructions, keep them exactly as shown in your output.
  """

  # Default identity, used when no Soul resides in the acting world (preserves
  # the original system prompt exactly).
  @default_identity "You are a conversational assistant."
  @system_prompt @default_identity <> "\n" <> @base_instructions

  @doc """
  Builds a realization packet from primitives, analysis, and unified context.

  Returns a list of ChatML messages ready for tokenization. The user message
  is rendered as natural language rather than JSON.
  """
  def build(primitives, analysis \\ %ChunkAnalysis{}, opts \\ [])

  def build(primitives, analysis, opts) when is_list(opts) do
    unified_context = Keyword.get(opts, :unified_context, %{})
    tone = extract_tone(analysis, opts)
    system_prompt = system_prompt_for(unified_context)

    dump_debug_json(primitives, analysis, unified_context)

    user_message =
      [
        render_analysis(analysis, unified_context),
        render_context(unified_context),
        render_grounds(unified_context, analysis, opts),
        render_plan(primitives),
        render_instructions(tone, opts)
      ]
      |> List.flatten()
      |> Enum.reject(&is_nil/1)
      |> Enum.reject(&(&1 == ""))
      |> Enum.join("\n")

    Logger.info("RealizationPacket prompt: #{byte_size(user_message)} chars, #{length(primitives)} primitives")

    dump_dir = Path.join([File.cwd!(), "tmp", "realization_packets"])
    ts = System.system_time(:millisecond)
    File.write(Path.join(dump_dir, "#{ts}_prompt.txt"), "=== SYSTEM ===\n#{system_prompt}\n=== USER ===\n#{user_message}")

    [
      %{role: "system", content: system_prompt},
      %{role: "user", content: user_message}
    ]
  end

  # A World respects the Souls residing in it: if the acting resident's Soul was
  # resolved into the unified context, its constitution becomes the identity the
  # model generates under, followed by the constant mechanical instructions.
  # No soul -> the generic default (unchanged behavior).
  defp system_prompt_for(unified_context) do
    case Map.get(unified_context, :soul) do
      %Brain.Soul{} = soul ->
        case Brain.Soul.system_prompt(soul) do
          nil -> @system_prompt
          constitution -> constitution <> "\n\n" <> @base_instructions
        end

      _ ->
        @system_prompt
    end
  end

  # --- Analysis rendering ---

  defp render_analysis(analysis, unified_context) do
    case multi_chunk_analyses(unified_context) do
      nil -> render_single_analysis(analysis)
      all_analyses -> render_multi_chunk_analysis(all_analyses, analysis)
    end
  end

  defp multi_chunk_analyses(unified_context) when is_map(unified_context) do
    case Map.get(unified_context, :all_analyses) do
      list when is_list(list) and length(list) > 1 -> list
      _ -> nil
    end
  end

  defp multi_chunk_analyses(_), do: nil

  defp render_multi_chunk_analysis(all_analyses, primary) do
    raw_intro = "The user said #{length(all_analyses)} things:"

    chunk_lines =
      all_analyses
      |> Enum.with_index(1)
      |> Enum.flat_map(fn {chunk, idx} -> render_chunk_line(chunk, idx, primary) end)

    primary_lines = render_primary_pointer(primary, all_analyses)

    [raw_intro | chunk_lines] ++ primary_lines
  end

  defp render_chunk_line(chunk, idx, primary) do
    text = chunk_field(chunk, :text)
    profile = chunk_field(chunk, :profile)
    intent = chunk_field(chunk, :intent)
    speech_act = chunk_field(chunk, :speech_act)
    entities = chunk_field(chunk, :entities) || []

    intent_label = if match?(%ChunkProfile{derived_label: l} when l != "", profile),
      do: profile.derived_label, else: intent

    base = "#{idx}. \"#{text}\""

    annotations =
      [
        format_chunk_intent(intent_label),
        format_chunk_speech_act(speech_act),
        format_chunk_entities(entities),
        if(primary?(chunk, primary), do: "primary chunk for response", else: nil)
      ]
      |> Enum.reject(&is_nil/1)
      |> Enum.reject(&(&1 == ""))

    line =
      case annotations do
        [] -> base
        anns -> "#{base} -- #{Enum.join(anns, "; ")}"
      end

    [line]
  end

  defp render_primary_pointer(primary, all_analyses) do
    primary_idx =
      Enum.find_index(all_analyses, fn chunk -> primary?(chunk, primary) end)

    case primary_idx do
      nil -> []
      idx -> ["Focus the response on chunk #{idx + 1}."]
    end
  end

  defp render_single_analysis(analysis) do
    lines = []

    text = safe_get(analysis, :text)
    lines = if text, do: lines ++ ["The user said: \"#{text}\""], else: lines

    profile = safe_get(analysis, :profile)
    intent = safe_get(analysis, :intent)
    confidence = safe_get(analysis, :confidence)

    lines = lines ++ render_intent_with_profile(profile, intent, confidence)

    lines = lines ++ render_entities(analysis)
    lines = lines ++ render_speech_act(analysis)
    lines = lines ++ render_sentiment(analysis)

    strategy = safe_get(analysis, :response_strategy)
    lines = if strategy, do: lines ++ ["Response strategy: #{strategy}."], else: lines

    lines = lines ++ render_beliefs(analysis)
    lines = lines ++ render_slots(analysis)

    lines
  end

  defp render_intent_with_profile(%ChunkProfile{} = profile, _intent, _confidence) do
    label = profile.derived_label
    conf_str = if is_number(profile.confidence) and profile.confidence > 0.0,
      do: " (confidence #{Float.round(profile.confidence * 1.0, 2)})", else: ""

    axes_parts =
      [
        if(profile.domain not in [:unknown, nil], do: "domain: #{profile.domain}"),
        if(profile.modality not in [:declarative, nil], do: "modality: #{profile.modality}"),
        if(profile.response_posture not in [:direct, nil], do: "posture: #{profile.response_posture}"),
        if(profile.urgency not in [:low, nil], do: "urgency: #{profile.urgency}")
      ]
      |> Enum.reject(&is_nil/1)

    base = "Intent: #{label}#{conf_str}."
    if axes_parts != [], do: [base, Enum.join(axes_parts, "; ") <> "."], else: [base]
  end

  defp render_intent_with_profile(_, intent, confidence) do
    if intent do
      conf_str = if is_number(confidence), do: " (confidence #{Float.round(confidence * 1.0, 2)})", else: ""
      ["Intent: #{intent}#{conf_str}."]
    else
      []
    end
  end

  defp chunk_field(chunk, key) do
    Map.get(chunk, key) || Map.get(chunk, to_string(key))
  end

  defp format_chunk_intent(nil), do: nil
  defp format_chunk_intent(""), do: nil
  defp format_chunk_intent(intent), do: "intent: #{intent}"

  defp format_chunk_speech_act(nil), do: nil

  defp format_chunk_speech_act(speech_act) when is_map(speech_act) do
    category = Map.get(speech_act, :category) || Map.get(speech_act, "category")
    sub_type = Map.get(speech_act, :sub_type) || Map.get(speech_act, "sub_type")

    is_question =
      Map.get(speech_act, :is_question) || Map.get(speech_act, "is_question") || false

    parts = [stringify_atom(category), stringify_atom(sub_type)] |> Enum.reject(&is_nil/1)
    base = if parts == [], do: nil, else: "speech act: #{Enum.join(parts, "/")}"

    cond do
      is_question and base != nil -> "#{base} (question)"
      is_question -> "speech act: question"
      true -> base
    end
  end

  defp format_chunk_speech_act(_), do: nil

  defp format_chunk_entities([]), do: nil

  defp format_chunk_entities(entities) do
    parts =
      entities
      |> Enum.map(fn e ->
        value = Map.get(e, :value) || Map.get(e, "value") || "?"
        type = Map.get(e, :entity_type) || Map.get(e, "entity_type") || ""
        if type == "", do: value, else: "#{value} (#{type})"
      end)
      |> Enum.reject(&(&1 == "" or &1 == "?"))

    case parts do
      [] -> nil
      parts -> "entities: #{Enum.join(parts, ", ")}"
    end
  end

  defp primary?(chunk, primary) do
    chunk_idx = chunk_field(chunk, :chunk_index)
    primary_idx = safe_get(primary, :chunk_index)
    primary_text = safe_get(primary, :text)
    chunk_text = chunk_field(chunk, :text)

    cond do
      chunk_idx != nil and primary_idx != nil -> chunk_idx == primary_idx
      chunk_text != nil and primary_text != nil -> chunk_text == primary_text
      true -> false
    end
  end

  defp render_entities(analysis) do
    entities = safe_get(analysis, :entities, [])

    case entities do
      [] ->
        []

      entities ->
        parts =
          Enum.map(entities, fn e ->
            value = Map.get(e, :value) || Map.get(e, "value") || "?"
            type = Map.get(e, :entity_type) || Map.get(e, :entity) || Map.get(e, "entity_type") || ""
            conf = Map.get(e, :confidence) || Map.get(e, "confidence")

            conf_str = if is_number(conf), do: ", confidence #{Float.round(conf * 1.0, 2)}", else: ""
            "#{value} (#{type}#{conf_str})"
          end)

        ["Entities: #{Enum.join(parts, "; ")}."]
    end
  end

  defp render_speech_act(analysis) do
    sa = safe_get(analysis, :speech_act)

    case sa do
      nil ->
        []

      sa when is_struct(sa) or is_map(sa) ->
        category = Map.get(sa, :category) || Map.get(sa, "category")
        sub_type = Map.get(sa, :sub_type) || Map.get(sa, "sub_type")
        is_question = Map.get(sa, :is_question) || Map.get(sa, "is_question") || false

        parts = [stringify_atom(category), stringify_atom(sub_type)] |> Enum.reject(&is_nil/1)
        question_note = if is_question, do: " (question)", else: ""

        if parts != [] do
          ["Speech act: #{Enum.join(parts, ", ")}#{question_note}."]
        else
          []
        end

      _ ->
        []
    end
  end

  defp render_sentiment(analysis) do
    s = safe_get(analysis, :sentiment)

    case s do
      %{label: label, confidence: conf} when not is_nil(label) ->
        conf_str = if is_number(conf), do: " (confidence #{Float.round(conf * 1.0, 2)})", else: ""
        ["Sentiment: #{label}#{conf_str}."]

      _ ->
        []
    end
  end

  defp render_beliefs(analysis) do
    beliefs = safe_get(analysis, :related_beliefs, []) || []

    case beliefs do
      [] ->
        []

      beliefs ->
        parts =
          Enum.map(beliefs, fn b ->
            subject = Map.get(b, :subject) || Map.get(b, "subject") || "?"
            predicate = Map.get(b, :predicate) || Map.get(b, "predicate") || ""
            object = Map.get(b, :object) || Map.get(b, "object") || ""
            "#{subject} #{predicate} #{object}" |> String.trim()
          end)

        ["Known facts: #{Enum.join(parts, "; ")}."]
    end
  end

  defp render_slots(analysis) do
    slots = safe_get(analysis, :slots)

    case slots do
      nil ->
        []

      s when is_struct(s) or is_map(s) ->
        missing = Map.get(s, :missing_required) || Map.get(s, "missing_required") || []

        if missing != [] do
          ["Missing required information: #{Enum.join(Enum.map(missing, &to_string/1), ", ")}."]
        else
          []
        end

      _ ->
        []
    end
  end

  # --- Context rendering ---

  defp render_context(unified_context) when unified_context in [nil, %{}], do: []

  defp render_context(unified_context) when is_map(unified_context) do
    lines = []

    lines = lines ++ render_enriched_data(unified_context)
    lines = lines ++ render_per_chunk_facts(unified_context)
    lines = lines ++ render_accumulator(unified_context)
    lines = lines ++ render_episodes(unified_context)

    lines
  end

  defp render_per_chunk_facts(ctx) do
    facts_map = Map.get(ctx, :per_chunk_facts)
    all_analyses = Map.get(ctx, :all_analyses)

    cond do
      not is_map(facts_map) or facts_map == %{} ->
        []

      true ->
        index_to_position = build_chunk_position_map(all_analyses)

        facts_map
        |> Enum.sort_by(fn {chunk_index, _} -> chunk_index || 0 end)
        |> Enum.flat_map(fn {chunk_index, facts} -> render_chunk_facts(chunk_index, facts, index_to_position) end)
    end
  end

  defp build_chunk_position_map(nil), do: %{}

  defp build_chunk_position_map(all_analyses) when is_list(all_analyses) do
    all_analyses
    |> Enum.with_index(1)
    |> Enum.reduce(%{}, fn {chunk, position}, acc ->
      idx = chunk_field(chunk, :chunk_index)
      if idx != nil, do: Map.put(acc, idx, position), else: acc
    end)
  end

  defp build_chunk_position_map(_), do: %{}

  defp render_chunk_facts(_chunk_index, [], _positions), do: []

  defp render_chunk_facts(chunk_index, facts, positions) when is_list(facts) do
    fact_texts = facts_to_texts(facts)

    case fact_texts do
      [] ->
        []

      texts ->
        position = Map.get(positions, chunk_index)

        prefix =
          case position do
            nil -> "Known facts about chunk #{chunk_index}"
            pos -> "Known facts for chunk #{pos}"
          end

        ["#{prefix}: #{Enum.join(texts, "; ")}."]
    end
  end

  defp render_chunk_facts(_chunk_index, _facts, _positions), do: []

  defp facts_to_texts(facts) do
    facts
    |> Enum.map(&fact_to_text/1)
    |> Enum.reject(&(&1 == nil or &1 == ""))
    |> Enum.uniq()
  end

  defp fact_to_text(%{fact: fact}) when is_binary(fact), do: fact
  defp fact_to_text(%{"fact" => fact}) when is_binary(fact), do: fact

  defp fact_to_text(%{subject: s, predicate: p, object: o}),
    do: [s, p, o] |> Enum.map(&to_string/1) |> Enum.join(" ") |> String.trim()

  defp fact_to_text(%{"subject" => s, "predicate" => p, "object" => o}),
    do: [s, p, o] |> Enum.map(&to_string/1) |> Enum.join(" ") |> String.trim()

  defp fact_to_text(text) when is_binary(text), do: text

  defp fact_to_text(map) when is_map(map) do
    Map.get(map, :text) || Map.get(map, "text") || nil
  end

  defp fact_to_text(_), do: nil

  defp render_enriched_data(ctx) do
    enrichment = Map.get(ctx, :enrichment, %{})
    enriched_data = if is_map(enrichment), do: Map.get(enrichment, :enriched_data, %{}), else: %{}

    if is_map(enriched_data) and enriched_data != %{} do
      pairs =
        enriched_data
        |> Enum.reject(fn {k, _} -> to_string(k) in ["raw", "__struct__"] end)
        |> Enum.reject(fn {_, v} -> is_nil(v) or v == "" or v == [] or (is_map(v) and map_size(v) == 0) end)
        |> Enum.map(fn {k, v} -> render_data_value(to_string(k), v) end)

      if pairs != [] do
        ["Available data: #{Enum.join(pairs, ", ")}."]
      else
        []
      end
    else
      []
    end
  end

  defp render_data_value(key, value) when is_binary(value), do: "#{key} is #{value}"
  defp render_data_value(key, value) when is_number(value), do: "#{key} is #{value}"
  defp render_data_value(key, value) when is_boolean(value), do: "#{key} is #{value}"
  defp render_data_value(key, value) when is_list(value), do: "#{key}: #{inspect(value)}"
  defp render_data_value(key, value) when is_map(value) do
    inner =
      value
      |> Enum.reject(fn {_, v} -> is_nil(v) or v == "" end)
      |> Enum.map(fn {k, v} -> "#{k}=#{format_short(v)}" end)
      |> Enum.join(", ")

    "#{key}: (#{inner})"
  end
  defp render_data_value(key, value), do: "#{key} is #{format_short(value)}"

  defp render_accumulator(ctx) do
    acc = Map.get(ctx, :accumulator, %{})

    cond do
      not is_map(acc) or acc == %{} -> []
      Map.get(acc, :should_hedge, false) -> ["The system is uncertain, so hedge the response with appropriate qualifiers."]
      true -> []
    end
  end

  defp render_episodes(ctx) do
    memory = Map.get(ctx, :memory, %{})
    episodes = if is_map(memory), do: Map.get(memory, :similar_episodes, []), else: []

    case episodes do
      [] ->
        []

      episodes ->
        summaries =
          episodes
          |> Enum.take(3)
          |> Enum.map(fn ep ->
            {state, similarity} = extract_episode_info(ep)
            sim_str = if is_number(similarity), do: " (similarity #{Float.round(similarity * 1.0, 2)})", else: ""
            "\"#{state}\"#{sim_str}"
          end)
          |> Enum.reject(&(&1 == "\"\""))

        if summaries != [] do
          ["Similar past interactions: #{Enum.join(summaries, "; ")}."]
        else
          []
        end
    end
  end

  defp extract_episode_info({episode, score}) when is_map(episode) do
    state = Map.get(episode, :state) || Map.get(episode, "state") || ""
    {to_string(state), score}
  end

  defp extract_episode_info(episode) when is_map(episode) do
    state = Map.get(episode, :state) || Map.get(episode, "state") || ""
    similarity = Map.get(episode, :similarity) || Map.get(episode, "similarity")
    {to_string(state), similarity}
  end

  defp extract_episode_info(_), do: {"", nil}

  # --- Plan rendering ---

  defp render_plan(primitives) when primitives in [nil, []], do: []

  defp render_plan(primitives) do
    steps =
      primitives
      |> Enum.with_index(1)
      |> Enum.map(fn {p, i} -> "#{i}. #{render_primitive(p)}" end)

    ["", "Response plan:"] ++ steps
  end

  defp render_primitive(%Primitive{type: type, variant: variant, content: content}) do
    label = if variant, do: "#{type}/#{variant}", else: to_string(type)
    desc = describe_primitive(type, variant, content)
    "[#{label}] #{desc}"
  end

  defp describe_primitive(:framing, :informative, _content), do: "Frame the response informatively."
  defp describe_primitive(:framing, :affirmative, _content), do: "Affirm the user's statement."
  defp describe_primitive(:framing, :negative, _content), do: "Gently correct or deny."
  defp describe_primitive(:framing, :reframe, _content), do: "Reframe the question thoughtfully."

  defp describe_primitive(:framing, :boundary, content) do
    alt = Map.get(content, :alternative) || Map.get(content, "alternative")
    if alt, do: "Acknowledge the boundary. Suggest: #{alt}.", else: "Acknowledge the limitation."
  end

  defp describe_primitive(:content, :factual, content) do
    facts = Map.get(content, :facts) || Map.get(content, "facts") || []

    if facts != [] do
      fact_texts = Enum.map(facts, fn
        %{fact: t} -> t
        %{"fact" => t} -> t
        t when is_binary(t) -> t
        _ -> nil
      end) |> Enum.reject(&is_nil/1)

      "Present these facts: #{Enum.join(fact_texts, "; ")}."
    else
      "Present factual information."
    end
  end

  defp describe_primitive(:content, :enriched, content) do
    placeholders = Map.get(content, :available_placeholders) || Map.get(content, "available_placeholders") || []
    topic = Map.get(content, :topic) || Map.get(content, "topic")

    placeholder_str =
      placeholders
      |> Enum.reject(&(&1 in ["raw", "daily_forecasts"]))
      |> Enum.map(&"$#{&1}")
      |> Enum.join(", ")

    cond do
      topic && placeholder_str != "" ->
        "Present the #{topic} data using these placeholders: #{placeholder_str}."
      placeholder_str != "" ->
        "Present the data using these placeholders: #{placeholder_str}."
      true ->
        "Present the available data."
    end
  end

  defp describe_primitive(:content, :reflective, _content), do: "Reflect on what the user shared."
  defp describe_primitive(:content, :explanatory, _content), do: "Explain the concept clearly."
  defp describe_primitive(:content, :narrative, _content), do: "Share relevant context narratively."
  defp describe_primitive(:content, :creative, _content), do: "Respond creatively."
  defp describe_primitive(:content, :action_result, _content), do: "Report the action result."

  defp describe_primitive(:content, :report, content) do
    topic = Map.get(content, :topic) || Map.get(content, "topic")

    subject =
      if is_binary(topic) and topic != "", do: "on #{topic}", else: "on what you were asked"

    "Report your findings #{subject}: the findings first, then how confident you are."
  end

  defp describe_primitive(:acknowledgment, :social, _content), do: "Acknowledge socially."
  defp describe_primitive(:acknowledgment, :general, _content), do: "Acknowledge the input."
  defp describe_primitive(:acknowledgment, :action, _content), do: "Acknowledge the action request."
  defp describe_primitive(:acknowledgment, :learning, _content), do: "Acknowledge that you learned something."
  defp describe_primitive(:acknowledgment, :repair, _content), do: "Acknowledge the need for repair."

  defp describe_primitive(:attunement, :empathy, _content), do: "Show empathy."
  defp describe_primitive(:attunement, :interest, _content), do: "Show interest."
  defp describe_primitive(:attunement, :concern, _content), do: "Express concern."

  defp describe_primitive(:follow_up, :clarification, content) do
    missing = Map.get(content, :missing_slots) || Map.get(content, "missing_slots") || []

    if missing != [] do
      "Ask for clarification about: #{Enum.join(Enum.map(missing, &to_string/1), ", ")}."
    else
      "Ask a clarifying question."
    end
  end

  defp describe_primitive(:follow_up, :elaboration, _content), do: "Invite the user to elaborate."
  defp describe_primitive(:follow_up, :continuation, _content), do: "Invite further conversation."
  defp describe_primitive(:follow_up, :correction_invite, _content), do: "Invite correction if needed."
  defp describe_primitive(:follow_up, :context_probe, _content), do: "Probe for more context."

  defp describe_primitive(type, variant, _content) do
    label = if variant, do: "#{type}/#{variant}", else: to_string(type)
    "Handle #{label} appropriately."
  end

  # --- Grounds (directive mode only) ---
  #
  # Why an order can be answered the way it is about to be answered. Everything
  # here was already computed and then discarded before reaching the prompt:
  # the justification and evidence chains `Brain.Response.ContextBuilder` builds
  # under :graph had no consumer anywhere in the codebase, and neither did the
  # per-chunk epistemic status. For a report to a commanding officer, "what
  # supports this" is exactly the part worth saying.
  #
  # Chat is untouched — without mode: :directive this renders nothing.

  @max_ground_lines 10

  defp render_grounds(unified_context, analysis, opts) do
    if Keyword.get(opts, :mode, :chat) == :directive do
      graph = if is_map(unified_context), do: Map.get(unified_context, :graph, %{}), else: %{}

      grounds =
        (justification_lines(graph) ++
           evidence_lines(graph) ++
           epistemic_lines(analysis) ++
           task_frame_lines(Keyword.get(opts, :directive_assessment)))
        |> Enum.reject(&(&1 in [nil, ""]))
        |> Enum.take(@max_ground_lines)

      if grounds == [], do: [], else: ["", "Grounds for this report:"] ++ grounds
    else
      []
    end
  end

  defp justification_lines(graph) when is_map(graph) do
    graph
    |> Map.get(:justification_chains, [])
    |> List.wrap()
    |> Enum.map(fn chain -> "- believed because: #{fact_to_text(chain)}" end)
  end

  defp justification_lines(_), do: []

  defp evidence_lines(graph) when is_map(graph) do
    graph
    |> Map.get(:evidence_chains, [])
    |> List.wrap()
    |> Enum.map(fn chain -> "- evidence: #{fact_to_text(chain)}" end)
  end

  defp evidence_lines(_), do: []

  defp epistemic_lines(%ChunkAnalysis{epistemic_status: status})
       when status in [:contradicted, "contradicted"] do
    ["- NOTE: the order's premise conflicts with what is currently believed."]
  end

  defp epistemic_lines(%ChunkAnalysis{epistemic_status: status})
       when status in [:verified, "verified"] do
    ["- the order's premise matches what is currently believed."]
  end

  defp epistemic_lines(_), do: []

  # "Who must do what", straight from the semantic-role frames the pipeline
  # already labelled — so the report answers the order that was actually given
  # rather than the topic it happens to mention.
  defp task_frame_lines(%Assessment{task_frames: %{srl_triples: triples}})
       when is_list(triples) and triples != [] do
    triples
    |> Enum.take(2)
    |> Enum.map(fn frame -> "- the order asks: #{fact_to_text(frame)}" end)
  end

  defp task_frame_lines(_), do: []

  # --- Instructions ---

  defp render_instructions(tone, opts) do
    verbosity = Keyword.get(opts, :verbosity, "medium")

    lines = []
    lines = lines ++ ["", "Write a #{tone}, #{verbosity}-length response."]
    lines = lines ++ ["Do not invent facts or capabilities not mentioned above."]
    lines = lines ++ directive_instructions(opts)

    lines
  end

  # An order is answered, not conversed with: the reply goes to the officer who
  # issued it, so greetings, offers of further help and questions back are all
  # wrong register. Anything the deterministic assessment flagged but did not
  # block on is stated plainly rather than glossed over.
  defp directive_instructions(opts) do
    if Keyword.get(opts, :mode, :chat) == :directive do
      assessment = Keyword.get(opts, :directive_assessment)

      base = [
        "This is a reply to an ORDER from a commanding officer. Report findings directly:",
        "no greeting, no offer of further help, no question back."
      ]

      base ++ constraint_instruction(assessment) ++ advisory_instruction(assessment)
    else
      []
    end
  end

  # Bounds the issuer stated on the order itself. They are not inferred, so they
  # are stated to the model as binding rather than as context.
  defp constraint_instruction(%Assessment{constraints: [_ | _] = constraints}) do
    ["The order carries these constraints, which bind your response:"] ++
      Enum.map(constraints, &"- #{&1}")
  end

  defp constraint_instruction(_), do: []

  defp advisory_instruction(%Assessment{advisories: [_ | _] = advisories}) do
    texts =
      advisories
      |> Enum.map(fn
        {:incapable_action, intent} -> "you have no capability registered for #{intent}"
        {:missing_slots, slots} -> "you were not given: #{Enum.join(slots, ", ")}"
        other -> inspect(other)
      end)
      |> Enum.join("; ")

    ["State plainly, without apologising, that #{texts}."]
  end

  defp advisory_instruction(_), do: []

  # --- Debug JSON dumps (not sent to Ouro) ---

  defp dump_debug_json(primitives, analysis, unified_context) do
    analysis_data = serialize_analysis_for_debug(analysis)
    context_data = serialize_context_for_debug(unified_context)
    plan_data = Enum.map(primitives, &serialize_primitive_for_debug/1)

    analysis_json = Jason.encode!(analysis_data, pretty: true)
    context_json = Jason.encode!(context_data, pretty: true)
    plan_json = Jason.encode!(plan_data, pretty: true)

    dump_dir = Path.join([File.cwd!(), "tmp", "realization_packets"])
    File.mkdir_p!(dump_dir)
    ts = System.system_time(:millisecond)

    File.write!(Path.join(dump_dir, "#{ts}_plan.json"), plan_json)
    File.write!(Path.join(dump_dir, "#{ts}_analysis.json"), analysis_json)
    File.write!(Path.join(dump_dir, "#{ts}_context.json"), context_json)

    Logger.info(
      "RealizationPacket debug dump: tmp/realization_packets/#{ts}_*.json | " <>
        "plan=#{byte_size(plan_json)} analysis=#{byte_size(analysis_json)} " <>
        "context=#{byte_size(context_json)}"
    )
  rescue
    e -> Logger.warning("Failed to dump debug JSON: #{inspect(e)}")
  end

  defp serialize_analysis_for_debug(analysis) do
    profile = safe_get(analysis, :profile)

    base = %{
      "text" => safe_get(analysis, :text),
      "intent" => safe_get(analysis, :intent),
      "confidence" => safe_get(analysis, :confidence, 0.5),
      "response_strategy" => safe_get(analysis, :response_strategy) |> stringify_atom(),
      "entities" => safe_get(analysis, :entities, []) |> Enum.map(&serialize_value/1),
      "sentiment" => serialize_value(safe_get(analysis, :sentiment)),
      "speech_act" => serialize_value(safe_get(analysis, :speech_act))
    }

    if match?(%ChunkProfile{}, profile) do
      Map.put(base, "profile", %{
        "derived_label" => profile.derived_label,
        "domain" => stringify_atom(profile.domain),
        "modality" => stringify_atom(profile.modality),
        "response_posture" => stringify_atom(profile.response_posture),
        "engagement_level" => stringify_atom(profile.engagement_level),
        "confidence" => profile.confidence
      })
    else
      base
    end
  end

  defp serialize_context_for_debug(ctx) when ctx in [nil, %{}], do: nil
  defp serialize_context_for_debug(ctx), do: serialize_value(ctx)

  @doc "Serializes a primitive to a map for JSON encoding. Used by debug dumps and DecompressorCollector."
  def serialize_primitive(%Primitive{} = p), do: serialize_primitive_for_debug(p)

  defp serialize_primitive_for_debug(%Primitive{} = p) do
    content =
      p.content
      |> Enum.map(fn {k, v} -> {to_string(k), serialize_value(v)} end)
      |> Map.new()

    %{
      "type" => to_string(p.type),
      "variant" => if(p.variant, do: to_string(p.variant)),
      "content" => content
    }
  end

  # --- Value serialization (for debug dumps) ---

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
    |> Enum.reject(fn {k, v} -> drop_from_debug?(k, v) end)
    |> Map.new()
  end

  def serialize_value(v) when is_tuple(v), do: Tuple.to_list(v) |> serialize_value()
  def serialize_value(v), do: inspect(v)

  @positional_keys ~w(source_tokens start_pos end_pos token_index
    graph_neighbor_count graph_neighbors graph_relationships graph_type
    disambiguation_source embedding embeddings vector vectors tfidf_vector)

  defp drop_from_debug?(_key, nil), do: true
  defp drop_from_debug?(_key, []), do: true
  defp drop_from_debug?(_key, v) when is_map(v) and map_size(v) == 0, do: true
  defp drop_from_debug?(key, _v) when key in @positional_keys, do: true
  defp drop_from_debug?(_key, list) when is_list(list), do: all_numeric?(list)
  defp drop_from_debug?(_key, _v), do: false

  defp all_numeric?([]), do: false
  defp all_numeric?(list), do: Enum.all?(list, &is_number/1)

  # --- Helpers ---

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

  defp format_short(v) when is_binary(v), do: v
  defp format_short(v) when is_number(v), do: to_string(v)
  defp format_short(v) when is_atom(v), do: to_string(v)
  defp format_short(v), do: inspect(v, limit: 5, printable_limit: 50)

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
