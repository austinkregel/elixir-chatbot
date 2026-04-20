defmodule Brain.Response.ContentSpecifier do
  @moduledoc """
  Fills primitives with concrete, grounded content from knowledge stores.

  For each primitive in the discourse plan, the Content Specifier populates
  the `content` map with real data from the appropriate source:

  - **Knowledge-grounded types** query FactDatabase, BeliefStore, or
    enrichment services for external data.
  - **Analysis-grounded types** extract content from the analysis context
    that was already computed (entities, sentiment, speech act).
  - **Context-dependent types** pull from conversation history or
    slot detection results.
  - **Pass-through types** need minimal content and pass through unchanged.
  """

  alias Brain.Response.{Primitive, PrimitiveTypes}
  alias Brain.Analysis.ChunkAnalysis

  require Logger

  @doc """
  Specifies content for all primitives in a discourse plan.

  Takes the ordered list of primitives (from DiscoursePlanner) and the
  analysis context, and returns primitives with fully populated content.
  Validates each primitive against PrimitiveTypes after specification.
  """
  def specify(primitives, analysis, opts \\ [])

  def specify(primitives, %ChunkAnalysis{} = analysis, opts) when is_list(primitives) do
    primitives
    |> Enum.map(&specify_primitive(&1, analysis, opts))
    |> Enum.map(&validate_content/1)
  end

  def specify(primitives, _analysis, _opts) when is_list(primitives), do: primitives

  defp specify_primitive(%Primitive{type: :content, variant: :factual} = p, analysis, opts) do
    facts = retrieve_facts(analysis, opts)

    p
    |> Primitive.merge_content(%{
      facts: facts,
      entity_context: analysis.entities,
      source: determine_fact_source(facts),
      confidence: analysis.confidence || 0.5
    })
    |> Map.put(:confidence, if(facts != [], do: 0.8, else: 0.3))
  end

  defp specify_primitive(%Primitive{type: :content, variant: :narrative} = p, analysis, _opts) do
    beliefs = retrieve_beliefs(analysis)
    filtered = apply_disclosure_filter(beliefs)

    p
    |> Primitive.merge_content(%{
      beliefs: filtered,
      belief_count: length(filtered),
      has_high_confidence: Enum.any?(filtered, &(Map.get(&1, :confidence, 0) > 0.7)),
      has_uncertain: Enum.any?(filtered, &(Map.get(&1, :confidence, 0) < 0.5))
    })
    |> Map.put(:confidence, if(filtered != [], do: 0.7, else: 0.4))
  end

  defp specify_primitive(%Primitive{type: :content, variant: :explanatory} = p, analysis, opts) do
    facts = retrieve_facts(analysis, opts)
    topic = extract_topic(analysis)

    p
    |> Primitive.merge_content(%{
      topic: topic,
      explanation_parts: build_explanation_parts(facts, analysis),
      knowledge_boundary: if(facts == [], do: :unknown, else: :partial)
    })
  end

  defp specify_primitive(%Primitive{type: :content, variant: :reflective} = p, analysis, _opts) do
    key_elements = extract_key_elements(analysis)
    emotional_tone = extract_emotional_tone(analysis)

    p
    |> Primitive.merge_content(%{
      understood_meaning: summarize_meaning(analysis),
      emotional_tone: emotional_tone,
      key_elements: key_elements
    })
  end

  defp specify_primitive(%Primitive{type: :content, variant: :action_result} = p, analysis, opts) do
    action = analysis.intent
    capability = check_action_capability(action, opts)

    p
    |> Primitive.merge_content(%{
      action: action,
      capability: capability,
      result: capability_to_result(capability),
      details: %{}
    })
  end

  defp specify_primitive(%Primitive{type: :content, variant: :creative} = p, analysis, _opts) do
    Primitive.merge_content(p, %{
      prompt_type: infer_creative_type(analysis),
      engagement_level: :moderate,
      text: analysis.text
    })
  end

  defp specify_primitive(%Primitive{type: :content, variant: :enriched} = p, analysis, opts) do
    unified_context = Keyword.get(opts, :unified_context, %{})
    enrichment = if is_map(unified_context), do: Map.get(unified_context, :enrichment, %{}), else: %{}
    enriched_data = if is_map(enrichment), do: Map.get(enrichment, :enriched_data, %{}), else: %{}

    available_fields = enriched_data
    |> Map.keys()
    |> Enum.reject(&(&1 == :raw))
    |> Enum.map(&to_string/1)

    p
    |> Primitive.merge_content(%{
      enriched_data: enriched_data,
      available_placeholders: available_fields,
      intent: analysis.intent,
      entities: analysis.entities,
      topic: extract_topic(analysis),
      confidence: if(enriched_data != %{}, do: 0.9, else: 0.3)
    })
    |> Map.put(:confidence, if(enriched_data != %{}, do: 0.9, else: 0.3))
  end

  defp specify_primitive(%Primitive{type: :framing, variant: :affirmative} = p, analysis, opts) do
    facts = retrieve_facts(analysis, opts)
    confirmed = List.first(facts)

    Primitive.merge_content(p, %{
      confirmed_fact: confirmed,
      topic: extract_topic(analysis)
    })
  end

  defp specify_primitive(%Primitive{type: :framing, variant: :negative} = p, analysis, opts) do
    facts = retrieve_facts(analysis, opts)
    actual = List.first(facts)

    Primitive.merge_content(p, %{
      actual_fact: actual,
      user_claim: analysis.text,
      topic: extract_topic(analysis)
    })
  end

  defp specify_primitive(%Primitive{type: :framing, variant: :informative} = p, analysis, _opts) do
    Primitive.merge_content(p, %{topic: extract_topic(analysis)})
  end

  defp specify_primitive(%Primitive{type: :framing, variant: :boundary} = p, analysis, _opts) do
    capability = check_action_capability(analysis.intent, [])

    Primitive.merge_content(p, %{
      capability: capability,
      alternative: suggest_alternative(analysis)
    })
  end

  defp specify_primitive(%Primitive{type: :framing, variant: :reframe} = p, analysis, _opts) do
    Primitive.merge_content(p, %{
      original_question_type: :opinion,
      offered_alternative: suggest_alternative(analysis)
    })
  end

  defp specify_primitive(%Primitive{type: :hedging} = p, analysis, _opts) do
    confidence = p.content[:confidence_level] || analysis.confidence || 0.5
    source = p.content[:confidence_source] || :analysis

    Primitive.merge_content(p, %{
      confidence_level: confidence,
      confidence_source: source
    })
  end

  defp specify_primitive(%Primitive{type: :attunement, variant: :empathy} = p, analysis, _opts) do
    sentiment = analysis.sentiment || %{}

    Primitive.merge_content(p, %{
      sentiment_label: Map.get(sentiment, :label, :negative),
      intensity: categorize_intensity(Map.get(sentiment, :confidence, 0.5)),
      context: analysis.text
    })
  end

  defp specify_primitive(%Primitive{type: :attunement, variant: :validation} = p, analysis, _opts) do
    Primitive.merge_content(p, %{
      experience_summary: analysis.text,
      emotional_tone: extract_emotional_tone(analysis)
    })
  end

  defp specify_primitive(%Primitive{type: :attunement, variant: :interest} = p, analysis, _opts) do
    Primitive.merge_content(p, %{
      topic: extract_topic(analysis),
      engagement_type: :curious
    })
  end

  defp specify_primitive(%Primitive{type: :attunement, variant: :concern} = p, _analysis, _opts) do
    Primitive.merge_content(p, %{
      frustration_source: :system_interaction,
      repair_possible: true
    })
  end

  defp specify_primitive(%Primitive{type: :follow_up, variant: :clarification} = p, analysis, _opts) do
    slots = analysis.slots
    missing = get_missing_slots(slots)

    Primitive.merge_content(p, %{
      missing_slots: missing,
      ambiguity_type: if(missing != [], do: :missing_slots, else: :unclear_intent),
      partial_understanding: summarize_partial(analysis),
      intent: analysis.intent
    })
  end

  defp specify_primitive(%Primitive{type: :follow_up, variant: :elaboration} = p, analysis, _opts) do
    Primitive.merge_content(p, %{
      topic: extract_topic(analysis),
      aspect_to_explore: suggest_exploration_aspect(analysis)
    })
  end

  defp specify_primitive(%Primitive{type: :follow_up, variant: :context_probe} = p, analysis, _opts) do
    Primitive.merge_content(p, %{
      possible_interpretations: [],
      conversation_context: analysis.text
    })
  end

  defp specify_primitive(%Primitive{type: :follow_up, variant: :correction_invite} = p, analysis, _opts) do
    Primitive.merge_content(p, %{
      uncertain_claim: nil,
      confidence: analysis.confidence || 0.5
    })
  end

  defp specify_primitive(%Primitive{type: :follow_up, variant: :continuation} = p, _analysis, _opts) do
    Primitive.merge_content(p, %{context: :general})
  end

  defp specify_primitive(%Primitive{type: :contradiction_response} = p, analysis, _opts) do
    beliefs = analysis.related_beliefs || []
    existing = List.first(beliefs)

    Primitive.merge_content(p, %{
      existing_belief: existing,
      new_claim: analysis.text,
      conflict_type: :factual,
      belief_confidence: if(existing, do: Map.get(existing, :confidence, 0.5), else: 0.0)
    })
  end

  defp specify_primitive(%Primitive{type: :acknowledgment, variant: :social} = p, analysis, _opts) do
    sub_type = p.content[:speech_act_sub_type] ||
      get_in_safe(analysis, [:speech_act, :sub_type]) || :unknown

    Primitive.merge_content(p, %{speech_act_sub_type: sub_type})
  end

  defp specify_primitive(%Primitive{type: :acknowledgment, variant: :action} = p, analysis, _opts) do
    action = analysis.intent
    capability = check_action_capability(action, [])

    Primitive.merge_content(p, %{
      action: action,
      capability: capability,
      status: capability_to_result(capability)
    })
  end

  defp specify_primitive(%Primitive{type: :acknowledgment, variant: :learning} = p, analysis, _opts) do
    entities = analysis.entities || []
    learned = Enum.map(entities, fn e ->
      %{type: Map.get(e, :type) || Map.get(e, :entity_type),
        value: Map.get(e, :value) || Map.get(e, :text)}
    end)

    Primitive.merge_content(p, %{
      learned_fact: List.first(learned),
      learned_entities: learned,
      confirmed: true
    })
  end

  defp specify_primitive(%Primitive{type: :acknowledgment, variant: :repair} = p, _analysis, _opts) do
    Primitive.merge_content(p, %{
      what_went_wrong: :misunderstanding,
      adjustment: :retry
    })
  end

  defp specify_primitive(%Primitive{type: :acknowledgment, variant: :general} = p, analysis, _opts) do
    Primitive.merge_content(p, %{user_input_summary: analysis.text})
  end

  defp specify_primitive(p, _analysis, _opts), do: p

  defp validate_content(%Primitive{} = p) do
    if PrimitiveTypes.valid?(p) do
      Primitive.merge_content(p, %{content_complete: true})
    else
      missing = PrimitiveTypes.required_content(p.type, p.variant)
                |> Enum.reject(&Map.has_key?(p.content, &1))

      Logger.debug("Primitive #{p.type}/#{p.variant} missing required content: #{inspect(missing)}")
      Primitive.merge_content(p, %{content_complete: false, missing_fields: missing})
    end
  end

  # --- Knowledge retrieval helpers ---

  defp retrieve_facts(analysis, opts) do
    query = analysis.text || ""
    entities = analysis.entities || []

    enriched_facts = Keyword.get(opts, :enriched_facts, [])
    if enriched_facts != [] do
      enriched_facts
    else
      try do
        if fact_retriever_available?() do
          case Brain.Response.FactRetriever.get_facts_for_query(query, entities) do
            facts when is_list(facts) and facts != [] -> facts
            _ -> []
          end
        else
          []
        end
      rescue
        _ -> []
      catch
        :exit, _ -> []
      end
    end
  end

  defp retrieve_beliefs(analysis) do
    try do
      if belief_store_available?() do
        case Brain.Epistemic.BeliefStore.query_beliefs(limit: 20) do
          beliefs when is_list(beliefs) -> beliefs
          _ -> []
        end
      else
        analysis.related_beliefs || []
      end
    rescue
      _ -> analysis.related_beliefs || []
    catch
      :exit, _ -> analysis.related_beliefs || []
    end
  end

  defp apply_disclosure_filter(beliefs) when is_list(beliefs) do
    try do
      if Code.ensure_loaded?(Brain.Epistemic.DisclosurePolicy) and
           function_exported?(Brain.Epistemic.DisclosurePolicy, :filter_discloseable, 1) do
        Brain.Epistemic.DisclosurePolicy.filter_discloseable(beliefs)
      else
        beliefs
      end
    rescue
      _ -> beliefs
    end
  end

  defp apply_disclosure_filter(beliefs), do: beliefs

  defp fact_retriever_available? do
    try do
      Brain.Response.FactRetriever.available?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp belief_store_available? do
    try do
      Brain.Epistemic.BeliefStore.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp determine_fact_source([]), do: :none
  defp determine_fact_source(facts) when is_list(facts) do
    if Enum.any?(facts, &is_map/1) do
      :fact_database
    else
      :unknown
    end
  end
  defp determine_fact_source(_), do: :unknown

  @generic_labels ~w(unknown query define factual general default greeting
    farewell thanks apology good bad hello hi hey)

  defp extract_topic(%ChunkAnalysis{intent: intent, entities: entities}) do
    entity_topics = (entities || [])
    |> Enum.take(4)
    |> Enum.map(&(Map.get(&1, :value) || Map.get(&1, :text) || ""))
    |> Enum.reject(&(&1 == ""))
    |> Enum.reject(&stopword?/1)
    |> Enum.reject(&generic_label?/1)
    |> Enum.reject(&(String.length(&1) < 3))
    |> Enum.take(2)

    cond do
      entity_topics != [] ->
        Enum.join(entity_topics, ", ")

      is_binary(intent) and intent != "" ->
        label = intent |> String.split(".") |> List.last() |> String.replace("_", " ")
        if generic_label?(label), do: nil, else: label

      true ->
        nil
    end
  end

  defp generic_label?(word) when is_binary(word) do
    String.downcase(word) in @generic_labels
  end

  defp extract_key_elements(%ChunkAnalysis{entities: entities, intent: intent}) do
    entity_values = Enum.map(entities || [], &(Map.get(&1, :value) || Map.get(&1, :text)))
    %{entities: entity_values, intent: intent}
  end

  defp extract_emotional_tone(%ChunkAnalysis{sentiment: nil}), do: :neutral
  defp extract_emotional_tone(%ChunkAnalysis{sentiment: %{label: label}}), do: label
  defp extract_emotional_tone(_), do: :neutral

  defp summarize_meaning(%ChunkAnalysis{} = analysis) do
    entity_names = (analysis.entities || [])
    |> Enum.take(3)
    |> Enum.map(&(Map.get(&1, :value) || Map.get(&1, :text) || ""))
    |> Enum.reject(&(&1 == ""))
    |> Enum.reject(&stopword?/1)

    sentiment = analysis.sentiment || %{}
    sentiment_label = Map.get(sentiment, :label)
    intent = analysis.intent

    cond do
      entity_names != [] ->
        "you're talking about #{Enum.join(entity_names, " and ")}"

      is_binary(intent) and intent != "" ->
        domain = intent |> String.split(".") |> List.last() |> String.replace("_", " ")
        if generic_label?(domain) do
          summarize_from_sentiment_or_shift(analysis)
        else
          "you're asking about #{domain}"
        end

      sentiment_label in [:negative, :positive] ->
        descriptor = sentiment_to_descriptor(sentiment_label)
        "you're feeling #{descriptor}"

      is_binary(analysis.text) and analysis.text != "" ->
        shift_perspective(analysis.text)

      true ->
        "that"
    end
  end

  @perspective_map %{
    "i" => "you", "i'm" => "you're", "i've" => "you've", "i'd" => "you'd",
    "i'll" => "you'll", "im" => "you're", "ive" => "you've",
    "my" => "your", "me" => "you", "myself" => "yourself",
    "mine" => "yours", "we" => "you", "we're" => "you're",
    "we've" => "you've", "we'd" => "you'd", "we'll" => "you'll",
    "our" => "your", "ours" => "yours", "ourselves" => "yourselves",
    "us" => "you", "am" => "are"
  }

  defp shift_perspective(text) when is_binary(text) do
    text
    |> String.split(~r/\b/, include_captures: true)
    |> Enum.map(fn token ->
      lower = String.downcase(token)
      Map.get(@perspective_map, lower, token)
    end)
    |> Enum.join()
  end

  @stopwords ~w(a an the is are was were am be been being do does did
    have has had having will would shall should may might can could
    to of in for on at by with from and or but not no nor so yet
    it its this that these those he she they them his her their
    i me my we us our you your if then than very too also just)

  defp stopword?(word) when is_binary(word) do
    String.downcase(word) in @stopwords
  end

  defp summarize_from_sentiment_or_shift(%ChunkAnalysis{sentiment: %{label: label}})
       when label in [:negative, :positive] do
    "you're feeling #{sentiment_to_descriptor(label)}"
  end

  defp summarize_from_sentiment_or_shift(%ChunkAnalysis{text: text})
       when is_binary(text) and text != "" do
    shift_perspective(text)
  end

  defp summarize_from_sentiment_or_shift(_), do: "that"

  defp sentiment_to_descriptor(:negative), do: "frustrated"
  defp sentiment_to_descriptor(:positive), do: "good about something"
  defp sentiment_to_descriptor(_), do: "something"

  defp summarize_partial(%ChunkAnalysis{entities: entities, intent: intent}) do
    %{entities: entities, intent: intent}
  end

  defp build_explanation_parts(facts, _analysis) when is_list(facts) and facts != [] do
    Enum.map(facts, fn fact ->
      cond do
        is_map(fact) -> Map.get(fact, :fact) || Map.get(fact, "fact") || inspect(fact)
        is_binary(fact) -> fact
        true -> inspect(fact)
      end
    end)
  end

  defp build_explanation_parts(_, _), do: []

  defp check_action_capability(_intent, _opts), do: :unknown

  defp capability_to_result(capability) do
    case capability do
      :capable -> :pending
      _ -> :incapable
    end
  end

  defp suggest_alternative(%ChunkAnalysis{intent: intent}) do
    if is_binary(intent) do
      domain = intent |> String.split(".") |> List.first()
      "ask about #{domain}"
    else
      nil
    end
  end

  defp suggest_exploration_aspect(%ChunkAnalysis{entities: entities}) do
    case entities do
      [first | _] -> Map.get(first, :value) || Map.get(first, :text)
      _ -> nil
    end
  end

  defp infer_creative_type(%ChunkAnalysis{intent: intent}) do
    cond do
      is_binary(intent) and String.contains?(intent, "joke") -> :joke
      is_binary(intent) and String.contains?(intent, "story") -> :story
      true -> :hypothetical
    end
  end

  defp categorize_intensity(confidence) when confidence > 0.8, do: :strong
  defp categorize_intensity(confidence) when confidence > 0.5, do: :moderate
  defp categorize_intensity(_), do: :mild

  defp get_missing_slots(nil), do: []
  defp get_missing_slots(%{missing_required: m}) when is_list(m), do: m
  defp get_missing_slots(_), do: []

  defp get_in_safe(struct, keys) when is_struct(struct), do: get_in_safe(Map.from_struct(struct), keys)
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
