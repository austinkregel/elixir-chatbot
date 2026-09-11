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

  alias Brain.Directive.Assessment
  alias Brain.Response.{Primitive, PrimitiveTypes}
  alias Brain.Analysis.{ChunkAnalysis, ChunkProfile}

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
    action = profile_label_or_intent(analysis)
    capability = check_action_capability(action, opts)

    p
    |> Primitive.merge_content(%{
      action: action,
      capability: capability,
      result: capability_to_result(capability),
      details: %{}
    })
  end

  # Answering an order. Everything here is grounded in what the pipeline and
  # the directive assessment already established — beliefs, epistemic status,
  # and any advisory finding (a capability the agent lacks, a parameter it was
  # not given) that the report should be able to state rather than paper over.
  defp specify_primitive(%Primitive{type: :content, variant: :report} = p, analysis, opts) do
    assessment = Keyword.get(opts, :directive_assessment)

    Primitive.merge_content(p, %{
      topic: report_topic(analysis),
      intent: analysis.intent,
      beliefs: analysis.related_beliefs || [],
      epistemic_status: analysis.epistemic_status,
      capability: assessment && assessment.capability,
      advisories: advisory_texts(assessment)
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

    enrichment =
      if is_map(unified_context), do: Map.get(unified_context, :enrichment, %{}), else: %{}

    enriched_data = if is_map(enrichment), do: Map.get(enrichment, :enriched_data, %{}), else: %{}

    available_fields =
      enriched_data
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
    capability = check_action_capability(profile_label_or_intent(analysis), [])

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

  defp specify_primitive(
         %Primitive{type: :follow_up, variant: :clarification} = p,
         analysis,
         _opts
       ) do
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

  defp specify_primitive(
         %Primitive{type: :follow_up, variant: :context_probe} = p,
         analysis,
         _opts
       ) do
    Primitive.merge_content(p, %{
      possible_interpretations: [],
      conversation_context: analysis.text
    })
  end

  defp specify_primitive(
         %Primitive{type: :follow_up, variant: :correction_invite} = p,
         analysis,
         _opts
       ) do
    Primitive.merge_content(p, %{
      uncertain_claim: nil,
      confidence: analysis.confidence || 0.5
    })
  end

  defp specify_primitive(
         %Primitive{type: :follow_up, variant: :continuation} = p,
         _analysis,
         _opts
       ) do
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
    sub_type =
      p.content[:speech_act_sub_type] ||
        get_in_safe(analysis, [:speech_act, :sub_type]) || :unknown

    Primitive.merge_content(p, %{speech_act_sub_type: sub_type})
  end

  defp specify_primitive(%Primitive{type: :acknowledgment, variant: :action} = p, analysis, _opts) do
    action = profile_label_or_intent(analysis)
    capability = check_action_capability(action, [])

    Primitive.merge_content(p, %{
      action: action,
      capability: capability,
      status: capability_to_result(capability)
    })
  end

  defp specify_primitive(
         %Primitive{type: :acknowledgment, variant: :learning} = p,
         analysis,
         _opts
       ) do
    entities = analysis.entities || []

    learned =
      Enum.map(entities, fn e ->
        %{
          type: Map.get(e, :type) || Map.get(e, :entity_type),
          value: Map.get(e, :value) || Map.get(e, :text)
        }
      end)

    Primitive.merge_content(p, %{
      learned_fact: List.first(learned),
      learned_entities: learned,
      confirmed: true
    })
  end

  defp specify_primitive(
         %Primitive{type: :acknowledgment, variant: :repair} = p,
         _analysis,
         _opts
       ) do
    Primitive.merge_content(p, %{
      what_went_wrong: :misunderstanding,
      adjustment: :retry
    })
  end

  defp specify_primitive(
         %Primitive{type: :acknowledgment, variant: :general} = p,
         analysis,
         _opts
       ) do
    Primitive.merge_content(p, %{user_input_summary: analysis.text})
  end

  defp specify_primitive(p, _analysis, _opts), do: p

  defp profile_label_or_intent(%ChunkAnalysis{profile: %ChunkProfile{derived_label: label}})
       when is_binary(label) and label != "", do: label

  defp profile_label_or_intent(%ChunkAnalysis{intent: intent}), do: intent

  defp validate_content(%Primitive{} = p) do
    if PrimitiveTypes.valid?(p) do
      Primitive.merge_content(p, %{content_complete: true})
    else
      missing =
        PrimitiveTypes.required_content(p.type, p.variant)
        |> Enum.reject(&Map.has_key?(p.content, &1))

      Logger.debug(
        "Primitive #{p.type}/#{p.variant} missing required content: #{inspect(missing)}"
      )

      Primitive.merge_content(p, %{content_complete: false, missing_fields: missing})
    end
  end

  # --- Knowledge retrieval helpers ---

  defp retrieve_facts(analysis, opts) do
    enriched_facts = Keyword.get(opts, :enriched_facts, [])
    unified_context = Keyword.get(opts, :unified_context, %{})

    cond do
      enriched_facts != [] ->
        enriched_facts

      true ->
        case prefetched_facts_for(analysis, unified_context) do
          facts when is_list(facts) and facts != [] ->
            facts

          _ ->
            lookup_chunk = pick_lookup_chunk(analysis, unified_context)
            query = (lookup_chunk && Map.get(lookup_chunk, :text)) || analysis.text || ""

            entities =
              (lookup_chunk && Map.get(lookup_chunk, :entities)) || analysis.entities || []

            do_fact_lookup(query, entities)
        end
    end
  end

  defp prefetched_facts_for(analysis, unified_context) when is_map(unified_context) do
    facts_map = Map.get(unified_context, :per_chunk_facts, %{})

    if is_map(facts_map) do
      lookup_chunk = pick_lookup_chunk(analysis, unified_context)
      lookup_idx = lookup_chunk && Map.get(lookup_chunk, :chunk_index)
      analysis_idx = Map.get(analysis, :chunk_index)

      Map.get(facts_map, lookup_idx) || Map.get(facts_map, analysis_idx) || []
    else
      []
    end
  end

  defp prefetched_facts_for(_analysis, _unified_context), do: []

  defp pick_lookup_chunk(analysis, unified_context) when is_map(unified_context) do
    case Map.get(unified_context, :question_chunk) do
      nil -> nil
      question_chunk -> if same_chunk?(question_chunk, analysis), do: nil, else: question_chunk
    end
  end

  defp pick_lookup_chunk(_analysis, _unified_context), do: nil

  defp same_chunk?(a, b) do
    a_idx = Map.get(a, :chunk_index)
    b_idx = Map.get(b, :chunk_index)

    cond do
      a_idx != nil and b_idx != nil -> a_idx == b_idx
      true -> Map.get(a, :text) == Map.get(b, :text)
    end
  end

  defp do_fact_lookup(query, entities) do
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

  # NB: `query_beliefs/1` replies `{:ok, beliefs}`, not a bare list, so the previous
  # `beliefs when is_list(beliefs)` clause never matched and this function returned
  # `[]` on every call where the store was actually up -- no belief has ever reached
  # a narrative primitive through here. Unwrapping the tuple is what makes the
  # disclosure filter below load-bearing rather than decorative.
  defp retrieve_beliefs(analysis) do
    try do
      if belief_store_available?() do
        case Brain.Epistemic.BeliefStore.query_beliefs(limit: 20) do
          {:ok, beliefs} when is_list(beliefs) -> beliefs
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

  # Drops beliefs the disclosure policy says must not be said: the @never_disclose
  # predicates (:password, :ssn, :credit_card, :health_condition, ...) and anything
  # below the 0.3 confidence floor. Without this, low-confidence beliefs reach the
  # prompt as though they were established fact, which is exactly how a small model
  # ends up stating them confidently.
  #
  # NB: this used to call `filter_discloseable/1`, which only has a clause for
  # `%SelfKnowledgeAssessment{}` -- passing a list raised FunctionClauseError into a
  # blanket `rescue -> beliefs`, so the filter silently did nothing at all.
  # `evaluate_disclosure/1` is the function that actually takes a belief.
  defp apply_disclosure_filter(beliefs) when is_list(beliefs) do
    Enum.filter(beliefs, &discloseable?/1)
  end

  defp apply_disclosure_filter(beliefs), do: beliefs

  defp discloseable?(%Brain.Epistemic.Types.Belief{} = belief) do
    Brain.Epistemic.DisclosurePolicy.evaluate_disclosure(belief).should_disclose
  rescue
    e ->
      # Withhold on error: a belief we cannot evaluate is one we should not state.
      Logger.warning(
        "ContentSpecifier: disclosure evaluation failed, withholding belief: " <>
          Exception.message(e)
      )

      false
  end

  # Not a Belief struct (e.g. a pre-resolved map from analysis.related_beliefs) --
  # the policy has nothing to say about it, so pass it through unchanged.
  defp discloseable?(_other), do: true

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

  defp extract_topic(%ChunkAnalysis{profile: %ChunkProfile{} = profile, entities: entities}) do
    entity_topics = extract_entity_topics(entities)

    cond do
      entity_topics != [] ->
        Enum.join(entity_topics, ", ")

      profile.domain not in [:unknown, nil] ->
        label = to_string(profile.domain)
        if generic_label?(label), do: nil, else: label

      true ->
        nil
    end
  end

  defp extract_topic(%ChunkAnalysis{intent: intent, entities: entities}) do
    entity_topics = extract_entity_topics(entities)

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

  defp extract_entity_topics(entities) do
    (entities || [])
    |> Enum.take(4)
    |> Enum.map(&(Map.get(&1, :value) || Map.get(&1, :text) || ""))
    |> Enum.reject(&(&1 == ""))
    |> Enum.reject(&stopword?/1)
    |> Enum.reject(&generic_label?/1)
    |> Enum.reject(&(String.length(&1) < 3))
    |> Enum.take(2)
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
    entity_names =
      (analysis.entities || [])
      |> Enum.take(3)
      |> Enum.map(&(Map.get(&1, :value) || Map.get(&1, :text) || ""))
      |> Enum.reject(&(&1 == ""))
      |> Enum.reject(&stopword?/1)

    sentiment = analysis.sentiment || %{}
    sentiment_label = Map.get(sentiment, :label)
    profile = analysis.profile

    domain_label =
      cond do
        match?(%ChunkProfile{domain: d} when d not in [:unknown, nil], profile) ->
          label = to_string(profile.domain)
          if generic_label?(label), do: nil, else: label

        is_binary(analysis.intent) and analysis.intent != "" ->
          label = analysis.intent |> String.split(".") |> List.last() |> String.replace("_", " ")
          if generic_label?(label), do: nil, else: label

        true ->
          nil
      end

    cond do
      entity_names != [] ->
        "you're talking about #{Enum.join(entity_names, " and ")}"

      domain_label != nil ->
        "you're asking about #{domain_label}"

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
    "i" => "you",
    "i'm" => "you're",
    "i've" => "you've",
    "i'd" => "you'd",
    "i'll" => "you'll",
    "im" => "you're",
    "ive" => "you've",
    "my" => "your",
    "me" => "you",
    "myself" => "yourself",
    "mine" => "yours",
    "we" => "you",
    "we're" => "you're",
    "we've" => "you've",
    "we'd" => "you'd",
    "we'll" => "you'll",
    "our" => "your",
    "ours" => "yours",
    "ourselves" => "yourselves",
    "us" => "you",
    "am" => "are"
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

  defp sentiment_to_descriptor(:negative), do: "frustrated"
  defp sentiment_to_descriptor(:positive), do: "good about something"
  defp sentiment_to_descriptor(_), do: "something"

  defp summarize_partial(%ChunkAnalysis{entities: entities, intent: intent}) do
    %{entities: entities, intent: intent}
  end

  defp build_explanation_parts(facts, _analysis) when is_list(facts) and facts != [] do
    Enum.map(facts, fn fact ->
      cond do
        is_map(fact) -> fact_display(fact)
        is_binary(fact) -> fact
        true -> inspect(fact)
      end
    end)
  end

  defp build_explanation_parts(_, _), do: []

  # Facts reach here either as %FactDatabase.Fact{} structs (atom :fact) or as
  # legacy raw maps that used the string "fact" key. Dispatch on shape once
  # instead of probing both keys at the access site.
  defp fact_display(%{fact: f}) when not is_nil(f), do: f
  defp fact_display(%{"fact" => f}) when not is_nil(f), do: f
  defp fact_display(fact), do: inspect(fact)

  # Real capability: resolve the intent to a service and check its
  # credentials/registration via the Dispatcher (the shared source of truth),
  # instead of a hardcoded :unknown.
  defp check_action_capability(intent, opts),
    do: Brain.Services.Dispatcher.action_capability(intent, opts)

  defp capability_to_result(capability) do
    case capability do
      :capable -> :pending
      _ -> :incapable
    end
  end

  defp suggest_alternative(%ChunkAnalysis{profile: %ChunkProfile{domain: domain}})
       when domain not in [:unknown, nil] do
    "ask about #{domain}"
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

  defp infer_creative_type(%ChunkAnalysis{profile: %ChunkProfile{derived_label: label}})
       when is_binary(label) and label != "" do
    cond do
      String.contains?(label, "joke") -> :joke
      String.contains?(label, "story") -> :story
      true -> :hypothetical
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

  # Unlike extract_topic/1 this never falls back to the classified domain.
  # For free-form orders that classification is frequently wrong (see
  # `Brain.Directive.Assessor`), and telling the model to "report your findings
  # on account" for "Investigate the anomaly at sector 7" would launder a bad
  # guess into the prompt. No entity-grounded topic means no topic, and the
  # report instruction falls back to "what you were asked".
  defp report_topic(%ChunkAnalysis{entities: entities}) do
    case extract_entity_topics(entities) do
      [] -> nil
      topics -> Enum.join(topics, ", ")
    end
  end

  defp report_topic(_), do: nil

  defp advisory_texts(nil), do: []

  defp advisory_texts(%Assessment{advisories: advisories}) do
    Enum.map(advisories, fn
      {:incapable_action, intent} -> "no capability registered for #{intent}"
      {:missing_slots, slots} -> "not given: #{Enum.join(slots, ", ")}"
      other -> inspect(other)
    end)
  end

  defp advisory_texts(_), do: []

  defp get_in_safe(struct, keys) when is_struct(struct),
    do: get_in_safe(Map.from_struct(struct), keys)

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
