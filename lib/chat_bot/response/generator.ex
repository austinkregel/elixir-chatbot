defmodule ChatBot.Response.Generator do
  @moduledoc """
  Unified response generation entry point.

  This module orchestrates response generation by:
  - Delegating to TemplateStore for template-based responses
  - Delegating to FactRetriever for factual queries
  - Delegating to MemoryAugmented for learning-based responses
  - Handling domain-specific response generation
  - Handling expressive speech act responses (greetings, farewells, etc.)

  Brain should use this module instead of implementing response logic directly.
  """

  require Logger

  alias ChatBot.Response.{TemplateStore, MemoryAugmented, FactRetriever, Composer}
  alias ChatBot.Analysis.IntentRegistry

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Generate a response for the given intent and entities.

  Returns:
  - {:ok, response, :domain} for domain-specific responses
  - {:ok, response, :memory_augmented} for memory-based responses
  - {:ok, response, :template} for template-based responses
  - {:ok, response, :fallback} for fallback responses
  """
  def generate(intent, entities, query_text \\ nil) do
    # Try domain-specific first, then memory, then template, then fallback
    case generate_domain_response(intent, entities, query_text) do
      {:ok, response} ->
        {:ok, response, :domain}

      :not_handled ->
        case try_memory_augmented(intent, entities) do
          {:ok, response} ->
            {:ok, response, :memory_augmented}

          :not_handled ->
            case try_template_response(intent, entities) do
              {:ok, response} ->
                {:ok, response, :template}

              :not_handled ->
                response = generate_fallback(intent, entities)
                {:ok, response, :fallback}
            end
        end
    end
  end

  @doc """
  Generate a response for an expressive speech act.
  Used for greetings, farewells, thanks, apologies, etc.
  """
  def generate_expressive(speech_act) when is_map(speech_act) do
    sub_type = Map.get(speech_act, :sub_type)
    TemplateStore.get_expressive_response(sub_type)
  end

  def generate_expressive(_), do: nil

  @doc """
  Generate a combined response for an analysis model with multiple speech acts.

  This handles multi-chunk inputs where we may have expressives (greetings)
  combined with directives (questions/commands).
  """
  def generate_from_analysis(analysis_model, intent, entities, query_text) do
    speech_acts =
      analysis_model.analyses
      |> Enum.map(& &1.speech_act)

    expressives =
      speech_acts
      |> Enum.filter(&(&1.category == :expressive))
      |> Enum.uniq_by(& &1.sub_type)

    directives = Enum.filter(speech_acts, &(&1.category == :directive))

    has_substantive_content =
      length(directives) > 0 or
        (intent != nil and intent != "" and
           not IntentRegistry.greeting?(intent))

    response_parts = []
    response_types = []

    # Add expressive acknowledgment if present
    {response_parts, response_types} =
      if length(expressives) > 0 do
        expressive = List.first(expressives)
        expressive_response = generate_expressive(expressive)

        if expressive_response do
          {[expressive_response | response_parts], [:expressive | response_types]}
        else
          {response_parts, response_types}
        end
      else
        {response_parts, response_types}
      end

    # Add substantive response for directives/questions/commands
    {response_parts, response_types} =
      if has_substantive_content do
        {:ok, substantive_response, response_type} = generate(intent, entities, query_text)
        {[substantive_response | response_parts], [response_type | response_types]}
      else
        {response_parts, response_types}
      end

    valid_parts =
      response_parts
      |> Enum.reverse()
      |> Enum.filter(&(&1 != nil and &1 != ""))

    # Determine primary response type
    primary_type =
      cond do
        :memory_augmented in response_types -> :memory_augmented
        :domain in response_types -> :domain
        :template in response_types -> :template
        :expressive in response_types -> :expressive
        true -> :fallback
      end

    response =
      case valid_parts do
        [] ->
          {:ok, resp, _} = generate(intent, entities, query_text)
          resp

        [single] ->
          single

        parts ->
          # Combine parts intelligently
          weave_response_parts(parts, Enum.reverse(response_types))
      end

    {response, primary_type}
  end

  # ============================================================================
  # Domain-Specific Response Handlers
  # ============================================================================

  defp generate_domain_response(intent, entities, query_text)

  defp generate_domain_response("weather.query", entities, _query_text) do
    location = find_entity_value(entities, "location")

    response =
      if location do
        # Try template first
        case try_template_with_slots("weather.query", entities) do
          {:ok, resp} ->
            resp

          :not_handled ->
            "Let me check the weather for #{location}. The current conditions are partly cloudy with a temperature around 72°F."
        end
      else
        "What location would you like the weather for?"
      end

    {:ok, response}
  end

  defp generate_domain_response("weather" <> _, entities, query_text) do
    generate_domain_response("weather.query", entities, query_text)
  end

  defp generate_domain_response("music.play", entities, _query_text) do
    artist = find_entity_value(entities, "music-artist")
    song = find_entity_value(entities, "song")

    response =
      cond do
        artist ->
          case try_template_with_slots("music.play", entities) do
            {:ok, resp} -> resp
            :not_handled -> "Playing music by #{artist} for you now."
          end

        song ->
          "Playing #{song} for you now."

        true ->
          "What would you like me to play?"
      end

    {:ok, response}
  end

  defp generate_domain_response("device.control", entities, _query_text) do
    device = find_entity_value(entities, "device")
    action = find_entity_value(entities, "action") || find_entity_value(entities, "locks-status")

    response =
      cond do
        device && action -> "I'll #{action} the #{device} for you."
        device -> "What would you like me to do with the #{device}?"
        true -> "Which device would you like me to control?"
      end

    {:ok, response}
  end

  defp generate_domain_response("news.query", entities, _query_text) do
    topic = find_entity_value(entities, "topic")

    response =
      if topic do
        "Here are the latest headlines about #{topic}."
      else
        "Here are today's top headlines."
      end

    {:ok, response}
  end

  defp generate_domain_response("reminder.create", entities, _query_text) do
    content = find_entity_value(entities, "content")
    date = find_entity_value(entities, "date")

    response =
      cond do
        content && date -> "I'll remind you about #{content} on #{date}."
        content -> "When would you like to be reminded about #{content}?"
        true -> "What would you like me to remind you about?"
      end

    {:ok, response}
  end

  defp generate_domain_response("question.factual", entities, query_text) do
    if FactRetriever.available?() do
      entity_names = extract_entity_names_for_facts(entities)
      query_str = query_text || ""
      facts = FactRetriever.get_facts_for_query(query_str, entity_names)

      if facts != [] do
        response = format_factual_response(query_str, facts)
        {:ok, response}
      else
        :not_handled
      end
    else
      :not_handled
    end
  end

  defp generate_domain_response(_intent, _entities, _query_text) do
    :not_handled
  end

  # ============================================================================
  # Template and Memory-Based Response Helpers
  # ============================================================================

  defp try_memory_augmented(intent, entities) do
    case MemoryAugmented.generate(intent, entities) do
      {:ok, response, _metadata} ->
        {:ok, response}

      _ ->
        :not_handled
    end
  end

  defp try_template_response(intent, entities) do
    if TemplateStore.ready?() do
      case TemplateStore.get_random_template(intent) do
        nil ->
          :not_handled

        template ->
          response = TemplateStore.substitute_slots(template, entities)
          {:ok, response}
      end
    else
      :not_handled
    end
  end

  defp try_template_with_slots(intent, entities) do
    if TemplateStore.ready?() do
      case TemplateStore.get_random_template(intent) do
        nil -> :not_handled
        template -> {:ok, TemplateStore.substitute_slots(template, entities)}
      end
    else
      :not_handled
    end
  end

  defp generate_fallback(intent, entities) do
    if intent && intent != "" do
      "I understood that you're asking about #{IntentRegistry.humanize(intent)}#{format_entities(entities)}."
    else
      "I'm not sure I understand. Could you rephrase that?"
    end
  end

  # ============================================================================
  # Factual Response Formatting
  # ============================================================================

  defp format_factual_response(query_text, facts)
       when is_binary(query_text) and query_text != "" do
    alias ChatBot.ML.Tokenizer
    alias ChatBot.ML.POSTagger

    query_tokens = Tokenizer.tokenize(query_text)
    speech_act = ChatBot.Analysis.SpeechActClassifier.classify(query_text)
    is_question = Map.get(speech_act, :is_question, false)

    query_numbers =
      query_tokens
      |> Enum.filter(fn t -> t.type == :number end)
      |> Enum.map(fn t -> parse_number(t.text) end)
      |> Enum.reject(&is_nil/1)

    content_words = extract_content_words_with_pos(query_tokens)

    if is_question and length(query_numbers) > 0 and length(content_words) > 0 do
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

  defp format_factual_response(_query_text, facts) do
    formatted = FactRetriever.format_facts(facts, 2)
    "Here's what I know: #{Enum.join(formatted, ". ")}."
  end

  defp extract_content_words_with_pos(tokens) do
    alias ChatBot.ML.POSTagger

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
    Enum.find(facts, fn fact ->
      fact_text = String.downcase(fact.fact)
      entity_text = String.downcase(fact.entity)

      Enum.any?(content_words, fn word ->
        String.contains?(fact_text, word) or String.contains?(entity_text, word)
      end)
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

      numbers_to_check = if relevant_numbers != [], do: relevant_numbers, else: fact_numbers

      if stated_number in numbers_to_check do
        {:match, stated_number}
      else
        {:mismatch, List.first(numbers_to_check)}
      end
    end
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  defp find_entity_value(entities, entity_type) when is_list(entities) do
    entity =
      Enum.find(entities, fn e ->
        e[:entity_type] == entity_type
      end)

    if entity do
      entity[:value] || entity["value"]
    else
      nil
    end
  end

  defp find_entity_value(_, _), do: nil

  defp extract_entity_names_for_facts(entities) when is_list(entities) do
    entities
    |> Enum.map(fn e ->
      e[:value] || e["value"] || ""
    end)
    |> Enum.filter(&(&1 != ""))
  end

  defp extract_entity_names_for_facts(_), do: []

  defp format_entities([]), do: ""

  defp format_entities(entities) do
    entity_str =
      entities
      |> Enum.map(fn e ->
        entity_type = e[:entity_type] || "unknown"
        entity_value = e[:value] || ""
        "#{entity_type}: #{entity_value}"
      end)
      |> Enum.join(", ")

    " (with #{entity_str})"
  end

  defp weave_response_parts(parts, types) do
    # Use Composer for sophisticated multi-part response weaving
    valid_parts = Enum.filter(parts, &(&1 != nil and &1 != ""))

    if length(valid_parts) <= 1 do
      # No need for weaving with single part
      Enum.join(valid_parts, " ")
    else
      # Build analyses for Composer
      analyses = build_analyses_for_composer(valid_parts, types)
      Composer.weave_multi_chunk_response(valid_parts, analyses)
    end
  end

  defp build_analyses_for_composer(parts, types) do
    # Pad types to match parts length if needed
    padded_types =
      if length(types) < length(parts) do
        types ++ List.duplicate(:assertive, length(parts) - length(types))
      else
        types
      end

    Enum.zip(parts, padded_types)
    |> Enum.map(fn {_part, type} ->
      %{
        speech_act: %{
          category: type_to_category(type),
          is_question: false,
          sub_type: nil
        }
      }
    end)
  end

  defp type_to_category(:expressive), do: :expressive
  defp type_to_category(:domain), do: :directive
  defp type_to_category(:template), do: :assertive
  defp type_to_category(:memory_augmented), do: :assertive
  defp type_to_category(_), do: :assertive
end
