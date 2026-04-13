defmodule Brain.Analysis.SpeechActClassifier do
  @moduledoc "Classifies the pragmatic function (speech act) of text chunks.\n\nBased on Searle's taxonomy:\n- Assertives: statements, claims, reports\n- Directives: requests, commands, questions\n- Commissives: promises, offers\n- Expressives: thanks, apologies, greetings\n- Declaratives: performatives\n\nUses multiple analysis passes to build a robust understanding:\n1. Intent Classification - trained model prediction\n2. Structural Analysis - sentence structure (questions, imperatives)\n3. Keyword Analysis - domain-specific keywords\n4. Entity Context - what entities suggest about intent\n5. Pragmatic Markers - politeness, urgency, etc.\n\nResults are combined using confidence-weighted voting.\n"

  alias Brain.Analysis.{SpeechActResult, IntentRegistry}
  alias Brain.ML.{IntentClassifierSimple, IntentArbitrator, POSTagger, Tokenizer}
  alias Brain.ML.LSTM.UnifiedModel

  alias Brain.Memory.Store
  alias Brain.Memory.Embedder
  require Logger

  @doc """
  Classifies the speech act of the given text using multiple analysis passes.

  Returns a SpeechActResult struct.

  ## Options
    - `:entities` - Pre-extracted entities to use for arbitration (default: [])
  """
  def classify(text, opts \\ []) when is_binary(text) do
    entities = Keyword.get(opts, :entities, [])
    analyses = run_all_analyses(text, entities)
    combine_analyses(analyses, text)
  end

  @doc "Returns detailed analysis with all individual scores for debugging/learning.\n"
  def analyze(text, opts \\ []) when is_binary(text) do
    entities = Keyword.get(opts, :entities, [])
    analyses = run_all_analyses(text, entities)
    result = combine_analyses(analyses, text)

    %{
      result: result,
      intent_classification: analyses.intent,
      structural_analysis: analyses.structural,
      keyword_analysis: analyses.keyword,
      pragmatic_analysis: analyses.pragmatic,
      memory_analysis: analyses.memory
    }
  end

  defp run_all_analyses(text, entities) do
    normalized = normalize_text(text)

    %{
      intent: analyze_with_intent_model(text, entities),
      structural: analyze_structure(text, normalized),
      keyword: analyze_keywords(normalized),
      pragmatic: analyze_pragmatics(text, normalized),
      memory: analyze_with_memory(text)
    }
  end

  defp analyze_with_intent_model(text, entities) do
    case classify_with_lstm(text, entities) do
      {:ok, result} -> result
      {:error, _} -> classify_with_tfidf(text)
    end
  end

  defp classify_with_lstm(text, entities) do
    if lstm_available?() do
      case UnifiedModel.classify_intent(text) do
        {:ok, %{label: lstm_intent, confidence: lstm_conf, scores: lstm_scores}} ->
          if registered_intent?(lstm_intent) do
            tfidf_result = safe_tfidf_classify(text)
            arbitrate_intent(text, lstm_intent, lstm_conf, lstm_scores, tfidf_result, entities)
          else
            {:error, :unregistered_lstm_intent}
          end

        {:ok, {intent, confidence}} ->
          if registered_intent?(intent) do
            tfidf_result = safe_tfidf_classify(text)
            arbitrate_intent(text, intent, confidence, [], tfidf_result, entities)
          else
            {:error, :unregistered_lstm_intent}
          end

        {:error, _} = error ->
          error
      end
    else
      {:error, :lstm_not_available}
    end
  end

  defp arbitrate_intent(text, lstm_intent, lstm_conf, lstm_scores, tfidf_result, entities) do
    features =
      IntentArbitrator.extract_features(%{
        lstm: %{intent: lstm_intent, confidence: lstm_conf, scores: lstm_scores},
        tfidf: tfidf_result,
        text: text,
        entities: entities
      })

    tfidf_conf = if is_map(tfidf_result), do: tfidf_result[:confidence] || 0.0, else: 0.0

    case IntentArbitrator.arbitrate(features) do
      {:tfidf, arb_conf} ->
        use_tfidf_result(tfidf_result, arb_conf, lstm_intent, lstm_conf, lstm_scores)

      {:lstm, _arb_conf} ->
        {:ok, build_lstm_result(lstm_intent, lstm_conf, lstm_scores)}

      {:error, reason} ->
        Logger.warning("IntentArbitrator unavailable (#{reason}), falling back to confidence comparison")
        if lstm_conf >= tfidf_conf do
          {:ok, build_lstm_result(lstm_intent, lstm_conf, lstm_scores)}
        else
          use_tfidf_result(tfidf_result, tfidf_conf, lstm_intent, lstm_conf, lstm_scores)
        end
    end
  end

  defp use_tfidf_result(tfidf_result, arb_conf, lstm_intent, lstm_conf, lstm_scores) do
    tfidf_intent = if is_map(tfidf_result), do: tfidf_result[:intent]
    tfidf_conf = if is_map(tfidf_result), do: tfidf_result[:confidence] || 0.0, else: 0.0
    tfidf_str = if tfidf_intent, do: to_string(tfidf_intent), else: ""

    if tfidf_str != "" and registered_intent?(tfidf_str) do
      {category, sub_type} = intent_to_speech_act(tfidf_str)

      {:ok,
       %{
         intent: tfidf_str,
         category: category,
         sub_type: sub_type,
         confidence: tfidf_conf,
         second_score: lstm_conf,
         margin: abs(tfidf_conf - lstm_conf),
         top_k: [],
         source: :arbitrator_tfidf,
         arbitrator_confidence: arb_conf
       }}
    else
      {:ok, build_lstm_result(lstm_intent, lstm_conf, lstm_scores)}
    end
  end

  defp safe_tfidf_classify(text) do
    try do
      case IntentClassifierSimple.classify(text, with_details: true, top_k: 5) do
        {:ok, %{intent: intent, confidence: conf} = result} ->
          top_k = Map.get(result, :top_k, [])
          scores = Enum.map(top_k, fn %{intent: i, score: s} -> {i, s} end)
          margin = Map.get(result, :margin, 0.0)
          %{intent: intent, confidence: conf, scores: scores, margin: margin}

        {:ok, {intent, conf}} ->
          %{intent: intent, confidence: conf, scores: []}

        _ ->
          nil
      end
    rescue
      _ -> nil
    catch
      _, _ -> nil
    end
  end

  defp registered_intent?(intent) when is_binary(intent) do
    if intent == "unknown" do
      false
    else
    case IntentRegistry.get(intent) do
      nil ->
        IntentRegistry.list_intents()
        |> Enum.any?(fn registered ->
          String.starts_with?(intent, registered <> ".") or
            String.starts_with?(registered, intent <> ".")
        end)

      _ ->
        true
    end
    end
  end

  defp registered_intent?(_), do: false

  defp build_lstm_result(intent, confidence, scores) do
    {category, sub_type} = intent_to_speech_act(intent)

    top_k =
      scores
      |> Enum.sort_by(fn {_label, score} -> -score end)
      |> Enum.take(5)
      |> Enum.map(fn {label, score} -> %{intent: label, score: score} end)

    second_score =
      case top_k do
        [_, %{score: s} | _] -> s
        _ -> 0.0
      end

    %{
      intent: intent,
      category: category,
      sub_type: sub_type,
      confidence: confidence,
      second_score: second_score,
      margin: confidence - second_score,
      top_k: top_k,
      source: :lstm
    }
  end

  defp classify_with_tfidf(text) do
    case IntentClassifierSimple.classify(text, with_details: true, top_k: 5) do
      {:ok, %{intent: intent, confidence: confidence} = result} ->
        {category, sub_type} = intent_to_speech_act(intent)

        %{
          intent: intent,
          category: category,
          sub_type: sub_type,
          confidence: confidence,
          second_score: Map.get(result, :second_score, 0.0),
          margin: Map.get(result, :margin, 0.0),
          top_k: Map.get(result, :top_k, []),
          source: :tfidf
        }

      {:error, _} ->
        %{
          intent: nil,
          category: nil,
          sub_type: nil,
          confidence: 0.0,
          second_score: 0.0,
          margin: 0.0,
          top_k: [],
          source: :tfidf
        }
    end
  end

  defp lstm_available? do
    case Process.get(:lstm_available_check) do
      nil ->
        available =
          try do
            Code.ensure_loaded?(UnifiedModel) and UnifiedModel.ready?()
          rescue
            _ -> false
          catch
            _, _ -> false
          end

        Process.put(:lstm_available_check, {available, System.monotonic_time(:second)})
        available

      {cached_result, checked_at} ->
        if System.monotonic_time(:second) - checked_at > 10 do
          Process.delete(:lstm_available_check)
          lstm_available?()
        else
          cached_result
        end
    end
  end

  defp analyze_structure(text, normalized) do
    is_question = has_question_structure?(text, normalized)
    is_exclamatory = Tokenizer.ends_with_exclamation?(text)
    is_declarative = Tokenizer.ends_with_period?(text)
    is_continuation = has_continuation_structure?(text, normalized)
    is_imperative = has_imperative_start?(normalized)

    has_modal = has_modal_verb?(normalized)

    {category, sub_type, confidence} =
      cond do
        is_continuation ->
          {:assertive, :continuation, 0.75}

        is_question and has_modal ->
          {:directive, :request_action, 0.8}

        is_question ->
          {:directive, :request_information, 0.85}

        is_imperative and not is_question ->
          {:directive, :command, 0.75}

        is_declarative and not is_question ->
          {:assertive, :statement, 0.7}

        is_exclamatory ->
          {:expressive, :general, 0.5}

        true ->
          {:assertive, :statement, 0.2}
      end

    %{
      is_question: is_question,
      is_imperative: is_imperative,
      is_exclamatory: is_exclamatory,
      is_declarative: is_declarative,
      is_continuation: is_continuation,
      has_modal: has_modal,
      category: category,
      sub_type: sub_type,
      confidence: confidence,
      source: :structural
    }
  end

  defp analyze_keywords(normalized) do
    case Brain.ML.SpeechActClassifierSimple.classify(normalized) do
      {:ok, %{label: label, confidence: confidence}} when confidence > 0.2 ->
        %{
          scores: %{label => confidence},
          category: label,
          sub_type: :general,
          confidence: confidence,
          source: :keyword
        }

      _ ->
        %{
          scores: %{},
          category: nil,
          sub_type: nil,
          confidence: 0.0,
          source: :keyword
        }
    end
  end

  defp analyze_pragmatics(_text, normalized) do
    words = String.split(normalized)
    word_count = length(words)
    is_short = word_count <= 3
    is_very_short = word_count <= 2

    %{
      has_please: false,
      has_thanks: false,
      has_urgency: false,
      has_hedging: false,
      is_short_utterance: is_short,
      is_very_short: is_very_short,
      is_backchannel: false,
      is_compliment: false,
      has_acknowledgment: false,
      pragmatic_sub_type: nil,
      expressive_score: 0.0,
      source: :pragmatic
    }
  end

  defp analyze_with_memory(text) do
    case query_memory_for_classification(text) do
      {:ok, [_ | _] = results} ->
        {category, sub_type, confidence} = vote_on_memory_results(results)

        %{
          category: category,
          sub_type: sub_type,
          confidence: confidence,
          similar_count: length(results),
          source: :memory
        }

      _ ->
        %{
          category: nil,
          sub_type: nil,
          confidence: 0.0,
          similar_count: 0,
          source: :memory
        }
    end
  end

  defp query_memory_for_classification(text) do
    store_pid = Process.whereis(Brain.Memory.Store)
    embedder_pid = Process.whereis(Brain.Memory.Embedder)

    cond do
      store_pid == nil ->
        {:error, :store_not_running}

      embedder_pid == nil ->
        {:error, :embedder_not_running}

      not Embedder.ready?() ->
        {:error, :embedder_not_ready}

      true ->
        task = Task.async(fn -> Store.query_similar(text, 5) end)

        case Task.yield(task, 500) || Task.shutdown(task, :brutal_kill) do
          {:ok, result} -> result
          nil -> {:error, :timeout}
        end
    end
  rescue
    _ -> {:error, :not_available}
  end

  defp vote_on_memory_results(results) do
    all_tags =
      results
      |> Enum.flat_map(fn {episode, similarity} ->
        Enum.map(episode.tags, fn tag -> {tag, similarity} end)
      end)

    tag_scores =
      all_tags
      |> Enum.group_by(fn {tag, _} -> tag end, fn {_, sim} -> sim end)
      |> Enum.map(fn {tag, sims} -> {tag, Enum.sum(sims)} end)
      |> Enum.sort_by(fn {_, score} -> -score end)

    case tag_scores do
      [{top_tag, score} | _] ->
        {category, sub_type} = tag_to_speech_act(top_tag)
        max_possible = length(results) * 1.0
        confidence = min(score / max_possible, 1.0)
        {category, sub_type, confidence}

      [] ->
        {:assertive, :statement, 0.0}
    end
  end

  defp tag_to_speech_act(tag) do
    case IntentRegistry.get(tag) do
      nil ->
        {:assertive, :statement}

      _meta ->
        category = IntentRegistry.category(tag) || :assertive
        speech_act = IntentRegistry.speech_act(tag) || :statement
        {category, speech_act}
    end
  end

  defp combine_analyses(analyses, _text) do
    votes = collect_votes(analyses)
    weighted_votes = apply_weights(votes)
    {category, sub_type, confidence} = determine_winner(weighted_votes, analyses)
    indicators = collect_indicators(analyses)

    is_imperative_from_intent =
      analyses.intent.sub_type == :command and analyses.intent.confidence > 0.3

    is_imperative = is_imperative_from_intent or analyses.structural.is_imperative

    SpeechActResult.new(category, sub_type, confidence,
      indicators: indicators,
      is_question: analyses.structural.is_question,
      is_imperative: is_imperative,
      intent_confidence: analyses.intent[:confidence]
    )
  end

  defp collect_votes(analyses) do
    votes = []

    votes =
      if analyses.intent.confidence > 0.2 and analyses.intent.category != nil do
        vote =
          {analyses.intent.category, analyses.intent.sub_type, analyses.intent.confidence, :model}

        [vote | votes]
      else
        votes
      end

    votes =
      if analyses.structural.confidence > 0.3 do
        vote =
          {analyses.structural.category, analyses.structural.sub_type,
           analyses.structural.confidence, :structural}

        [vote | votes]
      else
        votes
      end

    votes =
      if analyses.keyword.confidence > 0.3 and analyses.keyword.category != nil do
        vote =
          {analyses.keyword.category, analyses.keyword.sub_type, analyses.keyword.confidence,
           :keyword}

        [vote | votes]
      else
        votes
      end

    votes =
      if analyses.pragmatic.expressive_score > 0.5 do
        sub_type = analyses.pragmatic.pragmatic_sub_type || :general
        vote = {:expressive, sub_type, analyses.pragmatic.expressive_score, :pragmatic}
        [vote | votes]
      else
        votes
      end

    votes =
      if analyses.memory.confidence > 0.3 and analyses.memory.category != nil do
        vote =
          {analyses.memory.category, analyses.memory.sub_type, analyses.memory.confidence,
           :memory}

        [vote | votes]
      else
        votes
      end

    votes
  end

  defp apply_weights(votes) do
    source_weights = %{
      model: 1.5,
      memory: 1.4,
      keyword: 1.2,
      structural: 1.0,
      pragmatic: 0.8
    }

    Enum.map(votes, fn {category, sub_type, confidence, source} ->
      weight = Map.get(source_weights, source, 1.0)
      weighted_confidence = confidence * weight
      {category, sub_type, weighted_confidence, source}
    end)
  end

  defp determine_winner(weighted_votes, analyses) do
    if Enum.empty?(weighted_votes) do
      {:assertive, :statement, 0.4}
    else
      by_category =
        weighted_votes
        |> Enum.group_by(fn {cat, _, _, _} -> cat end)

      category_scores =
        Enum.into(by_category, %{}, fn {cat, votes} ->
          total = Enum.sum(Enum.map(votes, fn {_, _, conf, _} -> conf end))
          {cat, total}
        end)

      {winning_category, _} = Enum.max_by(category_scores, fn {_, score} -> score end)
      category_votes = Map.get(by_category, winning_category, [])

      {_, winning_sub_type, _, _} =
        Enum.max_by(category_votes, fn {_, _, conf, _} -> conf end)

      max_conf = Enum.max(Enum.map(weighted_votes, fn {_, _, conf, _} -> conf end))

      avg_conf =
        Enum.sum(Enum.map(weighted_votes, fn {_, _, conf, _} -> conf end)) /
          length(weighted_votes)

      confidence = min((max_conf + avg_conf) / 2, 1.0)

      model_expressive =
        analyses.intent.category == :expressive and analyses.intent.confidence > 0.4

      keyword_expressive = analyses.keyword.category == :expressive

      confidence =
        if model_expressive and keyword_expressive do
          min(confidence + 0.2, 1.0)
        else
          confidence
        end

      {winning_category, winning_sub_type, confidence}
    end
  end

  defp collect_indicators(analyses) do
    indicators = []

    indicators =
      if analyses.intent.intent do
        ["intent:#{analyses.intent.intent}" | indicators]
      else
        indicators
      end

    indicators =
      if analyses.structural.is_question do
        ["question_structure" | indicators]
      else
        indicators
      end

    indicators =
      if analyses.intent.sub_type == :command and analyses.intent.confidence > 0.3 do
        ["imperative_from_intent" | indicators]
      else
        indicators
      end

    indicators =
      if Map.get(analyses.structural, :is_continuation, false) do
        ["continuation_structure" | indicators]
      else
        indicators
      end

    indicators =
      if Map.get(analyses.pragmatic, :is_backchannel, false) do
        ["backchannel" | indicators]
      else
        indicators
      end

    indicators =
      if Map.get(analyses.pragmatic, :is_compliment, false) do
        ["compliment" | indicators]
      else
        indicators
      end

    indicators =
      if analyses.keyword.confidence > 0.3 do
        ["keyword:#{analyses.keyword.sub_type}" | indicators]
      else
        indicators
      end

    indicators =
      if analyses.memory.confidence > 0.3 do
        ["memory:#{analyses.memory.similar_count}_similar" | indicators]
      else
        indicators
      end

    indicators
  end

  defp normalize_text(text) do
    text
    |> String.downcase()
    |> String.trim()
  end

  defp has_question_structure?(text, normalized) do
    Tokenizer.ends_with_question?(text) or starts_with_interrogative_pos?(normalized)
  end

  defp starts_with_interrogative_pos?(normalized) do
    first_word =
      normalized
      |> Tokenizer.tokenize_words()
      |> List.first("")
      |> String.downcase()

    first_word in Tokenizer.question_words()
  end


  defp has_modal_verb?(normalized) do
    words = String.split(normalized)

    case POSTagger.load_model() do
      {:ok, model} ->
        predictions = POSTagger.predict(words, model)
        Enum.any?(predictions, fn {_word, tag} -> tag == "AUX" end)

      {:error, _} ->
        false
    end
  end

  defp has_imperative_start?(normalized) do
    imperative_verbs = expanded_imperative_verbs()

    words = Tokenizer.split_words(normalized)
    starter = List.first(words, "")

    first_action_word =
      case starter do
        "please" -> Enum.at(words, 1, "")
        "kindly" -> Enum.at(words, 1, "")
        _ -> starter
      end

    MapSet.member?(imperative_verbs, first_action_word)
  end

  @seed_imperative_verbs ~w(
    tell show give get find search look check
    turn set make create open close start stop
    play pause skip next list read send call
    help explain describe calculate remember
  )

  defp expanded_imperative_verbs do
    if Process.whereis(Brain.ML.Lexicon) do
      @seed_imperative_verbs
      |> Enum.flat_map(fn verb ->
        syns = Brain.ML.Lexicon.synonyms(verb, :verb)
        [verb | Enum.take(syns, 3)]
      end)
      |> MapSet.new()
    else
      MapSet.new(@seed_imperative_verbs)
    end
  end

  defp has_continuation_structure?(text, _normalized) do
    no_terminal = not Tokenizer.ends_with_terminal_punctuation?(text)
    ends_with_comma = String.last(String.trim_trailing(text)) == ","
    trailing_ellipsis = Tokenizer.ends_with_ellipsis?(text)
    no_terminal and (ends_with_comma or trailing_ellipsis)
  end

  defp intent_to_speech_act(intent) do
    # Look up speech act from the IntentRegistry (exact match first)
    case IntentRegistry.get(intent) do
      nil ->
        # Try prefix match in both directions:
        # 1. Intent is more specific than registry key (e.g. "weather.query.today" matches "weather.query")
        # 2. Intent is broader than registry key (e.g. "weather" matches "weather.query")
        find_registry_prefix_match(intent)

      _meta ->
        category = IntentRegistry.category(intent) || :assertive
        sub_type = IntentRegistry.speech_act(intent) || :statement
        {category, sub_type}
    end
  end

  defp find_registry_prefix_match(intent) do
    all_intents = IntentRegistry.list_intents()

    # First try: intent is more specific than a registry key
    # e.g. classified "smalltalk.greetings.hello" matches registered "smalltalk.greetings"
    specific_matches =
      all_intents
      |> Enum.filter(fn registered -> String.starts_with?(intent, registered <> ".") end)
      |> Enum.sort_by(&(-String.length(&1)))

    # Second try: intent is broader than registry keys
    # e.g. classified "weather" matches registered "weather.query", "weather.condition"
    # Sort by shortest (most general child) first
    broad_matches =
      all_intents
      |> Enum.filter(fn registered -> String.starts_with?(registered, intent <> ".") end)
      |> Enum.sort_by(&String.length/1)

    best = List.first(specific_matches) || List.first(broad_matches)

    case best do
      nil ->
        {:assertive, :statement}

      matched ->
        category = IntentRegistry.category(matched) || :assertive
        sub_type = IntentRegistry.speech_act(matched) || :statement
        {category, sub_type}
    end
  end

end
