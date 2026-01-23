defmodule ChatBot.Analysis.SpeechActClassifier do
  @moduledoc """
  Classifies the pragmatic function (speech act) of text chunks.

  Based on Searle's taxonomy:
  - Assertives: statements, claims, reports
  - Directives: requests, commands, questions
  - Commissives: promises, offers
  - Expressives: thanks, apologies, greetings
  - Declaratives: performatives

  Uses multiple analysis passes to build a robust understanding:
  1. Intent Classification - trained model prediction
  2. Structural Analysis - sentence structure (questions, imperatives)
  3. Keyword Analysis - domain-specific keywords
  4. Entity Context - what entities suggest about intent
  5. Pragmatic Markers - politeness, urgency, etc.

  Results are combined using confidence-weighted voting.
  """

  alias ChatBot.Analysis.SpeechActResult
  alias ChatBot.ML.IntentClassifierSimple
  alias ChatBot.ML.Tokenizer

  require Logger

  # Intent to speech act mapping
  @intent_to_speech_act %{
    # Welcome/Default intents
    "Default.Welcome.Intent" => {:expressive, :greeting},
    "Default.Fallback.Intent" => {:assertive, :statement},

    # Greetings
    "smalltalk.greetings.hello" => {:expressive, :greeting},
    "smalltalk.greetings.goodmorning" => {:expressive, :greeting},
    "smalltalk.greetings.goodevening" => {:expressive, :greeting},
    "smalltalk.greetings.nice_to_meet_you" => {:expressive, :greeting},
    "smalltalk.greetings.nice_to_see_you" => {:expressive, :greeting},
    "smalltalk.greetings.nice_to_talk_to_you" => {:expressive, :greeting},
    "smalltalk.greetings.whatsup" => {:expressive, :greeting},

    # Farewells
    "smalltalk.greetings.bye" => {:expressive, :farewell},
    "smalltalk.greetings.goodnight" => {:expressive, :farewell},

    # Thanks
    "smalltalk.appraisal.thank_you" => {:expressive, :thanks},

    # Apologies
    "smalltalk.appraisal.sorry" => {:expressive, :apology},
    "smalltalk.dialog.sorry" => {:expressive, :apology},

    # How are you (question about wellbeing)
    "smalltalk.greetings.how_are_you" => {:directive, :request_information},

    # Agent questions (asking about the bot)
    "smalltalk.agent.acquaintance" => {:directive, :request_information},
    "smalltalk.agent.age" => {:directive, :request_information},
    "smalltalk.agent.name" => {:directive, :request_information},
    "smalltalk.agent.can_you_help" => {:directive, :request_action},
    "smalltalk.agent.there" => {:directive, :request_information},

    # User statements about themselves
    "smalltalk.user.name" => {:assertive, :statement},
    "smalltalk.user.age" => {:assertive, :statement},
    "smalltalk.user.location" => {:assertive, :statement},

    # Confirmations
    "smalltalk.confirmation.yes" => {:assertive, :confirmation},
    "smalltalk.confirmation.no" => {:assertive, :denial},

    # Music
    "music.play" => {:directive, :command},
    "music.stop" => {:directive, :command},
    "music.pause" => {:directive, :command},
    "music.next" => {:directive, :command},
    "music.previous" => {:directive, :command},

    # Smart home
    "smarthome.lights.on" => {:directive, :command},
    "smarthome.lights.off" => {:directive, :command},
    "smarthome.lights.dim" => {:directive, :command},
    "smarthome.temperature" => {:directive, :command},

    # Default fallbacks by prefix
    "smalltalk.appraisal" => {:expressive, :general},
    "smalltalk.agent" => {:directive, :request_information},
    "smalltalk.user" => {:assertive, :statement},
    "smalltalk.greetings" => {:expressive, :greeting},
    "smalltalk.dialog" => {:directive, :general},
    "smalltalk.emotions" => {:expressive, :general},
    "music" => {:directive, :command},
    "smarthome" => {:directive, :command},
    "weather" => {:directive, :request_information},
    "reminder" => {:directive, :command},
    "timer" => {:directive, :command},
    "navigation" => {:directive, :request_information}
  }

  # Keyword patterns for different speech acts
  @greeting_keywords ~w(hello hi hey howdy greetings hola yo sup)
  @farewell_keywords ~w(bye goodbye cya farewell later goodnight)
  @thanks_keywords ~w(thanks thank appreciate grateful)
  @sorry_keywords ~w(sorry apologies apologize pardon)
  @question_words ~w(what where when why who whom whose which how)
  @imperative_starters ~w(
    tell show give get find search look check
    turn set make create open close start stop
    play pause skip next previous list read
    send call text message remind schedule
    help explain describe calculate
  )
  @modal_verbs ~w(can could would will should might may)

  @doc """
  Classifies the speech act of the given text using multiple analysis passes.

  Returns a SpeechActResult struct.
  """
  def classify(text) when is_binary(text) do
    # Run all analysis passes
    analyses = run_all_analyses(text)

    # Combine results using weighted voting
    combine_analyses(analyses, text)
  end

  @doc """
  Returns detailed analysis with all individual scores for debugging/learning.
  """
  def analyze(text) when is_binary(text) do
    analyses = run_all_analyses(text)
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

  # ============================================================================
  # Multi-Pass Analysis
  # ============================================================================

  defp run_all_analyses(text) do
    normalized = normalize_text(text)

    %{
      intent: analyze_with_intent_model(text),
      structural: analyze_structure(text, normalized),
      keyword: analyze_keywords(normalized),
      pragmatic: analyze_pragmatics(text, normalized),
      memory: analyze_with_memory(text)
    }
  end

  # Pass 1: Intent Model Classification
  defp analyze_with_intent_model(text) do
    case IntentClassifierSimple.classify(text) do
      {:ok, %{intent: intent, confidence: confidence}} ->
        {category, sub_type} = intent_to_speech_act(intent)

        %{
          intent: intent,
          category: category,
          sub_type: sub_type,
          confidence: confidence,
          source: :model
        }

      {:error, _} ->
        %{
          intent: nil,
          category: nil,
          sub_type: nil,
          confidence: 0.0,
          source: :model
        }
    end
  end

  # Pass 2: Structural Analysis
  defp analyze_structure(text, normalized) do
    is_question = has_question_structure?(text, normalized)
    is_imperative = has_imperative_structure?(normalized)
    is_exclamatory = String.ends_with?(String.trim(text), "!")
    is_declarative = String.ends_with?(String.trim(text), ".")
    has_modal = has_modal_verb?(normalized)

    {category, sub_type, confidence} =
      cond do
        is_question and has_modal ->
          {:directive, :request_action, 0.8}

        is_question ->
          {:directive, :request_information, 0.85}

        is_imperative ->
          {:directive, :command, 0.8}

        is_declarative and not is_question and not is_imperative ->
          {:assertive, :statement, 0.7}

        is_exclamatory ->
          {:expressive, :general, 0.5}

        true ->
          {:assertive, :statement, 0.4}
      end

    %{
      is_question: is_question,
      is_imperative: is_imperative,
      is_exclamatory: is_exclamatory,
      is_declarative: is_declarative,
      has_modal: has_modal,
      category: category,
      sub_type: sub_type,
      confidence: confidence,
      source: :structural
    }
  end

  # Pass 3: Keyword Analysis
  defp analyze_keywords(normalized) do
    # Remove punctuation for keyword matching using Tokenizer (no regex)
    words = Tokenizer.tokenize_words(normalized)
    first_word = List.first(words) || ""

    # Check for greeting keywords
    greeting_score = keyword_match_score(words, @greeting_keywords)

    # Check for farewell keywords
    farewell_score = keyword_match_score(words, @farewell_keywords)

    # Check for thanks keywords
    thanks_score = keyword_match_score(words, @thanks_keywords)

    # Check for sorry keywords
    sorry_score = keyword_match_score(words, @sorry_keywords)

    # Check for question starters
    question_score = if first_word in @question_words, do: 0.8, else: 0.0

    # Check for imperative starters
    imperative_score = if first_word in @imperative_starters, do: 0.8, else: 0.0

    # Find the highest scoring category
    scores = [
      {:greeting, greeting_score},
      {:farewell, farewell_score},
      {:thanks, thanks_score},
      {:apology, sorry_score},
      {:question, question_score},
      {:command, imperative_score}
    ]

    {best_type, best_score} = Enum.max_by(scores, fn {_, score} -> score end)

    {category, sub_type} =
      case best_type do
        :greeting -> {:expressive, :greeting}
        :farewell -> {:expressive, :farewell}
        :thanks -> {:expressive, :thanks}
        :apology -> {:expressive, :apology}
        :question -> {:directive, :request_information}
        :command -> {:directive, :command}
        _ -> {:assertive, :statement}
      end

    %{
      scores: Map.new(scores),
      category: if(best_score > 0.3, do: category, else: nil),
      sub_type: if(best_score > 0.3, do: sub_type, else: nil),
      confidence: best_score,
      source: :keyword
    }
  end

  # Pass 4: Pragmatic Markers Analysis
  defp analyze_pragmatics(_text, normalized) do
    words = String.split(normalized)

    # Check for politeness markers
    has_please = "please" in words
    has_thanks = Enum.any?(words, &(&1 in @thanks_keywords))

    # Check for urgency markers
    has_urgency = Enum.any?(words, &(&1 in ~w(now immediately urgent asap quickly)))

    # Check for hedging (uncertainty)
    has_hedging = Enum.any?(words, &(&1 in ~w(maybe perhaps possibly might)))

    # Check for discourse markers
    has_greeting_marker =
      Enum.any?(words, fn word ->
        word in @greeting_keywords or word in @farewell_keywords
      end)

    # Short utterances (1-3 words) are often expressives
    is_short = length(words) <= 3

    # Determine if this looks like an expressive based on pragmatics
    expressive_score =
      cond do
        has_greeting_marker and is_short -> 0.9
        has_thanks and is_short -> 0.85
        has_please -> 0.3
        true -> 0.0
      end

    %{
      has_please: has_please,
      has_thanks: has_thanks,
      has_urgency: has_urgency,
      has_hedging: has_hedging,
      is_short_utterance: is_short,
      expressive_score: expressive_score,
      source: :pragmatic
    }
  end

  # Pass 5: Memory-Based Classification (Cognitive Memory System)
  defp analyze_with_memory(text) do
    # Try to use the cognitive memory system for retrieval-based classification
    # This is optional and falls back gracefully if memory isn't available
    case query_memory_for_classification(text) do
      {:ok, [_ | _] = results} ->
        # Vote based on tags of similar episodes
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
    # Check if memory store and embedder are available and ready
    # Use Process.whereis which is non-blocking
    store_pid = Process.whereis(ChatBot.Memory.Store)
    embedder_pid = Process.whereis(ChatBot.Memory.Embedder)

    cond do
      store_pid == nil ->
        {:error, :store_not_running}

      embedder_pid == nil ->
        {:error, :embedder_not_running}

      not ChatBot.Memory.Embedder.ready?() ->
        # Embedder is still building vocabulary or busy - skip memory query
        {:error, :embedder_not_ready}

      true ->
        # Use a short timeout to avoid blocking the pipeline
        task = Task.async(fn -> ChatBot.Memory.Store.query_similar(text, 5) end)

        case Task.yield(task, 500) || Task.shutdown(task, :brutal_kill) do
          {:ok, result} -> result
          nil -> {:error, :timeout}
        end
    end
  rescue
    _ -> {:error, :not_available}
  end

  defp vote_on_memory_results(results) do
    # Extract tags from similar episodes and vote
    all_tags =
      results
      |> Enum.flat_map(fn {episode, similarity} ->
        Enum.map(episode.tags, fn tag -> {tag, similarity} end)
      end)

    # Group by tag and sum weighted similarity
    tag_scores =
      all_tags
      |> Enum.group_by(fn {tag, _} -> tag end, fn {_, sim} -> sim end)
      |> Enum.map(fn {tag, sims} -> {tag, Enum.sum(sims)} end)
      |> Enum.sort_by(fn {_, score} -> -score end)

    case tag_scores do
      [{top_tag, score} | _] ->
        # Map tag to category/sub_type
        {category, sub_type} = tag_to_speech_act(top_tag)
        # Normalize confidence
        max_possible = length(results) * 1.0
        confidence = min(score / max_possible, 1.0)
        {category, sub_type, confidence}

      [] ->
        {:assertive, :statement, 0.0}
    end
  end

  defp tag_to_speech_act(tag) do
    # Map common intent/tag patterns to speech acts
    cond do
      String.contains?(tag, "greeting") or String.contains?(tag, "hello") ->
        {:expressive, :greeting}

      String.contains?(tag, "bye") or String.contains?(tag, "farewell") ->
        {:expressive, :farewell}

      String.contains?(tag, "thank") ->
        {:expressive, :thanks}

      String.contains?(tag, "sorry") or String.contains?(tag, "apolog") ->
        {:expressive, :apology}

      String.contains?(tag, "weather") or String.contains?(tag, "time") ->
        {:directive, :request_information}

      String.contains?(tag, "play") or String.contains?(tag, "stop") or
          String.contains?(tag, "turn") ->
        {:directive, :command}

      String.contains?(tag, "question") or String.contains?(tag, "what") or
          String.contains?(tag, "how") ->
        {:directive, :request_information}

      true ->
        {:assertive, :statement}
    end
  end

  # ============================================================================
  # Result Combination
  # ============================================================================

  defp combine_analyses(analyses, _text) do
    # Collect votes from each analysis pass
    votes = collect_votes(analyses)

    # Apply voting weights based on confidence
    weighted_votes = apply_weights(votes)

    # Determine winner
    {category, sub_type, confidence} = determine_winner(weighted_votes, analyses)

    # Collect indicators for debugging
    indicators = collect_indicators(analyses)

    SpeechActResult.new(category, sub_type, confidence,
      indicators: indicators,
      is_question: analyses.structural.is_question,
      is_imperative: analyses.structural.is_imperative
    )
  end

  defp collect_votes(analyses) do
    votes = []

    # Vote from intent model (if confident enough)
    votes =
      if analyses.intent.confidence > 0.2 and analyses.intent.category != nil do
        vote =
          {analyses.intent.category, analyses.intent.sub_type, analyses.intent.confidence, :model}

        [vote | votes]
      else
        votes
      end

    # Vote from structural analysis
    votes =
      if analyses.structural.confidence > 0.3 do
        vote =
          {analyses.structural.category, analyses.structural.sub_type,
           analyses.structural.confidence, :structural}

        [vote | votes]
      else
        votes
      end

    # Vote from keyword analysis
    votes =
      if analyses.keyword.confidence > 0.3 and analyses.keyword.category != nil do
        vote =
          {analyses.keyword.category, analyses.keyword.sub_type, analyses.keyword.confidence,
           :keyword}

        [vote | votes]
      else
        votes
      end

    # Vote from pragmatic analysis (for expressives)
    votes =
      if analyses.pragmatic.expressive_score > 0.5 do
        vote = {:expressive, :general, analyses.pragmatic.expressive_score, :pragmatic}
        [vote | votes]
      else
        votes
      end

    # Vote from memory-based analysis (cognitive memory system)
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
    # Weight multipliers for different sources
    source_weights = %{
      # Trained model gets higher weight
      model: 1.5,
      # Memory-based retrieval is learned from examples
      memory: 1.4,
      # Keywords are reliable signals
      keyword: 1.2,
      # Structural is baseline
      structural: 1.0,
      # Pragmatic is supportive
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
      # No confident votes - default to statement
      {:assertive, :statement, 0.4}
    else
      # Group votes by category
      by_category =
        weighted_votes
        |> Enum.group_by(fn {cat, _, _, _} -> cat end)

      # Sum weighted confidence per category
      category_scores =
        Enum.into(by_category, %{}, fn {cat, votes} ->
          total = Enum.sum(Enum.map(votes, fn {_, _, conf, _} -> conf end))
          {cat, total}
        end)

      # Find winning category
      {winning_category, _} = Enum.max_by(category_scores, fn {_, score} -> score end)

      # Find best sub_type within winning category
      category_votes = Map.get(by_category, winning_category, [])

      {_, winning_sub_type, _, _} =
        Enum.max_by(category_votes, fn {_, _, conf, _} -> conf end)

      # Calculate combined confidence
      max_conf = Enum.max(Enum.map(weighted_votes, fn {_, _, conf, _} -> conf end))

      avg_conf =
        Enum.sum(Enum.map(weighted_votes, fn {_, _, conf, _} -> conf end)) /
          length(weighted_votes)

      confidence = min((max_conf + avg_conf) / 2, 1.0)

      # Special case: if model strongly says expressive and keywords agree, boost confidence
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

    # Intent indicator
    indicators =
      if analyses.intent.intent do
        ["intent:#{analyses.intent.intent}" | indicators]
      else
        indicators
      end

    # Structural indicators
    indicators =
      if analyses.structural.is_question do
        ["question_structure" | indicators]
      else
        indicators
      end

    indicators =
      if analyses.structural.is_imperative do
        ["imperative_structure" | indicators]
      else
        indicators
      end

    # Keyword indicators
    indicators =
      if analyses.keyword.confidence > 0.3 do
        ["keyword:#{analyses.keyword.sub_type}" | indicators]
      else
        indicators
      end

    # Memory indicators
    indicators =
      if analyses.memory.confidence > 0.3 do
        ["memory:#{analyses.memory.similar_count}_similar" | indicators]
      else
        indicators
      end

    indicators
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp normalize_text(text) do
    text
    |> String.downcase()
    |> String.trim()
  end

  defp has_question_structure?(text, normalized) do
    String.ends_with?(String.trim(text), "?") or
      starts_with_question_word?(normalized)
  end

  defp starts_with_question_word?(normalized) do
    first_word = normalized |> String.split() |> List.first() || ""
    first_word in @question_words
  end

  defp has_imperative_structure?(normalized) do
    first_word = normalized |> String.split() |> List.first() || ""
    first_word in @imperative_starters
  end

  defp has_modal_verb?(normalized) do
    words = String.split(normalized)
    Enum.any?(words, fn word -> word in @modal_verbs end)
  end

  defp keyword_match_score(words, keywords) do
    matches = Enum.count(words, fn word -> word in keywords end)

    cond do
      matches >= 2 -> 0.95
      matches == 1 -> 0.8
      true -> 0.0
    end
  end

  defp intent_to_speech_act(intent) do
    # First try exact match
    case Map.get(@intent_to_speech_act, intent) do
      nil ->
        # Try prefix matching
        find_prefix_match(intent)

      result ->
        result
    end
  end

  defp find_prefix_match(intent) do
    # Find the longest matching prefix
    matching_prefixes =
      @intent_to_speech_act
      |> Enum.filter(fn {prefix, _} -> String.starts_with?(intent, prefix) end)
      |> Enum.sort_by(fn {prefix, _} -> -String.length(prefix) end)

    case matching_prefixes do
      [{_prefix, result} | _] -> result
      # Default
      [] -> {:assertive, :statement}
    end
  end

  # ============================================================================
  # Legacy API Compatibility
  # ============================================================================

  @doc """
  Maps speech act types to categories. Used for backwards compatibility.
  """
  def map_to_speech_act(type, _normalized, _text) do
    case type do
      :question -> {:directive, :request_information}
      :request_action -> {:directive, :request_action}
      :request_information -> {:directive, :request_information}
      :command -> {:directive, :command}
      :greeting -> {:expressive, :greeting}
      :farewell -> {:expressive, :farewell}
      :thanks -> {:expressive, :thanks}
      :apology -> {:expressive, :apology}
      :promise -> {:commissive, :promise}
      :offer -> {:commissive, :offer}
      :statement -> {:assertive, :statement}
      _ -> {:assertive, :statement}
    end
  end
end
