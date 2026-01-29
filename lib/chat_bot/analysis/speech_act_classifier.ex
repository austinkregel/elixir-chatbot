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

  alias ChatBot.Analysis.{SpeechActResult, IntentRegistry}
  alias ChatBot.ML.{IntentClassifierSimple, POSTagger, Tokenizer}

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
    "smalltalk.user.introduction" => {:assertive, :statement},
    "smalltalk.user.origin" => {:assertive, :statement},

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

  # ============================================================================
  # ML-Based Classification
  # Speech acts are determined by the trained intent classifier and POS analysis
  # No keyword matching - all patterns are learned from training data
  # ============================================================================

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
    is_exclamatory = Tokenizer.ends_with_exclamation?(text)
    is_declarative = Tokenizer.ends_with_period?(text)

    # Continuation detection: no terminal punctuation, or ends with continuation marker
    is_continuation = has_continuation_structure?(text, normalized)

    has_modal = has_modal_verb?(normalized)

    {category, sub_type, confidence} =
      cond do
        # Continuation: incomplete thought, expecting more
        is_continuation ->
          {:assertive, :continuation, 0.75}

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
      is_continuation: is_continuation,
      has_modal: has_modal,
      category: category,
      sub_type: sub_type,
      confidence: confidence,
      source: :structural
    }
  end

  # Pass 3: Disabled - All patterns learned from training data
  # Keyword matching has been removed in favor of ML-based classification
  defp analyze_keywords(_normalized) do
    # Return neutral results - let the ML model (intent classifier) drive speech act detection
    # All speech act patterns are learned from training data, not keyword lists
    %{
      scores: %{},
      category: nil,
      sub_type: nil,
      confidence: 0.0,
      source: :keyword
    }
  end

  # Pass 4: Pragmatic Markers Analysis (Structural Only)
  # Keyword-based detection removed - ML model handles pattern recognition
  defp analyze_pragmatics(_text, normalized) do
    words = String.split(normalized)
    word_count = length(words)

    # Structural features only - no keyword matching
    # Short utterances (1-3 words) are often expressives
    is_short = word_count <= 3
    is_very_short = word_count <= 2

    # Return structural features only - let ML model determine speech act type
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
    # Look up speech act from IntentRegistry
    case IntentRegistry.get(tag) do
      nil ->
        # Tag not in registry, default to statement
        {:assertive, :statement}

      _meta ->
        category = IntentRegistry.category(tag) || :assertive
        speech_act = IntentRegistry.speech_act(tag) || :statement
        {category, speech_act}
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

    # Infer is_imperative from either structural analysis OR intent classification
    # If the intent classifier detected a command intent, treat it as imperative
    is_imperative_from_intent =
      analyses.intent.sub_type == :command and analyses.intent.confidence > 0.3

    is_imperative = analyses.structural.is_imperative or is_imperative_from_intent

    SpeechActResult.new(category, sub_type, confidence,
      indicators: indicators,
      is_question: analyses.structural.is_question,
      is_imperative: is_imperative
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

    # Vote from pragmatic analysis (for expressives, including new sub-types)
    votes =
      if analyses.pragmatic.expressive_score > 0.5 do
        # Use the specific pragmatic sub_type if detected
        sub_type = analyses.pragmatic.pragmatic_sub_type || :general
        vote = {:expressive, sub_type, analyses.pragmatic.expressive_score, :pragmatic}
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

    # Continuation indicator
    indicators =
      if Map.get(analyses.structural, :is_continuation, false) do
        ["continuation_structure" | indicators]
      else
        indicators
      end

    # Backchannel indicator
    indicators =
      if Map.get(analyses.pragmatic, :is_backchannel, false) do
        ["backchannel" | indicators]
      else
        indicators
      end

    # Compliment indicator
    indicators =
      if Map.get(analyses.pragmatic, :is_compliment, false) do
        ["compliment" | indicators]
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
    # Use punctuation + POS-based detection (no keyword lists)
    Tokenizer.ends_with_question?(text) or starts_with_interrogative_pos?(normalized)
  end

  defp starts_with_interrogative_pos?(_normalized) do
    # POS-based interrogative detection has been disabled because:
    # 1. The POS tagger can't distinguish personal pronouns (I, you) from
    #    interrogative pronouns (what, who) - both are tagged as PRON
    # 2. Question detection now relies solely on punctuation (question mark)
    #    which is more reliable
    # 3. The intent classifier is trained on question patterns and will
    #    correctly identify questions through ML
    false
  end

  defp has_imperative_structure?(normalized) do
    # Use POS tagger to detect if first word is a verb (imperative)
    words = normalized |> String.split() |> Enum.take(2)

    case words do
      [] ->
        false

      _ ->
        case POSTagger.load_model() do
          {:ok, model} ->
            predictions = POSTagger.predict(words, model)
            # Check if first token is tagged as VERB (command/imperative)
            case predictions do
              [{_word, "VERB"} | _] -> true
              _ -> false
            end

          {:error, _} ->
            false
        end
    end
  end

  defp has_modal_verb?(normalized) do
    # Use POS tagger to detect AUX (auxiliary/modal verbs)
    words = String.split(normalized)

    case POSTagger.load_model() do
      {:ok, model} ->
        predictions = POSTagger.predict(words, model)
        Enum.any?(predictions, fn {_word, tag} -> tag == "AUX" end)

      {:error, _} ->
        false
    end
  end

  defp has_continuation_structure?(text, _normalized) do
    # Use only punctuation analysis - no keyword lists
    # No terminal punctuation (. ! ?) suggests incomplete thought
    no_terminal = not Tokenizer.ends_with_terminal_punctuation?(text)

    # Ends with a comma (incomplete sentence)
    ends_with_comma = String.last(String.trim_trailing(text)) == ","

    # Trailing ellipsis suggests more coming
    trailing_ellipsis = Tokenizer.ends_with_ellipsis?(text)

    # Consider it a continuation if punctuation indicates it
    no_terminal and (ends_with_comma or trailing_ellipsis)
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
      # Response optionality types
      :backchannel -> {:expressive, :backchannel}
      :compliment -> {:expressive, :compliment}
      :acknowledgment -> {:expressive, :acknowledgment}
      :continuation -> {:assertive, :continuation}
      _ -> {:assertive, :statement}
    end
  end
end
