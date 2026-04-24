defmodule Brain.Analysis.SpeechActClassifier do
  @moduledoc """
  Classifies the pragmatic function (speech act) of text chunks.

  Based on Searle's taxonomy:
  - Assertives: statements, claims, reports
  - Directives: requests, commands, questions
  - Commissives: promises, offers
  - Expressives: thanks, apologies, greetings
  - Declaratives: performatives

  Uses four cheap analysis voters in pass 1 (this module's `classify/2`):

  1. Structural Analysis - sentence structure (questions, imperatives)
  2. Keyword Analysis - domain-specific keywords
  3. Pragmatic Markers - politeness, urgency, etc.
  4. Memory Analysis - similar prior episodes

  The `:intent_full` model is **not** invoked here. The pipeline runs full
  feature extraction and `MicroClassifiers.classify_vector(:intent_full, ...)`
  in pass 2, then calls `refine_with_intent/3` to splice that intent vote
  back into the speech-act combiner. This keeps the model's runtime feature
  distribution identical to its training distribution.

  Results are combined using confidence-weighted voting.
  """

  alias Brain.Analysis.SpeechActResult
  alias Brain.ML.{POSTagger, Tokenizer}

  alias Brain.Memory.Store
  alias Brain.Memory.Embedder
  require Logger

  @doc """
  Classifies the speech act of the given text using the four cheap voters
  (structural, keyword, pragmatic, memory).

  Returns a `SpeechActResult` whose `:raw_analyses` field carries the per-voter
  analyses so a later call to `refine_with_intent/3` can splice in a
  full-features intent vote without re-running the cheap voters.

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

  @doc """
  Refines a pass-1 `SpeechActResult` with an intent prediction obtained by
  running `Brain.ML.MicroClassifiers.classify_vector(:intent_full, ...)` on a
  full feature vector built from the complete chunk analysis (pass 2).

  Splices the intent vote into the original analyses map (carried on
  `:raw_analyses`) and re-runs `combine_analyses/2`. Returns a new
  `SpeechActResult`.

  No-ops (returns the original result unchanged) when:
  - `intent` is nil or an empty string,
  - `confidence` is non-positive, or
  - the result has no `:raw_analyses` (e.g. it was constructed manually for
    testing rather than via `classify/2`).
  """
  @spec refine_with_intent(SpeechActResult.t(), String.t() | nil, number()) :: SpeechActResult.t()
  def refine_with_intent(%SpeechActResult{} = result, nil, _conf), do: result
  def refine_with_intent(%SpeechActResult{} = result, "", _conf), do: result
  def refine_with_intent(%SpeechActResult{raw_analyses: nil} = result, _intent, _conf), do: result

  def refine_with_intent(%SpeechActResult{raw_analyses: analyses} = result, intent, confidence)
      when is_binary(intent) and is_number(confidence) do
    if confidence <= 0 do
      result
    else
      norm = normalize_text(Map.get(analyses, :original_text, ""))

      if feature_intent_conflicts_strong_lexicon?(norm, intent) do
        result
      else
        conf = confidence / 1
        {category, sub_type} = intent_to_speech_act(intent)

        intent_analysis = %{
          intent: intent,
          category: category,
          sub_type: sub_type,
          confidence: conf,
          second_score: 0.0,
          margin: conf,
          top_k: [%{intent: intent, score: conf}],
          source: :feature_vector
        }

        updated_analyses = Map.put(analyses, :intent, intent_analysis)
        combine_analyses(updated_analyses, Map.get(analyses, :original_text, ""))
      end
    end
  end

  def refine_with_intent(%SpeechActResult{} = result, _intent, _conf), do: result

  defp run_all_analyses(text, _entities) do
    normalized = normalize_text(text)

    %{
      original_text: text,
      intent: empty_intent_result(),
      structural: analyze_structure(text, normalized),
      keyword: analyze_keywords(normalized),
      pragmatic: analyze_pragmatics(text, normalized),
      memory: analyze_with_memory(text)
    }
  end

  defp empty_intent_result do
    %{
      intent: nil,
      category: nil,
      sub_type: nil,
      confidence: 0.0,
      second_score: 0.0,
      margin: 0.0,
      top_k: [],
      source: :feature_vector
    }
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

  defp tag_to_speech_act(_tag) do
    {:assertive, :statement}
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
      intent_confidence: analyses.intent[:confidence],
      raw_analyses: analyses
    )
  end

  defp collect_votes(analyses) do
    normalized =
      analyses
      |> Map.get(:original_text, "")
      |> normalize_text()

    votes =
      case lexicon_vote(normalized) do
        nil -> []
        vote -> [vote]
      end

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

  # High-precision surface cues that must not be overridden by a coarse TF-IDF
  # centroid (trained on dialogue acts as strings) when that model misfires on
  # short imperatives, greetings, or farewells.
  # Do not let pass-2 `intent_full` overwrite clear greeting/farewell surface cues
  # (short utterances where the centroid model often misfires).
  defp feature_intent_conflicts_strong_lexicon?(norm, intent) when is_binary(norm) and is_binary(intent) do
    domain =
      case String.split(intent, ".", parts: 2) do
        [d, _] -> d
        _ -> intent
      end

    expressive_domains = ~w(smalltalk greeting)

    (lexicon_surface_greeting?(norm) or lexicon_surface_farewell?(norm)) and
      domain not in expressive_domains
  end

  defp feature_intent_conflicts_strong_lexicon?(_, _), do: false

  defp lexicon_first_clause(norm) when is_binary(norm) do
    norm
    |> String.split("?", parts: 2)
    |> hd()
    |> String.trim()
  end

  defp lexicon_surface_greeting?(norm) when is_binary(norm) do
    fc = lexicon_first_clause(norm)
    String.length(fc) < 80 and Regex.match?(~r/^(hello|hi|hey)\b/, fc)
  end

  defp lexicon_surface_farewell?(norm) when is_binary(norm) do
    String.length(norm) < 120 and
      Regex.match?(
        ~r/\b(bye|goodbye|see you|see ya|farewell|cya)\b/,
        norm
      )
  end

  defp lexicon_vote(norm) when is_binary(norm) do
    t = String.trim(norm)
    if t == "", do: nil, else: lexicon_vote_clauses(t)
  end

  defp lexicon_vote_clauses(t) do
    first_clause =
      t
      |> String.split("?", parts: 2)
      |> hd()
      |> String.trim()

    cond do
      Regex.match?(~r/^play\b/, t) ->
        {:directive, :command, 0.9, :lexicon}

      Regex.match?(~r/^turn on\b|^set (a )?reminder\b|^remind me\b|^schedule\b/, t) ->
        {:directive, :command, 0.88, :lexicon}

      Regex.match?(~r/^find me\b.+\b(music|jazz|songs?|tracks?)\b/, t) ->
        {:directive, :request_action, 0.85, :lexicon}

      String.length(first_clause) < 80 and Regex.match?(~r/^(hello|hi|hey)\b/, first_clause) ->
        {:expressive, :greeting, 0.86, :lexicon}

      String.length(t) < 120 and
          Regex.match?(
            ~r/\b(bye|goodbye|see you|see ya|farewell|cya)\b/,
            t
          ) ->
        {:expressive, :farewell, 0.87, :lexicon}

      true ->
        nil
    end
  end

  defp apply_weights(votes) do
    source_weights = %{
      lexicon: 1.45,
      model: 1.5,
      memory: 1.4,
      keyword: 1.05,
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
        kw_st = Map.get(analyses.keyword, :sub_type, :none)
        ["keyword:#{kw_st}" | indicators]
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

  defp intent_to_speech_act(intent) when is_binary(intent) do
    domain =
      case String.split(intent, ".", parts: 2) do
        [d, _] -> d
        _ -> intent
      end

    cond do
      domain in ~w(greeting farewell thanks apology compliment) -> {:expressive, String.to_atom(domain)}
      domain in ~w(question request query search) -> {:directive, :request_information}
      domain in ~w(command action set turn) -> {:directive, :command}
      domain in ~w(promise offer) -> {:commissive, :offer}
      true -> {:assertive, :statement}
    end
  end

  defp intent_to_speech_act(_), do: {:assertive, :statement}

end
