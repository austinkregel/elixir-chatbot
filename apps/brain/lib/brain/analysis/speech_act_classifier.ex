defmodule Brain.Analysis.SpeechActClassifier do
  @moduledoc """
  Classifies the pragmatic function (speech act) of text chunks.

  Based on Searle's taxonomy:
  - Assertives: statements, claims, reports
  - Directives: requests, commands, questions
  - Commissives: promises, offers
  - Expressives: thanks, apologies, greetings
  - Declaratives: performatives

  Uses five cheap analysis voters in pass 1 (this module's `classify/2`):

  1. Structural Analysis - sentence structure (questions, imperatives)
  2. Keyword Analysis - domain-specific keywords
  3. Pragmatic Markers - politeness, urgency, etc.
  4. Memory Analysis - similar prior episodes
  5. Lexical Evidence - what the words say about the speech act, learned from
     the intent training rows (`Brain.Analysis.SpeechActTable`'s `evidence`):
     a one-word "Hello" is read as a greeting from every greeting intent's
     utterances. It votes when the words are positively associated with a
     speech act, with that association's strength.

  The structural voter's exclamation and the keyword voter measure a category
  only, so they vote for it without a sub-type; the winning category's
  sub-type is that of its strongest vote naming one, and `:general` when no
  vote does.

  The `:intent_full` model is **not** invoked here. The pipeline runs full
  feature extraction and `MicroClassifiers.classify_vector(:intent_full, ...)`
  in pass 2, then calls `refine_with_intent/3` to splice that intent vote
  back into the speech-act combiner. This keeps the model's runtime feature
  distribution identical to its training distribution. The intent's speech
  act comes from `Brain.Analysis.SpeechActTable`.

  Results are combined using confidence-weighted voting.
  """

  alias Brain.Analysis.{LexicalEvidence, SpeechActResult, SpeechActTable}
  alias Brain.Lattice
  alias Brain.Lattice.Candidate, as: LatticeCandidate
  alias Brain.ML.{POSTagger, Tokenizer}

  alias Brain.Memory.Store
  require Logger

  @doc """
  Classifies the speech act of the given text using the four cheap voters
  (structural, keyword, pragmatic, memory).

  Returns a `SpeechActResult` whose `:raw_analyses` field carries the per-voter
  analyses so a later call to `refine_with_intent/3` can splice in a
  full-features intent vote without re-running the cheap voters.

  ## Options
    - `:world_id` - Required. The world whose episodic memory the memory voter reads.
    - `:entities` - Pre-extracted entities to use for arbitration (default: [])
    - `:language` - the `Brain.Language` code whose speech-act evidence the
      lexical voter reads. Absent means English; an unsupported code raises.
    - `:speech_act_evidence` - a `Brain.Analysis.LexicalEvidence` model used
      in place of the loaded table's `evidence`. Training passes each row's
      out-of-fold model (`Brain.Analysis.SpeechActTable.fold_evidence/3`).
  """
  def classify(text, opts) when is_binary(text) do
    entities = Keyword.get(opts, :entities, [])
    analyses = run_all_analyses(text, entities, world_id!(opts, :classify), evidence!(opts))
    combine_analyses(analyses, text)
  end

  @doc "Returns detailed analysis with all individual scores for debugging/learning.\n\nTakes the same options as `classify/2`; `:world_id` is required.\n"
  def analyze(text, opts) when is_binary(text) do
    entities = Keyword.get(opts, :entities, [])
    analyses = run_all_analyses(text, entities, world_id!(opts, :analyze), evidence!(opts))
    result = combine_analyses(analyses, text)

    %{
      result: result,
      intent_classification: analyses.intent,
      structural_analysis: analyses.structural,
      keyword_analysis: analyses.keyword,
      pragmatic_analysis: analyses.pragmatic,
      memory_analysis: analyses.memory,
      lexical_analysis: analyses.lexical
    }
  end

  defp evidence!(opts) do
    case Keyword.fetch(opts, :speech_act_evidence) do
      {:ok, %LexicalEvidence{} = evidence} ->
        evidence

      {:ok, other} ->
        raise ArgumentError,
              "SpeechActClassifier: :speech_act_evidence must be a LexicalEvidence model, got #{inspect(other, limit: 3)}"

      :error ->
        language =
          case Keyword.fetch(opts, :language) do
            {:ok, code} -> Brain.Language.validate!(code)
            :error -> Brain.Language.resolve!(nil)
          end

        SpeechActTable.loaded!(language).evidence
    end
  end

  @doc """
  Refines a pass-1 `SpeechActResult` with the `:intent_full` lattice the
  pipeline classified from the chunk's full feature vector (pass 2).

  The best intent's speech act is read from `table`, and that vote is spliced
  into the original analyses map (carried on `:raw_analyses`) before
  `combine_analyses/2` re-runs. Returns a new `SpeechActResult`. An intent the
  table does not hold raises.

  Returns the original result unchanged when the lattice is empty or its best
  candidate has no positive confidence.
  """
  @spec refine_with_intent(SpeechActResult.t(), Lattice.t(), SpeechActTable.t()) :: SpeechActResult.t()
  def refine_with_intent(%SpeechActResult{} = result, %Lattice{} = lattice, %SpeechActTable{} = table) do
    if Lattice.empty?(lattice), do: result, else: lattice_best_refinement(result, lattice, table)
  end

  defp lattice_best_refinement(%SpeechActResult{} = result, %Lattice{} = lattice, table) do
    case Lattice.best(lattice) do
      nil ->
        result

      %LatticeCandidate{confidence: c} when c <= 0 ->
        result

      %LatticeCandidate{label: intent, confidence: conf} ->
        intent_str = if is_binary(intent), do: intent, else: to_string(intent)
        refine_with_intent_from_lattice(result, intent_str, conf * 1.0, lattice, table)
    end
  end

  defp refine_with_intent_from_lattice(
         %SpeechActResult{raw_analyses: analyses},
         intent,
         confidence,
         %Lattice{} = lattice,
         table
       )
       when is_binary(intent) and is_number(confidence) do
    {category, sub_type} = SpeechActTable.lookup!(table, intent)
    second = Lattice.second(lattice)

    second_score =
      if second, do: second.confidence * 1.0, else: 0.0

    margin = Lattice.margin(lattice)

    top_k =
      Lattice.to_top_k(lattice)
      |> Enum.map(fn %{intent: i, score: s} -> %{intent: i, score: s} end)

    intent_analysis = %{
      intent: intent,
      category: category,
      sub_type: sub_type,
      confidence: confidence,
      second_score: second_score,
      margin: margin,
      top_k: top_k,
      source: :feature_vector
    }

    updated_analyses = Map.put(analyses, :intent, intent_analysis)
    combine_analyses(updated_analyses, Map.get(analyses, :original_text, ""))
  end

  defp world_id!(opts, function) do
    case Keyword.get(opts, :world_id) do
      world_id when is_binary(world_id) and world_id != "" ->
        world_id

      other ->
        raise ArgumentError,
              "SpeechActClassifier.#{function} requires a :world_id option, the world whose " <>
                "memory the memory voter reads; got #{inspect(other)}"
    end
  end

  defp run_all_analyses(text, _entities, world_id, evidence) do
    normalized = normalize_text(text)

    %{
      original_text: text,
      intent: empty_intent_result(),
      structural: analyze_structure(text, normalized),
      keyword: analyze_keywords(normalized),
      pragmatic: analyze_pragmatics(text, normalized),
      memory: analyze_with_memory(text, world_id),
      lexical: analyze_lexical(text, evidence)
    }
  end

  defp analyze_lexical(text, evidence) do
    case SpeechActTable.read(evidence, text) do
      {:ok, {category, sub_type}, value} ->
        %{category: category, sub_type: sub_type, confidence: value, source: :lexical}

      :none ->
        %{category: nil, sub_type: nil, confidence: 0.0, source: :lexical}
    end
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

        # An exclamation says the act is expressive, not which expressive.
        is_exclamatory ->
          {:expressive, nil, 0.5}

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
      # The keyword classifier is trained on categories, so it names no sub-type.
      {:ok, %{label: label, confidence: confidence}} when confidence > 0.2 ->
        %{
          scores: %{label => confidence},
          category: label,
          sub_type: nil,
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

  @please_tokens MapSet.new(~w(please pls plz kindly))
  @thanks_tokens MapSet.new(~w(thanks thank thankyou thx))
  @urgency_tokens MapSet.new(~w(urgent asap immediately emergency now quickly hurry))
  @hedge_tokens MapSet.new(~w(maybe perhaps possibly might could probably somewhat))
  @backchannel_tokens MapSet.new(~w(ok okay sure yeah yes no nah right hmm mhm uh huh yep nope))
  @compliment_tokens MapSet.new(~w(great awesome amazing wonderful excellent nice good brilliant fantastic))
  @ack_tokens MapSet.new(~w(ok okay got understood acknowledged roger noted))
  @commissive_tokens MapSet.new(~w(will shall promise guarantee commit pledge vow ensure))
  @offer_tokens MapSet.new(~w(offer help assist volunteer provide lend))
  @future_aux MapSet.new(~w(will shall gonna going))
  @conditional_tokens MapSet.new(~w(if would could should might))

  defp analyze_pragmatics(_text, normalized) do
    tokens = Tokenizer.tokenize(normalized)
    word_count = length(tokens)
    is_short = word_count <= 3
    is_very_short = word_count <= 2
    token_set = MapSet.new(tokens)

    has_please = not MapSet.disjoint?(token_set, @please_tokens)
    has_thanks = not MapSet.disjoint?(token_set, @thanks_tokens)
    has_urgency = not MapSet.disjoint?(token_set, @urgency_tokens)
    has_hedging = not MapSet.disjoint?(token_set, @hedge_tokens)
    is_backchannel = is_very_short and not MapSet.disjoint?(token_set, @backchannel_tokens)
    is_compliment = not MapSet.disjoint?(token_set, @compliment_tokens)
    has_acknowledgment = is_short and not MapSet.disjoint?(token_set, @ack_tokens)

    has_commissive = not MapSet.disjoint?(token_set, @commissive_tokens)
    has_offer = not MapSet.disjoint?(token_set, @offer_tokens)
    has_future = not MapSet.disjoint?(token_set, @future_aux)
    has_conditional = not MapSet.disjoint?(token_set, @conditional_tokens)

    first_person_subject =
      case tokens do
        ["i" | _] -> true
        ["we" | _] -> true
        ["i'll" | _] -> true
        ["we'll" | _] -> true
        ["i'm" | _] -> true
        _ -> false
      end

    {pragmatic_sub_type, expressive_score} =
      cond do
        has_commissive and first_person_subject ->
          {:promise, 0.0}

        has_offer and first_person_subject ->
          {:offer, 0.0}

        has_future and first_person_subject and not has_hedging ->
          {:commitment, 0.0}

        has_thanks ->
          {:thanks, 0.7}

        is_compliment ->
          {:compliment, 0.6}

        is_backchannel ->
          {:backchannel, 0.3}

        has_acknowledgment ->
          {:acknowledgment, 0.2}

        has_hedging and has_conditional ->
          {:hedged_intent, 0.0}

        true ->
          {nil, 0.0}
      end

    %{
      has_please: has_please,
      has_thanks: has_thanks,
      has_urgency: has_urgency,
      has_hedging: has_hedging,
      is_short_utterance: is_short,
      is_very_short: is_very_short,
      is_backchannel: is_backchannel,
      is_compliment: is_compliment,
      has_acknowledgment: has_acknowledgment,
      has_commissive: has_commissive,
      has_offer: has_offer,
      has_future: has_future,
      first_person_subject: first_person_subject,
      pragmatic_sub_type: pragmatic_sub_type,
      expressive_score: expressive_score,
      source: :pragmatic
    }
  end

  # A world whose memory holds nothing like the text gives no vote; that is
  # an answer, not a failure.
  defp analyze_with_memory(text, world_id) do
    case query_memory_for_classification(text, world_id) do
      [_ | _] = results ->
        {category, sub_type, confidence} = vote_on_memory_results(results)

        %{
          category: category,
          sub_type: sub_type,
          confidence: confidence,
          similar_count: length(results),
          source: :memory
        }

      [] ->
        %{
          category: nil,
          sub_type: nil,
          confidence: 0.0,
          similar_count: 0,
          source: :memory
        }
    end
  end

  # The query is embedded by the world's embedder, inside `Store.query_similar/3`.
  # It runs in the caller and is not cut short: while the world's vocabulary is
  # being rebuilt it waits for the rebuild, whose own bound raises
  # (`Brain.Worlds.Embedder.await_ready/2`). A query that fails raises here.
  defp query_memory_for_classification(text, world_id) do
    if Process.whereis(Brain.Memory.Store) == nil do
      raise "SpeechActClassifier: the memory voter needs Brain.Memory.Store, which is not running"
    end

    case Store.query_similar(text, 5, world_id: world_id, rerank: false) do
      {:ok, results} ->
        results

      {:error, reason} ->
        raise "SpeechActClassifier: the memory query in world #{inspect(world_id)} failed: #{inspect(reason)}"
    end
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

  @tag_speech_act_map %{
    "question" => {:directive, :question},
    "request" => {:directive, :request},
    "command" => {:directive, :command},
    "instruction" => {:directive, :instruction},
    "greeting" => {:expressive, :greeting},
    "farewell" => {:expressive, :farewell},
    "thanks" => {:expressive, :thanks},
    "apology" => {:expressive, :apology},
    "compliment" => {:expressive, :compliment},
    "congratulation" => {:expressive, :congratulation},
    "promise" => {:commissive, :promise},
    "offer" => {:commissive, :offer},
    "commitment" => {:commissive, :commitment},
    "threat" => {:commissive, :threat},
    "agreement" => {:commissive, :agreement},
    "declaration" => {:declarative, :declaration},
    "announcement" => {:declarative, :announcement},
    "naming" => {:declarative, :naming},
    "statement" => {:assertive, :statement},
    "claim" => {:assertive, :claim},
    "report" => {:assertive, :report},
    "opinion" => {:assertive, :opinion},
    "acknowledgment" => {:expressive, :acknowledgment}
  }

  defp tag_to_speech_act(tag) when is_atom(tag) do
    tag_to_speech_act(to_string(tag))
  end

  defp tag_to_speech_act(tag) when is_binary(tag) do
    Map.get(@tag_speech_act_map, tag, {:assertive, :statement})
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
    votes =
      if analyses.intent.confidence > 0.2 and analyses.intent.category != nil do
        [{analyses.intent.category, analyses.intent.sub_type, analyses.intent.confidence, :model}]
      else
        []
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
      cond do
        Map.get(analyses.pragmatic, :has_commissive, false) and
            Map.get(analyses.pragmatic, :first_person_subject, false) ->
          sub_type = analyses.pragmatic.pragmatic_sub_type || :promise
          vote = {:commissive, sub_type, 0.65, :pragmatic}
          [vote | votes]

        Map.get(analyses.pragmatic, :has_offer, false) and
            Map.get(analyses.pragmatic, :first_person_subject, false) ->
          sub_type = analyses.pragmatic.pragmatic_sub_type || :offer
          vote = {:commissive, sub_type, 0.55, :pragmatic}
          [vote | votes]

        Map.get(analyses.pragmatic, :has_future, false) and
            Map.get(analyses.pragmatic, :first_person_subject, false) and
            not Map.get(analyses.pragmatic, :has_hedging, false) ->
          vote = {:commissive, :commitment, 0.45, :pragmatic}
          [vote | votes]

        true ->
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

    # The words' evidence votes whenever it is positive, with its strength.
    votes =
      if analyses.lexical.category != nil do
        [{analyses.lexical.category, analyses.lexical.sub_type, analyses.lexical.confidence, :lexical} | votes]
      else
        votes
      end

    votes
  end

  defp apply_weights(votes) do
    # The lexical evidence's strength is taken as it is, unweighted.
    source_weights = %{
      model: 1.5,
      memory: 1.4,
      keyword: 1.05,
      structural: 1.0,
      lexical: 1.0,
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

      # The sub-type of the category's strongest vote that names one; a vote
      # that measures only the category names none.
      winning_sub_type =
        case Enum.filter(category_votes, fn {_, sub_type, _, _} -> sub_type != nil end) do
          [] -> :general
          naming -> naming |> Enum.max_by(fn {_, _, conf, _} -> conf end) |> elem(1)
        end

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
        ["keyword:#{analyses.keyword.category}" | indicators]
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

  @imperative_verbs_cache :speech_act_imperative_verbs

  # The seeds plus the synonyms sharing each seed's dominant verb sense.
  #
  # The criterion is the sense, not a count. `Brain.Lexicon.synonyms/3` flattens
  # every sense of a polysemous verb together, so expanding through it reaches
  # `check` -> `curb`, `chit`, `tick off` as readily as `verify`. Measured against
  # the speech act corpus's directive labels, taking every synonym of every sense
  # costs 9.6 points of precision (68.9% -> 59.3%) to gain 2.7 of recall.
  #
  # Memoised: the seeds are a compile-time constant and WordNet does not change
  # between `mix atlas.seed` runs, but this is read once per utterance.
  defp expanded_imperative_verbs do
    table = imperative_verbs_table()

    case :ets.lookup(table, :set) do
      [{:set, set}] ->
        set

      [] ->
        set =
          @seed_imperative_verbs
          |> Enum.flat_map(fn verb ->
            [verb | Brain.Lexicon.dominant_sense_synonyms(verb, :verb)]
          end)
          |> MapSet.new()

        :ets.insert(table, {:set, set})
        set
    end
  end

  defp imperative_verbs_table do
    :ets.new(@imperative_verbs_cache, [:set, :public, :named_table, read_concurrency: true])
  rescue
    ArgumentError -> @imperative_verbs_cache
  end

  defp has_continuation_structure?(text, _normalized) do
    no_terminal = not Tokenizer.ends_with_terminal_punctuation?(text)
    ends_with_comma = String.last(String.trim_trailing(text)) == ","
    trailing_ellipsis = Tokenizer.ends_with_ellipsis?(text)
    no_terminal and (ends_with_comma or trailing_ellipsis)
  end
end
