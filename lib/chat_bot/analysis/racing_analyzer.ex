defmodule ChatBot.Analysis.RacingAnalyzer do
  @moduledoc """
  Runs multiple interpretation paths in parallel with early-exit.

  Instead of sequential analysis stages, this module:
  1. Launches multiple analyzers concurrently
  2. Monitors for early threshold crossing
  3. First to reach confidence threshold "wins"
  4. Runner-ups are kept warm as alternatives

  Supports three fast-path triggers:
  - Memory match (high similarity to past interaction)
  - Confidence threshold (single analyzer hits 90%+)
  - Pattern recognition (structural features reliably indicate intent)

  Pattern and keyword triggers are loaded from data/pattern_triggers.json
  to keep intent recognition data-driven and trainable.
  """

  alias ChatBot.Analysis.{
    Interpretation,
    AnalyzerResult,
    ActivationPool,
    AnalyzerCalibration,
    HeuristicStore,
    SelfKnowledgeAnalyzer,
    Progress
  }

  alias ChatBot.ML.IntentClassifierSimple
  alias ChatBot.Memory.Store, as: MemoryStore

  require Logger

  @early_exit_threshold 0.90
  @fast_path_threshold 0.85
  @analyzer_timeout 2000
  @pattern_triggers_file "data/pattern_triggers.json"

  # Cache for loaded pattern triggers (loaded once per process)
  @pattern_triggers_key :racing_analyzer_pattern_triggers

  @doc """
  Races multiple analyzers to interpret the input.

  Returns an Interpretation with the winning intent and alternatives.

  Options:
  - :user_id - User ID for user-scoped heuristics
  - :cohort_id - Cohort ID for cohort-scoped heuristics
  - :skip_heuristics - Skip heuristic fast path (for testing)
  - :skip_memory - Skip memory similarity check
  """
  def race(text, opts \\ []) when is_binary(text) do
    start_time = System.monotonic_time(:millisecond)
    user_id = Keyword.get(opts, :user_id)
    cohort_id = Keyword.get(opts, :cohort_id)

    # Step 1: Check fast path (heuristics)
    unless Keyword.get(opts, :skip_heuristics, false) do
      case check_fast_path(text, user_id, cohort_id) do
        {:fast_path, interpretation} ->
          elapsed = System.monotonic_time(:millisecond) - start_time

          Logger.debug("Fast path hit", %{
            intent: interpretation.intent,
            source: interpretation.source,
            elapsed_ms: elapsed
          })

          # Report fast path hit to debug inspector
          Progress.report(opts, :racing_complete, %{
            fast_path: true,
            fast_path_source: interpretation.source,
            intent: interpretation.intent,
            activation: interpretation.activation,
            elapsed_ms: elapsed,
            results: [],
            early_exit: false
          })

          return_with_timing(interpretation, start_time)

        :no_match ->
          :continue
      end
    end

    # Step 2: Launch racing analyzers
    {results, early_exit_triggered} = run_analyzers_with_early_exit(text, opts)

    # Step 3: Calibrate and normalize results
    calibrated_results = calibrate_results(results)

    # Step 4: Apply safeguards against common misclassifications
    corrected_results = apply_intent_safeguards(text, calibrated_results)

    # Step 5: Build interpretation from results
    interpretation =
      text
      |> Interpretation.from_analyzer_results(corrected_results)
      |> ActivationPool.normalize_with_alternatives()

    elapsed = System.monotonic_time(:millisecond) - start_time

    Logger.debug("Racing complete", %{
      intent: interpretation.intent,
      activation: interpretation.activation,
      alternatives: length(interpretation.alternatives),
      elapsed_ms: elapsed
    })

    # Report racing results to debug inspector
    Progress.report(opts, :racing_complete, %{
      fast_path: false,
      fast_path_source: nil,
      intent: interpretation.intent,
      activation: interpretation.activation,
      elapsed_ms: elapsed,
      early_exit: early_exit_triggered,
      results:
        Enum.map(corrected_results, fn r ->
          %{
            analyzer: r.analyzer,
            intent: r.intent,
            raw_score: r.raw_score,
            calibrated: r.calibrated_activation,
            indicators: r.indicators || []
          }
        end),
      alternatives:
        Enum.map(interpretation.alternatives || [], fn alt ->
          %{intent: alt.intent, activation: alt.activation, source: alt.source}
        end)
    })

    interpretation
  end

  @doc """
  Checks if any fast path trigger fires.

  Returns {:fast_path, interpretation} or :no_match
  """
  def check_fast_path(text, user_id, cohort_id) do
    # Check heuristics first (fastest)
    case check_heuristics(text, user_id, cohort_id) do
      {:ok, heuristic, confidence} when confidence >= @fast_path_threshold ->
        interpretation =
          Interpretation.new(heuristic.conclusion.intent, text, confidence, :heuristic)
          |> Interpretation.with_heuristic(heuristic.id, heuristic.scope)

        {:fast_path, interpretation}

      _ ->
        # Check memory similarity
        case check_memory_similarity(text) do
          {:ok, intent, confidence} when confidence >= @fast_path_threshold ->
            interpretation = Interpretation.new(intent, text, confidence, :memory_match)
            {:fast_path, interpretation}

          _ ->
            :no_match
        end
    end
  end

  # Private functions

  defp check_heuristics(text, user_id, cohort_id) do
    if Process.whereis(HeuristicStore) do
      HeuristicStore.match_best(text, user_id, cohort_id)
    else
      {:error, :store_not_running}
    end
  rescue
    _ -> {:error, :heuristic_error}
  end

  defp check_memory_similarity(text) do
    if Process.whereis(MemoryStore) do
      case MemoryStore.query_similar(text, 3) do
        {:ok, [{episode, similarity} | _]} when similarity >= @fast_path_threshold ->
          # Extract intent from episode tags
          intent = extract_intent_from_tags(episode.tags)
          {:ok, intent, similarity}

        _ ->
          {:error, :no_match}
      end
    else
      {:error, :store_not_running}
    end
  rescue
    _ -> {:error, :memory_error}
  end

  defp extract_intent_from_tags(tags) do
    # Find the first tag that looks like an intent
    intent_tag =
      Enum.find(tags, fn tag ->
        String.contains?(tag, ".") and not String.starts_with?(tag, "conv_")
      end)

    intent_tag || "unknown"
  end

  defp run_analyzers_with_early_exit(text, opts) do
    user_id = Keyword.get(opts, :user_id)

    # Create tasks for each analyzer
    analyzers = [
      {:model, fn -> analyze_with_model(text) end},
      {:structural, fn -> analyze_structure(text) end},
      {:keyword, fn -> analyze_keywords(text) end},
      {:pattern_recognition, fn -> analyze_patterns(text) end}
    ]

    # Add memory analyzer unless skipped
    analyzers =
      unless Keyword.get(opts, :skip_memory, false) do
        [{:memory_similarity, fn -> analyze_memory(text, opts) end} | analyzers]
      else
        analyzers
      end

    # Add self-knowledge analyzer for meta-cognitive queries (epistemic system)
    analyzers =
      unless Keyword.get(opts, :skip_epistemic, false) do
        [{:self_knowledge, fn -> analyze_self_knowledge(text, user_id) end} | analyzers]
      else
        analyzers
      end

    # Launch all analyzers
    tasks =
      Enum.map(analyzers, fn {name, fun} ->
        {name, Task.async(fun)}
      end)

    # Collect results with early exit - returns {results, early_exit_triggered?}
    collect_with_early_exit(tasks, [], @analyzer_timeout, false)
  end

  defp collect_with_early_exit([], results, _timeout, early_exit_triggered),
    do: {results, early_exit_triggered}

  defp collect_with_early_exit(tasks, results, timeout, early_exit_triggered) do
    # Wait for any task to complete
    case Task.yield_many(tasks |> Enum.map(&elem(&1, 1)), timeout) do
      yielded_results ->
        # Process completed tasks
        {completed, pending} =
          Enum.zip(tasks, yielded_results)
          |> Enum.split_with(fn {_, {_task, result}} -> result != nil end)

        new_results =
          Enum.flat_map(completed, fn {{_name, _}, {_task, {:ok, result}}} ->
            case result do
              %AnalyzerResult{} = r -> [r]
              {:ok, r} -> [r]
              _ -> []
            end
          end)

        all_results = results ++ new_results

        # Check for early exit condition
        if should_early_exit?(all_results) do
          # Kill remaining tasks
          Enum.each(pending, fn {{_, task}, _} -> Task.shutdown(task, :brutal_kill) end)
          {all_results, true}
        else
          # Continue waiting for remaining tasks
          remaining_tasks =
            pending
            |> Enum.map(fn {{name, task}, _} -> {name, task} end)

          collect_with_early_exit(remaining_tasks, all_results, timeout, early_exit_triggered)
        end
    end
  end

  defp should_early_exit?(results) do
    Enum.any?(results, fn r ->
      r.raw_score >= @early_exit_threshold
    end)
  end

  defp calibrate_results(results) do
    Enum.map(results, fn result ->
      if Process.whereis(AnalyzerCalibration) do
        {calibrated, error} = AnalyzerCalibration.calibrate(result.analyzer, result.raw_score)
        AnalyzerResult.with_calibration(result, calibrated, error)
      else
        # No calibration available, use raw score
        result
      end
    end)
  end

  # Individual analyzer implementations

  defp analyze_with_model(text) do
    case IntentClassifierSimple.classify(text) do
      {:ok, %{intent: intent, confidence: confidence}} ->
        AnalyzerResult.new(:model, intent, confidence,
          confidence_estimate: confidence,
          indicators: ["ml_model"]
        )

      {:error, _} ->
        AnalyzerResult.new(:model, nil, 0.0)
    end
  rescue
    _ -> AnalyzerResult.new(:model, nil, 0.0)
  end

  defp analyze_structure(text) do
    # Structural analysis based on sentence form
    is_question = String.ends_with?(String.trim(text), "?")
    words = String.split(String.downcase(text))
    first_word = List.first(words) || ""

    question_words = ~w(what where when why who whom whose which how)

    imperative_words =
      ~w(tell show give get find search look check turn set make create open close start stop play pause)

    {intent, confidence, indicators} =
      cond do
        is_question and first_word in question_words ->
          {"question.factual", 0.75, ["question_mark", "wh_word"]}

        is_question ->
          {"question.general", 0.65, ["question_mark"]}

        first_word in imperative_words ->
          {"command.general", 0.70, ["imperative_verb"]}

        length(words) <= 3 and first_word in ~w(hi hello hey) ->
          {"smalltalk.greeting", 0.80, ["short_utterance", "greeting_word"]}

        true ->
          {"statement.general", 0.40, ["declarative"]}
      end

    AnalyzerResult.new(:structural, intent, confidence,
      confidence_estimate: confidence,
      indicators: indicators
    )
  end

  defp analyze_keywords(text) do
    # Load keyword patterns from data file (cached)
    keyword_patterns = get_keyword_patterns()
    lower = String.downcase(text)

    best_match =
      keyword_patterns
      |> Enum.map(fn {intent, keywords, base_conf} ->
        matches = Enum.count(keywords, &String.contains?(lower, &1))

        if matches > 0 do
          # Boost confidence for multiple matches
          confidence = min(base_conf + matches * 0.05, 0.95)
          {intent, confidence, matches}
        else
          nil
        end
      end)
      |> Enum.reject(&is_nil/1)
      |> Enum.max_by(fn {_, conf, _} -> conf end, fn -> nil end)

    case best_match do
      {intent, confidence, matches} ->
        AnalyzerResult.new(:keyword, intent, confidence,
          confidence_estimate: confidence,
          indicators: ["keyword_match:#{matches}"]
        )

      nil ->
        AnalyzerResult.new(:keyword, nil, 0.0)
    end
  end

  defp analyze_patterns(text) do
    # Load token patterns from data file (cached)
    patterns = get_token_patterns()
    tokens = ChatBot.ML.Tokenizer.tokenize_normalized(text, expand_contractions: true)

    best_match =
      Enum.find_value(patterns, fn {token_sequences, intent, confidence} ->
        if matches_any_token_pattern?(tokens, token_sequences) do
          {intent, confidence}
        else
          nil
        end
      end)

    case best_match do
      {intent, confidence} ->
        AnalyzerResult.new(:pattern_recognition, intent, confidence,
          confidence_estimate: confidence,
          indicators: ["pattern_match"]
        )

      nil ->
        AnalyzerResult.new(:pattern_recognition, nil, 0.0)
    end
  end

  # ============================================================================
  # Pattern Data Loading (from JSON)
  # ============================================================================

  # Get keyword patterns, loading from file if not cached
  defp get_keyword_patterns do
    case Process.get(@pattern_triggers_key) do
      %{keywords: keywords} -> keywords
      nil -> load_and_cache_triggers().keywords
    end
  end

  # Get token patterns, loading from file if not cached
  defp get_token_patterns do
    case Process.get(@pattern_triggers_key) do
      %{patterns: patterns} -> patterns
      nil -> load_and_cache_triggers().patterns
    end
  end

  # Load triggers from JSON file and cache in process dictionary
  defp load_and_cache_triggers do
    triggers = load_pattern_triggers()
    Process.put(@pattern_triggers_key, triggers)
    triggers
  end

  # Load pattern triggers from JSON file
  defp load_pattern_triggers do
    paths_to_try = [
      @pattern_triggers_file,
      Path.join(File.cwd!(), @pattern_triggers_file)
    ]

    result =
      Enum.find_value(paths_to_try, fn path ->
        if File.exists?(path) do
          case File.read(path) do
            {:ok, contents} ->
              case Jason.decode(contents) do
                {:ok, data} -> {:ok, data}
                {:error, _} -> nil
              end

            {:error, _} ->
              nil
          end
        end
      end)

    case result do
      {:ok, data} ->
        %{
          keywords: parse_keyword_patterns(data),
          patterns: parse_token_patterns(data)
        }

      nil ->
        Logger.warning("Pattern triggers file not found, using empty patterns")
        %{keywords: [], patterns: []}
    end
  end

  # Parse keyword patterns from JSON data
  defp parse_keyword_patterns(data) do
    (data["keywords"] || [])
    |> Enum.map(fn entry ->
      {
        entry["intent"],
        entry["keywords"],
        entry["base_confidence"]
      }
    end)
  end

  # Parse token sequence patterns from JSON data
  defp parse_token_patterns(data) do
    (data["patterns"] || [])
    |> Enum.map(fn entry ->
      {
        entry["token_sequences"],
        entry["intent"],
        entry["confidence"]
      }
    end)
  end

  # Check if tokens start with any of the given token patterns
  defp matches_any_token_pattern?(tokens, token_patterns) do
    Enum.any?(token_patterns, fn pattern ->
      starts_with_tokens?(tokens, pattern)
    end)
  end

  # Check if the token list starts with the given pattern
  defp starts_with_tokens?(tokens, pattern) when length(tokens) >= length(pattern) do
    tokens
    |> Enum.take(length(pattern))
    |> Enum.zip(pattern)
    |> Enum.all?(fn {token, expected} -> token == expected end)
  end

  defp starts_with_tokens?(_, _), do: false

  defp analyze_memory(text, opts) do
    if Process.whereis(MemoryStore) do
      case MemoryStore.query_similar(text, 5) do
        {:ok, [_ | _] = results} ->
          # Weight by similarity
          {best_episode, best_similarity} = hd(results)
          intent = extract_intent_from_tags(best_episode.tags)

          # Report memory query results to debug inspector
          Progress.report(opts, :memory_query, %{
            query_text: String.slice(text, 0, 100),
            match_count: length(results),
            top_similarity: best_similarity,
            matches:
              Enum.map(results, fn {ep, sim} ->
                %{
                  episode_id: ep.id,
                  similarity: Float.round(sim, 3),
                  tags: Enum.take(ep.tags, 5),
                  state_preview: String.slice(ep.state || "", 0, 50)
                }
              end)
          })

          AnalyzerResult.new(:memory_similarity, intent, best_similarity,
            confidence_estimate: best_similarity,
            indicators: ["memory_match"],
            metadata: %{episode_id: best_episode.id, match_count: length(results)}
          )

        _ ->
          Progress.report(opts, :memory_query, %{
            query_text: String.slice(text, 0, 100),
            match_count: 0,
            top_similarity: 0.0,
            matches: []
          })

          AnalyzerResult.new(:memory_similarity, nil, 0.0)
      end
    else
      AnalyzerResult.new(:memory_similarity, nil, 0.0)
    end
  rescue
    _ -> AnalyzerResult.new(:memory_similarity, nil, 0.0)
  end

  defp analyze_self_knowledge(text, user_id) do
    # Use the SelfKnowledgeAnalyzer for meta-cognitive queries
    SelfKnowledgeAnalyzer.analyze(text, user_id: user_id)
  rescue
    _ -> AnalyzerResult.new(:self_knowledge, nil, 0.0)
  end

  defp return_with_timing(interpretation, start_time) do
    elapsed = System.monotonic_time(:millisecond) - start_time
    current_metadata = Map.get(interpretation, :metadata) || %{}

    %{interpretation | metadata: Map.put(current_metadata, :racing_time_ms, elapsed)}
  end

  # Safeguards against common misclassifications
  # For example, "Hello" should not be classified as music.play just because
  # Adele's song "Hello" is in the training data
  defp apply_intent_safeguards(text, results) do
    lower = String.downcase(String.trim(text))
    words = String.split(lower)
    first_word = List.first(words) || ""

    # Common greeting words that should NOT be classified as music
    greeting_words = ~w(hello hi hey howdy greetings hiya)

    # Check if this looks like a greeting being misclassified
    is_likely_greeting =
      first_word in greeting_words and
        length(words) <= 6 and
        not String.contains?(lower, "play")

    if is_likely_greeting do
      # Apply corrections: penalize music.play, boost greeting
      Enum.map(results, fn result ->
        cond do
          # If classified as music.play, heavily penalize
          result.intent == "music.play" ->
            %{
              result
              | raw_score: result.raw_score * 0.1,
                calibrated_activation: result.calibrated_activation * 0.1
            }

          # If classified as greeting, boost
          result.intent == "smalltalk.greeting" or
              String.starts_with?(result.intent || "", "smalltalk.greeting") ->
            %{result | raw_score: min(result.raw_score * 1.5, 0.95)}

          # Otherwise keep as is
          true ->
            result
        end
      end)
    else
      # Check if this is a music play request being incorrectly classified as greeting
      # Only apply if text starts with "play" or contains explicit music keywords
      is_likely_music =
        String.starts_with?(lower, "play ") or
          (String.contains?(lower, "play") and
             String.contains?(lower, ["song", "music", "album", "artist"]))

      if is_likely_music do
        # Apply corrections: penalize greeting, boost music.play
        Enum.map(results, fn result ->
          cond do
            result.intent == "smalltalk.greeting" ->
              %{
                result
                | raw_score: result.raw_score * 0.3,
                  calibrated_activation: result.calibrated_activation * 0.3
              }

            result.intent == "music.play" ->
              %{result | raw_score: min(result.raw_score * 1.3, 0.95)}

            true ->
              result
          end
        end)
      else
        results
      end
    end
  end
end
