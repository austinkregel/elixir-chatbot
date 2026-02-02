defmodule Brain.Analysis.RacingAnalyzer do
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

  alias Brain.Analysis.{
    Interpretation,
    AnalyzerResult,
    ActivationPool,
    AnalyzerCalibration,
    HeuristicStore,
    SelfKnowledgeAnalyzer,
    Progress
  }

  alias Brain.ML.{IntentClassifierSimple, Tokenizer}
  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.Telemetry

  require Logger

  @early_exit_threshold 0.90
  @fast_path_threshold 0.85
  @analyzer_timeout 2000
  @default_pattern_triggers_file "data/pattern_triggers.json"

  # Cache for loaded pattern triggers (loaded once per process)
  @pattern_triggers_key :racing_analyzer_pattern_triggers

  # Get pattern triggers file path from config or use default
  defp pattern_triggers_file do
    Application.get_env(:brain, :pattern_triggers_file, @default_pattern_triggers_file)
  end

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
    Telemetry.span(:racing_analysis, %{text_length: String.length(text)}, fn ->
      do_race(text, opts)
    end)
  end

  defp do_race(text, opts) do
    start_time = System.monotonic_time(:millisecond)
    world_id = Keyword.get(opts, :world_id, "default")
    user_id = Keyword.get(opts, :user_id)
    cohort_id = Keyword.get(opts, :cohort_id)

    # Step 1: Check fast path (heuristics)
    unless Keyword.get(opts, :skip_heuristics, false) do
      case check_fast_path(text, world_id, user_id, cohort_id) do
        {:fast_path, interpretation} ->
          elapsed = System.monotonic_time(:millisecond) - start_time

          Logger.debug("Fast path hit", %{
            intent: interpretation.intent,
            source: interpretation.source,
            elapsed_ms: elapsed
          })

          # Emit early exit telemetry for fast path
          Telemetry.emit_racing_early_exit(interpretation.source, interpretation.activation, elapsed)

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

    # Emit early exit telemetry if triggered
    if early_exit_triggered do
      winner = Enum.find(corrected_results, &(&1.raw_score >= @early_exit_threshold))

      if winner do
        Telemetry.emit_racing_early_exit(winner.analyzer, winner.raw_score, elapsed)
      end
    end

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

  Requires world_id for proper world isolation of learned heuristics.
  Returns {:fast_path, interpretation} or :no_match
  """
  def check_fast_path(text, world_id, user_id, cohort_id) do
    # Check heuristics first (fastest)
    case check_heuristics(text, world_id, user_id, cohort_id) do
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

  # Backward compatibility - will be removed
  def check_fast_path(text, user_id, cohort_id) do
    Logger.warning("check_fast_path/3 is deprecated, use check_fast_path/4 with world_id")
    check_fast_path(text, "default", user_id, cohort_id)
  end

  # Private functions

  defp check_heuristics(text, world_id, user_id, cohort_id) do
    if Process.whereis(HeuristicStore) do
      HeuristicStore.match_best(text, world_id, user_id, cohort_id)
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
    is_question = Tokenizer.ends_with_question?(text)
    words = Tokenizer.tokenize_normalized(text)
    first_word = List.first(words) || ""
    text_lower = String.downcase(text)

    question_words = ~w(what where when why who whom whose which how)

    imperative_words =
      ~w(tell show give get find search look check turn set make create open close start stop play pause)

    # Personal/conversational question patterns - not factual
    personal_patterns = [
      "your name",
      "you called",
      "are you",
      "do you",
      "can you",
      "will you",
      "would you",
      "how are",
      "how do you",
      "what do you",
      "who are you",
      "what are you",
      "where are you from"
    ]

    # Check if this is a personal/conversational question
    is_personal = Enum.any?(personal_patterns, &String.contains?(text_lower, &1))

    {intent, confidence, indicators} =
      cond do
        # Personal questions about the bot itself
        is_question and is_personal ->
          {"question.personal", 0.70, ["question_mark", "personal_pattern"]}

        # WH-questions about external facts (not personal)
        is_question and first_word in question_words ->
          {"question.factual", 0.65, ["question_mark", "wh_word"]}

        is_question ->
          {"question.general", 0.60, ["question_mark"]}

        first_word in imperative_words ->
          {"command.general", 0.70, ["imperative_verb"]}

        length(words) <= 3 and first_word in ~w(hi hello hey) ->
          {"smalltalk.greetings.hello", 0.80, ["short_utterance", "greeting_word"]}

        true ->
          {"statement.general", 0.40, ["declarative"]}
      end

    AnalyzerResult.new(:structural, intent, confidence,
      confidence_estimate: confidence,
      indicators: indicators
    )
  end

  defp analyze_keywords(_text) do
    # Keyword-based analysis disabled - use model classification instead.
    # Keyword matching bypasses the classification system and creates
    # tight coupling to specific intent naming conventions.
    # If intent detection is inaccurate, improve training data rather than
    # adding keyword heuristics.
    AnalyzerResult.new(:keyword, nil, 0.0)
  end

  defp analyze_patterns(text) do
    # Load token patterns from data file (cached)
    patterns = get_token_patterns()
    tokens = Brain.ML.Tokenizer.tokenize_normalized(text, expand_contractions: true)

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
    triggers_file = pattern_triggers_file()
    paths_to_try = [
      triggers_file,
      Path.join(File.cwd!(), triggers_file)
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
        handle_missing_file("Pattern triggers file not found", triggers_file)
        %{keywords: [], patterns: []}
    end
  end

  defp handle_missing_file(message, path) do
    if Application.get_env(:brain, :strict_file_checks, false) do
      raise "#{message}: #{path}"
    else
      Logger.warning(message)
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
  # Intent safeguards removed - the classifier should be the source of truth.
  # If misclassifications occur (e.g., "Hello" as music.play due to Adele's song),
  # the solution is to improve training data, not add keyword heuristics.
  defp apply_intent_safeguards(_text, results), do: results
end
