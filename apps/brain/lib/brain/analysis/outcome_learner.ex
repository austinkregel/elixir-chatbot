defmodule Brain.Analysis.OutcomeLearner do
  @moduledoc "Learns from conversation outcomes to refine heuristics.\n\nAfter each conversation turn, this module:\n1. Assesses whether the response succeeded\n2. Updates heuristic success/failure counts\n3. Creates new heuristics from successful slow-path patterns\n4. Deprecates failing heuristics\n5. Updates analyzer calibration data\n\nHeuristics are scoped appropriately:\n- User-specific patterns stay user-scoped\n- Patterns that work across users may be promoted to cohort/global\n"

  alias Brain.Analysis.{
    Interpretation,
    HeuristicStore,
    AnalyzerCalibration,
    ActivationPool,
    IntentRegistry
  }

  alias Brain.ML.Tokenizer
  require Logger

  @min_successes_for_heuristic %{
    global: 3,
    cohort: 4,
    user: 5
  }
  @pattern_similarity_threshold 0.85
  defmodule PendingPattern do
    defstruct [
      :pattern,
      :conclusion,
      :scope,
      :scope_id,
      success_count: 0,
      failure_count: 0,
      examples: [],
      first_seen: nil,
      last_seen: nil
    ]
  end

  @doc "Records the outcome of a conversation turn for learning.\n\nOptions:\n- :world_id - Required. Training world ID for world-scoped heuristics\n- :user_id - User ID for user-scoped learning\n- :cohort_id - Cohort ID for cohort-scoped learning\n- :user_feedback - Explicit feedback (:positive, :negative, :correction, nil)\n"
  def learn_from_outcome(%Interpretation{} = interp, response, opts \\ []) do
    world_id = Keyword.get(opts, :world_id)
    user_id = Keyword.get(opts, :user_id)
    cohort_id = Keyword.get(opts, :cohort_id)
    user_feedback = Keyword.get(opts, :user_feedback)

    if is_nil(world_id) do
      Logger.warning("OutcomeLearner.learn_from_outcome called without world_id")
    end

    outcome = assess_outcome(interp, response, user_feedback)

    Logger.debug("Learning from outcome", %{
      intent: interp.intent,
      outcome: outcome,
      world_id: world_id,
      from_heuristic: Interpretation.from_heuristic?(interp)
    })

    if Interpretation.from_heuristic?(interp) do
      update_heuristic_outcome(interp.triggering_heuristic_id, outcome)
    end

    update_analyzer_calibration(interp, outcome)

    if outcome == :success and not Interpretation.from_heuristic?(interp) do
      maybe_create_heuristic(interp, world_id, user_id, cohort_id)
    end

    track_pattern(interp, outcome, user_id, cohort_id)

    # Generate training example from high-confidence confirmed outcomes
    if outcome in [:success, :likely_success] and interp.activation >= 0.8 and
         interp.intent != nil and interp.text != nil do
      maybe_buffer_training_example(interp.text, interp.intent)
    end

    outcome
  end

  @doc "Assesses whether a response was successful.\n\nUses multiple signals:\n- Explicit user feedback\n- Implicit signals (user continued conversation, didn't correct)\n- Task completion indicators\n"
  def assess_outcome(%Interpretation{} = interp, response, user_feedback) do
    cond do
      user_feedback == :positive ->
        :success

      user_feedback == :negative ->
        :failure

      user_feedback == :correction ->
        :failure

      is_clarification_response?(response) ->
        :clarification

      interp.activation >= 0.8 and not Interpretation.has_missing_required?(interp) ->
        :likely_success

      interp.activation < 0.4 ->
        :likely_failure

      true ->
        :uncertain
    end
  end

  @doc "Extracts a learnable pattern from an interpretation.\n"
  def extract_pattern(%Interpretation{} = interp) do
    text = interp.text
    lower = String.downcase(text)
    words = String.split(lower)
    first_word = List.first(words) || ""
    word_count = length(words)

    pattern =
      %{}
      |> maybe_add_first_word(first_word, interp.source)
      |> maybe_add_keywords(lower, interp.intent)
      |> maybe_add_word_count(word_count)
      |> maybe_add_phrase(lower, interp.intent)

    conclusion = %{
      intent: interp.intent,
      confidence_boost: calculate_appropriate_boost(interp)
    }

    {pattern, conclusion}
  end

  @doc "Determines the appropriate scope for a new heuristic.\n"
  def determine_scope(pattern, _user_id, _cohort_id) do
    cond do
      Map.has_key?(pattern, :phrase) ->
        :user

      Map.has_key?(pattern, :keywords) and length(pattern.keywords) >= 2 ->
        :cohort

      Map.has_key?(pattern, :first_word) ->
        :global

      true ->
        :user
    end
  end

  defp update_heuristic_outcome(heuristic_id, outcome) do
    case outcome do
      :success ->
        HeuristicStore.record_success(heuristic_id)

      :likely_success ->
        HeuristicStore.record_success(heuristic_id)

      :failure ->
        HeuristicStore.record_failure(heuristic_id)

      :likely_failure ->
        HeuristicStore.record_failure(heuristic_id)

      _ ->
        :ok
    end
  end

  defp update_analyzer_calibration(%Interpretation{} = interp, outcome) do
    was_correct =
      case outcome do
        :success -> true
        :likely_success -> true
        :failure -> false
        :likely_failure -> false
        _ -> nil
      end

    if was_correct != nil and Process.whereis(AnalyzerCalibration) do
      AnalyzerCalibration.track_outcome(
        interp.source,
        interp.raw_activation,
        was_correct
      )

      Enum.each(interp.analyzer_results, fn result ->
        AnalyzerCalibration.track_outcome(
          result.analyzer,
          result.raw_score,
          was_correct
        )
      end)
    end
  end

  defp maybe_create_heuristic(%Interpretation{} = interp, world_id, user_id, cohort_id) do
    {pattern, conclusion} = extract_pattern(interp)

    if map_size(pattern) >= 1 and world_id do
      scope = determine_scope(pattern, user_id, cohort_id)
      scope_id = get_scope_id(scope, user_id, cohort_id)

      unless similar_heuristic_exists?(pattern, scope, scope_id) do
        examples_key = pattern_key(pattern, scope, scope_id)
        success_count = get_pattern_success_count(examples_key)

        min_required = Map.get(@min_successes_for_heuristic, scope, 5)

        if success_count >= min_required do
          create_heuristic(pattern, conclusion, world_id, scope, scope_id)
        else
          increment_pattern_success(examples_key)
        end
      end
    end
  end

  defp create_heuristic(pattern, conclusion, world_id, scope, scope_id) do
    opts = [world_id: world_id, scope: scope, scope_id: scope_id, source: :learned]

    case HeuristicStore.add_heuristic(pattern, conclusion, opts) do
      {:ok, heuristic} ->
        Logger.info("Created new heuristic from learning", %{
          id: heuristic.id,
          world_id: world_id,
          scope: scope,
          pattern: pattern,
          intent: conclusion.intent
        })

        {:ok, heuristic}

      error ->
        Logger.warning("Failed to create heuristic", %{error: error})
        error
    end
  end

  defp similar_heuristic_exists?(pattern, scope, scope_id) do
    existing = HeuristicStore.match_scope(scope, scope_id, pattern_to_sample_text(pattern))

    Enum.any?(existing, fn {h, _conf} ->
      pattern_similarity(h.pattern, pattern) >= @pattern_similarity_threshold
    end)
  end

  defp pattern_similarity(pattern1, pattern2) do
    keys1 = Map.keys(pattern1) |> MapSet.new()
    keys2 = Map.keys(pattern2) |> MapSet.new()

    common = MapSet.intersection(keys1, keys2) |> MapSet.size()
    total = MapSet.union(keys1, keys2) |> MapSet.size()

    if total == 0 do
      0.0
    else
      common / total
    end
  end

  defp pattern_to_sample_text(pattern) do
    cond do
      Map.has_key?(pattern, :phrase) -> pattern.phrase
      Map.has_key?(pattern, :keywords) -> Enum.join(pattern.keywords, " ")
      Map.has_key?(pattern, :first_word) -> hd(pattern.first_word)
      true -> ""
    end
  end

  defp maybe_buffer_training_example(text, intent) do
    if Brain.ML.TrainingExampleBuffer.ready?() do
      Brain.ML.TrainingExampleBuffer.add_example(text, intent)
    end
  rescue
    _ -> :ok
  end

  defp track_pattern(%Interpretation{} = interp, outcome, user_id, cohort_id) do
    if outcome in [:success, :likely_success] do
      {pattern, _conclusion} = extract_pattern(interp)
      scope = determine_scope(pattern, user_id, cohort_id)
      scope_id = get_scope_id(scope, user_id, cohort_id)
      key = pattern_key(pattern, scope, scope_id)

      increment_pattern_success(key)
    end
  end

  defp get_scope_id(:user, user_id, _) do
    user_id
  end

  defp get_scope_id(:cohort, _, cohort_id) do
    cohort_id
  end

  defp get_scope_id(:global, _, _) do
    nil
  end

  defp pattern_key(pattern, scope, scope_id) do
    :erlang.phash2({pattern, scope, scope_id})
  end

  defp get_pattern_success_count(key) do
    case Process.get({:pattern_success, key}) do
      nil -> 0
      count -> count
    end
  end

  defp increment_pattern_success(key) do
    current = get_pattern_success_count(key)
    Process.put({:pattern_success, key}, current + 1)
  end

  defp maybe_add_first_word(pattern, first_word, source) do
    if source in [:structural, :pattern_recognition] and first_word != "" do
      Map.put(pattern, :first_word, [first_word])
    else
      pattern
    end
  end

  defp maybe_add_keywords(pattern, text, intent) do
    keywords = extract_domain_keywords(text, intent)

    if keywords != [] do
      Map.put(pattern, :keywords, keywords)
    else
      pattern
    end
  end

  defp maybe_add_word_count(pattern, word_count) do
    if word_count <= 5 do
      Map.put(pattern, :word_count, 1..max(word_count + 2, 5))
    else
      pattern
    end
  end

  defp maybe_add_phrase(pattern, text, intent) do
    if String.length(text) <= 30 and specific_intent?(intent) do
      Map.put(pattern, :phrase, text)
    else
      pattern
    end
  end

  defp extract_domain_keywords(text, intent) do
    domain =
      intent
      |> String.split(".")
      |> List.first()

    text_words =
      text
      |> Tokenizer.tokenize_normalized(min_length: 2)
      |> MapSet.new()

    domain_words =
      domain
      |> Tokenizer.tokenize_normalized(min_length: 2)

    Enum.filter(domain_words, &MapSet.member?(text_words, &1))
  end

  defp specific_intent?(intent) do
    IntentRegistry.specific?(intent)
  end

  defp calculate_appropriate_boost(%Interpretation{} = interp) do
    base_boost =
      case interp.source do
        :pattern_recognition -> 0.3
        :keyword -> 0.25
        :structural -> 0.2
        :model -> 0.25
        _ -> 0.15
      end

    if interp.activation >= 0.8 do
      min(base_boost + 0.1, ActivationPool.max_boost_for(:learned_user))
    else
      base_boost
    end
  end

  defp is_clarification_response?(response) when is_binary(response) do
    case Brain.ML.MicroClassifiers.classify(:clarification_response, response) do
      {:ok, "clarification", score} when score > 0.3 -> true
      _ -> false
    end
  end

  defp is_clarification_response?(%{type: :clarification}) do
    true
  end

  defp is_clarification_response?(%{type: :disambiguation}) do
    true
  end

  defp is_clarification_response?(%{type: :missing_slot}) do
    true
  end

  defp is_clarification_response?(_) do
    false
  end
end
