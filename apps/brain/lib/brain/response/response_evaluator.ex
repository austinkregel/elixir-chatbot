defmodule Brain.Response.ResponseEvaluator do
  @moduledoc """
  Evaluates a realized response against the original analysis context and
  the system's epistemic state.

  Scores the response on 9 dimensions (0.0 to 1.0):

  1. **Speech act alignment** -- does the response's classified speech act
     match the input's? Uses `Pipeline.process` on the actual Ouro output.
  2. **Confidence alignment** -- does hedging match system confidence?
  3. **Content coverage** -- does the response address entities and intent?
     Uses `Tokenizer` overlap instead of `String.contains?`.
  4. **Content completeness** -- are all primitives structurally valid?
  5. **Slot coverage** -- are missing slots addressed with clarification?
  6. **Naturalness** -- no repetition, reasonable length.
  7. **Echo avoidance** -- response doesn't parrot input.
  8. **Belief grounding** -- claims in the response align with JTMS-backed
     beliefs and pass disclosure policy.
  9. **Epistemic consistency** -- the response maintains the system's
     epistemic integrity (no JTMS contradictions, no stance drift).

  The evaluator produces a `%Score{}` that the `RefinementLoop` uses to
  decide whether to iterate and which stage to re-run.
  """

  alias Brain.Analysis.{ChunkAnalysis, Pipeline}
  alias Brain.Response.PrimitiveTypes
  alias Brain.ML.Tokenizer
  alias Brain.Epistemic.{BeliefStore, JTMS, DisclosurePolicy, StanceTracker}

  require Logger

  @convergence_threshold 0.7
  @silence_threshold 0.35

  defmodule Score do
    @moduledoc false
    defstruct [
      speech_act_alignment: 0.0,
      confidence_alignment: 0.0,
      content_coverage: 0.0,
      content_completeness: 0.0,
      slot_coverage: 0.0,
      naturalness: 0.0,
      echo_avoidance: 0.0,
      belief_grounding: 0.0,
      epistemic_consistency: 0.0,
      overall: 0.0,
      weakest_dimension: nil,
      converged: false,
      silence_preferred: false
    ]
  end

  @doc """
  Evaluates the realized response against the analysis context.

  Returns a `%Score{}` with per-dimension scores and the overall score.
  """
  def evaluate(primitives, response, %ChunkAnalysis{} = analysis) when is_list(primitives) do
    response_analysis = analyze_response(response)

    speech_act = score_speech_act_alignment(response_analysis, analysis)
    confidence = score_confidence_alignment(primitives, analysis)
    coverage = score_content_coverage(primitives, response, analysis)
    completeness = score_content_completeness(primitives)
    slots = score_slot_coverage(primitives, analysis)
    natural = score_naturalness(response)
    echo = score_echo_avoidance(response, analysis)
    grounding = score_belief_grounding(response_analysis, primitives, analysis)
    epistemic = score_epistemic_consistency(response_analysis, analysis)

    overall = weighted_average([
      {speech_act, 0.12},
      {confidence, 0.10},
      {coverage, 0.12},
      {completeness, 0.10},
      {slots, 0.08},
      {natural, 0.12},
      {echo, 0.10},
      {grounding, 0.16},
      {epistemic, 0.10}
    ])

    dimensions = %{
      speech_act_alignment: speech_act,
      confidence_alignment: confidence,
      content_coverage: coverage,
      content_completeness: completeness,
      slot_coverage: slots,
      naturalness: natural,
      echo_avoidance: echo,
      belief_grounding: grounding,
      epistemic_consistency: epistemic
    }

    weakest = dimensions
    |> Enum.min_by(fn {_k, v} -> v end)
    |> elem(0)

    %Score{
      speech_act_alignment: speech_act,
      confidence_alignment: confidence,
      content_coverage: coverage,
      content_completeness: completeness,
      slot_coverage: slots,
      naturalness: natural,
      echo_avoidance: echo,
      belief_grounding: grounding,
      epistemic_consistency: epistemic,
      overall: overall,
      weakest_dimension: weakest,
      converged: overall >= @convergence_threshold,
      silence_preferred: overall < @silence_threshold
    }
  end

  def evaluate(_, _, _), do: %Score{converged: true, overall: 0.5}

  @doc "Maps a weak dimension to the pipeline stage that should be re-run."
  def dimension_to_stage(:speech_act_alignment), do: :discourse_planner
  def dimension_to_stage(:confidence_alignment), do: :content_specifier
  def dimension_to_stage(:content_coverage), do: :content_specifier
  def dimension_to_stage(:content_completeness), do: :content_specifier
  def dimension_to_stage(:slot_coverage), do: :content_specifier
  def dimension_to_stage(:naturalness), do: :surface_realizer
  def dimension_to_stage(:echo_avoidance), do: :content_specifier
  def dimension_to_stage(:belief_grounding), do: :content_specifier
  def dimension_to_stage(:epistemic_consistency), do: :discourse_planner
  def dimension_to_stage(_), do: :surface_realizer

  # --- Response analysis ---

  defp analyze_response(response) when is_binary(response) and response != "" do
    Pipeline.process(response)
  rescue
    e ->
      Logger.debug("ResponseEvaluator: Pipeline.process on response failed: #{Exception.message(e)}")
      nil
  catch
    :exit, _ -> nil
  end

  defp analyze_response(_), do: nil

  # --- Dimension scorers ---

  defp score_speech_act_alignment(response_analysis, analysis) do
    input_speech_act = analysis.speech_act || %{}
    input_category = Map.get(input_speech_act, :category)
    input_is_question = Map.get(input_speech_act, :is_question, false)

    response_category = extract_response_speech_act_category(response_analysis)

    cond do
      response_category == nil ->
        score_speech_act_from_primitives_fallback(input_category, input_is_question)

      compatible_speech_acts?(input_category, input_is_question, response_category) ->
        1.0

      true ->
        0.3
    end
  end

  defp extract_response_speech_act_category(nil), do: nil

  defp extract_response_speech_act_category(response_model) do
    analyses = response_model.analyses || []

    case analyses do
      [primary | _] ->
        speech_act = primary.speech_act || %{}
        Map.get(speech_act, :category)

      [] ->
        nil
    end
  end

  defp compatible_speech_acts?(input_cat, input_is_question, response_cat) do
    cond do
      input_is_question and response_cat in [:assertive, :commissive] -> true
      input_cat == :expressive and response_cat == :expressive -> true
      input_cat == :expressive and response_cat == :assertive -> true
      input_cat == :directive and response_cat in [:assertive, :commissive] -> true
      input_cat == :assertive and response_cat in [:assertive, :expressive, :commissive] -> true
      input_cat == response_cat -> true
      true -> false
    end
  end

  defp score_speech_act_from_primitives_fallback(input_category, input_is_question) do
    cond do
      input_is_question -> 0.6
      input_category == :expressive -> 0.7
      true -> 0.5
    end
  end

  defp score_confidence_alignment(primitives, analysis) do
    system_conf = analysis.confidence || 0.5
    has_hedging = Enum.any?(primitives, &(&1.type == :hedging))

    cond do
      system_conf < 0.4 and has_hedging -> 1.0
      system_conf < 0.4 and not has_hedging -> 0.4
      system_conf >= 0.8 and has_hedging -> 0.5
      system_conf >= 0.8 and not has_hedging -> 1.0
      system_conf >= 0.5 and system_conf < 0.8 -> 0.8
      true -> 0.6
    end
  end

  defp score_content_completeness(primitives) do
    if primitives == [] do
      0.5
    else
      valid_count = Enum.count(primitives, &PrimitiveTypes.valid?/1)
      valid_count / length(primitives)
    end
  end

  defp score_content_coverage(primitives, response, analysis) do
    entities = analysis.entities || []
    entity_values = entities
    |> Enum.map(&(Map.get(&1, :value) || Map.get(&1, :text) || ""))
    |> Enum.reject(&(&1 == ""))

    entity_score = if entity_values == [] do
      1.0
    else
      response_tokens = Tokenizer.tokenize_normalized(response) |> MapSet.new()

      matched = Enum.count(entity_values, fn val ->
        val_tokens = Tokenizer.tokenize_normalized(val) |> MapSet.new()
        overlap = MapSet.intersection(response_tokens, val_tokens) |> MapSet.size()
        overlap >= max(MapSet.size(val_tokens), 1)
      end)

      min(matched / length(entity_values), 1.0)
    end

    has_substantive = Enum.any?(primitives, &(&1.type == :content))
    content_score = if has_substantive, do: 0.8, else: 0.4

    (entity_score * 0.4 + content_score * 0.6)
  end

  defp score_slot_coverage(primitives, analysis) do
    slots = analysis.slots
    missing = get_missing_slots(slots)

    if missing == [] do
      1.0
    else
      has_clarification = Enum.any?(primitives, fn p ->
        p.type == :follow_up and p.variant == :clarification
      end)

      if has_clarification do
        mentioned_slots = primitives
        |> Enum.filter(&(&1.type == :follow_up and &1.variant == :clarification))
        |> Enum.flat_map(&(Map.get(&1.content, :missing_slots, [])))

        if mentioned_slots != [] do
          overlap = Enum.count(missing, &(&1 in mentioned_slots))
          min(overlap / length(missing), 1.0)
        else
          0.6
        end
      else
        0.3
      end
    end
  end

  defp score_naturalness(response) when is_binary(response) do
    tokens = try do
      Tokenizer.tokenize(response)
    rescue
      _ -> String.split(response)
    end

    word_count = length(tokens)

    length_score = cond do
      word_count == 0 -> 0.1
      word_count < 2 -> 0.4
      word_count > 100 -> 0.5
      word_count > 50 -> 0.7
      true -> 1.0
    end

    words = if is_list(tokens) do
      Enum.map(tokens, fn
        t when is_map(t) -> Map.get(t, :text, "")
        t when is_binary(t) -> t
        _ -> ""
      end)
    else
      String.split(response)
    end

    repetition_score = if length(words) > 3 do
      unique = words |> Enum.map(&String.downcase/1) |> Enum.uniq() |> length()
      min(unique / length(words) + 0.2, 1.0)
    else
      1.0
    end

    (length_score * 0.5 + repetition_score * 0.5)
  end

  defp score_naturalness(_), do: 0.3

  defp score_echo_avoidance(response, %ChunkAnalysis{text: input_text})
       when is_binary(response) and is_binary(input_text) do
    input_words = input_text |> String.downcase() |> String.split()
    response_words = response |> String.downcase() |> String.split()

    if length(input_words) < 4, do: 1.0, else: do_score_echo(input_words, response_words)
  end

  defp score_echo_avoidance(_, _), do: 1.0

  # --- Belief grounding ---

  defp score_belief_grounding(response_analysis, _primitives, _analysis) do
    response_entities = extract_response_entities(response_analysis)

    if response_entities == [] do
      0.7
    else
      if epistemic_services_ready?() do
        scores = Enum.map(response_entities, &score_entity_belief/1)
        Enum.sum(scores) / max(length(scores), 1)
      else
        0.7
      end
    end
  end

  defp extract_response_entities(nil), do: []

  defp extract_response_entities(response_model) do
    analyses = response_model.analyses || []

    analyses
    |> Enum.flat_map(fn a -> a.entities || [] end)
    |> Enum.uniq_by(fn e ->
      {Map.get(e, :entity_type), Map.get(e, :value)}
    end)
  end

  defp score_entity_belief(entity) do
    value = Map.get(entity, :value) || ""
    entity_type = Map.get(entity, :entity_type) || "unknown"

    if value == "" do
      0.7
    else
      predicate = safe_to_atom(entity_type)

      case BeliefStore.query_beliefs(subject: :world, predicate: predicate, min_confidence: 0.3) do
        {:ok, beliefs} when beliefs != [] ->
          score_against_beliefs(value, beliefs)

        _ ->
          0.7
      end
    end
  rescue
    _ -> 0.7
  catch
    :exit, _ -> 0.7
  end

  defp score_against_beliefs(claim_value, beliefs) do
    scores = Enum.map(beliefs, fn belief ->
      belief_object = to_string(belief.object)

      claim_tokens = Tokenizer.tokenize_normalized(claim_value) |> MapSet.new()
      belief_tokens = Tokenizer.tokenize_normalized(belief_object) |> MapSet.new()
      overlap = MapSet.intersection(claim_tokens, belief_tokens) |> MapSet.size()
      max_possible = max(MapSet.size(belief_tokens), 1)

      is_related = overlap >= max(div(max_possible, 3), 1)

      cond do
        not is_related ->
          0.7

        is_related and check_belief_justified?(belief) ->
          check_disclosure_score(belief)

        is_related ->
          0.5

        true ->
          0.7
      end
    end)

    if scores == [], do: 0.7, else: Enum.min(scores)
  end

  defp check_belief_justified?(belief) do
    case belief.node_id do
      nil -> true
      node_id ->
        case JTMS.is_in?(node_id) do
          true -> true
          false -> false
          {:error, _} -> true
        end
    end
  rescue
    _ -> true
  catch
    :exit, _ -> true
  end

  defp check_disclosure_score(belief) do
    case DisclosurePolicy.evaluate_disclosure(belief) do
      %{should_disclose: true} -> 1.0
      %{should_disclose: false} -> 0.0
      _ -> 1.0
    end
  rescue
    _ -> 1.0
  catch
    :exit, _ -> 1.0
  end

  # --- Epistemic consistency ---

  defp score_epistemic_consistency(response_analysis, analysis) do
    if not epistemic_services_ready?() do
      0.8
    else
      consistency_score = check_jtms_consistency()
      drift_score = check_stance_drift(response_analysis, analysis)

      consistency_score * 0.6 + drift_score * 0.4
    end
  end

  defp check_jtms_consistency do
    case JTMS.check_consistency() do
      {:ok, :consistent} -> 1.0
      {:error, {:contradiction, _node_id}} -> 0.2
      _ -> 0.8
    end
  rescue
    _ -> 0.8
  catch
    :exit, _ -> 0.8
  end

  defp check_stance_drift(response_analysis, analysis) do
    if not StanceTracker.ready?() do
      0.8
    else
      topics = extract_topics(response_analysis, analysis)

      if topics == [] do
        1.0
      else
        drift_scores = Enum.map(topics, fn topic ->
          case StanceTracker.check_drift("current", topic) do
            {:ok, :no_drift} ->
              1.0

            {:ok, %{exceeds_threshold: true, absolute_drift: drift}} ->
              max(1.0 - drift, 0.0)

            {:ok, %{exceeds_threshold: false}} ->
              1.0

            _ ->
              0.8
          end
        end)

        Enum.sum(drift_scores) / max(length(drift_scores), 1)
      end
    end
  rescue
    _ -> 0.8
  catch
    :exit, _ -> 0.8
  end

  defp extract_topics(response_analysis, analysis) do
    input_entities = (analysis.entities || [])
    |> Enum.map(&(Map.get(&1, :value) || ""))
    |> Enum.reject(&(&1 == ""))

    response_entities = extract_response_entities(response_analysis)
    |> Enum.map(&(Map.get(&1, :value) || ""))
    |> Enum.reject(&(&1 == ""))

    (input_entities ++ response_entities)
    |> Enum.uniq()
    |> Enum.take(5)
  end

  # --- Helpers ---

  defp epistemic_services_ready? do
    BeliefStore.ready?() and JTMS.ready?()
  rescue
    _ -> false
  catch
    :exit, _ -> false
  end

  defp safe_to_atom(str) when is_binary(str) do
    String.to_existing_atom(str)
  rescue
    ArgumentError -> :unknown
  end

  defp safe_to_atom(atom) when is_atom(atom), do: atom
  defp safe_to_atom(_), do: :unknown

  defp do_score_echo(input_words, response_words) do
    max_run = longest_common_run(input_words, response_words)

    cond do
      max_run >= length(input_words) * 0.8 -> 0.1
      max_run >= 6 -> 0.2
      max_run >= 4 -> 0.5
      true -> 1.0
    end
  end

  defp longest_common_run(needle, haystack) do
    needle_len = length(needle)

    if needle_len == 0 or haystack == [] do
      0
    else
      haystack
      |> Enum.chunk_every(needle_len, 1, :discard)
      |> Enum.reduce(0, fn window, best ->
        run = count_matching_prefix(needle, window, 0)
        max(run, best)
      end)
    end
  end

  defp count_matching_prefix([], _, count), do: count
  defp count_matching_prefix(_, [], count), do: count
  defp count_matching_prefix([a | rest_a], [b | rest_b], count) do
    if a == b, do: count_matching_prefix(rest_a, rest_b, count + 1), else: count
  end

  defp weighted_average(scores_and_weights) do
    {sum, weight_sum} = Enum.reduce(scores_and_weights, {0.0, 0.0}, fn {score, weight}, {s, w} ->
      {s + score * weight, w + weight}
    end)

    if weight_sum > 0, do: sum / weight_sum, else: 0.0
  end

  defp get_missing_slots(nil), do: []
  defp get_missing_slots(%{missing_required: m}) when is_list(m), do: m
  defp get_missing_slots(_), do: []
end
