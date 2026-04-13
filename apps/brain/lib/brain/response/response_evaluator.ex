defmodule Brain.Response.ResponseEvaluator do
  @moduledoc """
  Evaluates a realized response against the original analysis context.

  Scores the response on 7 dimensions (0.0 to 1.0):

  1. **Speech act alignment** -- does the response's pragmatic function match
     the input? A question should get informative content, not just an ack.
  2. **Confidence alignment** -- does the hedging language match the system's
     actual confidence? High confidence shouldn't be hedged; low confidence
     shouldn't be stated as fact.
  3. **Content coverage** -- does the response address the entities and intent?
  4. **Slot coverage** -- if clarification was needed, does the response ask
     about the right missing slots?
  5. **Naturalness** -- no repeated phrases, reasonable length, no awkward
     transitions.
  6. **Echo avoidance** -- the response should not parrot back the user's
     input verbatim.

  The evaluator produces a `%Score{}` that the RefinementLoop uses to decide
  whether to iterate and which stage to re-run.
  """

  alias Brain.Analysis.ChunkAnalysis
  alias Brain.Response.PrimitiveTypes
  alias Brain.ML.Tokenizer

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
    speech_act = score_speech_act_alignment(primitives, analysis)
    confidence = score_confidence_alignment(primitives, analysis)
    coverage = score_content_coverage(primitives, response, analysis)
    completeness = score_content_completeness(primitives)
    slots = score_slot_coverage(primitives, analysis)
    natural = score_naturalness(response)
    echo = score_echo_avoidance(response, analysis)

    overall = weighted_average([
      {speech_act, 0.15},
      {confidence, 0.15},
      {coverage, 0.15},
      {completeness, 0.15},
      {slots, 0.10},
      {natural, 0.15},
      {echo, 0.15}
    ])

    dimensions = %{
      speech_act_alignment: speech_act,
      confidence_alignment: confidence,
      content_coverage: coverage,
      content_completeness: completeness,
      slot_coverage: slots,
      naturalness: natural,
      echo_avoidance: echo
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
  def dimension_to_stage(_), do: :surface_realizer

  # --- Dimension scorers ---

  defp score_speech_act_alignment(primitives, analysis) do
    speech_act = analysis.speech_act || %{}
    category = Map.get(speech_act, :category)
    is_question = Map.get(speech_act, :is_question, false)

    types = Enum.map(primitives, & &1.type)
    has_content = :content in types
    has_follow_up = :follow_up in types
    has_ack = :acknowledgment in types
    has_framing = :framing in types

    score = cond do
      is_question and (has_content or has_framing) -> 1.0
      is_question and has_follow_up -> 0.7
      category == :expressive and has_ack -> 1.0
      category == :assertive and (has_content or has_ack) -> 0.9
      category == :directive and (has_content or has_ack or has_follow_up) -> 0.8
      has_ack -> 0.5
      true -> 0.3
    end

    score
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
      response_lower = String.downcase(response)
      matched = Enum.count(entity_values, fn val ->
        String.downcase(val) |> String.contains?(response_lower) or
          String.contains?(response_lower, String.downcase(val))
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

  # --- Helpers ---

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
