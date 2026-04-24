defmodule Brain.Response.RefinementLoop do
  @moduledoc """
  Orchestrates iterative response refinement toward convergence.

  Runs the full synthesis pipeline (plan -> specify -> realize -> evaluate),
  then iteratively re-runs the weakest stage until the response converges
  or the iteration limit is reached.

  This is the "fit to 1" mechanism: each iteration tightens the response
  toward better alignment with the system's actual knowledge and the
  input's communicative needs.

  ## Pipeline

      DiscoursePlanner.plan(model)
      -> ContentSpecifier.specify(primitives, analysis, opts)
      -> SurfaceRealizer.realize(primitives, opts)
      -> ResponseEvaluator.evaluate(primitives, response, analysis)
      -> if not converged: refine weakest stage and repeat
  """

  alias Brain.Response.{DiscoursePlanner, ContentSpecifier, SurfaceRealizer, ResponseEvaluator}
  alias Brain.Response.ResponseEvaluator.Score
  alias Brain.Analysis.{InternalModel, ChunkPriority}

  require Logger

  @max_iterations 3

  @doc """
  Generates a response for the given analysis model using iterative refinement.

  Returns `{:ok, response, metadata}` where metadata includes the final
  score, iteration count, and rendered primitives.

  Options:
    - `:unified_context` - rich context map from ContextBuilder
    - `:max_iterations` - override default iteration limit
  """
  def generate(%InternalModel{} = model, opts \\ []) do
    analyses = model.analyses || []
    max_iter = Keyword.get(opts, :max_iterations, @max_iterations)

    if analyses == [] do
      {:error, :empty_analysis}
    else
      primary_analysis = select_primary_analysis(analyses)
      initial_plan = DiscoursePlanner.plan(model, opts)

      result = iterate(initial_plan, primary_analysis, opts, 1, max_iter, nil)

      case result do
        {:ok, {:ouro_dry_run, _messages} = response, primitives, _score, iterations} ->
          Logger.info("RefinementLoop: dry_run_ouro=true, returning ChatML messages without evaluation")
          {:ok, response, %{
            score: nil,
            iterations: iterations,
            primitives: primitives,
            method: :ouro_dry_run
          }}

        {:ok, response, primitives, score, iterations} ->
          if score != nil and score.silence_preferred do
            Logger.info("RefinementLoop: silence preferred (score=#{Float.round(score.overall, 2)})")
            {:ok, nil, %{
              score: score,
              iterations: iterations,
              primitives: primitives,
              method: :silence_preferred
            }}
          else
            {:ok, response, %{
              score: score,
              iterations: iterations,
              primitives: primitives,
              method: :synthesis_pipeline
            }}
          end

        {:error, reason} ->
          Logger.warning("RefinementLoop failed: #{inspect(reason)}")
          {:error, reason}
      end
    end
  end

  @doc """
  Runs a single pass of the synthesis pipeline without iteration.
  Useful for testing individual stages.
  """
  def single_pass(%InternalModel{} = model, opts \\ []) do
    analyses = model.analyses || []
    primary = select_primary_analysis(analyses)
    primitives = DiscoursePlanner.plan(model, opts)
    specified = ContentSpecifier.specify(primitives, primary, opts)
    realize_opts = Keyword.merge(opts, [analysis: primary])

    case SurfaceRealizer.realize(specified, realize_opts) do
      {:ok, rendered, response} ->
        score = ResponseEvaluator.evaluate(rendered, response, primary)
        {:ok, response, %{score: score, primitives: rendered, iterations: 1}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp iterate(plan, analysis, opts, iteration, max_iter, best_so_far) do
    specified = ContentSpecifier.specify(plan, analysis, opts)
    realize_opts = Keyword.merge(opts, [analysis: analysis])

    case SurfaceRealizer.realize(specified, realize_opts) do
      {:ok, rendered, {:ouro_dry_run, _messages} = response} ->
        {:ok, response, rendered, nil, iteration}

      {:ok, rendered, response} ->
        score = ResponseEvaluator.evaluate(rendered, response, analysis)

        current = %{response: response, primitives: rendered, score: score, iteration: iteration}
        best = pick_best(best_so_far, current)

        if score.converged or iteration >= max_iter do
          {:ok, best.response, best.primitives, best.score, iteration}
        else
          Logger.debug("RefinementLoop iteration #{iteration}: overall=#{Float.round(score.overall, 2)}, weakest=#{score.weakest_dimension}")

          refined_plan = refine(plan, score, analysis, opts)
          iterate(refined_plan, analysis, opts, iteration + 1, max_iter, best)
        end

      {:error, reason} ->
        if best_so_far do
          Logger.warning("RefinementLoop: realization failed at iteration #{iteration}, using best so far")
          {:ok, best_so_far.response, best_so_far.primitives, best_so_far.score, iteration}
        else
          {:error, reason}
        end
    end
  end

  defp refine(plan, %Score{weakest_dimension: dimension}, analysis, opts) do
    stage = ResponseEvaluator.dimension_to_stage(dimension)

    case stage do
      :discourse_planner ->
        adjust_plan(plan, dimension, analysis)

      :content_specifier ->
        adjust_content(plan, dimension, analysis, opts)

      :surface_realizer ->
        transform_for_realization(plan, dimension, analysis, opts)
    end
  end

  defp transform_for_realization(plan, dimension, _analysis, opts) do
    unified_context = Keyword.get(opts, :unified_context, %{})

    case dimension do
      :naturalness ->
        merge_compatible_primitives(plan)

      _ ->
        maybe_simplify(plan, unified_context)
    end
  end

  defp merge_compatible_primitives(plan) do
    {merged, _} =
      Enum.reduce(plan, {[], nil}, fn primitive, {acc, prev} ->
        if prev && mergeable?(prev, primitive) do
          grouped = %{prev | content: Map.merge(prev.content, primitive.content)}
          {acc, grouped}
        else
          if prev do
            {acc ++ [prev], primitive}
          else
            {acc, primitive}
          end
        end
      end)

    merged =
      case Enum.at(plan, -1) do
        nil -> merged
        last ->
          if merged == [] or List.last(merged) != last do
            merged ++ [last]
          else
            merged
          end
      end

    if merged == [], do: plan, else: merged
  end

  defp mergeable?(a, b) do
    (a.type == :hedging and b.type == :content) or
      (a.type == :attunement and b.type == :follow_up and b.variant == :clarification) or
      (a.type == :contradiction_response and b.type == :follow_up and b.variant == :correction_invite)
  end

  defp maybe_simplify(plan, _unified_context) do
    optional_types = [:transition, :attunement]

    essential = Enum.reject(plan, &(&1.type in optional_types))

    if length(essential) >= 1, do: essential, else: plan
  end

  defp adjust_plan(plan, :speech_act_alignment, analysis) do
    speech_act = analysis.speech_act || %{}
    is_question = Map.get(speech_act, :is_question, false)
    has_content = Enum.any?(plan, &(&1.type == :content))

    if is_question and not has_content do
      plan ++ [Brain.Response.Primitive.new(:content, :reflective)]
    else
      plan
    end
  end

  defp adjust_plan(plan, _, _analysis), do: plan

  defp adjust_content(plan, :confidence_alignment, analysis, _opts) do
    conf = analysis.confidence || 0.5
    has_hedging = Enum.any?(plan, &(&1.type == :hedging))

    cond do
      conf < 0.4 and not has_hedging ->
        hedging = Brain.Response.Primitive.new(:hedging, nil, %{confidence_level: conf})
        case Enum.find_index(plan, &(&1.type in [:content, :framing])) do
          nil -> [hedging | plan]
          idx -> List.insert_at(plan, idx, hedging)
        end

      conf >= 0.8 and has_hedging ->
        Enum.reject(plan, &(&1.type == :hedging))

      true ->
        plan
    end
  end

  defp adjust_content(plan, :content_coverage, _analysis, _opts) do
    has_content = Enum.any?(plan, &(&1.type == :content))

    if not has_content do
      plan ++ [Brain.Response.Primitive.new(:content, :reflective)]
    else
      plan
    end
  end

  defp adjust_content(plan, :slot_coverage, analysis, _opts) do
    has_clarification = Enum.any?(plan, fn p ->
      p.type == :follow_up and p.variant == :clarification
    end)

    slots = analysis.slots
    missing = get_missing_slots(slots)

    if missing != [] and not has_clarification do
      plan ++ [Brain.Response.Primitive.new(:follow_up, :clarification)]
    else
      plan
    end
  end

  defp adjust_content(plan, _, _, _), do: plan

  defp select_primary_analysis(analyses) do
    respondable =
      Enum.filter(analyses, fn a ->
        a.response_strategy in [:can_respond, :hedged_response]
      end)

    case respondable do
      [] -> ChunkPriority.select_primary(analyses)
      respondable -> ChunkPriority.select_primary(respondable)
    end
  end

  defp pick_best(nil, current), do: current
  defp pick_best(best, current) do
    if current.score.overall >= best.score.overall, do: current, else: best
  end

  defp get_missing_slots(nil), do: []
  defp get_missing_slots(%{missing_required: m}) when is_list(m), do: m
  defp get_missing_slots(_), do: []
end
