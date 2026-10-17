defmodule Brain.Response.OuroRealizer do
  @moduledoc """
  Realizes a full primitive plan into natural language using the Ouro LoopLM model.

  Sends the complete primitive plan to Ouro as a structured packet. The model's
  iterative latent refinement (4 recurrence passes) fuses the primitives into
  coherent, natural prose.

  Returns `{:ok, text, metadata}` on success, or `{:error, reason}` on any
  failure. Structural validation (empty, degenerate, length) is applied here;
  semantic evaluation is deferred to `ResponseEvaluator` so the
  `RefinementLoop` can iterate.
  """

  alias Brain.ML.Ouro.Model, as: OuroModel
  alias Brain.Response.{RealizationPacket, ConstraintEnforcer}
  alias Brain.Analysis.ChunkAnalysis

  require Logger

  @doc """
  Realizes a primitive plan using the Ouro model.

  Returns `{:ok, response_text, metadata}` on success,
  or `{:error, reason}` on failure.

  `primitives` is the list of content-specified `%Primitive{}` structs.
  `analysis` is the primary `%ChunkAnalysis{}` for constraint building.
  `opts` may contain `:unified_context` for rich context serialization.

  The metadata carries `:recurring_pattern`, the turn's recurring pattern the
  realization packet named (`RealizationPacket.recurring_pattern/1`), or `nil`.
  """
  def realize(primitives, analysis \\ %ChunkAnalysis{}, opts \\ [])

  def realize([], _analysis, _opts) do
    {:error, :empty_primitive_list}
  end

  def realize(primitives, analysis, opts) when is_list(primitives) do
    messages = RealizationPacket.build(primitives, analysis, opts)
    recurring_pattern = RealizationPacket.recurring_pattern(Keyword.get(opts, :unified_context))

    if Keyword.get(opts, :dry_run_ouro, false) do
      Logger.info("OuroRealizer: dry_run_ouro=true, returning ChatML messages without calling Ouro")

      {:ok, :ouro_dry_run,
       %{
         messages: messages,
         source: :dry_run,
         primitive_count: length(primitives),
         recurring_pattern: recurring_pattern
       }}
    else
      gen_opts = [
        max_new_tokens: Keyword.get(opts, :max_new_tokens, 200),
        temperature: Keyword.get(opts, :temperature, 0.6),
        repetition_penalty: Keyword.get(opts, :repetition_penalty, 1.3)
      ]

      do_generate(messages, gen_opts, primitives, recurring_pattern)
    end
  end

  defp do_generate(messages, gen_opts, primitives, recurring_pattern) do
    case OuroModel.generate(messages, gen_opts) do
      {:ok, text} ->
        Logger.info("OuroRealizer: generated #{String.length(text)} chars, validating structural constraints")
        validate_and_return(text, primitives, recurring_pattern)

      :fallback ->
        Logger.error("OuroRealizer: Ouro model not loaded")
        {:error, :ouro_not_loaded}

      {:error, reason} ->
        Logger.error("Ouro generation failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp validate_and_return(text, primitives, recurring_pattern) do
    case ConstraintEnforcer.validate(text, primitives) do
      {:ok, validated_text} ->
        metadata = %{
          source: :ouro,
          model: "ouro-1.4b",
          primitive_count: length(primitives),
          recurring_pattern: recurring_pattern
        }

        {:ok, validated_text, metadata}

      {:rejected, reason} ->
        Logger.error("OuroRealizer: structurally degenerate output -- #{reason}")
        {:error, {:structural_rejection, reason}}
    end
  end
end
