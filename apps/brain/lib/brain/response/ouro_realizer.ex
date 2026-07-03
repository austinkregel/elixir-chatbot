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

  alias Brain.ML.Generation
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
  """
  def realize(primitives, analysis \\ %ChunkAnalysis{}, opts \\ [])

  def realize([], _analysis, _opts) do
    {:error, :empty_primitive_list}
  end

  def realize(primitives, analysis, opts) when is_list(primitives) do
    messages = RealizationPacket.build(primitives, analysis, opts)

    if Keyword.get(opts, :dry_run_ouro, false) do
      Logger.info("OuroRealizer: dry_run_ouro=true, returning ChatML messages without calling Ouro")

      {:ok, :ouro_dry_run,
       %{
         messages: messages,
         source: :dry_run,
         primitive_count: length(primitives)
       }}
    else
      gen_opts = [
        max_new_tokens: Keyword.get(opts, :max_new_tokens, 200),
        temperature: Keyword.get(opts, :temperature, 0.6),
        repetition_penalty: Keyword.get(opts, :repetition_penalty, 1.3)
      ]

      do_generate(messages, gen_opts, primitives)
    end
  end

  defp do_generate(messages, gen_opts, primitives) do
    # Backend-agnostic: routes through the configured generation backend
    # (OpenAI-compatible / Ouro sidecar / null), not a hard-wired Ouro dependency.
    case Generation.generate(messages, gen_opts) do
      {:ok, text} when is_binary(text) and text != "" ->
        Logger.info("Realizer: generated #{String.length(text)} chars, validating structural constraints")
        validate_and_return(text, primitives)

      {:ok, ""} ->
        {:error, {:backend_unavailable, Generation.name(), :empty_output}}

      {:error, reason} ->
        Logger.error("Generation failed (backend=#{Generation.name()}): #{inspect(reason)}")
        {:error, {:backend_unavailable, Generation.name(), reason}}
    end
  end

  defp validate_and_return(text, primitives) do
    case ConstraintEnforcer.validate(text, primitives) do
      {:ok, validated_text} ->
        metadata = %{
          source: :ouro,
          model: "ouro-1.4b",
          primitive_count: length(primitives)
        }

        {:ok, validated_text, metadata}

      {:rejected, reason} ->
        Logger.error("OuroRealizer: structurally degenerate output -- #{reason}")
        {:error, {:structural_rejection, reason}}
    end
  end
end
