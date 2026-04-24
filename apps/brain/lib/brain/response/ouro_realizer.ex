defmodule Brain.Response.OuroRealizer do
  @moduledoc """
  Realizes a full primitive plan into natural language using the Ouro LoopLM model.

  Sends the complete primitive plan to Ouro as a structured packet. The model's
  iterative latent refinement (4 recurrence passes) fuses the primitives into
  coherent, natural prose.

  Returns `{:ok, text, metadata}` on success, or `{:error, reason}` on any
  failure. There is no `:fallback` -- Ouro is the sole realization path.
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

      do_generate(messages, gen_opts, primitives, opts)
    end
  end

  defp do_generate(messages, gen_opts, primitives, opts) do
    case OuroModel.generate(messages, gen_opts) do
      {:ok, text} ->
        Logger.info("OuroRealizer: generated #{String.length(text)} chars, validating")
        unified_context = Keyword.get(opts, :unified_context, %{})
        validate_and_return(text, primitives, unified_context)

      :fallback ->
        Logger.error("OuroRealizer: Ouro model not loaded")
        {:error, :ouro_not_loaded}

      {:error, reason} ->
        Logger.error("Ouro generation failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp validate_and_return(text, primitives, unified_context) do
    case ConstraintEnforcer.validate(text, primitives, unified_context: unified_context) do
      {:ok, validated_text} ->
        metadata = %{
          source: :ouro,
          model: "ouro-1.4b",
          primitive_count: length(primitives)
        }

        {:ok, validated_text, metadata}

      {:rejected, reason} ->
        Logger.error("OuroRealizer: output rejected -- #{reason} #{inspect(text)}")
        {:error, {:constraint_rejected, reason}}
    end
  end
end
