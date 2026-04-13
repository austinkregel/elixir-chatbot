defmodule Brain.Response.SurfaceRealizer do
  @moduledoc """
  Renders content-specified primitives into natural language text via the
  Ouro LoopLM model.

  The full primitive plan is sent as a structured packet to Ouro, whose
  iterative latent refinement (4 recurrence passes) fuses the primitives
  into coherent, natural prose.

  If Ouro is not loaded, the system raises -- there is no template fallback.
  """

  alias Brain.Response.{Primitive, DecompressorCollector, OuroRealizer}

  require Logger

  @doc """
  Renders a list of content-specified primitives into text via Ouro.

  Options:
    - `:analysis` - ChunkAnalysis for Ouro realization packet context
    - `:unified_context` - rich context map from ContextBuilder
  """
  def realize(primitives, opts \\ [])

  def realize(primitives, opts) when is_list(primitives) do
    analysis = Keyword.get(opts, :analysis)

    case try_ouro_realization(primitives, analysis, opts) do
      {:ok, text, _metadata} ->
        Logger.info("SurfaceRealizer: Ouro realized #{length(primitives)} primitives")

        rendered =
          Enum.map(primitives, fn p ->
            p |> Primitive.render(text) |> Map.put(:source, :ouro)
          end)

        collect_plan(primitives, text, opts)
        collect_pairs(primitives, text)
        {rendered, text}

      {:error, reason} ->
        raise "Ouro realization failed: #{inspect(reason)}"
    end
  end

  defp try_ouro_realization(primitives, analysis, opts) do
    OuroRealizer.realize(primitives, analysis || %Brain.Analysis.ChunkAnalysis{}, opts)
  end

  defp collect_pairs(primitives, _text) do
    Enum.each(primitives, fn p ->
      collect_pair(p, p.rendered || "")
    end)
  end

  defp collect_pair(p, text) do
    if Code.ensure_loaded?(DecompressorCollector) and function_exported?(DecompressorCollector, :collect, 2) do
      DecompressorCollector.collect(p, text)
    end
  rescue
    _ -> :ok
  end

  defp collect_plan(primitives, response, opts) do
    if Code.ensure_loaded?(DecompressorCollector) and
         function_exported?(DecompressorCollector, :collect_plan, 3) do
      DecompressorCollector.collect_plan(primitives, response, opts)
    end
  rescue
    _ -> :ok
  end
end
