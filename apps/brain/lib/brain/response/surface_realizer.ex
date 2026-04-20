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
      {:ok, :ouro_dry_run, %{messages: messages}} ->
        Logger.info("SurfaceRealizer: dry_run_ouro=true, returning ChatML messages without rendering")
        {:ok, primitives, {:ouro_dry_run, messages}}

      {:ok, text, _metadata} ->
        Logger.info("SurfaceRealizer: Ouro realized #{length(primitives)} primitives")

        rendered =
          Enum.map(primitives, fn p ->
            p |> Primitive.render(text) |> Map.put(:source, :ouro)
          end)

        collect_plan(primitives, text, opts)
        collect_pairs(primitives, text)
        {:ok, rendered, text}

      {:error, reason} ->
        Logger.warning("SurfaceRealizer: Ouro realization failed: #{inspect(reason)}, trying enriched fallback")
        try_enriched_fallback(primitives, reason, opts)
    end
  end

  defp try_ouro_realization(primitives, analysis, opts) do
    OuroRealizer.realize(primitives, analysis || %Brain.Analysis.ChunkAnalysis{}, opts)
  end

  defp try_enriched_fallback(primitives, original_reason, _opts) do
    enriched_primitive = Enum.find(primitives, &(&1.type == :content and &1.variant == :enriched))

    if enriched_primitive do
      text = build_enriched_placeholder_text(enriched_primitive)

      rendered =
        Enum.map(primitives, fn p ->
          p |> Primitive.render(text) |> Map.put(:source, :enriched_fallback)
        end)

      {:ok, rendered, text}
    else
      {:error, original_reason}
    end
  end

  defp build_enriched_placeholder_text(%Primitive{content: content}) do
    available = Map.get(content, :available_placeholders, [])
    topic = Map.get(content, :topic)

    placeholder_parts =
      available
      |> Enum.reject(&(&1 in ["raw", "daily_forecasts"]))
      |> Enum.map(fn field -> "$#{field}" end)

    cond do
      topic && placeholder_parts != [] ->
        "Here's what I found for #{topic}: #{Enum.join(placeholder_parts, ", ")}."

      placeholder_parts != [] ->
        "Here's what I found: #{Enum.join(placeholder_parts, ", ")}."

      true ->
        "I found some information but couldn't format it properly."
    end
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
