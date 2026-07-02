defmodule Brain.Response.SurfaceRealizer do
  @moduledoc """
  Renders content-specified primitives into natural language text via the
  Ouro LoopLM model.

  The full primitive plan is sent as a structured packet to Ouro, whose
  iterative latent refinement (4 recurrence passes) fuses the primitives
  into coherent, natural prose.

  If Ouro is not loaded, the system raises -- there is no template fallback.
  """

  alias Brain.Response.{Primitive, DecompressorCollector, OuroRealizer,
                         ResponseSystemRouter, PhraseLatticeRealizer, LatticeScorer}

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
    domain = extract_domain(analysis)

    {system, domain_config} = ResponseSystemRouter.route(domain)

    case system do
      :lattice ->
        case try_lattice_realization(primitives, analysis, domain_config, opts) do
          {:ok, _, _} = result -> result
          _ -> try_synthesizer_fallback(primitives, analysis, opts)
        end

      :ouro ->
        try_ouro_or_fallback(primitives, analysis, opts)

      :synthesizer ->
        try_synthesizer_fallback(primitives, analysis, opts)

      :template ->
        case try_enriched_fallback(primitives, :template_configured, opts) do
          {:ok, _, _} = result -> result
          {:error, _} -> try_synthesizer_fallback(primitives, analysis, opts)
        end
    end
  end

  defp extract_domain(nil), do: nil
  defp extract_domain(%{intent: intent}) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [domain, _] -> domain
      _ -> nil
    end
  end
  defp extract_domain(_), do: nil

  defp try_lattice_realization(primitives, analysis, domain_config, opts) do
    feature_vector = lattice_feature_vector(opts, analysis)
    intent = lattice_intent(opts, analysis)

    input_sentiment = extract_sentiment(analysis)
    tone_bias = Map.get(domain_config.tone_vectors, domain_config.tone_bias, List.duplicate(0.5, 10))
    desired_tone = LatticeScorer.compute_desired_tone(input_sentiment, tone_bias, domain_config.mirror_coefficient)

    lattice_opts =
      opts
      |> Keyword.merge(desired_tone: desired_tone)
      |> Keyword.put(:feature_vector, feature_vector)
      |> maybe_put_lattice_intent(intent)

    PhraseLatticeRealizer.realize(primitives, lattice_opts)
  end

  defp lattice_feature_vector(opts, analysis) do
    case Keyword.get(opts, :feature_vector) do
      fv when is_list(fv) and fv != [] -> fv
      _ -> extract_analysis_feature_vector(analysis)
    end
  end

  defp lattice_intent(opts, analysis) do
    case Keyword.get(opts, :intent) do
      intent when is_binary(intent) and intent != "" -> intent
      _ -> if analysis, do: Map.get(analysis, :intent), else: nil
    end
  end

  defp extract_analysis_feature_vector(nil), do: []

  defp extract_analysis_feature_vector(%{feature_vector: fv}) when is_list(fv), do: fv
  defp extract_analysis_feature_vector(analysis) when is_map(analysis), do: Map.get(analysis, :feature_vector, [])
  defp extract_analysis_feature_vector(_), do: []

  defp maybe_put_lattice_intent(opts, intent) when is_binary(intent) and intent != "" do
    Keyword.put(opts, :intent, intent)
  end

  defp maybe_put_lattice_intent(opts, _), do: opts

  defp extract_sentiment(nil), do: [0.0, 0.0, 1.0, 0.5, 0.0]
  defp extract_sentiment(%{sentiment: %{scores: scores}}) when is_map(scores) do
    [
      Map.get(scores, :positive, 0.0),
      Map.get(scores, :negative, 0.0),
      Map.get(scores, :neutral, 1.0),
      Map.get(scores, :confidence, 0.5),
      Map.get(scores, :polarity_magnitude, 0.0)
    ]
  end
  defp extract_sentiment(_), do: [0.0, 0.0, 1.0, 0.5, 0.0]

  defp try_ouro_or_fallback(primitives, analysis, opts) do
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

  defp try_synthesizer_fallback(primitives, analysis, opts) do
    intent = if analysis, do: Map.get(analysis, :intent), else: nil
    entities = if analysis, do: Map.get(analysis, :entities, []), else: []

    synth_opts = [
      confidence: (analysis && Map.get(analysis, :confidence)) || 0.5,
      context: Keyword.get(opts, :unified_context, %{})
    ]

    case Brain.Response.Synthesizer.synthesize(intent, entities, synth_opts) do
      {:ok, text} when is_binary(text) and text != "" ->
        Logger.info("SurfaceRealizer: template fallback via Synthesizer for #{inspect(intent)}")

        rendered =
          Enum.map(primitives, fn p ->
            p |> Primitive.render(text) |> Map.put(:source, :synthesizer_template)
          end)

        {:ok, rendered, text}

      _ ->
        {:error, :synthesizer_fallback_failed}
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
