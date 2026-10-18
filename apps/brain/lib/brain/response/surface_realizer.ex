defmodule Brain.Response.SurfaceRealizer do
  @moduledoc """
  Renders content-specified primitives into natural language text via the
  Ouro LoopLM model.

  The full primitive plan is sent as a structured packet to Ouro, whose
  iterative latent refinement (4 recurrence passes) fuses the primitives
  into coherent, natural prose.

  If Ouro is not loaded, the system raises -- there is no template fallback.
  """

  alias Brain.Memory.RecurringPattern
  alias Brain.Response.{Primitive, DecompressorCollector, OuroRealizer,
                         ResponseSystemRouter, PhraseLatticeRealizer, LatticeScorer}

  require Logger

  @doc """
  Renders a list of content-specified primitives into text via Ouro.

  Options:
    - `:world_id` - required: the world the response is realized in; what it
      collects for training belongs to that world. Absent raises.
    - `:analysis` - ChunkAnalysis for Ouro realization packet context
    - `:unified_context` - rich context map from ContextBuilder

  A plan holding a primitive planned for a service's error
  (`Brain.Response.DiscoursePlanner.enrichment_error_primitive/2`) is worded
  from that reply's templates
  (`Brain.Response.Synthesizer.synthesize_service_reply/1`), whatever system
  the domain is realized by, since the reply states exactly what was not done
  and names the items the service found; every primitive of the plan is
  rendered with that text, with source `:service_reply`.

  Returns `{:ok, rendered, text, recurring_pattern_use}` or
  `{:error, reason}`. `recurring_pattern_use` is how the text drew on the
  turn's recurring pattern (`t:Brain.Memory.RecurringPattern.pattern_use/0`): the
  synthesizer adapted it from the pattern, or the realization packet Ouro
  realized it from named the pattern; `nil` when the text drew on none.
  Realizing reports nothing: a refined turn realizes on each pass, and the
  caller that settles on the turn's response reports the use
  (`Brain.Memory.RecurringPattern.used/3`).
  """
  @spec realize([Primitive.t()], keyword()) ::
          {:ok, [Primitive.t()], String.t() | {:ouro_dry_run, list()}, RecurringPattern.pattern_use() | nil}
          | {:error, term()}
  def realize(primitives, opts) when is_list(primitives) do
    world_id!(opts)

    case Enum.find(primitives, &service_reply?/1) do
      nil -> realize_by_system(primitives, opts)
      reply -> realize_service_reply(primitives, reply)
    end
  end

  @doc """
  Whether the primitive answers a service's error
  (`Brain.Response.DiscoursePlanner.enrichment_error_primitive/2`).
  """
  @spec service_reply?(Primitive.t()) :: boolean()
  def service_reply?(%Primitive{content: %{enrichment_error: _}}), do: true
  def service_reply?(_), do: false

  defp realize_service_reply(primitives, %Primitive{content: content}) do
    {:ok, text} = Brain.Response.Synthesizer.synthesize_service_reply(content)

    rendered =
      Enum.map(primitives, fn p ->
        p |> Primitive.render(text) |> Map.put(:source, :service_reply)
      end)

    {:ok, rendered, text, nil}
  end

  defp realize_by_system(primitives, opts) do
    analysis = Keyword.get(opts, :analysis)
    domain = extract_domain(analysis)

    {system, domain_config} = ResponseSystemRouter.route(domain)

    case system do
      :lattice ->
        case try_lattice_realization(primitives, analysis, domain_config, opts) do
          {:ok, rendered, text} -> {:ok, rendered, text, nil}
          {:error, lattice_reason} -> try_synthesizer_fallback(primitives, analysis, opts, {:lattice, lattice_reason})
        end

      :ouro ->
        try_ouro_or_fallback(primitives, analysis, opts)

      :synthesizer ->
        try_synthesizer_fallback(primitives, analysis, opts, :synthesizer_configured)

      :template ->
        case try_enriched_fallback(primitives, :template_configured, opts) do
          {:ok, _, _, _} = result -> result
          {:error, reason} -> try_synthesizer_fallback(primitives, analysis, opts, {:template, reason})
        end
    end
  end

  defp world_id!(opts) do
    case Keyword.fetch(opts, :world_id) do
      {:ok, world_id} when is_binary(world_id) and world_id != "" ->
        world_id

      _ ->
        raise ArgumentError,
              "SurfaceRealizer.realize needs :world_id, the world the response is realized in, got: #{inspect(Keyword.get(opts, :world_id))}"
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
      {:ok, :ouro_dry_run, %{messages: messages, recurring_pattern: recurring_pattern}} ->
        Logger.info("SurfaceRealizer: dry_run_ouro=true, returning ChatML messages without rendering")
        {:ok, primitives, {:ouro_dry_run, messages}, recurring_pattern_use(recurring_pattern, :realization_packet)}

      {:ok, text, %{recurring_pattern: recurring_pattern}} ->
        Logger.info("SurfaceRealizer: Ouro realized #{length(primitives)} primitives")

        rendered =
          Enum.map(primitives, fn p ->
            p |> Primitive.render(text) |> Map.put(:source, :ouro)
          end)

        # The realized pairs and plan are decompressor training data; a run
        # with side effects off keeps them out of it.
        if Keyword.get(opts, :side_effects, true) do
          collect_plan(primitives, text, opts)
          collect_pairs(primitives, world_id!(opts))
        end

        {:ok, rendered, text, recurring_pattern_use(recurring_pattern, :realization_packet)}

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

      {:ok, rendered, text, nil}
    else
      {:error, original_reason}
    end
  end

  @doc """
  The options the synthesizer fallback is called with: the analysis's
  confidence, the unified context, and the turn's recurring pattern and similar
  episodes, which ContextBuilder retrieved into the context's memory. A context
  built without memory has neither.
  """
  @spec synthesizer_opts(map() | nil, keyword()) :: keyword()
  def synthesizer_opts(analysis, opts) do
    unified_context = Keyword.get(opts, :unified_context, %{})

    [
      confidence: (analysis && Map.get(analysis, :confidence)) || 0.5,
      context: unified_context,
      recurring_pattern: get_in(unified_context, [:memory, :recurring_pattern]),
      similar_episodes: get_in(unified_context, [:memory, :similar_episodes]) || []
    ]
  end

  # `prior` is why the domain's own system did not realize the plan, kept in
  # the error when the synthesizer cannot either.
  defp try_synthesizer_fallback(primitives, analysis, opts, prior) do
    intent = if analysis, do: Map.get(analysis, :intent), else: nil
    entities = if analysis, do: Map.get(analysis, :entities, []), else: []

    case Brain.Response.Synthesizer.synthesize(intent, entities, synthesizer_opts(analysis, opts)) do
      {:ok, text, recurring_pattern} when is_binary(text) and text != "" ->
        Logger.info("SurfaceRealizer: template fallback via Synthesizer for #{inspect(intent)}")

        rendered =
          Enum.map(primitives, fn p ->
            p |> Primitive.render(text) |> Map.put(:source, :synthesizer_template)
          end)

        {:ok, rendered, text, recurring_pattern_use(recurring_pattern, :synthesizer)}

      :not_synthesized ->
        {:error, {:synthesizer_fallback_failed, prior}}
    end
  end

  defp recurring_pattern_use(nil, _used_by), do: nil
  defp recurring_pattern_use(%RecurringPattern{} = pattern, used_by), do: {pattern, used_by}

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

  defp collect_pairs(primitives, world_id) do
    Enum.each(primitives, fn p ->
      collect_pair(p, p.rendered || "", world_id)
    end)
  end

  defp collect_pair(p, text, world_id) do
    if Code.ensure_loaded?(DecompressorCollector) and function_exported?(DecompressorCollector, :collect, 3) do
      DecompressorCollector.collect(p, text, world_id)
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
