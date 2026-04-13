defmodule Brain.Services.SystemStatus do
  @moduledoc """
  System status service integration — purely internal, no external API.

  Provides live system health, GenServer status, ML model readiness,
  and performance metrics for conversational status queries.

  ## Enrichment Fields

  This service provides the following fields for template enrichment:
  - `:component_count` - Total GenServer/component count (alias for genservers_total)
  - `:health_status` - Overall health label (e.g., "healthy", "degraded")
  - `:health_score` - Numeric health score (e.g., "95%")
  - `:genservers_running` - Count of running GenServers
  - `:genservers_total` - Total GenServer count
  - `:uptime` - Formatted uptime string (e.g., "2 hours, 15 minutes")
  - `:ml_models_summary` - Summary of ML model readiness
  - `:memory_summary` - Memory store episode/semantic counts
  - `:services_summary` - External services status
  - `:categories_summary` - Per-category health breakdown

  ## Intent Discovery

  Supported intents are discovered dynamically from IntentRegistry.
  Any intent with `"enrichment_sources": ["system_stats"]` in its
  registry metadata will be routed to this service. No hardcoded
  intent lists are maintained.
  """

  @behaviour Brain.Services.Service

  alias Brain.Analysis.IntentRegistry

  require Logger

  # ============================================================================
  # Service Behaviour Implementation
  # ============================================================================

  @impl true
  def name, do: :system_stats

  @impl true
  def display_name, do: "System Status"

  @impl true
  def description do
    "Live system health, GenServer status, and ML model readiness"
  end

  @impl true
  def required_credentials, do: []

  @impl true
  def supported_intents do
    if IntentRegistry.ready?() do
      IntentRegistry.intents_with_enrichment_source("system_stats")
    else
      []
    end
  end

  @impl true
  def provides_fields do
    [
      :component_count,
      :health_status,
      :health_score,
      :genservers_running,
      :genservers_total,
      :uptime,
      :ml_models_summary,
      :memory_summary,
      :services_summary,
      :categories_summary
    ]
  end

  @impl true
  def enrich(_intent, _slots, _credentials) do
    health = safe_get_health_indicators()
    memory = safe_get_memory_status()
    models = safe_get_ml_summary()

    {:ok,
     %{
       component_count: to_string(health.genservers_total),
       health_status: format_health_status(health.health_status),
       health_score: "#{health.health_score}%",
       genservers_running: to_string(health.genservers_running),
       genservers_total: to_string(health.genservers_total),
       uptime: format_uptime(health.uptime_seconds),
       ml_models_summary: models,
       memory_summary: memory,
       services_summary: safe_get_services_summary(),
       categories_summary: safe_get_categories_summary()
     }}
  end

  @impl true
  def health_check(_credentials) do
    # Internal service — always healthy if the BEAM is running
    :ok
  end

  # ============================================================================
  # Data Fetchers (with graceful fallbacks)
  # ============================================================================

  defp safe_get_health_indicators do
    Brain.SystemStatus.get_health_indicators()
  rescue
    _ ->
      %{
        health_score: 0,
        health_status: :unknown,
        genservers_running: 0,
        genservers_total: 0,
        uptime_seconds: 0
      }
  end

  defp safe_get_memory_status do
    status = Brain.SystemStatus.get_memory_store_status()
    episodes = Map.get(status, :episodes, 0)
    semantics = Map.get(status, :semantics, 0)

    if status.ready do
      "#{episodes} episodes, #{semantics} semantic memories"
    else
      "memory store initializing"
    end
  rescue
    _ -> "unavailable"
  end

  defp safe_get_ml_summary do
    models = Brain.SystemStatus.get_ml_models_status()

    loaded =
      [:intent_classifier, :entity_extractor]
      |> Enum.count(fn key ->
        get_in(models, [key, :loaded]) == true
      end)

    lstm_ready =
      [:unified_model, :response_scorer]
      |> Enum.count(fn key ->
        get_in(models, [key, :ready]) == true
      end)

    "#{loaded}/2 classical models loaded, #{lstm_ready}/3 LSTM models ready"
  rescue
    _ -> "unavailable"
  end

  defp safe_get_services_summary do
    services_status = Brain.SystemStatus.get_services_status()
    services = Map.get(services_status, :services, [])
    configured = Enum.count(services, & &1.configured)
    total = length(services)

    if total > 0 do
      "#{configured}/#{total} services configured"
    else
      "no external services registered"
    end
  rescue
    _ -> "unavailable"
  end

  defp safe_get_categories_summary do
    genservers = Brain.SystemStatus.get_all_genservers_status()
    categories = Map.get(genservers, :categories, %{})

    categories
    |> Enum.map(fn {category, servers} ->
      running = Enum.count(servers, fn {_mod, s} -> s.running end)
      total = map_size(servers)
      "#{category}: #{running}/#{total}"
    end)
    |> Enum.join(", ")
  rescue
    _ -> "unavailable"
  end

  # ============================================================================
  # Formatting Helpers
  # ============================================================================

  defp format_health_status(:healthy), do: "healthy"
  defp format_health_status(:degraded), do: "degraded"
  defp format_health_status(:warning), do: "warning"
  defp format_health_status(:critical), do: "critical"
  defp format_health_status(other) when is_atom(other), do: to_string(other)
  defp format_health_status(_), do: "unknown"

  defp format_uptime(seconds) when is_integer(seconds) and seconds >= 0 do
    days = div(seconds, 86400)
    hours = div(rem(seconds, 86400), 3600)
    minutes = div(rem(seconds, 3600), 60)

    parts =
      [
        {days, "day", "days"},
        {hours, "hour", "hours"},
        {minutes, "minute", "minutes"}
      ]
      |> Enum.filter(fn {val, _, _} -> val > 0 end)
      |> Enum.map(fn
        {1, singular, _} -> "1 #{singular}"
        {n, _, plural} -> "#{n} #{plural}"
      end)

    case parts do
      [] -> "less than a minute"
      _ -> Enum.join(parts, ", ")
    end
  end

  defp format_uptime(_), do: "unknown"
end
