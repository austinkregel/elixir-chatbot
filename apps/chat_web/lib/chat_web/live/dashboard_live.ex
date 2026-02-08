defmodule ChatWeb.DashboardLive do
  @moduledoc """
  Operational dashboard for monitoring GenServer statuses, performance metrics,
  and system health indicators.

  Provides real-time visibility into:
  - All GenServers organized by category (Core, Epistemic, Analysis, ML, Storage)
  - Performance metrics (processing times, throughput, queue sizes)
  - Health indicators (uptime, error rates, overall health score)
  - World-specific memory and knowledge stats
  """

  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.KnowledgeStore

  # Refresh interval in milliseconds
  @refresh_interval_ms 2_000

  @default_expanded [:core, :epistemic, :analysis, :ml, :knowledge, :learning, :storage, :metrics, :code_analysis]

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Start periodic refresh
      :timer.send_interval(@refresh_interval_ms, self(), :refresh_dashboard)
    end

    {:ok, socket}
  end

  @impl true
  def handle_params(_params, _uri, socket) do
    # Load initial data with world context
    world_id = socket.assigns.current_world_id

    socket =
      socket
      |> assign(:genserver_status, load_genserver_status())
      |> assign(:performance_metrics, load_performance_metrics())
      |> assign(:health_indicators, load_health_indicators())
      |> assign(:ml_models_status, load_ml_models_status())
      |> assign(:readiness_details, load_readiness_details(world_id))
      |> assign(:training_worlds_status, load_training_worlds_status())
      |> assign(:world_memory_stats, load_world_memory_stats(world_id))
      |> assign(:world_models_status, load_world_models_status(world_id))
      |> assign(:code_analysis_status, load_code_analysis_status())
      |> assign(:last_updated, DateTime.utc_now())
      |> assign(:expanded_categories, MapSet.new(@default_expanded))
      |> assign(:auto_refresh, true)

    {:noreply, socket}
  end

  defp load_world_memory_stats(world_id) do
    episodes =
      case MemoryStore.all_episodes(world_id: world_id) do
        {:ok, eps} -> length(eps)
        _ -> 0
      end

    semantics =
      case MemoryStore.all_semantics(world_id: world_id) do
        {:ok, sems} -> length(sems)
        _ -> 0
      end

    knowledge =
      case KnowledgeStore.get_world_knowledge(world_id) do
        k when is_map(k) -> map_size(k)
        _ -> 0
      end

    %{
      episodes: episodes,
      semantics: semantics,
      knowledge_categories: knowledge
    }
  end

  @impl true
  def handle_info(:refresh_dashboard, socket) do
    if socket.assigns.auto_refresh do
      world_id = socket.assigns.current_world_id

      socket =
        socket
        |> assign(:genserver_status, load_genserver_status())
        |> assign(:performance_metrics, load_performance_metrics())
        |> assign(:health_indicators, load_health_indicators())
        |> assign(:ml_models_status, load_ml_models_status())
        |> assign(:readiness_details, load_readiness_details(world_id))
        |> assign(:training_worlds_status, load_training_worlds_status())
        |> assign(:world_memory_stats, load_world_memory_stats(world_id))
        |> assign(:world_models_status, load_world_models_status(world_id))
        |> assign(:code_analysis_status, load_code_analysis_status())
        |> assign(:last_updated, DateTime.utc_now())

      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end

  def handle_info({:world_context_changed, world_id}, socket) do
    # World was changed from another LiveView or tab - sync our data
    {:noreply, reload_world_data(socket, world_id)}
  end

  @impl true
  def handle_event("toggle_auto_refresh", _params, socket) do
    new_auto_refresh = !socket.assigns.auto_refresh
    {:noreply, assign(socket, :auto_refresh, new_auto_refresh)}
  end

  def handle_event("manual_refresh", _params, socket) do
    world_id = socket.assigns.current_world_id

    socket =
      socket
      |> assign(:genserver_status, load_genserver_status())
      |> assign(:performance_metrics, load_performance_metrics())
      |> assign(:health_indicators, load_health_indicators())
      |> assign(:ml_models_status, load_ml_models_status())
      |> assign(:readiness_details, load_readiness_details(world_id))
      |> assign(:training_worlds_status, load_training_worlds_status())
      |> assign(:world_memory_stats, load_world_memory_stats(world_id))
      |> assign(:world_models_status, load_world_models_status(world_id))
      |> assign(:code_analysis_status, load_code_analysis_status())
      |> assign(:last_updated, DateTime.utc_now())

    {:noreply, socket}
  end

  def handle_event("switch_world", %{"world_id" => world_id}, socket) do
    # World context hook already updated current_world_id and broadcast the change
    # Reload all world-specific data
    {:noreply, reload_world_data(socket, world_id)}
  end

  def handle_event("refresh_worlds", _params, socket) do
    # World context hook already refreshed available_worlds
    {:noreply, socket}
  end

  def handle_event("toggle_category", %{"category" => category}, socket) do
    category = String.to_existing_atom(category)
    expanded = socket.assigns.expanded_categories

    new_expanded =
      if MapSet.member?(expanded, category) do
        MapSet.delete(expanded, category)
      else
        MapSet.put(expanded, category)
      end

    {:noreply, assign(socket, :expanded_categories, new_expanded)}
  end

  def handle_event("reload_training_worlds", _params, socket) do
    case World.Manager.reload_persisted_worlds() do
      {:ok, loaded} ->
        socket =
          socket
          |> assign(:training_worlds_status, load_training_worlds_status())
          |> put_flash(:info, "Reloaded #{loaded} world(s) from disk")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to reload: #{inspect(reason)}")}
    end
  end

  defp reload_world_data(socket, world_id) do
    socket
    |> assign(:world_memory_stats, load_world_memory_stats(world_id))
    |> assign(:world_models_status, load_world_models_status(world_id))
    |> assign(:readiness_details, load_readiness_details(world_id))
  end

  # ============================================================================
  # Data Loading Functions
  # ============================================================================

  defp load_genserver_status do
    Brain.SystemStatus.get_all_genservers_status()
  end

  defp load_performance_metrics do
    Brain.SystemStatus.get_performance_metrics()
  end

  defp load_health_indicators do
    Brain.SystemStatus.get_health_indicators()
  end

  defp load_ml_models_status do
    Brain.SystemStatus.get_ml_models_status()
  end

  defp load_readiness_details(world_id) do
    Brain.SystemStatus.get_readiness_details(world_id: world_id)
  end

  defp load_training_worlds_status do
    Brain.SystemStatus.get_training_worlds_status()
  end

  defp load_world_models_status(world_id) do
    Brain.SystemStatus.get_world_models_status(world_id)
  end

  defp load_code_analysis_status do
    Brain.SystemStatus.get_code_analysis_status()
  end

  # ============================================================================
  # Helper Functions for Template
  # ============================================================================

  def category_label(:core), do: "Core Systems"
  def category_label(:epistemic), do: "Epistemic System"
  def category_label(:analysis), do: "Analysis System"
  def category_label(:ml), do: "Machine Learning"
  def category_label(:knowledge), do: "Knowledge Expansion"
  def category_label(:learning), do: "Training Worlds"
  def category_label(:storage), do: "Storage"
  def category_label(:metrics), do: "Metrics & Telemetry"
  def category_label(:code_analysis), do: "Code Analysis"
  def category_label(other), do: to_string(other) |> String.capitalize()

  def category_icon(:core), do: "hero-cpu-chip"
  def category_icon(:epistemic), do: "hero-light-bulb"
  def category_icon(:analysis), do: "hero-chart-bar"
  def category_icon(:ml), do: "hero-sparkles"
  def category_icon(:knowledge), do: "hero-book-open"
  def category_icon(:learning), do: "hero-academic-cap"
  def category_icon(:storage), do: "hero-circle-stack"
  def category_icon(:metrics), do: "hero-chart-pie"
  def category_icon(:code_analysis), do: "hero-code-bracket"
  def category_icon(_), do: "hero-cube"

  def status_color(:ready), do: "text-success"
  def status_color(:running), do: "text-success"
  def status_color(:initializing), do: "text-warning"
  def status_color(:building_vocabulary), do: "text-warning"
  def status_color(:tokenizing), do: "text-warning"
  def status_color(:building_frequencies), do: "text-warning"
  def status_color(:calculating_idf), do: "text-warning"
  def status_color(:loading), do: "text-warning"
  def status_color(:busy), do: "text-warning"
  def status_color(:idle), do: "text-info"
  def status_color(:not_started), do: "text-error"
  def status_color(_), do: "text-base-content/50"

  # Note: status_dot_color is no longer used, we use <.status_dot> component instead
  def status_dot_color(:ready), do: "bg-success"
  def status_dot_color(:running), do: "bg-success"
  def status_dot_color(:initializing), do: "bg-warning"
  def status_dot_color(:building_vocabulary), do: "bg-warning"
  def status_dot_color(:tokenizing), do: "bg-warning"
  def status_dot_color(:building_frequencies), do: "bg-warning"
  def status_dot_color(:calculating_idf), do: "bg-warning"
  def status_dot_color(:loading), do: "bg-warning"
  def status_dot_color(:busy), do: "bg-warning"
  def status_dot_color(:idle), do: "bg-info"
  def status_dot_color(:not_started), do: "bg-error"
  def status_dot_color(_), do: "bg-base-content/50"

  def health_status_color(:healthy), do: "text-success"
  def health_status_color(:degraded), do: "text-warning"
  def health_status_color(:warning), do: "text-warning"
  def health_status_color(:critical), do: "text-error"
  def health_status_color(_), do: "text-base-content/50"

  def health_badge_class(:healthy), do: "badge-success"
  def health_badge_class(:degraded), do: "badge-warning"
  def health_badge_class(:warning), do: "badge-warning"
  def health_badge_class(:critical), do: "badge-error"
  def health_badge_class(_), do: "badge-ghost"

  # Maps health status to UI component variant atoms
  def health_variant(:healthy), do: :success
  def health_variant(:degraded), do: :warning
  def health_variant(:warning), do: :warning
  def health_variant(:critical), do: :error
  def health_variant(_), do: :default

  def format_bytes(nil), do: "-"
  def format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  def format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"
  def format_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024), 2)} MB"

  def format_uptime(seconds) when seconds < 60, do: "#{seconds}s"
  def format_uptime(seconds) when seconds < 3600, do: "#{div(seconds, 60)}m #{rem(seconds, 60)}s"

  def format_uptime(seconds) when seconds < 86400 do
    hours = div(seconds, 3600)
    minutes = div(rem(seconds, 3600), 60)
    "#{hours}h #{minutes}m"
  end

  def format_uptime(seconds) do
    days = div(seconds, 86400)
    hours = div(rem(seconds, 86400), 3600)
    "#{days}d #{hours}h"
  end

  def format_rate(nil), do: "-"
  def format_rate(rate) when is_float(rate), do: "#{Float.round(rate, 1)}/min"
  def format_rate(rate), do: "#{rate}/min"

  def format_ms(nil), do: "-"
  def format_ms(ms) when is_float(ms), do: "#{Float.round(ms, 1)}ms"
  def format_ms(ms), do: "#{ms}ms"

  def format_datetime(nil), do: "-"

  def format_datetime(%DateTime{} = dt) do
    Calendar.strftime(dt, "%H:%M:%S")
  end

  def category_servers(categories, category) do
    Map.get(categories, category, %{})
    |> Enum.sort_by(fn {_module, status} -> status.name end)
  end

  def count_running_in_category(categories, category) do
    Map.get(categories, category, %{})
    |> Enum.count(fn {_module, status} -> status.running end)
  end

  def count_total_in_category(categories, category) do
    Map.get(categories, category, %{}) |> map_size()
  end

  # Category styling helpers
  def category_bg_class(:core), do: "bg-primary/10"
  def category_bg_class(:epistemic), do: "bg-secondary/10"
  def category_bg_class(:analysis), do: "bg-accent/10"
  def category_bg_class(:ml), do: "bg-warning/10"
  def category_bg_class(:knowledge), do: "bg-cyan-500/10"
  def category_bg_class(:learning), do: "bg-error/10"
  def category_bg_class(:storage), do: "bg-info/10"
  def category_bg_class(:metrics), do: "bg-success/10"
  def category_bg_class(:code_analysis), do: "bg-violet-500/10"
  def category_bg_class(_), do: "bg-base-200"

  def category_text_class(:core), do: "text-primary"
  def category_text_class(:epistemic), do: "text-secondary"
  def category_text_class(:analysis), do: "text-accent"
  def category_text_class(:ml), do: "text-warning"
  def category_text_class(:knowledge), do: "text-cyan-500"
  def category_text_class(:learning), do: "text-error"
  def category_text_class(:storage), do: "text-info"
  def category_text_class(:metrics), do: "text-success"
  def category_text_class(:code_analysis), do: "text-violet-500"
  def category_text_class(_), do: "text-base-content"

  # Badge variant based on status
  def status_badge_variant(:ready), do: :success
  def status_badge_variant(:running), do: :success
  def status_badge_variant(:initializing), do: :warning
  def status_badge_variant(:building_vocabulary), do: :warning
  def status_badge_variant(:tokenizing), do: :warning
  def status_badge_variant(:building_frequencies), do: :warning
  def status_badge_variant(:calculating_idf), do: :warning
  def status_badge_variant(:loading), do: :warning
  def status_badge_variant(:busy), do: :warning
  def status_badge_variant(:idle), do: :info
  def status_badge_variant(:not_started), do: :error
  def status_badge_variant(_), do: :default

  # Format stat values for display
  def format_stat_value(value) when is_binary(value), do: value
  def format_stat_value(value) when is_integer(value), do: Integer.to_string(value)
  def format_stat_value(value) when is_float(value), do: Float.round(value, 2) |> to_string()
  def format_stat_value(value) when is_boolean(value), do: to_string(value)
  def format_stat_value(value) when is_list(value), do: "[#{length(value)}]"
  def format_stat_value(value) when is_map(value), do: "{#{map_size(value)}}"
  def format_stat_value(value), do: inspect(value)

  # ============================================================================
  # ML Model Status Helpers
  # ============================================================================

  def model_status_variant(%{exists: true, loaded: true}), do: :success
  def model_status_variant(%{exists: true, loaded: false}), do: :warning
  def model_status_variant(%{exists: false}), do: :error
  def model_status_variant(%{loaded: true}), do: :success
  def model_status_variant(%{loaded: false}), do: :error
  def model_status_variant(_), do: :default

  def model_status_label(%{exists: true, loaded: true}), do: "Loaded"
  def model_status_label(%{exists: true, loaded: false}), do: "Not Loaded"
  def model_status_label(%{exists: false}), do: "Not Trained"
  def model_status_label(%{loaded: true}), do: "Loaded"
  def model_status_label(%{loaded: false}), do: "Not Loaded"
  def model_status_label(_), do: "Unknown"

  def format_model_datetime(nil), do: "Never"

  def format_model_datetime(%DateTime{} = dt) do
    Calendar.strftime(dt, "%Y-%m-%d %H:%M")
  end

  def format_model_datetime({{year, month, day}, {hour, min, _sec}}) do
    "#{year}-#{String.pad_leading("#{month}", 2, "0")}-#{String.pad_leading("#{day}", 2, "0")} #{String.pad_leading("#{hour}", 2, "0")}:#{String.pad_leading("#{min}", 2, "0")}"
  end

  def format_model_datetime(_), do: "-"

  def training_status_variant(:completed), do: :success
  def training_status_variant(:in_progress), do: :warning
  def training_status_variant(:failed), do: :error
  def training_status_variant(_), do: :default

  def training_status_label(:completed), do: "Completed"
  def training_status_label(:in_progress), do: "In Progress"
  def training_status_label(:failed), do: "Failed"
  def training_status_label(nil), do: "Never Run"
  def training_status_label(_), do: "Unknown"

  def model_name(:pos_model), do: "POS Tagger"
  def model_name(:entity_model), do: "Entity Model"
  def model_name(:classifier), do: "Intent Classifier"
  def model_name(:gazetteer), do: "Gazetteer"
  def model_name(:intent_classifier), do: "Intent Classifier (Agent)"
  def model_name(:entity_extractor), do: "Entity Extractor (Agent)"
  def model_name(:pos_tagger), do: "POS Tagger"
  def model_name(:entity_trainer), do: "Entity Trainer"
  def model_name(:unified_model), do: "Unified LSTM"
  def model_name(:multi_task_model), do: "Multi-Task LSTM"
  def model_name(:response_scorer), do: "Response Scorer"

  def model_name(other),
    do: other |> to_string() |> String.replace("_", " ") |> String.capitalize()

  # Get list of file-based models for display
  def file_based_models(ml_models_status) do
    [:pos_model, :entity_model, :classifier, :gazetteer]
    |> Enum.map(fn key -> {key, Map.get(ml_models_status, key)} end)
    |> Enum.filter(fn {_k, v} -> v != nil end)
  end

  # Get list of agent-based models for display
  def agent_based_models(ml_models_status) do
    [:intent_classifier, :entity_extractor]
    |> Enum.map(fn key -> {key, Map.get(ml_models_status, key)} end)
    |> Enum.filter(fn {_k, v} -> v != nil end)
  end

  # Get list of LSTM models for display
  def lstm_models(ml_models_status) do
    [:unified_model, :multi_task_model, :response_scorer]
    |> Enum.map(fn key -> {key, Map.get(ml_models_status, key)} end)
    |> Enum.filter(fn {_k, v} -> v != nil end)
  end

  # Get corpus size info
  def corpus_info do
    try do
      size_info = Brain.ML.CorpusManager.size_by_category()

      %{
        total: Brain.ML.CorpusManager.format_bytes(size_info.total),
        utilization: Brain.ML.CorpusManager.utilization_percent(),
        categories: %{
          training: Brain.ML.CorpusManager.format_bytes(size_info.training_data),
          models: Brain.ML.CorpusManager.format_bytes(size_info.ml_models),
          evaluation: Brain.ML.CorpusManager.format_bytes(size_info.evaluation),
          worlds: Brain.ML.CorpusManager.format_bytes(size_info.training_worlds),
          knowledge: Brain.ML.CorpusManager.format_bytes(size_info.knowledge)
        }
      }
    rescue
      _ -> %{total: "N/A", utilization: 0.0, categories: %{}}
    end
  end

  # Get training stats from performance metrics
  def get_training_stats(performance_metrics) do
    Map.get(performance_metrics, :training, %{})
  end

  # Get all categories including learning, metrics, and code_analysis categories
  def all_categories do
    [:core, :epistemic, :analysis, :ml, :knowledge, :learning, :storage, :metrics, :code_analysis]
  end

  # ============================================================================
  # Embedder Status Helpers
  # ============================================================================

  @doc """
  Determine the status dot indicator for the embedder.
  Idle is shown as info (blue), building as warning, ready as success.
  """
  def embedder_status_for_dot(%{ready: true}), do: :ready
  def embedder_status_for_dot(%{phase: :idle}), do: :idle
  def embedder_status_for_dot(%{phase: :not_started}), do: :not_started
  def embedder_status_for_dot(_), do: :initializing

  @doc """
  Check if the embedder is actively building vocabulary (should show progress).
  Returns false for idle state (on-demand, not yet used).
  """
  def embedder_is_building?(%{ready: true}), do: false
  def embedder_is_building?(%{phase: :idle}), do: false
  def embedder_is_building?(%{phase: :not_started}), do: false
  def embedder_is_building?(_), do: true

  # ============================================================================
  # World Embedder Status Helpers
  # ============================================================================

  @doc """
  Determine the status dot indicator for the world-specific embedder.
  """
  def world_embedder_status_for_dot(%{ready: true}), do: :ready
  def world_embedder_status_for_dot(%{phase: :not_initialized}), do: :idle
  def world_embedder_status_for_dot(%{phase: :table_not_ready}), do: :warning
  def world_embedder_status_for_dot(%{phase: :no_data}), do: :warning
  def world_embedder_status_for_dot(_), do: :initializing

  @doc """
  Check if the world embedder is actively building vocabulary.
  """
  def world_embedder_is_building?(%{ready: true}), do: false
  def world_embedder_is_building?(%{phase: :not_initialized}), do: false
  def world_embedder_is_building?(%{phase: :table_not_ready}), do: false
  def world_embedder_is_building?(%{phase: :no_data}), do: false
  def world_embedder_is_building?(%{phase: :ready}), do: false
  def world_embedder_is_building?(_), do: true
end
