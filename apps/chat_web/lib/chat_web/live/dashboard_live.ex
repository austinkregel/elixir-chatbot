defmodule ChatWeb.DashboardLive do
  @moduledoc "Operational dashboard for monitoring GenServer statuses, performance metrics,\nand system health indicators.\n\nProvides real-time visibility into:\n- All GenServers organized by category (Core, Epistemic, Analysis, ML, Storage)\n- Performance metrics (processing times, throughput, queue sizes)\n- Health indicators (uptime, error rates, overall health score)\n- World-specific memory and knowledge stats\n"

  alias Brain.ML.CorpusManager
  alias Brain.SystemStatus
  alias World.Manager
  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.KnowledgeStore
  @refresh_interval_ms 5000

  @default_expanded [
    :core,
    :epistemic,
    :analysis,
    :ml,
    :knowledge,
    :learning,
    :storage,
    :metrics,
    :code_analysis,
    :services
  ]

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      :timer.send_interval(@refresh_interval_ms, self(), :refresh_dashboard)
      Phoenix.PubSub.subscribe(Brain.PubSub, "evaluation:complete")
    end

    {:ok, socket}
  end

  @impl true
  def handle_params(_params, _uri, socket) do
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
      |> assign(:services_status, load_services_status())
      |> assign(:epistemic_metrics, load_epistemic_metrics())
      |> assign(:knowledge_health, load_knowledge_health())
      |> assign(:micro_classifiers_status, load_micro_classifiers_status())
      |> assign(:response_timing, load_response_timing())
      |> assign(:atlas_stats, load_atlas_stats())
      |> assign(:evaluation_summary, load_evaluation_summary())
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
        |> assign(:services_status, load_services_status())
        |> assign(:epistemic_metrics, load_epistemic_metrics())
        |> assign(:knowledge_health, load_knowledge_health())
        |> assign(:micro_classifiers_status, load_micro_classifiers_status())
        |> assign(:response_timing, load_response_timing())
        |> assign(:atlas_stats, load_atlas_stats())
        |> assign(:evaluation_summary, load_evaluation_summary())
        |> assign(:last_updated, DateTime.utc_now())

      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end

  def handle_info({:evaluation_complete, _payload}, socket) do
    {:noreply, assign(socket, :evaluation_summary, load_evaluation_summary())}
  end

  def handle_info({:world_context_changed, world_id}, socket) do
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
      |> assign(:services_status, load_services_status())
      |> assign(:epistemic_metrics, load_epistemic_metrics())
      |> assign(:knowledge_health, load_knowledge_health())
      |> assign(:micro_classifiers_status, load_micro_classifiers_status())
      |> assign(:response_timing, load_response_timing())
      |> assign(:atlas_stats, load_atlas_stats())
      |> assign(:evaluation_summary, load_evaluation_summary())
      |> assign(:last_updated, DateTime.utc_now())

    {:noreply, socket}
  end

  def handle_event("switch_world", %{"world_id" => world_id}, socket) do
    {:noreply, reload_world_data(socket, world_id)}
  end

  def handle_event("refresh_worlds", _params, socket) do
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
    case Manager.reload_persisted_worlds() do
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

  defp load_genserver_status do
    SystemStatus.get_all_genservers_status()
  end

  defp load_performance_metrics do
    SystemStatus.get_performance_metrics()
  end

  defp load_health_indicators do
    SystemStatus.get_health_indicators()
  end

  defp load_ml_models_status do
    SystemStatus.get_ml_models_status()
  end

  defp load_readiness_details(world_id) do
    SystemStatus.get_readiness_details(world_id: world_id)
  end

  defp load_training_worlds_status do
    SystemStatus.get_training_worlds_status()
  end

  defp load_world_models_status(world_id) do
    SystemStatus.get_world_models_status(world_id)
  end

  defp load_code_analysis_status do
    SystemStatus.get_code_analysis_status()
  end

  defp load_services_status do
    SystemStatus.get_services_status()
  end

  defp load_epistemic_metrics do
    metrics = Brain.Metrics.Aggregator.get_epistemic_metrics()
    Map.put(metrics, :miss_reasons, Brain.Metrics.Aggregator.get_verification_miss_reasons())
  end

  defp load_knowledge_health do
    fact_stats =
      try do
        if Brain.FactDatabase.ready?(), do: Brain.FactDatabase.stats(), else: %{}
      catch
        :exit, _ -> %{}
      end

    jtms_stats =
      try do
        if Brain.Epistemic.JTMS.ready?(), do: Brain.Epistemic.JTMS.stats(), else: %{}
      catch
        :exit, _ -> %{}
      end

    gazetteer_stats =
      try do
        Brain.ML.Gazetteer.stats()
      rescue
        _ -> %{}
      end

    belief_count =
      try do
        case Brain.Epistemic.BeliefStore.query_beliefs([]) do
          {:ok, beliefs} -> length(beliefs)
          _ -> 0
        end
      catch
        :exit, _ -> 0
      end

    epistemic = Brain.Metrics.Aggregator.get_epistemic_metrics()
    by_status = Map.get(epistemic, :by_status, %{})
    verified = Map.get(by_status, :verified, 0)
    total_checked = verified + Map.get(by_status, :uncertain, 0) + Map.get(by_status, :unchecked, 0)

    verification_coverage =
      if total_checked > 0, do: Float.round(verified / total_checked * 100, 1), else: 0.0

    enrichment = Brain.Metrics.Aggregator.get_enrichment_metrics()

    %{
      curated_facts: Map.get(fact_stats, :curated_facts, 0),
      learned_facts: Map.get(fact_stats, :learned_facts, 0),
      total_facts: Map.get(fact_stats, :total_facts, 0),
      beliefs: belief_count,
      jtms_nodes: Map.get(jtms_stats, :total_nodes, 0),
      jtms_justifications: Map.get(jtms_stats, :justifications, 0),
      gazetteer_entities: Map.get(gazetteer_stats, :entities, 0),
      gazetteer_types: Map.get(gazetteer_stats, :entity_types, 0),
      gazetteer_loaded: Map.get(gazetteer_stats, :loaded, false),
      verification_coverage: verification_coverage,
      fact_hit_rate: Map.get(enrichment, :fact_hit_rate, 0.0),
      semantic_hit_rate: Map.get(enrichment, :semantic_hit_rate, 0.0),
      avg_facts_per_response: Map.get(enrichment, :avg_facts_per_response, 0.0)
    }
  rescue
    _ ->
      %{
        curated_facts: 0, learned_facts: 0, total_facts: 0, beliefs: 0,
        jtms_nodes: 0, jtms_justifications: 0,
        gazetteer_entities: 0, gazetteer_types: 0, gazetteer_loaded: false,
        verification_coverage: 0.0,
        fact_hit_rate: 0.0, semantic_hit_rate: 0.0, avg_facts_per_response: 0.0
      }
  end

  defp load_micro_classifiers_status do
    SystemStatus.get_micro_classifiers_status()
  end

  defp load_response_timing do
    SystemStatus.get_response_timing()
  end

  defp load_atlas_stats do
    Atlas.Stats.get_overview()
  rescue
    _ ->
      %{
        connected: false,
        repo: %{credentials: 0, beliefs: 0, episodes: 0, semantic_facts: 0, review_candidates: 0, learned_facts: 0},
        graphs: %{},
        connection_pool: %{pool_size: 0, checked_out: 0, idle: 0},
        migrations: [],
        query_metrics: %{}
      }
  end

  defp load_evaluation_summary do
    aggregator_data = Brain.Metrics.Aggregator.get_evaluation_metrics()

    if aggregator_data == %{} do
      load_evaluation_from_store()
    else
      aggregator_data
    end
  rescue
    _ -> %{}
  end

  defp load_evaluation_from_store do
    ~w(intent ner sentiment speech_act)
    |> Enum.reduce(%{}, fn task, acc ->
      case Brain.ML.EvaluationStore.latest(task) do
        nil ->
          acc

        result ->
          Map.put(acc, task, %{
            accuracy: result["accuracy"] || 0.0,
            macro_f1: result["macro_f1"] || 0.0,
            weighted_f1: result["weighted_f1"] || 0.0,
            total_examples: result["total_examples"] || 0,
            duration_ms: result["duration_ms"],
            completed_at: result["timestamp"]
          })
      end
    end)
  rescue
    _ -> %{}
  end

  def category_label(:core) do
    "Core Systems"
  end

  def category_label(:epistemic) do
    "Epistemic System"
  end

  def category_label(:analysis) do
    "Analysis System"
  end

  def category_label(:ml) do
    "Machine Learning"
  end

  def category_label(:knowledge) do
    "Knowledge Expansion"
  end

  def category_label(:learning) do
    "Training Worlds"
  end

  def category_label(:storage) do
    "Storage"
  end

  def category_label(:metrics) do
    "Metrics & Telemetry"
  end

  def category_label(:code_analysis) do
    "Code Analysis"
  end

  def category_label(:services) do
    "External Services"
  end

  def category_label(:atlas) do
    "Atlas Database"
  end

  def category_label(other) do
    to_string(other) |> String.capitalize()
  end

  def category_icon(:core) do
    "hero-cpu-chip"
  end

  def category_icon(:epistemic) do
    "hero-light-bulb"
  end

  def category_icon(:analysis) do
    "hero-chart-bar"
  end

  def category_icon(:ml) do
    "hero-sparkles"
  end

  def category_icon(:knowledge) do
    "hero-book-open"
  end

  def category_icon(:learning) do
    "hero-academic-cap"
  end

  def category_icon(:storage) do
    "hero-circle-stack"
  end

  def category_icon(:metrics) do
    "hero-chart-pie"
  end

  def category_icon(:code_analysis) do
    "hero-code-bracket"
  end

  def category_icon(:services) do
    "hero-cloud"
  end

  def category_icon(:atlas) do
    "hero-map"
  end

  def category_icon(_) do
    "hero-cube"
  end

  # Epistemic status helpers for dashboard display
  def status_badge_class(:verified), do: "bg-success/20 text-success border border-success/30"
  def status_badge_class(:contradicted), do: "bg-error/20 text-error border border-error/30"
  def status_badge_class(:uncertain), do: "bg-warning/20 text-warning border border-warning/30"
  def status_badge_class(:unchecked), do: "bg-base-300/50 text-base-content/60 border border-base-300"
  def status_badge_class(_), do: "bg-base-300/50 text-base-content/60 border border-base-300"

  def status_icon(:verified), do: "hero-check-circle"
  def status_icon(:contradicted), do: "hero-x-circle"
  def status_icon(:uncertain), do: "hero-question-mark-circle"
  def status_icon(:unchecked), do: "hero-minus-circle"
  def status_icon(_), do: "hero-minus-circle"

  def reason_label(:no_subject), do: "No Subject"
  def reason_label(:no_facts), do: "No Facts"
  def reason_label(:no_beliefs), do: "No Beliefs"
  def reason_label(:low_confidence), do: "Low Confidence"
  def reason_label(other), do: to_string(other)

  def status_color(:ready) do
    "text-success"
  end

  def status_color(:running) do
    "text-success"
  end

  def status_color(:initializing) do
    "text-warning"
  end

  def status_color(:building_vocabulary) do
    "text-warning"
  end

  def status_color(:tokenizing) do
    "text-warning"
  end

  def status_color(:building_frequencies) do
    "text-warning"
  end

  def status_color(:calculating_idf) do
    "text-warning"
  end

  def status_color(:loading) do
    "text-warning"
  end

  def status_color(:busy) do
    "text-warning"
  end

  def status_color(:idle) do
    "text-info"
  end

  def status_color(:not_started) do
    "text-error"
  end

  def status_color(_) do
    "text-base-content/50"
  end

  def status_dot_color(:ready) do
    "bg-success"
  end

  def status_dot_color(:running) do
    "bg-success"
  end

  def status_dot_color(:initializing) do
    "bg-warning"
  end

  def status_dot_color(:building_vocabulary) do
    "bg-warning"
  end

  def status_dot_color(:tokenizing) do
    "bg-warning"
  end

  def status_dot_color(:building_frequencies) do
    "bg-warning"
  end

  def status_dot_color(:calculating_idf) do
    "bg-warning"
  end

  def status_dot_color(:loading) do
    "bg-warning"
  end

  def status_dot_color(:busy) do
    "bg-warning"
  end

  def status_dot_color(:idle) do
    "bg-info"
  end

  def status_dot_color(:not_started) do
    "bg-error"
  end

  def status_dot_color(_) do
    "bg-base-content/50"
  end

  def health_status_color(:healthy) do
    "text-success"
  end

  def health_status_color(:degraded) do
    "text-warning"
  end

  def health_status_color(:warning) do
    "text-warning"
  end

  def health_status_color(:critical) do
    "text-error"
  end

  def health_status_color(_) do
    "text-base-content/50"
  end

  def health_badge_class(:healthy) do
    "badge-success"
  end

  def health_badge_class(:degraded) do
    "badge-warning"
  end

  def health_badge_class(:warning) do
    "badge-warning"
  end

  def health_badge_class(:critical) do
    "badge-error"
  end

  def health_badge_class(_) do
    "badge-ghost"
  end

  def health_variant(:healthy) do
    :success
  end

  def health_variant(:degraded) do
    :warning
  end

  def health_variant(:warning) do
    :warning
  end

  def health_variant(:critical) do
    :error
  end

  def health_variant(_) do
    :default
  end

  def format_bytes(nil) do
    "-"
  end

  def format_bytes(bytes) when bytes < 1024 do
    "#{bytes} B"
  end

  def format_bytes(bytes) when bytes < 1024 * 1024 do
    "#{Float.round(bytes / 1024, 1)} KB"
  end

  def format_bytes(bytes) do
    "#{Float.round(bytes / (1024 * 1024), 2)} MB"
  end

  def format_uptime(seconds) when seconds < 60 do
    "#{seconds}s"
  end

  def format_uptime(seconds) when seconds < 3600 do
    "#{div(seconds, 60)}m #{rem(seconds, 60)}s"
  end

  def format_uptime(seconds) when seconds < 86_400 do
    hours = div(seconds, 3600)
    minutes = div(rem(seconds, 3600), 60)
    "#{hours}h #{minutes}m"
  end

  def format_uptime(seconds) do
    days = div(seconds, 86_400)
    hours = div(rem(seconds, 86_400), 3600)
    "#{days}d #{hours}h"
  end

  def format_rate(nil) do
    "-"
  end

  def format_rate(rate) when is_float(rate) do
    "#{Float.round(rate, 1)}/min"
  end

  def format_rate(rate) do
    "#{rate}/min"
  end

  def format_ms(nil) do
    "-"
  end

  def format_ms(ms) when is_float(ms) do
    "#{Float.round(ms, 1)}ms"
  end

  def format_ms(ms) do
    "#{ms}ms"
  end

  def format_datetime(nil) do
    "-"
  end

  def format_datetime(%DateTime{} = dt) do
    Calendar.strftime(dt, "%H:%M:%S")
  end

  def accuracy_color(accuracy) when is_number(accuracy) do
    cond do
      accuracy >= 0.8 -> "text-success"
      accuracy >= 0.6 -> "text-warning"
      true -> "text-error"
    end
  end

  def accuracy_color(_), do: "text-base-content/50"

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

  def category_bg_class(:core) do
    "bg-primary/10"
  end

  def category_bg_class(:epistemic) do
    "bg-secondary/10"
  end

  def category_bg_class(:analysis) do
    "bg-accent/10"
  end

  def category_bg_class(:ml) do
    "bg-warning/10"
  end

  def category_bg_class(:knowledge) do
    "bg-cyan-500/10"
  end

  def category_bg_class(:learning) do
    "bg-error/10"
  end

  def category_bg_class(:storage) do
    "bg-info/10"
  end

  def category_bg_class(:metrics) do
    "bg-success/10"
  end

  def category_bg_class(:code_analysis) do
    "bg-violet-500/10"
  end

  def category_bg_class(:services) do
    "bg-sky-500/10"
  end

  def category_bg_class(:atlas) do
    "bg-teal-500/10"
  end

  def category_bg_class(_) do
    "bg-base-200"
  end

  def category_text_class(:core) do
    "text-primary"
  end

  def category_text_class(:epistemic) do
    "text-secondary"
  end

  def category_text_class(:analysis) do
    "text-accent"
  end

  def category_text_class(:ml) do
    "text-warning"
  end

  def category_text_class(:knowledge) do
    "text-cyan-500"
  end

  def category_text_class(:learning) do
    "text-error"
  end

  def category_text_class(:storage) do
    "text-info"
  end

  def category_text_class(:metrics) do
    "text-success"
  end

  def category_text_class(:code_analysis) do
    "text-violet-500"
  end

  def category_text_class(:services) do
    "text-sky-500"
  end

  def category_text_class(:atlas) do
    "text-teal-500"
  end

  def category_text_class(_) do
    "text-base-content"
  end

  def status_badge_variant(:ready) do
    :success
  end

  def status_badge_variant(:running) do
    :success
  end

  def status_badge_variant(:initializing) do
    :warning
  end

  def status_badge_variant(:building_vocabulary) do
    :warning
  end

  def status_badge_variant(:tokenizing) do
    :warning
  end

  def status_badge_variant(:building_frequencies) do
    :warning
  end

  def status_badge_variant(:calculating_idf) do
    :warning
  end

  def status_badge_variant(:loading) do
    :warning
  end

  def status_badge_variant(:busy) do
    :warning
  end

  def status_badge_variant(:idle) do
    :info
  end

  def status_badge_variant(:not_started) do
    :error
  end

  def status_badge_variant(_) do
    :default
  end

  def format_stat_value(value) when is_binary(value) do
    value
  end

  def format_stat_value(value) when is_integer(value) do
    Integer.to_string(value)
  end

  def format_stat_value(value) when is_float(value) do
    Float.round(value, 2) |> to_string()
  end

  def format_stat_value(value) when is_boolean(value) do
    to_string(value)
  end

  def format_stat_value(value) when is_list(value) do
    "[#{length(value)}]"
  end

  def format_stat_value(value) when is_map(value) do
    "{#{map_size(value)}}"
  end

  def format_stat_value(value) do
    inspect(value)
  end

  def model_status_variant(%{exists: true, loaded: true}) do
    :success
  end

  def model_status_variant(%{exists: true, loaded: false}) do
    :warning
  end

  def model_status_variant(%{exists: false}) do
    :error
  end

  def model_status_variant(%{loaded: true}) do
    :success
  end

  def model_status_variant(%{loaded: false}) do
    :error
  end

  def model_status_variant(_) do
    :default
  end

  def model_status_label(%{exists: true, loaded: true}) do
    "Loaded"
  end

  def model_status_label(%{exists: true, loaded: false}) do
    "Not Loaded"
  end

  def model_status_label(%{exists: false}) do
    "Not Trained"
  end

  def model_status_label(%{loaded: true}) do
    "Loaded"
  end

  def model_status_label(%{loaded: false}) do
    "Not Loaded"
  end

  def model_status_label(_) do
    "Unknown"
  end

  def format_model_datetime(nil) do
    "Never"
  end

  def format_model_datetime(%DateTime{} = dt) do
    Calendar.strftime(dt, "%Y-%m-%d %H:%M")
  end

  def format_model_datetime({{year, month, day}, {hour, min, _sec}}) do
    "#{year}-#{String.pad_leading("#{month}", 2, "0")}-#{String.pad_leading("#{day}", 2, "0")} #{String.pad_leading("#{hour}", 2, "0")}:#{String.pad_leading("#{min}", 2, "0")}"
  end

  def format_model_datetime(_) do
    "-"
  end

  def training_status_variant(:completed) do
    :success
  end

  def training_status_variant(:in_progress) do
    :warning
  end

  def training_status_variant(:failed) do
    :error
  end

  def training_status_variant(_) do
    :default
  end

  def training_status_label(:completed) do
    "Completed"
  end

  def training_status_label(:in_progress) do
    "In Progress"
  end

  def training_status_label(:failed) do
    "Failed"
  end

  def training_status_label(nil) do
    "Never Run"
  end

  def training_status_label(_) do
    "Unknown"
  end

  def model_name(:pos_model) do
    "POS Tagger"
  end

  def model_name(:entity_model) do
    "Entity Model"
  end

  def model_name(:classifier) do
    "Intent Classifier"
  end

  def model_name(:gazetteer) do
    "Gazetteer"
  end

  def model_name(:intent_classifier) do
    "Intent Classifier (Agent)"
  end

  def model_name(:entity_extractor) do
    "Entity Extractor (Agent)"
  end

  def model_name(:pos_tagger) do
    "POS Tagger"
  end

  def model_name(:entity_trainer) do
    "Entity Trainer"
  end

  def model_name(:unified_model) do
    "Unified LSTM"
  end

  def model_name(:response_scorer) do
    "Response Scorer"
  end

  def model_name(:intent_arbitrator) do
    "Intent Arbitrator"
  end

  def model_name(other) do
    other |> to_string() |> String.replace("_", " ") |> String.capitalize()
  end

  def file_based_models(ml_models_status) do
    [:pos_model, :entity_model, :classifier, :gazetteer]
    |> Enum.map(fn key -> {key, Map.get(ml_models_status, key)} end)
    |> Enum.filter(fn {_k, v} -> v != nil end)
  end

  def agent_based_models(ml_models_status) do
    [:intent_classifier, :entity_extractor]
    |> Enum.map(fn key -> {key, Map.get(ml_models_status, key)} end)
    |> Enum.filter(fn {_k, v} -> v != nil end)
  end

  def lstm_models(ml_models_status) do
    [:unified_model, :response_scorer, :intent_arbitrator]
    |> Enum.map(fn key -> {key, Map.get(ml_models_status, key)} end)
    |> Enum.filter(fn {_k, v} -> v != nil end)
  end

  def corpus_info do
    try do
      size_info = CorpusManager.size_by_category()

      %{
        total: CorpusManager.format_bytes(size_info.total),
        utilization: CorpusManager.utilization_percent(),
        categories: %{
          training: CorpusManager.format_bytes(size_info.training_data),
          models: CorpusManager.format_bytes(size_info.ml_models),
          evaluation: CorpusManager.format_bytes(size_info.evaluation),
          worlds: CorpusManager.format_bytes(size_info.training_worlds),
          knowledge: CorpusManager.format_bytes(size_info.knowledge)
        }
      }
    rescue
      _ -> %{total: "N/A", utilization: 0.0, categories: %{}}
    end
  end

  def get_training_stats(performance_metrics) do
    Map.get(performance_metrics, :training, %{})
  end

  @doc """
  Returns all categories in display order.
  Uses the keys from genserver_status.categories if available, falling back to defaults.
  """
  def all_categories do
    [:core, :epistemic, :analysis, :ml, :knowledge, :learning, :services, :storage, :metrics, :code_analysis]
  end

  @doc """
  Returns categories that exist in the current genserver status data.
  This ensures we only display categories that have actual data.
  """
  def available_categories(genserver_status) do
    existing = Map.keys(genserver_status.categories) |> MapSet.new()
    all_categories() |> Enum.filter(&MapSet.member?(existing, &1))
  end

  @doc "Determine the status dot indicator for the embedder.\nIdle is shown as info (blue), building as warning, ready as success.\n"
  def embedder_status_for_dot(%{ready: true}) do
    :ready
  end

  def embedder_status_for_dot(%{phase: :idle}) do
    :idle
  end

  def embedder_status_for_dot(%{phase: :not_started}) do
    :not_started
  end

  def embedder_status_for_dot(_) do
    :initializing
  end

  @doc "Check if the embedder is actively building vocabulary (should show progress).\nReturns false for idle state (on-demand, not yet used).\n"
  def embedder_is_building?(%{ready: true}) do
    false
  end

  def embedder_is_building?(%{phase: :idle}) do
    false
  end

  def embedder_is_building?(%{phase: :not_started}) do
    false
  end

  def embedder_is_building?(_) do
    true
  end

  @doc "Determine the status dot indicator for the world-specific embedder.\n"
  def world_embedder_status_for_dot(%{ready: true}) do
    :ready
  end

  def world_embedder_status_for_dot(%{phase: :not_initialized}) do
    :idle
  end

  def world_embedder_status_for_dot(%{phase: :table_not_ready}) do
    :warning
  end

  def world_embedder_status_for_dot(%{phase: :no_data}) do
    :warning
  end

  def world_embedder_status_for_dot(_) do
    :initializing
  end

  @doc "Check if the world embedder is actively building vocabulary.\n"
  def world_embedder_is_building?(%{ready: true}) do
    false
  end

  def world_embedder_is_building?(%{phase: :not_initialized}) do
    false
  end

  def world_embedder_is_building?(%{phase: :table_not_ready}) do
    false
  end

  def world_embedder_is_building?(%{phase: :no_data}) do
    false
  end

  def world_embedder_is_building?(%{phase: :ready}) do
    false
  end

  def world_embedder_is_building?(_) do
    true
  end
end
