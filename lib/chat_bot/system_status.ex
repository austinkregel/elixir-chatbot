defmodule ChatBot.SystemStatus do
  @moduledoc """
  Reports the status of various background systems for UI display.
  Provides comprehensive monitoring of all GenServers in the application.
  """

  alias ChatBot.Memory.{Embedder, Store}

  # All GenServers organized by category
  @genserver_categories %{
    core: [
      {ChatBot.Brain, "Brain", :has_status},
      {ChatBot.Memory.Embedder, "Memory Embedder", :has_ready},
      {ChatBot.Memory.Store, "Memory Store", :has_stats}
    ],
    epistemic: [
      {ChatBot.Epistemic.JTMS, "JTMS", :has_ready_and_stats},
      {ChatBot.Epistemic.BeliefStore, "Belief Store", :has_ready_and_stats},
      {ChatBot.Epistemic.UserModelStore, "User Model Store", :has_ready_and_stats},
      {ChatBot.Epistemic.ContradictionHandler, "Contradiction Handler", :has_ready}
    ],
    analysis: [
      {ChatBot.Analysis.LearningStore, "Learning Store", :basic},
      {ChatBot.Analysis.AnalyzerCalibration, "Analyzer Calibration", :has_stats},
      {ChatBot.Analysis.HeuristicStore, "Heuristic Store", :has_stats}
    ],
    ml: [
      {ChatBot.ML.Gazetteer, "Gazetteer", :has_stats},
      {ChatBot.ML.InformalExpansions, "Informal Expansions", :has_ready}
    ],
    storage: [
      {ChatBot.KnowledgeStore, "Knowledge Store", :basic},
      {ChatBot.MemoryStore, "Memory Store (Legacy)", :basic}
    ]
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Returns a map of all system statuses (legacy format for compatibility).
  """
  def get_all do
    %{
      embedder: get_embedder_status(),
      memory_store: get_memory_store_status(),
      brain: get_brain_status(),
      nlp_pipeline: get_nlp_pipeline_status()
    }
  end

  @doc """
  Returns comprehensive status of all GenServers organized by category.
  """
  def get_all_genservers_status do
    started_at = System.monotonic_time(:millisecond)

    categories =
      @genserver_categories
      |> Enum.map(fn {category, servers} ->
        statuses =
          servers
          |> Enum.map(fn {module, name, type} ->
            {module, get_genserver_status(module, name, type)}
          end)
          |> Map.new()

        {category, statuses}
      end)
      |> Map.new()

    # Add subprocess supervisor status
    subprocess_status = get_subprocess_supervisor_status()

    elapsed = System.monotonic_time(:millisecond) - started_at

    %{
      categories: categories,
      subprocess_supervisor: subprocess_status,
      checked_at: DateTime.utc_now(),
      check_duration_ms: elapsed
    }
  end

  @doc """
  Returns performance metrics from the Metrics Aggregator.
  """
  def get_performance_metrics do
    if Code.ensure_loaded?(ChatBot.Metrics.Aggregator) do
      try do
        ChatBot.Metrics.Aggregator.get_metrics()
      catch
        :exit, _ -> default_metrics()
      end
    else
      default_metrics()
    end
  end

  @doc """
  Returns health indicators for the system.
  """
  def get_health_indicators do
    genserver_status = get_all_genservers_status()
    supervisor_info = get_supervisor_info()

    # Count running/total GenServers
    {running, total} = count_genserver_health(genserver_status.categories)

    # Calculate health score (0-100)
    health_score = if total > 0, do: round(running / total * 100), else: 0

    %{
      genservers_running: running,
      genservers_total: total,
      health_score: health_score,
      health_status: health_status_label(health_score),
      supervisor: supervisor_info,
      uptime_seconds: get_uptime_seconds(),
      checked_at: DateTime.utc_now()
    }
  end

  @doc """
  Returns the embedder status.
  """
  def get_embedder_status do
    if Process.whereis(Embedder) do
      ready = Embedder.ready?()

      %{
        running: true,
        ready: ready,
        status: if(ready, do: :ready, else: :building_vocabulary),
        label: if(ready, do: "Ready", else: "Building vocabulary...")
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started"
      }
    end
  end

  @doc """
  Returns the memory store status.
  """
  def get_memory_store_status do
    if Process.whereis(Store) do
      stats =
        try do
          Store.stats()
        catch
          :exit, _ -> %{episode_count: 0, semantic_count: 0}
        end

      %{
        running: true,
        ready: true,
        status: :ready,
        label: "Ready",
        episodes: Map.get(stats, :episode_count, 0),
        semantics: Map.get(stats, :semantic_count, 0)
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started",
        episodes: 0,
        semantics: 0
      }
    end
  end

  @doc """
  Returns the brain status.
  """
  def get_brain_status do
    if Process.whereis(ChatBot.Brain) do
      %{
        running: true,
        ready: true,
        status: :ready,
        label: "Ready"
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started"
      }
    end
  end

  @doc """
  Returns the NLP pipeline status.
  """
  def get_nlp_pipeline_status do
    # Check if models are loaded
    classifier_ready =
      try do
        ChatBot.ML.IntentClassifierSimple.is_loaded?()
      catch
        :exit, _ -> false
      end

    gazetteer_ready =
      try do
        ChatBot.ML.Gazetteer.is_loaded?()
      catch
        :exit, _ -> false
      end

    all_ready = classifier_ready and gazetteer_ready

    %{
      running: true,
      ready: all_ready,
      status: if(all_ready, do: :ready, else: :loading),
      label: if(all_ready, do: "Ready", else: "Loading models..."),
      components: %{
        intent_classifier: classifier_ready,
        gazetteer: gazetteer_ready
      }
    }
  end

  @doc """
  Returns true if all systems are ready.
  """
  def all_ready? do
    status = get_all()

    status.embedder.ready and
      status.memory_store.ready and
      status.brain.ready and
      status.nlp_pipeline.ready
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_genserver_status(module, name, type) do
    pid = Process.whereis(module)

    base_status = %{
      name: name,
      module: module,
      running: pid != nil,
      pid: pid,
      ready: false,
      status: :not_started,
      label: "Not started",
      stats: nil,
      memory_bytes: nil,
      message_queue_len: nil
    }

    if pid do
      # Get process info (non-blocking)
      process_info = get_process_info(pid)

      status_with_info = %{
        base_status
        | running: true,
          status: :running,
          label: "Running",
          memory_bytes: process_info[:memory],
          message_queue_len: process_info[:message_queue_len]
      }

      # Add type-specific information
      enhance_status(status_with_info, module, type)
    else
      base_status
    end
  end

  defp enhance_status(status, _module, :basic) do
    %{status | ready: true, status: :ready, label: "Ready"}
  end

  defp enhance_status(status, module, :has_ready) do
    ready = safe_call_ready(module)

    %{
      status
      | ready: ready,
        status: if(ready, do: :ready, else: :initializing),
        label: if(ready, do: "Ready", else: "Initializing...")
    }
  end

  defp enhance_status(status, module, :has_stats) do
    stats = safe_call_stats(module)

    %{
      status
      | ready: true,
        status: :ready,
        label: "Ready",
        stats: stats
    }
  end

  defp enhance_status(status, module, :has_ready_and_stats) do
    ready = safe_call_ready(module)
    stats = if ready, do: safe_call_stats(module), else: nil

    %{
      status
      | ready: ready,
        status: if(ready, do: :ready, else: :initializing),
        label: if(ready, do: "Ready", else: "Initializing..."),
        stats: stats
    }
  end

  defp enhance_status(status, module, :has_status) do
    # For Brain which has get_status
    brain_status = safe_call_brain_status(module)

    %{
      status
      | ready: true,
        status: :ready,
        label: "Ready",
        stats: brain_status
    }
  end

  defp safe_call_ready(module) do
    try do
      module.ready?()
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
      :exit, _ -> false
    end
  end

  defp safe_call_stats(module) do
    try do
      module.stats()
    catch
      :exit, {:timeout, _} -> nil
      :exit, {:noproc, _} -> nil
      :exit, _ -> nil
    end
  end

  defp safe_call_brain_status(module) do
    try do
      module.get_status()
    catch
      :exit, {:timeout, _} -> nil
      :exit, {:noproc, _} -> nil
      :exit, _ -> nil
    end
  end

  defp get_process_info(pid) do
    try do
      Process.info(pid, [:memory, :message_queue_len]) || []
    catch
      _, _ -> []
    end
  end

  defp get_subprocess_supervisor_status do
    pid = Process.whereis(ChatBot.Subprocesses.Supervisor)

    if pid do
      children =
        try do
          DynamicSupervisor.count_children(ChatBot.Subprocesses.Supervisor)
        catch
          :exit, _ -> %{active: 0, specs: 0, supervisors: 0, workers: 0}
        end

      %{
        running: true,
        pid: pid,
        active_children: children[:active] || 0,
        specs: children[:specs] || 0,
        workers: children[:workers] || 0,
        supervisors: children[:supervisors] || 0
      }
    else
      %{
        running: false,
        pid: nil,
        active_children: 0,
        specs: 0,
        workers: 0,
        supervisors: 0
      }
    end
  end

  defp get_supervisor_info do
    try do
      children = Supervisor.count_children(ChatBot.Supervisor)

      %{
        active: children[:active] || 0,
        specs: children[:specs] || 0,
        supervisors: children[:supervisors] || 0,
        workers: children[:workers] || 0
      }
    catch
      :exit, _ ->
        %{active: 0, specs: 0, supervisors: 0, workers: 0}
    end
  end

  defp count_genserver_health(categories) do
    Enum.reduce(categories, {0, 0}, fn {_category, servers}, {running, total} ->
      category_stats =
        Enum.reduce(servers, {0, 0}, fn {_module, status}, {r, t} ->
          {r + if(status.running, do: 1, else: 0), t + 1}
        end)

      {running + elem(category_stats, 0), total + elem(category_stats, 1)}
    end)
  end

  defp health_status_label(score) when score >= 90, do: :healthy
  defp health_status_label(score) when score >= 70, do: :degraded
  defp health_status_label(score) when score >= 50, do: :warning
  defp health_status_label(_score), do: :critical

  defp get_uptime_seconds do
    case :erlang.statistics(:wall_clock) do
      {uptime_ms, _} -> div(uptime_ms, 1000)
      _ -> 0
    end
  end

  defp default_metrics do
    %{
      brain_evaluate: %{count: 0, avg_ms: 0, min_ms: 0, max_ms: 0},
      pipeline_process: %{count: 0, avg_ms: 0, min_ms: 0, max_ms: 0},
      memory_query: %{count: 0, avg_ms: 0, min_ms: 0, max_ms: 0},
      errors: %{count: 0, rate_per_minute: 0.0}
    }
  end
end
