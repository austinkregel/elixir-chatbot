defmodule Brain.Analysis.LearningStore do
  @moduledoc """
  Stores and retrieves learned parameters for the analysis pipeline.

  Features:
  - Persists parameters to JSON file
  - Admin override/lock capability
  - Threshold tuning based on feedback
  - Version tracking for parameter changes

  Parameters can be:
  - Automatically learned from feedback
  - Manually set by admins
  - Locked to prevent automatic updates
  """

  use GenServer
  require Logger

  # Path resolved at runtime
  defp params_path, do: Brain.priv_path("analysis/learned_params.json")
  @save_debounce_ms 5_000

  # Client API

  @doc """
  Starts the learning store.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Gets parameters for a specific component.
  """
  def get_params(component) when is_binary(component) do
    GenServer.call(__MODULE__, {:get_params, component})
  catch
    :exit, _ ->
      # GenServer not running, load directly
      load_params_direct(component)
  end

  def get_params(component) when is_atom(component) do
    get_params(Atom.to_string(component))
  end

  @doc """
  Gets all parameters.
  """
  def get_all_params do
    GenServer.call(__MODULE__, :get_all_params)
  catch
    :exit, _ ->
      load_all_params_direct()
  end

  @doc """
  Updates parameters for a component.

  Options:
  - :admin - if true, sets the parameter as admin-set
  - :lock - if true, locks the parameter from automatic updates
  """
  def update_params(component, params, opts \\ []) do
    GenServer.call(__MODULE__, {:update_params, component, params, opts})
  end

  @doc """
  Records feedback for learning.
  """
  def record_feedback(feedback_type, data \\ %{}) do
    GenServer.cast(__MODULE__, {:record_feedback, feedback_type, data})
  end

  @doc """
  Locks a component's parameters from automatic updates.
  """
  def lock_params(component) do
    GenServer.call(__MODULE__, {:lock_params, component})
  end

  @doc """
  Unlocks a component's parameters for automatic updates.
  """
  def unlock_params(component) do
    GenServer.call(__MODULE__, {:unlock_params, component})
  end

  @doc """
  Resets parameters to defaults.
  """
  def reset_to_defaults(component \\ nil) do
    GenServer.call(__MODULE__, {:reset_to_defaults, component})
  end

  @doc """
  Gets feedback statistics.
  """
  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end

  @doc """
  Checks if the learning store is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    params = load_params_from_file()

    state = %{
      params: params,
      dirty: false,
      save_timer: nil
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:get_params, component}, _from, state) do
    result =
      case Map.get(state.params, component) do
        nil -> {:error, :not_found}
        params -> {:ok, params}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call(:get_all_params, _from, state) do
    {:reply, state.params, state}
  end

  @impl true
  def handle_call({:update_params, component, new_params, opts}, _from, state) do
    is_admin = Keyword.get(opts, :admin, false)
    should_lock = Keyword.get(opts, :lock, false)

    current = Map.get(state.params, component, %{})

    # Check if locked and not an admin update
    if Map.get(current, "admin_locked", false) and not is_admin do
      {:reply, {:error, :locked}, state}
    else
      updated =
        current
        |> Map.merge(new_params)
        |> Map.put("learned_at", DateTime.utc_now() |> DateTime.to_iso8601())
        |> maybe_lock(should_lock)

      new_params_map = Map.put(state.params, component, updated)
      new_state = schedule_save(%{state | params: new_params_map, dirty: true})

      {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:lock_params, component}, _from, state) do
    case Map.get(state.params, component) do
      nil ->
        {:reply, {:error, :not_found}, state}

      current ->
        updated = Map.put(current, "admin_locked", true)
        new_params = Map.put(state.params, component, updated)
        new_state = schedule_save(%{state | params: new_params, dirty: true})
        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:unlock_params, component}, _from, state) do
    case Map.get(state.params, component) do
      nil ->
        {:reply, {:error, :not_found}, state}

      current ->
        updated = Map.put(current, "admin_locked", false)
        new_params = Map.put(state.params, component, updated)
        new_state = schedule_save(%{state | params: new_params, dirty: true})
        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:reset_to_defaults, nil}, _from, state) do
    defaults = default_params()
    new_state = schedule_save(%{state | params: defaults, dirty: true})
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:reset_to_defaults, component}, _from, state) do
    defaults = default_params()

    case Map.get(defaults, component) do
      nil ->
        {:reply, {:error, :not_found}, state}

      default_component ->
        new_params = Map.put(state.params, component, default_component)
        new_state = schedule_save(%{state | params: new_params, dirty: true})
        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    stats = Map.get(state.params, "feedback_stats", %{})
    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_cast({:record_feedback, feedback_type, data}, state) do
    stats = Map.get(state.params, "feedback_stats", %{})

    updated_stats =
      case feedback_type do
        :successful_response ->
          stats
          |> Map.update("total_interactions", 1, &(&1 + 1))
          |> Map.update("successful_responses", 1, &(&1 + 1))

        :clarification_needed ->
          stats
          |> Map.update("total_interactions", 1, &(&1 + 1))
          |> Map.update("clarifications_needed", 1, &(&1 + 1))

        # Response optionality feedback types
        :response_deferred ->
          stats
          |> Map.update("total_interactions", 1, &(&1 + 1))
          |> Map.update("responses_deferred", 1, &(&1 + 1))

        :response_optional_deferred ->
          stats
          |> Map.update("total_interactions", 1, &(&1 + 1))
          |> Map.update("optional_responses_deferred", 1, &(&1 + 1))

        :response_optional_proceeded ->
          stats
          |> Map.update("total_interactions", 1, &(&1 + 1))
          |> Map.update("optional_responses_proceeded", 1, &(&1 + 1))

        :user_frustrated_by_silence ->
          # User showed frustration after we deferred - reduce deferral confidence
          update_optionality_learning(state.params, :reduce_deferral, data)

          stats
          |> Map.update("silence_frustrations", 1, &(&1 + 1))

        :user_confirmed_no_response_needed ->
          # User confirmed deferral was correct - reinforce pattern
          update_optionality_learning(state.params, :reinforce_deferral, data)

          stats
          |> Map.update("deferral_confirmations", 1, &(&1 + 1))

        _ ->
          stats
      end

    new_params = Map.put(state.params, "feedback_stats", updated_stats)
    new_state = schedule_save(%{state | params: new_params, dirty: true})

    {:noreply, new_state}
  end

  # Update response optionality learning based on feedback
  defp update_optionality_learning(params, :reduce_deferral, data) do
    # Reduce confidence for the pattern that caused frustration
    optionality_params = Map.get(params, "response_optionality", %{})
    pattern = Map.get(data, :pattern)

    # Reduce the relevant confidence threshold
    key =
      case pattern do
        :backchannel -> "backchannel_defer_confidence"
        :compliment -> "compliment_defer_confidence"
        :acknowledgment -> "acknowledgment_defer_confidence"
        _ -> nil
      end

    if key do
      current = Map.get(optionality_params, key, 0.7)
      # Reduce by 0.05, but don't go below 0.3
      new_value = max(current - 0.05, 0.3)
      updated = Map.put(optionality_params, key, new_value)
      Map.put(params, "response_optionality", updated)
    else
      params
    end
  end

  defp update_optionality_learning(params, :reinforce_deferral, data) do
    # Increase confidence for the pattern that was correctly deferred
    optionality_params = Map.get(params, "response_optionality", %{})
    pattern = Map.get(data, :pattern)

    key =
      case pattern do
        :backchannel -> "backchannel_defer_confidence"
        :compliment -> "compliment_defer_confidence"
        :acknowledgment -> "acknowledgment_defer_confidence"
        _ -> nil
      end

    if key do
      current = Map.get(optionality_params, key, 0.7)
      # Increase by 0.02, but don't go above 0.95
      new_value = min(current + 0.02, 0.95)
      updated = Map.put(optionality_params, key, new_value)
      Map.put(params, "response_optionality", updated)
    else
      params
    end
  end

  @impl true
  def handle_info(:save, state) do
    if state.dirty do
      save_params_to_file(state.params)
      {:noreply, %{state | dirty: false, save_timer: nil}}
    else
      {:noreply, %{state | save_timer: nil}}
    end
  end

  # Private functions

  defp schedule_save(%{save_timer: nil} = state) do
    timer = Process.send_after(self(), :save, @save_debounce_ms)
    %{state | save_timer: timer}
  end

  defp schedule_save(state), do: state

  defp maybe_lock(params, true), do: Map.put(params, "admin_locked", true)
  defp maybe_lock(params, _), do: params

  defp load_params_from_file do
    path = get_params_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, params} -> params
          {:error, _} -> default_params()
        end

      {:error, _} ->
        defaults = default_params()
        # Create the file with defaults
        save_params_to_file(defaults)
        defaults
    end
  end

  defp save_params_to_file(params) do
    path = get_params_path()

    # Ensure directory exists
    path |> Path.dirname() |> File.mkdir_p!()

    updated_params =
      params
      |> Map.put("last_updated", DateTime.utc_now() |> DateTime.to_iso8601())

    content = Jason.encode!(updated_params, pretty: true)
    File.write!(path, content)

    Logger.debug("Saved learning params to #{path}")
  end

  defp get_params_path do
    Application.get_env(:brain, :learning_params_path, params_path())
  end

  defp load_params_direct(component) do
    params = load_params_from_file()

    case Map.get(params, component) do
      nil -> {:error, :not_found}
      component_params -> {:ok, component_params}
    end
  end

  defp load_all_params_direct do
    load_params_from_file()
  end

  defp default_params do
    %{
      "version" => "1.0.0",
      "last_updated" => nil,
      "chunker" => %{
        "max_chunk_words" => 50,
        "min_chunk_words" => 3,
        "split_on_conjunctions" => true,
        "admin_locked" => false,
        "learned_at" => nil
      },
      "speech_acts" => %{
        "confidence_threshold" => 0.3,
        "admin_locked" => false,
        "learned_at" => nil
      },
      "discourse" => %{
        "confidence_threshold" => 0.4,
        "bot_names" => ["companion", "bot", "assistant", "ai", "echo"],
        "admin_locked" => false,
        "learned_at" => nil
      },
      "context_resolver" => %{
        "history_depth" => 5,
        "min_confidence_from_history" => 0.5,
        "admin_locked" => false,
        "learned_at" => nil
      },
      "feedback_stats" => %{
        "total_interactions" => 0,
        "successful_responses" => 0,
        "clarifications_needed" => 0,
        "responses_deferred" => 0,
        "optional_responses_deferred" => 0,
        "optional_responses_proceeded" => 0,
        "silence_frustrations" => 0,
        "deferral_confirmations" => 0
      },
      "response_optionality" => %{
        "admin_locked" => false,
        "learned_at" => nil,
        "gratitude_loop_threshold" => 2,
        "compliment_defer_confidence" => 0.6,
        "backchannel_defer_confidence" => 0.8,
        "acknowledgment_defer_confidence" => 0.7,
        "continuation_defer_confidence" => 0.9,
        "default_defer_threshold" => 0.7
      }
    }
  end
end
