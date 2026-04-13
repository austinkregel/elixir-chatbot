defmodule ChatWeb.SettingsLive do
  @moduledoc "Settings page for world management and entity administration.\n\nFeatures:\n- World management (create, delete, configure)\n- Gazetteer entity management\n- System configuration\n"

  alias Brain.Response.LSTMResponse
  alias Brain.ML.LSTM.UnifiedModel
  alias Phoenix.PubSub
  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias World.Manager, as: WorldManager
  alias World.Persistence, as: WorldPersistence
  alias Brain.Knowledge.LearningCenter
  alias Tasks.Source, as: TaskSource
  alias Brain.ML.Gazetteer
  alias Brain.ML.TrainingServer
  alias Brain.Response.TemplateStore
  alias Brain.Services.{Dispatcher, CredentialVault}

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      PubSub.subscribe(Brain.PubSub, "training:progress")
    end

    {:ok, socket}
  end

  @impl true
  def handle_params(params, _uri, socket) do
    section =
      case params["section"] do
        "entities" -> :entities
        "worlds" -> :worlds
        "training" -> :training
        "ml_training" -> :ml_training
        "templates" -> :templates
        "services" -> :services
        _ -> :worlds
      end

    socket =
      socket
      |> assign(:section, section)
      |> assign(:new_world_name, "")
      |> assign(:new_world_mode, "persistent")
      |> assign(:creating_world, false)
      |> assign(:entity_search, "")
      |> assign(:selected_entity_type, nil)
      |> assign(:new_entity_key, "")
      |> assign(:new_entity_value, "")
      |> assign(:new_entity_type, "location")
      |> assign(:training_sessions, [])
      |> assign(:expanded_session_id, nil)
      |> assign(:available_tasks, %{})
      |> assign(:selected_capability, :all)
      |> assign(:starting_training, false)
      |> assign(:tasks_loading, false)
      |> assign(:lc_stats, %{total_sessions: 0, active_agents: 0})
      |> assign(:ml_model_statuses, %{})
      |> assign(:ml_training_status, :idle)
      |> assign(:ml_selected_model, "unified")
      |> assign(:ml_epochs, "20")
      |> assign(:ml_head_epochs, "20")
      |> assign(:ml_batch_size, "32")
      |> assign(:ml_experiment_name, "")
      |> assign(:ml_training_log, [])
      |> assign(:ml_schedules, [])
      |> assign(:ml_schedule_interval, "24")
      |> assign(:ml_reloading, false)
      |> assign(:template_intents, [])
      |> assign(:selected_template_intent, nil)
      |> assign(:intent_templates, [])
      |> assign(:template_search, "")
      |> assign(:new_template_text, "")
      |> assign(:template_stats, %{})
      |> assign(:template_has_unsaved, false)
      |> assign(:services, [])
      |> assign(:service_credentials, %{})
      |> assign(:service_health_status, %{})
      |> assign(:service_checking, nil)
      |> load_section_data()

    {:noreply, socket}
  end

  defp load_section_data(socket) do
    case socket.assigns.section do
      :worlds -> load_worlds_data(socket)
      :entities -> load_entities_data(socket)
      :training -> load_training_data(socket)
      :ml_training -> load_ml_training_data(socket)
      :templates -> load_templates_data(socket)
      :services -> load_services_data(socket)
      _ -> socket
    end
  end

  defp load_worlds_data(socket) do
    worlds =
      try do
        WorldManager.list_worlds()
      rescue
        _ -> []
      end

    persisted =
      try do
        WorldPersistence.list_persisted_worlds()
      rescue
        _ -> []
      end

    socket
    |> assign(:worlds, worlds)
    |> assign(:persisted_worlds, persisted)
  end

  defp load_entities_data(socket) do
    world_id = socket.assigns.current_world_id
    entity_types = Gazetteer.list_types()
    world_overlay = Gazetteer.get_world_overlay(world_id)

    socket
    |> assign(:entity_types, entity_types)
    |> assign(:world_overlay, world_overlay)
    |> assign(:selected_entity_type, List.first(entity_types))
    |> load_type_entities()
  end

  defp load_type_entities(socket) do
    type = socket.assigns[:selected_entity_type]

    entities =
      if type do
        Gazetteer.list_by_type(type)
        |> Enum.map(fn {key, info} ->
          %{
            key: key,
            value: Map.get(info, :value) || Map.get(info, :original) || key,
            source: Map.get(info, :source, "unknown"),
            entity_type: Map.get(info, :entity_type) || Map.get(info, :type)
          }
        end)
      else
        []
      end

    assign(socket, :type_entities, entities)
  end

  defp load_training_data(socket) do
    sessions =
      try do
        LearningCenter.list_sessions()
      rescue
        _ -> []
      catch
        :exit, _ -> []
      end

    lc_stats =
      try do
        LearningCenter.stats()
      rescue
        _ -> %{total_sessions: 0, active_agents: 0}
      catch
        :exit, _ -> %{total_sessions: 0, active_agents: 0}
      end

    socket =
      socket
      |> assign(:training_sessions, sessions)
      |> assign(:lc_stats, lc_stats)
      |> assign(:available_tasks, socket.assigns[:available_tasks] || %{})
      |> assign(:tasks_loading, true)

    if connected?(socket) do
      self_pid = self()

      Task.start(fn ->
        available =
          try do
            case TaskSource.available_tasks() do
              {:ok, grouped} -> grouped
              _ -> %{}
            end
          rescue
            _ -> %{}
          catch
            :exit, _ -> %{}
          end

        send(self_pid, {:tasks_loaded, available})
      end)
    end

    socket
  end

  defp load_ml_training_data(socket) do
    unified_ready =
      try do
        UnifiedModel.ready?()
      rescue
        _ -> false
      catch
        :exit, _ -> false
      end

    response_ready =
      try do
        LSTMResponse.ready?()
      rescue
        _ -> false
      catch
        :exit, _ -> false
      end

    arbitrator_ready =
      try do
        Brain.ML.IntentArbitrator.ready?()
      rescue
        _ -> false
      catch
        :exit, _ -> false
      end

    model_statuses = %{
      unified_model: unified_ready,
      response_scorer: response_ready,
      intent_arbitrator: arbitrator_ready
    }

    training_status = TrainingServer.get_status()
    schedules = TrainingServer.list_schedules()

    socket
    |> assign(:ml_model_statuses, model_statuses)
    |> assign(:ml_training_status, training_status)
    |> assign(:ml_schedules, schedules)
  end

  defp load_templates_data(socket) do
    stats =
      try do
        TemplateStore.stats()
      rescue
        _ -> %{intent_count: 0, template_count: 0}
      catch
        :exit, _ -> %{intent_count: 0, template_count: 0}
      end

    intents =
      try do
        TemplateStore.list_intents() |> Enum.sort()
      rescue
        _ -> []
      catch
        :exit, _ -> []
      end

    has_unsaved =
      try do
        TemplateStore.has_unsaved_changes?()
      rescue
        _ -> false
      catch
        :exit, _ -> false
      end

    socket
    |> assign(:template_stats, stats)
    |> assign(:template_intents, intents)
    |> assign(:template_has_unsaved, has_unsaved)
    |> assign(
      :selected_template_intent,
      socket.assigns[:selected_template_intent] || List.first(intents)
    )
    |> load_intent_templates()
  end

  defp load_intent_templates(socket) do
    intent = socket.assigns[:selected_template_intent]

    templates =
      if intent do
        try do
          TemplateStore.list_templates_with_metadata(intent)
        rescue
          _ -> []
        catch
          :exit, _ -> []
        end
      else
        []
      end

    assign(socket, :intent_templates, templates)
  end

  defp load_services_data(socket) do
    world = socket.assigns[:current_world_id] || "default"

    services =
      try do
        Dispatcher.list_services(world: world)
      rescue
        _ -> []
      catch
        :exit, _ -> []
      end

    # Build credential status for each service
    service_credentials =
      Enum.reduce(services, %{}, fn service, acc ->
        creds =
          Enum.reduce(service.required_credentials, %{}, fn cred_key, inner_acc ->
            has_cred = CredentialVault.has_credential?(service.name, cred_key, world: world)
            Map.put(inner_acc, cred_key, has_cred)
          end)

        Map.put(acc, service.name, creds)
      end)

    socket
    |> assign(:services, services)
    |> assign(:service_credentials, service_credentials)
  end

  @impl true
  def handle_event("switch_world", %{"world_id" => _world_id}, socket) do
    {:noreply, load_section_data(socket)}
  end

  def handle_event("refresh_worlds", _params, socket) do
    {:noreply, socket}
  end

  def handle_event("switch_section", %{"section" => section}, socket) do
    {:noreply, push_patch(socket, to: ~p"/settings?section=#{section}")}
  end

  def handle_event("update_new_world", %{"name" => name, "mode" => mode}, socket) do
    {:noreply, socket |> assign(:new_world_name, name) |> assign(:new_world_mode, mode)}
  end

  def handle_event("create_world", _params, socket) do
    name = socket.assigns.new_world_name
    mode = String.to_existing_atom(socket.assigns.new_world_mode)

    if name != "" do
      socket = assign(socket, :creating_world, true)

      case WorldManager.create(name, mode: mode, base_world: "default") do
        {:ok, world} ->
          socket =
            socket
            |> assign(:creating_world, false)
            |> assign(:new_world_name, "")
            |> load_worlds_data()
            |> put_flash(:info, "Created world: #{world.name}")

          {:noreply, socket}

        {:error, reason} ->
          {:noreply,
           socket
           |> assign(:creating_world, false)
           |> put_flash(:error, "Failed to create world: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "World name is required")}
    end
  end

  @impl true
  def handle_event("delete_world", %{"id" => world_id}, socket) do
    if world_id != "default" do
      case WorldManager.destroy(world_id) do
        :ok ->
          {:noreply,
           socket |> load_worlds_data() |> put_flash(:info, "Deleted world: #{world_id}")}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to delete world: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Cannot delete the default world")}
    end
  end

  @impl true
  def handle_event("save_world", %{"id" => world_id}, socket) do
    case WorldManager.checkpoint(world_id) do
      :ok ->
        {:noreply, put_flash(socket, :info, "World saved: #{world_id}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to save world: #{inspect(reason)}")}
    end
  end

  @impl true
  def handle_event("load_world", %{"id" => _world_id}, socket) do
    case WorldManager.reload_persisted_worlds() do
      :ok ->
        {:noreply, socket |> load_worlds_data() |> put_flash(:info, "Reloaded persisted worlds")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to reload: #{inspect(reason)}")}
    end
  end

  @impl true
  def handle_event("select_entity_type", %{"type" => type}, socket) do
    {:noreply, socket |> assign(:selected_entity_type, type) |> load_type_entities()}
  end

  @impl true
  def handle_event("search_entities", %{"query" => query}, socket) do
    {:noreply, assign(socket, :entity_search, query)}
  end

  @impl true
  def handle_event("update_new_entity", params, socket) do
    socket =
      socket
      |> assign(:new_entity_key, params["key"] || socket.assigns.new_entity_key)
      |> assign(:new_entity_value, params["value"] || socket.assigns.new_entity_value)
      |> assign(:new_entity_type, params["type"] || socket.assigns.new_entity_type)

    {:noreply, socket}
  end

  @impl true
  def handle_event("add_entity", _params, socket) do
    world_id = socket.assigns.current_world_id
    key = socket.assigns.new_entity_key
    value = socket.assigns.new_entity_value
    type = socket.assigns.new_entity_type

    if key != "" do
      case Gazetteer.add_to_world(world_id, key, type, %{
             value:
               if(value == "") do
                 key
               else
                 value
               end,
             source: :admin,
             added_at: DateTime.utc_now()
           }) do
        :ok ->
          {:noreply,
           socket
           |> assign(:new_entity_key, "")
           |> assign(:new_entity_value, "")
           |> load_entities_data()
           |> put_flash(:info, "Added entity: #{key}")}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to add entity: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Entity key is required")}
    end
  end

  @impl true
  def handle_event("remove_entity", %{"key" => key}, socket) do
    world_id = socket.assigns.current_world_id

    case Gazetteer.remove_from_world(world_id, key) do
      :ok ->
        {:noreply, socket |> load_entities_data() |> put_flash(:info, "Removed entity: #{key}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to remove entity: #{inspect(reason)}")}
    end
  end

  @impl true
  def handle_event("refresh", _params, socket) do
    {:noreply, load_section_data(socket)}
  end

  def handle_event("select_capability", %{"capability" => capability}, socket) do
    capability = String.to_existing_atom(capability)
    {:noreply, assign(socket, :selected_capability, capability)}
  end

  def handle_event("start_task_training", _params, socket) do
    capability = socket.assigns.selected_capability
    socket = assign(socket, :starting_training, true)

    case LearningCenter.start_task_training(capability, max_tasks: 5) do
      {:ok, session} ->
        socket =
          socket
          |> assign(:starting_training, false)
          |> load_training_data()
          |> put_flash(:info, "Started training session: #{session.id}")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply,
         socket
         |> assign(:starting_training, false)
         |> put_flash(:error, "Failed to start training: #{inspect(reason)}")}
    end
  end

  def handle_event("toggle_session_detail", %{"id" => session_id}, socket) do
    current = socket.assigns.expanded_session_id

    new_id =
      if current == session_id do
        nil
      else
        session_id
      end

    {:noreply, assign(socket, :expanded_session_id, new_id)}
  end

  def handle_event("cancel_session", %{"id" => session_id}, socket) do
    case LearningCenter.cancel_session(session_id) do
      :ok ->
        {:noreply,
         socket |> load_training_data() |> put_flash(:info, "Cancelled session: #{session_id}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to cancel session: #{inspect(reason)}")}
    end
  end

  def handle_event("update_ml_training_form", params, socket) do
    socket =
      socket
      |> assign(:ml_selected_model, params["model_type"] || socket.assigns.ml_selected_model)
      |> assign(:ml_epochs, params["epochs"] || socket.assigns.ml_epochs)
      |> assign(:ml_head_epochs, params["head_epochs"] || socket.assigns.ml_head_epochs)
      |> assign(:ml_batch_size, params["batch_size"] || socket.assigns.ml_batch_size)
      |> assign(
        :ml_experiment_name,
        params["experiment_name"] || socket.assigns.ml_experiment_name
      )

    {:noreply, socket}
  end

  def handle_event("start_ml_training", _params, socket) do
    model_type =
      case socket.assigns.ml_selected_model do
        "tfidf" -> :tfidf
        "unified" -> :unified
        "response" -> :response
        "arbitrator" -> :arbitrator
        _ -> :unified
      end

    epochs = parse_integer(socket.assigns.ml_epochs, 20)
    head_epochs = parse_integer(socket.assigns.ml_head_epochs, 20)
    batch_size = parse_integer(socket.assigns.ml_batch_size, 32)
    experiment_name = socket.assigns.ml_experiment_name

    config =
      [epochs: epochs, head_epochs: head_epochs, batch_size: batch_size]
      |> then(fn cfg ->
        if experiment_name != "" do
          Keyword.put(cfg, :name, experiment_name)
        else
          cfg
        end
      end)

    case TrainingServer.start_training(model_type, config) do
      {:ok, _model_type} ->
        socket =
          socket
          |> load_ml_training_data()
          |> append_training_log("Started training #{model_type}")
          |> put_flash(:info, "Started #{model_type} training")

        {:noreply, socket}

      {:error, {:already_training, current}} ->
        {:noreply, put_flash(socket, :error, "Already training #{current}. Cancel it first.")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to start training: #{inspect(reason)}")}
    end
  end

  def handle_event("cancel_ml_training", _params, socket) do
    case TrainingServer.cancel() do
      :ok ->
        socket =
          socket
          |> load_ml_training_data()
          |> append_training_log("Training cancelled")
          |> put_flash(:info, "Training cancelled")

        {:noreply, socket}

      {:error, :not_training} ->
        {:noreply, put_flash(socket, :error, "No training in progress")}
    end
  end

  def handle_event("reload_ml_models", _params, socket) do
    socket = assign(socket, :ml_reloading, true)

    results =
      [:unified, :response, :arbitrator]
      |> Enum.map(fn model ->
        try do
          case model do
            :unified -> {model, UnifiedModel.reload()}
            :response -> {model, LSTMResponse.reload()}
            :arbitrator -> {model, Brain.ML.IntentArbitrator.reload()}
          end
        rescue
          e -> {model, {:error, Exception.message(e)}}
        catch
          :exit, reason -> {model, {:error, reason}}
        end
      end)

    successes = Enum.count(results, fn {_, res} -> res == :ok end)

    socket =
      socket
      |> assign(:ml_reloading, false)
      |> load_ml_training_data()
      |> append_training_log("Reloaded models (#{successes}/3 succeeded)")
      |> put_flash(:info, "Reloaded #{successes}/3 models")

    {:noreply, socket}
  end

  def handle_event("update_ml_schedule_interval", %{"interval" => interval}, socket) do
    {:noreply, assign(socket, :ml_schedule_interval, interval)}
  end

  def handle_event("add_ml_schedule", _params, socket) do
    model_type =
      case socket.assigns.ml_selected_model do
        "tfidf" -> :tfidf
        "unified" -> :unified
        "response" -> :response
        "arbitrator" -> :arbitrator
        _ -> :unified
      end

    interval_hours = parse_integer(socket.assigns.ml_schedule_interval, 24)
    epochs = parse_integer(socket.assigns.ml_epochs, 20)
    head_epochs = parse_integer(socket.assigns.ml_head_epochs, 20)
    batch_size = parse_integer(socket.assigns.ml_batch_size, 32)

    config = [epochs: epochs, head_epochs: head_epochs, batch_size: batch_size]

    case TrainingServer.schedule(model_type, config, interval_hours) do
      {:ok, schedule_id} ->
        socket =
          socket
          |> load_ml_training_data()
          |> append_training_log(
            "Scheduled #{model_type} every #{interval_hours}h (#{schedule_id})"
          )
          |> put_flash(:info, "Scheduled #{model_type} training every #{interval_hours} hours")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to schedule: #{inspect(reason)}")}
    end
  end

  def handle_event("cancel_ml_schedule", %{"id" => schedule_id}, socket) do
    case TrainingServer.cancel_schedule(schedule_id) do
      :ok ->
        socket =
          socket
          |> load_ml_training_data()
          |> append_training_log("Cancelled schedule #{schedule_id}")
          |> put_flash(:info, "Cancelled schedule")

        {:noreply, socket}

      {:error, :not_found} ->
        {:noreply, put_flash(socket, :error, "Schedule not found")}
    end
  end

  def handle_event("select_template_intent", %{"intent" => intent}, socket) do
    {:noreply,
     socket
     |> assign(:selected_template_intent, intent)
     |> load_intent_templates()}
  end

  def handle_event("search_templates", %{"query" => query}, socket) do
    {:noreply, assign(socket, :template_search, query)}
  end

  def handle_event("update_new_template", %{"text" => text}, socket) do
    {:noreply, assign(socket, :new_template_text, text)}
  end

  def handle_event("add_template", _params, socket) do
    intent = socket.assigns.selected_template_intent
    text = String.trim(socket.assigns.new_template_text)

    if intent && text != "" do
      case TemplateStore.add_template(intent, text) do
        {:ok, _template} ->
          {:noreply,
           socket
           |> assign(:new_template_text, "")
           |> load_intent_templates()
           |> assign(:template_has_unsaved, true)
           |> put_flash(:info, "Added template")}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to add template: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Template text is required")}
    end
  end

  def handle_event("remove_template", %{"text" => text}, socket) do
    intent = socket.assigns.selected_template_intent

    if intent do
      case TemplateStore.remove_template(intent, text) do
        :ok ->
          {:noreply,
           socket
           |> load_intent_templates()
           |> assign(:template_has_unsaved, true)
           |> put_flash(:info, "Removed template")}

        {:error, :not_found} ->
          {:noreply, put_flash(socket, :error, "Template not found")}
      end
    else
      {:noreply, socket}
    end
  end

  def handle_event("sync_templates", _params, socket) do
    case TemplateStore.sync_to_file() do
      {:ok, _path} ->
        {:noreply,
         socket
         |> assign(:template_has_unsaved, false)
         |> put_flash(:info, "Templates saved to file")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to save: #{inspect(reason)}")}
    end
  end

  # ============================================================================
  # Services Section Event Handlers
  # ============================================================================

  def handle_event("save_credential", %{"service" => service_name, "key" => key, "value" => value}, socket) do
    world = socket.assigns[:current_world_id] || "default"
    service_atom = String.to_existing_atom(service_name)
    key_atom = String.to_existing_atom(key)

    case CredentialVault.store(service_atom, key_atom, value, world: world) do
      :ok ->
        {:noreply,
         socket
         |> load_services_data()
         |> put_flash(:info, "Credential saved for #{service_name}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to save credential: #{inspect(reason)}")}
    end
  rescue
    ArgumentError ->
      {:noreply, put_flash(socket, :error, "Invalid service or credential key")}
  end

  def handle_event("delete_credential", %{"service" => service_name, "key" => key}, socket) do
    world = socket.assigns[:current_world_id] || "default"
    service_atom = String.to_existing_atom(service_name)
    key_atom = String.to_existing_atom(key)

    CredentialVault.delete(service_atom, key_atom, world: world)

    {:noreply,
     socket
     |> load_services_data()
     |> put_flash(:info, "Credential removed")}
  rescue
    ArgumentError ->
      {:noreply, put_flash(socket, :error, "Invalid service or credential key")}
  end

  def handle_event("check_service_health", %{"service" => service_name}, socket) do
    world = socket.assigns[:current_world_id] || "default"
    service_atom = String.to_existing_atom(service_name)

    # Mark as checking
    socket = assign(socket, :service_checking, service_atom)

    # Run health check
    result = Dispatcher.health_check(service_atom, world: world)

    status =
      case result do
        :ok -> :healthy
        {:error, :missing_credentials} -> :missing_credentials
        {:error, :invalid_credentials} -> :invalid_credentials
        {:error, reason} -> {:error, reason}
      end

    health_status = Map.put(socket.assigns.service_health_status, service_atom, status)

    flash_msg =
      case status do
        :healthy -> "#{service_name} is working correctly"
        :missing_credentials -> "Missing credentials for #{service_name}"
        :invalid_credentials -> "Invalid credentials for #{service_name}"
        {:error, reason} -> "#{service_name} error: #{inspect(reason)}"
      end

    flash_type = if status == :healthy, do: :info, else: :error

    {:noreply,
     socket
     |> assign(:service_health_status, health_status)
     |> assign(:service_checking, nil)
     |> put_flash(flash_type, flash_msg)}
  rescue
    ArgumentError ->
      {:noreply,
       socket
       |> assign(:service_checking, nil)
       |> put_flash(:error, "Invalid service name")}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, load_section_data(socket)}
  end

  def handle_info({:tasks_loaded, available}, socket) do
    {:noreply,
     socket
     |> assign(:available_tasks, available)
     |> assign(:tasks_loading, false)}
  end

  def handle_info({:training_started, model_type, _started_at}, socket) do
    socket =
      socket
      |> assign(:ml_training_status, {:training, model_type, DateTime.utc_now()})
      |> append_training_log("Training #{model_type} started")

    {:noreply, socket}
  end

  def handle_info({:training_complete, model_type, result}, socket) do
    message =
      case result do
        {:ok, _} -> "Training #{model_type} completed successfully"
        {:error, reason} -> "Training #{model_type} failed: #{inspect(reason)}"
      end

    socket =
      socket
      |> assign(:ml_training_status, :idle)
      |> load_ml_training_data()
      |> append_training_log(message)
      |> put_flash(:info, message)

    {:noreply, socket}
  end

  def handle_info({:training_cancelled, model_type}, socket) do
    socket =
      socket
      |> assign(:ml_training_status, :idle)
      |> append_training_log("Training #{model_type} cancelled")

    {:noreply, socket}
  end

  def handle_info({:schedule_added, _id, model_type, interval}, socket) do
    socket =
      socket
      |> load_ml_training_data()
      |> append_training_log("Schedule added: #{model_type} every #{interval}h")

    {:noreply, socket}
  end

  def handle_info({:schedule_cancelled, _id}, socket) do
    {:noreply, load_ml_training_data(socket)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-xl font-bold">Settings</h1>
            <p class="text-sm text-base-content/60">Manage worlds and entities</p>
          </div>
          <button phx-click="refresh" class="btn btn-ghost btn-sm">
            <.icon name="hero-arrow-path" class="size-4" /> Refresh
          </button>
        </div>
      </:page_header>

      <div class="p-4 sm:p-6">
        <!-- Section Tabs -->
        <div class="tabs tabs-boxed mb-6">
          <button
            phx-click="switch_section"
            phx-value-section="worlds"
            class={["tab gap-1", if(@section == :worlds, do: "tab-active", else: "")]}
          >
            <.icon name="hero-globe-alt" class="size-4" /> Worlds
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="entities"
            class={["tab gap-1", if(@section == :entities, do: "tab-active", else: "")]}
          >
            <.icon name="hero-tag" class="size-4" /> Entities
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="training"
            class={["tab gap-1", if(@section == :training, do: "tab-active", else: "")]}
          >
            <.icon name="hero-academic-cap" class="size-4" /> Training
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="ml_training"
            class={["tab gap-1", if(@section == :ml_training, do: "tab-active", else: "")]}
          >
            <.icon name="hero-cpu-chip" class="size-4" /> ML Models
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="templates"
            class={["tab gap-1", if(@section == :templates, do: "tab-active", else: "")]}
          >
            <.icon name="hero-chat-bubble-bottom-center-text" class="size-4" /> Templates
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="services"
            class={["tab gap-1", if(@section == :services, do: "tab-active", else: "")]}
          >
            <.icon name="hero-cloud" class="size-4" /> Services
          </button>
        </div>

    <!-- Content -->
        <%= case @section do %>
          <% :worlds -> %>
            <.worlds_section
              worlds={@worlds}
              persisted_worlds={@persisted_worlds}
              new_world_name={@new_world_name}
              new_world_mode={@new_world_mode}
              creating_world={@creating_world}
            />
          <% :entities -> %>
            <.entities_section
              entity_types={@entity_types}
              type_entities={@type_entities}
              world_overlay={@world_overlay}
              selected_entity_type={@selected_entity_type}
              entity_search={@entity_search}
              new_entity_key={@new_entity_key}
              new_entity_value={@new_entity_value}
              new_entity_type={@new_entity_type}
              current_world_id={@current_world_id}
            />
          <% :training -> %>
            <.training_section
              training_sessions={@training_sessions}
              expanded_session_id={@expanded_session_id}
              available_tasks={@available_tasks}
              selected_capability={@selected_capability}
              starting_training={@starting_training}
              tasks_loading={@tasks_loading}
              lc_stats={@lc_stats}
            />
          <% :ml_training -> %>
            <.ml_training_section
              model_statuses={@ml_model_statuses}
              training_status={@ml_training_status}
              selected_model={@ml_selected_model}
              epochs={@ml_epochs}
              head_epochs={@ml_head_epochs}
              batch_size={@ml_batch_size}
              experiment_name={@ml_experiment_name}
              training_log={@ml_training_log}
              schedules={@ml_schedules}
              schedule_interval={@ml_schedule_interval}
              reloading={@ml_reloading}
            />
          <% :templates -> %>
            <.templates_section
              intents={@template_intents}
              selected_intent={@selected_template_intent}
              templates={@intent_templates}
              search={@template_search}
              new_template_text={@new_template_text}
              stats={@template_stats}
              has_unsaved={@template_has_unsaved}
            />
          <% :services -> %>
            <.services_section
              services={@services}
              credentials={@service_credentials}
              health_status={@service_health_status}
              checking={@service_checking}
            />
        <% end %>
      </div>
    </.app_shell>
    """
  end

  defp worlds_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Create World -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">Create New World</h3>
        <form phx-change="update_new_world" phx-submit="create_world" class="flex flex-wrap gap-4">
          <input
            type="text"
            name="name"
            value={@new_world_name}
            placeholder="World name"
            class="input input-bordered flex-1 min-w-[200px]"
          />
          <select name="mode" class="select select-bordered">
            <option value="persistent" selected={@new_world_mode == "persistent"}>Persistent</option>
            <option value="ephemeral" selected={@new_world_mode == "ephemeral"}>Ephemeral</option>
          </select>
          <button type="submit" class="btn btn-primary" disabled={@creating_world}>
            <%= if @creating_world do %>
              <span class="loading loading-spinner loading-sm"></span>
            <% else %>
              <.icon name="hero-plus" class="size-4" />
            <% end %>
            Create World
          </button>
        </form>
      </div>

    <!-- Active Worlds -->
      <div class="bg-base-100 rounded-xl border border-base-300/50">
        <div class="p-4 border-b border-base-300">
          <h3 class="font-semibold">Active Worlds</h3>
        </div>
        <%= if length(@worlds) == 0 do %>
          <div class="p-8 text-center text-base-content/50">
            <.icon name="hero-globe-alt" class="size-12 mx-auto mb-4 text-base-content/30" />
            <p>No active worlds</p>
          </div>
        <% else %>
          <div class="divide-y divide-base-300/50">
            <%= for world <- @worlds do %>
              <div class="p-4 flex items-center justify-between hover:bg-base-200/50">
                <div>
                  <div class="font-medium">{world.name}</div>
                  <div class="text-sm text-base-content/60 font-mono">{world.id}</div>
                </div>
                <div class="flex items-center gap-2">
                  <span class={[
                    "badge badge-sm",
                    if(world.mode == :persistent, do: "badge-info", else: "badge-ghost")
                  ]}>
                    {world.mode}
                  </span>
                  <%= if world.mode == :persistent do %>
                    <button
                      phx-click="save_world"
                      phx-value-id={world.id}
                      class="btn btn-ghost btn-xs"
                      title="Save to disk"
                    >
                      <.icon name="hero-cloud-arrow-up" class="size-4" />
                    </button>
                  <% end %>
                  <%= if world.id != "default" do %>
                    <button
                      phx-click="delete_world"
                      phx-value-id={world.id}
                      class="btn btn-ghost btn-xs text-error"
                      title="Delete world"
                      data-confirm="Are you sure you want to delete this world?"
                    >
                      <.icon name="hero-trash" class="size-4" />
                    </button>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>

    <!-- Persisted Worlds (not loaded) -->
      <% not_loaded =
        Enum.filter(@persisted_worlds, fn pw -> not Enum.any?(@worlds, &(&1.id == pw.id)) end) %>
      <%= if length(not_loaded) > 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50">
          <div class="p-4 border-b border-base-300">
            <h3 class="font-semibold">Persisted Worlds (Not Loaded)</h3>
          </div>
          <div class="divide-y divide-base-300/50">
            <%= for world <- not_loaded do %>
              <div class="p-4 flex items-center justify-between hover:bg-base-200/50">
                <div>
                  <div class="font-medium text-base-content/70">{world.name}</div>
                  <div class="text-sm text-base-content/50 font-mono">{world.id}</div>
                </div>
                <button
                  phx-click="load_world"
                  phx-value-id={world.id}
                  class="btn btn-ghost btn-xs"
                >
                  <.icon name="hero-arrow-down-tray" class="size-4" /> Load
                </button>
              </div>
            <% end %>
          </div>
        </div>
      <% end %>
    </div>
    """
  end

  defp entities_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Add Entity -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">
          Add Entity to World: <span class="text-primary">{@current_world_id}</span>
        </h3>
        <form phx-change="update_new_entity" phx-submit="add_entity" class="flex flex-wrap gap-4">
          <input
            type="text"
            name="key"
            value={@new_entity_key}
            placeholder="Lookup key (e.g., 'new york')"
            class="input input-bordered flex-1 min-w-[200px]"
          />
          <input
            type="text"
            name="value"
            value={@new_entity_value}
            placeholder="Canonical value (optional)"
            class="input input-bordered flex-1 min-w-[200px]"
          />
          <select name="type" class="select select-bordered">
            <%= for type <- @entity_types do %>
              <option value={type} selected={@new_entity_type == type}>{type}</option>
            <% end %>
          </select>
          <button type="submit" class="btn btn-primary">
            <.icon name="hero-plus" class="size-4" /> Add
          </button>
        </form>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <!-- Entity Type Sidebar -->
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <h3 class="font-semibold mb-4">Entity Types</h3>
          <ul class="space-y-1">
            <%= for type <- @entity_types do %>
              <li>
                <button
                  phx-click="select_entity_type"
                  phx-value-type={type}
                  class={[
                    "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
                    if(type == @selected_entity_type,
                      do: "bg-primary/10 text-primary font-medium",
                      else: "hover:bg-base-200"
                    )
                  ]}
                >
                  {type}
                </button>
              </li>
            <% end %>
          </ul>
        </div>

    <!-- Entities List -->
        <div class="lg:col-span-3 bg-base-100 rounded-xl border border-base-300/50">
          <div class="p-4 border-b border-base-300 flex items-center gap-4">
            <h3 class="font-semibold">{@selected_entity_type || "Select a type"}</h3>
            <input
              type="text"
              placeholder="Search entities..."
              value={@entity_search}
              phx-keyup="search_entities"
              name="query"
              phx-debounce="150"
              class="input input-sm input-bordered flex-1 max-w-xs"
            />
          </div>
          <%= if length(@type_entities) == 0 do %>
            <div class="p-8 text-center text-base-content/50">
              <.icon name="hero-tag" class="size-12 mx-auto mb-4 text-base-content/30" />
              <p>No entities of this type</p>
            </div>
          <% else %>
            <% filtered = filter_entities(@type_entities, @entity_search) %>
            <div class="max-h-96 overflow-y-auto">
              <table class="table table-sm">
                <thead class="bg-base-200/50 sticky top-0">
                  <tr>
                    <th>Key</th>
                    <th>Value</th>
                    <th>Source</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  <%= for entity <- Enum.take(filtered, 100) do %>
                    <% is_from_overlay = is_overlay_entity(entity, @world_overlay) %>
                    <tr class="hover:bg-base-200/30">
                      <td class="font-medium">{entity.key}</td>
                      <td>{entity.value || entity.key}</td>
                      <td>
                        <span class={[
                          "badge badge-xs",
                          if(is_from_overlay, do: "badge-primary", else: "badge-ghost")
                        ]}>
                          {if is_from_overlay, do: "world", else: "global"}
                        </span>
                      </td>
                      <td>
                        <%= if is_from_overlay do %>
                          <button
                            phx-click="remove_entity"
                            phx-value-key={entity.key}
                            class="btn btn-ghost btn-xs text-error"
                            title="Remove from world"
                          >
                            <.icon name="hero-x-mark" class="size-4" />
                          </button>
                        <% end %>
                      </td>
                    </tr>
                  <% end %>
                </tbody>
              </table>
              <%= if length(filtered) > 100 do %>
                <div class="p-4 text-center text-sm text-base-content/60">
                  Showing 100 of {length(filtered)} entities
                </div>
              <% end %>
            </div>
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  defp training_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Stats Overview -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <.icon name="hero-academic-cap" class="size-5 text-primary" />
            </div>
            <div>
              <div class="text-2xl font-bold">{@lc_stats[:total_sessions] || 0}</div>
              <div class="text-sm text-base-content/60">Total Sessions</div>
            </div>
          </div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-warning/10 flex items-center justify-center">
              <.icon name="hero-cpu-chip" class="size-5 text-warning" />
            </div>
            <div>
              <div class="text-2xl font-bold">{@lc_stats[:active_agents] || 0}</div>
              <div class="text-sm text-base-content/60">Active Agents</div>
            </div>
          </div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center">
              <.icon name="hero-document-text" class="size-5 text-success" />
            </div>
            <div>
              <div class="text-2xl font-bold">{map_size(@available_tasks)}</div>
              <div class="text-sm text-base-content/60">Task Categories</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Start Training -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">Start Task-Based Training</h3>
        <p class="text-sm text-base-content/60 mb-4">
          Train child agents using curated NLP benchmark tasks. Select a capability to focus the training.
        </p>
        <div class="flex flex-wrap gap-4 items-end">
          <div class="form-control">
            <label class="label">
              <span class="label-text">Capability</span>
            </label>
            <select
              phx-change="select_capability"
              name="capability"
              class="select select-bordered"
            >
              <option value="all" selected={@selected_capability == :all}>All Capabilities</option>
              <option value="question_answering" selected={@selected_capability == :question_answering}>
                Question Answering
              </option>
              <option value="commonsense" selected={@selected_capability == :commonsense}>
                Commonsense Reasoning
              </option>
              <option value="sentiment" selected={@selected_capability == :sentiment}>
                Sentiment Analysis
              </option>
              <option value="reasoning" selected={@selected_capability == :reasoning}>
                Explanation & Reasoning
              </option>
            </select>
          </div>
          <button
            phx-click="start_task_training"
            class="btn btn-primary"
            disabled={@starting_training}
          >
            <%= if @starting_training do %>
              <span class="loading loading-spinner loading-sm"></span>
            <% else %>
              <.icon name="hero-play" class="size-4" />
            <% end %>
            Start Training
          </button>
        </div>
      </div>

      <!-- Active Sessions -->
      <div class="bg-base-100 rounded-xl border border-base-300/50">
        <div class="p-4 border-b border-base-300 flex items-center justify-between">
          <h3 class="font-semibold">Training Sessions</h3>
          <.link navigate={~p"/sessions"} class="text-sm text-primary hover:underline">
            View All Sessions
          </.link>
        </div>
        <%= if length(@training_sessions) == 0 do %>
          <div class="p-8 text-center text-base-content/50">
            <.icon name="hero-academic-cap" class="size-12 mx-auto mb-4 text-base-content/30" />
            <p>No active training sessions</p>
            <p class="text-sm mt-2">Start a training session above to begin</p>
          </div>
        <% else %>
          <div class="divide-y divide-base-300/50">
            <%= for session <- @training_sessions do %>
              <% is_expanded = @expanded_session_id == session.id %>
              <% failed_goals = Enum.count(session.goals, &(&1.status == :failed)) %>
              <% completed_goals = Enum.count(session.goals, &(&1.status == :completed)) %>
              <% total_goals = length(session.goals) %>
              <div class={["transition-colors", if(is_expanded, do: "bg-base-200/30", else: "hover:bg-base-200/50")]}>
                <!-- Session Header (clickable) -->
                <div
                  class="p-4 cursor-pointer"
                  phx-click="toggle_session_detail"
                  phx-value-id={session.id}
                >
                  <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-2">
                      <.icon
                        name={if is_expanded, do: "hero-chevron-down", else: "hero-chevron-right"}
                        class="size-4 text-base-content/40"
                      />
                      <div>
                        <div class="font-medium">{session.topic || "Untitled Session"}</div>
                        <div class="text-xs text-base-content/40 font-mono">{session.id}</div>
                      </div>
                    </div>
                    <div class="flex items-center gap-2">
                      <span class={["badge badge-sm", session_status_badge(session.status)]}>
                        {session.status}
                      </span>
                      <%= if session.status == :active do %>
                        <button
                          phx-click="cancel_session"
                          phx-value-id={session.id}
                          class="btn btn-ghost btn-xs text-error"
                          title="Cancel session"
                        >
                          <.icon name="hero-stop" class="size-4" />
                        </button>
                      <% end %>
                    </div>
                  </div>

                  <!-- Summary stats row -->
                  <div class="flex flex-wrap gap-3 ml-6 text-xs">
                    <span class="flex items-center gap-1 text-base-content/60">
                      <.icon name="hero-flag" class="size-3" />
                      {completed_goals}/{total_goals} goals
                    </span>
                    <%= if failed_goals > 0 do %>
                      <span class="flex items-center gap-1 text-error">
                        <.icon name="hero-exclamation-triangle" class="size-3" />
                        {failed_goals} failed
                      </span>
                    <% end %>
                    <%= if session.findings_count > 0 do %>
                      <span class="flex items-center gap-1 text-info">
                        <.icon name="hero-document-magnifying-glass" class="size-3" />
                        {session.findings_count} findings
                      </span>
                    <% end %>
                    <%= if session.approved_count > 0 do %>
                      <span class="flex items-center gap-1 text-success">
                        <.icon name="hero-check-circle" class="size-3" />
                        {session.approved_count} approved
                      </span>
                    <% end %>
                    <%= if session.rejected_count > 0 do %>
                      <span class="flex items-center gap-1 text-error/70">
                        <.icon name="hero-x-circle" class="size-3" />
                        {session.rejected_count} rejected
                      </span>
                    <% end %>
                    <%= if session.hypotheses_tested > 0 do %>
                      <span class="flex items-center gap-1 text-base-content/60">
                        <.icon name="hero-beaker" class="size-3" />
                        {session.hypotheses_tested} hypotheses
                      </span>
                    <% end %>
                    <%= if session.started_at do %>
                      <span class="text-base-content/40">
                        Started {Calendar.strftime(session.started_at, "%H:%M:%S")}
                      </span>
                    <% end %>
                    <%= if session.completed_at do %>
                      <span class="text-base-content/40">
                        Completed {Calendar.strftime(session.completed_at, "%H:%M:%S")}
                      </span>
                    <% end %>
                  </div>
                </div>

                <!-- Expanded Detail Panel -->
                <%= if is_expanded do %>
                  <div class="px-4 pb-4 ml-6 space-y-4">
                    <!-- Goals Detail -->
                    <div class="bg-base-100 rounded-lg border border-base-300/30">
                      <div class="px-3 py-2 border-b border-base-300/30">
                        <h4 class="text-sm font-semibold text-base-content/80">
                          Research Goals ({total_goals})
                        </h4>
                      </div>
                      <%= if total_goals == 0 do %>
                        <div class="p-3 text-sm text-base-content/50">No goals defined</div>
                      <% else %>
                        <div class="divide-y divide-base-300/20">
                          <%= for goal <- session.goals do %>
                            <div class="p-3">
                              <div class="flex items-start justify-between gap-2">
                                <div class="flex-1 min-w-0">
                                  <div class="flex items-center gap-2 mb-1">
                                    <span class={["badge badge-xs", goal_status_badge(goal.status)]}>
                                      {goal.status}
                                    </span>
                                    <span class="font-medium text-sm truncate">{goal.topic}</span>
                                    <%= if goal.priority != :normal do %>
                                      <span class={["badge badge-xs badge-outline", priority_badge(goal.priority)]}>
                                        {goal.priority}
                                      </span>
                                    <% end %>
                                  </div>
                                  <%= if length(goal.questions) > 0 do %>
                                    <div class="ml-2 mt-1 space-y-0.5">
                                      <%= for question <- goal.questions do %>
                                        <div class="text-xs text-base-content/50 flex items-start gap-1">
                                          <span class="text-base-content/30 shrink-0">Q:</span>
                                          <span>{display_value(question)}</span>
                                        </div>
                                      <% end %>
                                    </div>
                                  <% end %>
                                  <%= if map_size(goal.constraints) > 0 do %>
                                    <div class="flex flex-wrap gap-1 mt-1 ml-2">
                                      <%= for {key, val} <- goal.constraints do %>
                                        <span class="badge badge-xs badge-ghost">{key}: {display_value(val)}</span>
                                      <% end %>
                                    </div>
                                  <% end %>
                                </div>
                                <div class="shrink-0">
                                  <%= case goal.status do %>
                                    <% :completed -> %>
                                      <.icon name="hero-check-circle" class="size-5 text-success" />
                                    <% :failed -> %>
                                      <.icon name="hero-x-circle" class="size-5 text-error" />
                                    <% :in_progress -> %>
                                      <span class="loading loading-spinner loading-xs text-warning"></span>
                                    <% _ -> %>
                                      <.icon name="hero-clock" class="size-5 text-base-content/30" />
                                  <% end %>
                                </div>
                              </div>
                            </div>
                          <% end %>
                        </div>
                      <% end %>
                    </div>

                    <!-- Investigations Detail -->
                    <%= if length(session.investigations) > 0 do %>
                      <div class="bg-base-100 rounded-lg border border-base-300/30">
                        <div class="px-3 py-2 border-b border-base-300/30">
                          <h4 class="text-sm font-semibold text-base-content/80">
                            <.icon name="hero-beaker" class="size-4 inline-block mr-1" />
                            Scientific Investigations ({length(session.investigations)})
                          </h4>
                        </div>
                        <div class="divide-y divide-base-300/20">
                          <%= for investigation <- session.investigations do %>
                            <div class="p-3">
                              <div class="flex items-center justify-between mb-2">
                                <span class="font-medium text-sm">{investigation.topic}</span>
                                <div class="flex items-center gap-2">
                                  <span class={["badge badge-xs", investigation_status_badge(investigation.status)]}>
                                    {investigation.status}
                                  </span>
                                  <%= if investigation.conclusion do %>
                                    <span class={["badge badge-xs", conclusion_badge(investigation.conclusion)]}>
                                      {investigation.conclusion}
                                    </span>
                                  <% end %>
                                </div>
                              </div>
                              <!-- Hypotheses within investigation -->
                              <%= if length(investigation.hypotheses) > 0 do %>
                                <div class="ml-2 space-y-1">
                                  <%= for hypothesis <- investigation.hypotheses do %>
                                    <div class="flex items-start gap-2 text-xs">
                                      <span class={["badge badge-xs shrink-0 mt-0.5", hypothesis_status_badge(hypothesis.status)]}>
                                        {hypothesis.status}
                                      </span>
                                      <div class="min-w-0">
                                        <span class="text-base-content/70">{hypothesis.claim}</span>
                                        <%= if hypothesis.confidence > 0 do %>
                                          <span class="text-base-content/40 ml-1">
                                            ({Float.round(hypothesis.confidence * 100, 1)}% confidence)
                                          </span>
                                        <% end %>
                                      </div>
                                    </div>
                                  <% end %>
                                </div>
                              <% end %>
                              <!-- Evidence counts -->
                              <div class="flex flex-wrap gap-2 mt-2 text-xs text-base-content/50">
                                <span>{length(investigation.evidence)} evidence items</span>
                                <%= if length(investigation.control_evidence) > 0 do %>
                                  <span>{length(investigation.control_evidence)} control items</span>
                                <% end %>
                                <%= if investigation.concluded_at do %>
                                  <span>
                                    Concluded {Calendar.strftime(investigation.concluded_at, "%H:%M:%S")}
                                  </span>
                                <% end %>
                              </div>
                            </div>
                          <% end %>
                        </div>
                      </div>
                    <% end %>

                    <!-- Session Metrics Summary -->
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <div class="bg-base-100 rounded-lg border border-base-300/30 p-3 text-center">
                        <div class="text-lg font-bold">{session.findings_count}</div>
                        <div class="text-xs text-base-content/50">Findings</div>
                      </div>
                      <div class="bg-base-100 rounded-lg border border-base-300/30 p-3 text-center">
                        <div class="text-lg font-bold text-success">{session.approved_count}</div>
                        <div class="text-xs text-base-content/50">Approved</div>
                      </div>
                      <div class="bg-base-100 rounded-lg border border-base-300/30 p-3 text-center">
                        <div class="text-lg font-bold text-error">{session.rejected_count}</div>
                        <div class="text-xs text-base-content/50">Rejected</div>
                      </div>
                      <div class="bg-base-100 rounded-lg border border-base-300/30 p-3 text-center">
                        <div class="text-lg font-bold">
                          <%= if session.hypotheses_tested > 0 do %>
                            {Float.round(session.hypotheses_supported / session.hypotheses_tested * 100, 1)}%
                          <% else %>
                            N/A
                          <% end %>
                        </div>
                        <div class="text-xs text-base-content/50">Support Rate</div>
                      </div>
                    </div>
                  </div>
                <% end %>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>

      <!-- Available Task Categories -->
      <div class="bg-base-100 rounded-xl border border-base-300/50">
        <div class="p-4 border-b border-base-300">
          <h3 class="font-semibold">Available Task Categories</h3>
          <p class="text-sm text-base-content/60">Domain-specific NLP tasks from benchmarks</p>
        </div>
        <%= if @tasks_loading do %>
          <div class="p-8 text-center text-base-content/50">
            <span class="loading loading-spinner loading-lg text-primary"></span>
            <p class="mt-4">Scanning task files...</p>
            <p class="text-sm mt-2">This may take a moment on first load</p>
          </div>
        <% else %>
          <%= if map_size(@available_tasks) == 0 do %>
            <div class="p-8 text-center text-base-content/50">
              <.icon name="hero-document-text" class="size-12 mx-auto mb-4 text-base-content/30" />
              <p>No tasks available</p>
              <p class="text-sm mt-2">Check that domain task files are in data/domain_specific_tasks/</p>
            </div>
          <% else %>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
              <%= for {category, tasks} <- @available_tasks do %>
                <div class="bg-base-200/50 rounded-lg p-3">
                  <div class="flex items-center justify-between mb-2">
                    <span class="font-medium text-sm">{category}</span>
                    <span class="badge badge-sm badge-ghost">{length(tasks)} tasks</span>
                  </div>
                  <div class="text-xs text-base-content/60">
                    <%= for task <- Enum.take(tasks, 3) do %>
                      <div class="truncate">{task.task_id}</div>
                    <% end %>
                    <%= if length(tasks) > 3 do %>
                      <div class="text-primary">+{length(tasks) - 3} more</div>
                    <% end %>
                  </div>
                </div>
              <% end %>
            </div>
          <% end %>
        <% end %>
      </div>
    </div>
    """
  end

  defp ml_training_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Model Status Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center justify-between">
            <div>
              <div class="font-medium">Unified Model</div>
              <div class="text-sm text-base-content/60">Intent, NER, Sentiment, Speech Act</div>
            </div>
            <span class={[
              "badge",
              if(@model_statuses[:unified_model], do: "badge-success", else: "badge-ghost")
            ]}>
              {if @model_statuses[:unified_model], do: "Ready", else: "Not Ready"}
            </span>
          </div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center justify-between">
            <div>
              <div class="font-medium">Response Scorer</div>
              <div class="text-sm text-base-content/60">Query-response quality scoring</div>
            </div>
            <span class={[
              "badge",
              if(@model_statuses[:response_scorer], do: "badge-success", else: "badge-ghost")
            ]}>
              {if @model_statuses[:response_scorer], do: "Ready", else: "Not Ready"}
            </span>
          </div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center justify-between">
            <div>
              <div class="font-medium">Intent Arbitrator</div>
              <div class="text-sm text-base-content/60">LSTM vs TF-IDF meta-learner</div>
            </div>
            <span class={[
              "badge",
              if(@model_statuses[:intent_arbitrator], do: "badge-success", else: "badge-ghost")
            ]}>
              {if @model_statuses[:intent_arbitrator], do: "Ready", else: "Not Ready"}
            </span>
          </div>
        </div>
      </div>

      <!-- Reload Models -->
      <div class="flex justify-end">
        <button
          phx-click="reload_ml_models"
          class="btn btn-outline btn-sm"
          disabled={@reloading}
        >
          <%= if @reloading do %>
            <span class="loading loading-spinner loading-sm"></span>
          <% else %>
            <.icon name="hero-arrow-path" class="size-4" />
          <% end %>
          Reload All Models
        </button>
      </div>

      <!-- Training Form -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">Train ML Model</h3>
        <p class="text-sm text-base-content/60 mb-4">
          Start an async training job for a specific model. Training runs in the background.
        </p>
        <form phx-change="update_ml_training_form" phx-submit="start_ml_training">
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
            <div class="form-control">
              <label class="label">
                <span class="label-text">Model Type</span>
              </label>
              <select name="model_type" class="select select-bordered">
                <option value="unified" selected={@selected_model == "unified"}>
                  Unified LSTM
                </option>
                <option value="response" selected={@selected_model == "response"}>
                  Response Scorer
                </option>
                <option value="tfidf" selected={@selected_model == "tfidf"}>
                  TF-IDF Classifier
                </option>
                <option value="arbitrator" selected={@selected_model == "arbitrator"}>
                  Intent Arbitrator
                </option>
              </select>
            </div>
            <div class="form-control">
              <label class="label">
                <span class="label-text">Encoder Epochs</span>
              </label>
              <input
                type="number"
                name="epochs"
                value={@epochs}
                min="1"
                max="200"
                class="input input-bordered"
              />
            </div>
            <div class="form-control">
              <label class="label">
                <span class="label-text">Head Epochs</span>
              </label>
              <input
                type="number"
                name="head_epochs"
                value={@head_epochs}
                min="1"
                max="200"
                class="input input-bordered"
                title="Epochs for task heads (sentiment, speech act) trained with frozen encoder"
              />
            </div>
            <div class="form-control">
              <label class="label">
                <span class="label-text">Batch Size</span>
              </label>
              <input
                type="number"
                name="batch_size"
                value={@batch_size}
                min="1"
                max="256"
                class="input input-bordered"
              />
            </div>
            <div class="form-control">
              <label class="label">
                <span class="label-text">Experiment Name</span>
              </label>
              <input
                type="text"
                name="experiment_name"
                value={@experiment_name}
                placeholder="Optional"
                class="input input-bordered"
              />
            </div>
          </div>
          <div class="flex items-center gap-4">
            <%= case @training_status do %>
              <% :idle -> %>
                <button type="submit" class="btn btn-primary">
                  <.icon name="hero-play" class="size-4" /> Train
                </button>
              <% {:training, model_type, started_at} -> %>
                <div class="flex items-center gap-3">
                  <span class="loading loading-spinner loading-md text-warning"></span>
                  <div>
                    <div class="font-medium">Training {model_type}...</div>
                    <div class="text-xs text-base-content/60">
                      Started {Calendar.strftime(started_at, "%H:%M:%S")}
                    </div>
                  </div>
                  <button
                    type="button"
                    phx-click="cancel_ml_training"
                    class="btn btn-error btn-sm"
                  >
                    <.icon name="hero-stop" class="size-4" /> Cancel
                  </button>
                </div>
            <% end %>
          </div>
        </form>
      </div>

      <!-- Training Progress Log -->
      <%= if length(@training_log) > 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50">
          <div class="p-4 border-b border-base-300">
            <h3 class="font-semibold">Training Log</h3>
          </div>
          <div class="max-h-48 overflow-y-auto p-4 font-mono text-sm space-y-1">
            <%= for {message, timestamp} <- Enum.reverse(@training_log) do %>
              <div class="text-base-content/80">
                <span class="text-base-content/40">[{Calendar.strftime(timestamp, "%H:%M:%S")}]</span>
                {message}
              </div>
            <% end %>
          </div>
        </div>
      <% end %>

      <!-- Scheduling -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">Training Schedules</h3>
        <p class="text-sm text-base-content/60 mb-4">
          Schedule recurring training runs. Uses the model type and parameters from the form above.
        </p>
        <div class="flex flex-wrap gap-4 items-end mb-4">
          <div class="form-control">
            <label class="label">
              <span class="label-text">Interval (hours)</span>
            </label>
            <select
              name="interval"
              phx-change="update_ml_schedule_interval"
              class="select select-bordered"
            >
              <option value="1" selected={@schedule_interval == "1"}>Every 1 hour</option>
              <option value="6" selected={@schedule_interval == "6"}>Every 6 hours</option>
              <option value="12" selected={@schedule_interval == "12"}>Every 12 hours</option>
              <option value="24" selected={@schedule_interval == "24"}>Every 24 hours</option>
              <option value="48" selected={@schedule_interval == "48"}>Every 48 hours</option>
              <option value="168" selected={@schedule_interval == "168"}>Every 7 days</option>
            </select>
          </div>
          <button phx-click="add_ml_schedule" class="btn btn-outline btn-primary">
            <.icon name="hero-clock" class="size-4" /> Schedule
          </button>
        </div>

        <!-- Active Schedules -->
        <%= if length(@schedules) == 0 do %>
          <div class="text-sm text-base-content/50 p-4 text-center">
            No active schedules
          </div>
        <% else %>
          <div class="divide-y divide-base-300/50 border border-base-300/50 rounded-lg">
            <%= for schedule <- @schedules do %>
              <div class="p-3 flex items-center justify-between hover:bg-base-200/50">
                <div>
                  <span class="badge badge-sm badge-primary mr-2">{schedule.model_type}</span>
                  <span class="text-sm">Every {schedule.interval_hours} hour(s)</span>
                  <span class="text-xs text-base-content/50 ml-2 font-mono">{schedule.id}</span>
                </div>
                <button
                  phx-click="cancel_ml_schedule"
                  phx-value-id={schedule.id}
                  class="btn btn-ghost btn-xs text-error"
                  title="Cancel schedule"
                >
                  <.icon name="hero-x-mark" class="size-4" />
                </button>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  defp session_status_badge(:active) do
    "badge-warning"
  end

  defp session_status_badge(:completed) do
    "badge-success"
  end

  defp session_status_badge(:cancelled) do
    "badge-error"
  end

  defp session_status_badge(_) do
    "badge-ghost"
  end

  defp goal_status_badge(:completed), do: "badge-success"
  defp goal_status_badge(:failed), do: "badge-error"
  defp goal_status_badge(:in_progress), do: "badge-warning"
  defp goal_status_badge(:pending), do: "badge-ghost"
  defp goal_status_badge(_), do: "badge-ghost"

  defp priority_badge(:high), do: "badge-error"
  defp priority_badge(:low), do: "badge-ghost"
  defp priority_badge(_), do: ""

  defp investigation_status_badge(:concluded), do: "badge-success"
  defp investigation_status_badge(:evaluating), do: "badge-warning"
  defp investigation_status_badge(:gathering_evidence), do: "badge-info"
  defp investigation_status_badge(:planning), do: "badge-ghost"
  defp investigation_status_badge(_), do: "badge-ghost"

  defp conclusion_badge(:hypotheses_supported), do: "badge-success"
  defp conclusion_badge(:hypotheses_falsified), do: "badge-error"
  defp conclusion_badge(:inconclusive), do: "badge-warning"
  defp conclusion_badge(:mixed), do: "badge-warning"
  defp conclusion_badge(_), do: "badge-ghost"

  defp hypothesis_status_badge(:supported), do: "badge-success"
  defp hypothesis_status_badge(:falsified), do: "badge-error"
  defp hypothesis_status_badge(:inconclusive), do: "badge-warning"
  defp hypothesis_status_badge(:testing), do: "badge-info"
  defp hypothesis_status_badge(:untested), do: "badge-ghost"
  defp hypothesis_status_badge(_), do: "badge-ghost"

  defp display_value(val) when is_binary(val), do: val
  defp display_value(val) when is_atom(val), do: Atom.to_string(val)
  defp display_value(val) when is_number(val), do: to_string(val)
  defp display_value(%{text: text}) when is_binary(text), do: text
  defp display_value(%{claim: claim}) when is_binary(claim), do: claim
  defp display_value(val) when is_map(val), do: inspect(val, limit: 5, pretty: false)
  defp display_value(val) when is_list(val), do: Enum.map_join(val, ", ", &display_value/1)
  defp display_value(val), do: inspect(val)

  defp templates_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Stats & Actions Bar -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div class="flex flex-wrap gap-4">
            <div class="stat p-0">
              <div class="stat-title text-xs">Intents</div>
              <div class="stat-value text-lg">{Map.get(@stats, :intent_count, 0)}</div>
            </div>
            <div class="stat p-0">
              <div class="stat-title text-xs">Templates</div>
              <div class="stat-value text-lg">{Map.get(@stats, :template_count, 0)}</div>
            </div>
            <div class="stat p-0">
              <div class="stat-title text-xs">Admin Added</div>
              <div class="stat-value text-lg">{Map.get(@stats, :admin_template_count, 0)}</div>
            </div>
          </div>
          <div class="flex gap-2">
            <%= if @has_unsaved do %>
              <button phx-click="sync_templates" class="btn btn-primary btn-sm gap-1">
                <.icon name="hero-arrow-down-tray" class="size-4" />
                Save Changes
              </button>
            <% end %>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Intent List -->
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <h3 class="font-semibold mb-4">Intents</h3>
          <div class="mb-3">
            <input
              type="text"
              name="query"
              value={@search}
              placeholder="Search intents..."
              phx-debounce="300"
              phx-change="search_templates"
              class="input input-sm input-bordered w-full"
            />
          </div>
          <div class="overflow-y-auto max-h-[400px] space-y-1">
            <%= for intent <- filter_intents(@intents, @search) do %>
              <button
                phx-click="select_template_intent"
                phx-value-intent={intent}
                class={[
                  "w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors",
                  if(intent == @selected_intent,
                    do: "bg-primary text-primary-content",
                    else: "hover:bg-base-200")
                ]}
                title={intent}
              >
                {intent}
              </button>
            <% end %>
          </div>
        </div>

        <!-- Templates for Selected Intent -->
        <div class="lg:col-span-2 bg-base-100 rounded-xl border border-base-300/50 p-4">
          <h3 class="font-semibold mb-4">
            Templates
            <%= if @selected_intent do %>
              <span class="text-sm font-normal text-base-content/60">for {@selected_intent}</span>
            <% end %>
          </h3>

          <%= if @selected_intent do %>
            <!-- Add Template Form -->
            <form phx-submit="add_template" class="mb-4 flex gap-2">
              <input type="hidden" name="intent" value={@selected_intent} />
              <input
                type="text"
                name="text"
                value={@new_template_text}
                placeholder="Enter new template text..."
                phx-change="update_new_template"
                class="input input-bordered flex-1"
              />
              <button type="submit" class="btn btn-primary">
                <.icon name="hero-plus" class="size-4" /> Add
              </button>
            </form>

            <!-- Template List -->
            <div class="space-y-2 max-h-[400px] overflow-y-auto">
              <%= if length(@templates) == 0 do %>
                <div class="text-sm text-base-content/50 p-4 text-center">
                  No templates for this intent
                </div>
              <% else %>
                <%= for template <- @templates do %>
                  <div class="flex items-start gap-3 p-3 bg-base-200/50 rounded-lg group">
                    <div class="flex-1 min-w-0">
                      <p class="text-sm">{template.text}</p>
                      <div class="flex gap-2 mt-1">
                        <span class={[
                          "badge badge-xs",
                          if(template.source == :admin, do: "badge-primary", else: "badge-ghost")
                        ]}>
                          {template.source}
                        </span>
                        <%= if template.condition do %>
                          <span class="badge badge-xs badge-info" title={template.condition}>
                            conditional
                          </span>
                        <% end %>
                      </div>
                    </div>
                    <button
                      phx-click="remove_template"
                      phx-value-text={template.text}
                      class="btn btn-ghost btn-xs text-error opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Remove template"
                    >
                      <.icon name="hero-trash" class="size-4" />
                    </button>
                  </div>
                <% end %>
              <% end %>
            </div>
          <% else %>
            <div class="text-sm text-base-content/50 p-8 text-center">
              Select an intent to view its templates
            </div>
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  defp services_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Services Overview -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="font-semibold">External Services</h3>
            <p class="text-sm text-base-content/60">
              Configure API credentials for live data enrichment
            </p>
          </div>
          <div class="badge badge-outline gap-1">
            <.icon name="hero-shield-check" class="size-3" />
            Credentials are encrypted
          </div>
        </div>

        <div class="alert alert-info mb-4">
          <.icon name="hero-information-circle" class="size-5" />
          <span>
            API keys are stored encrypted and never exposed in responses or logs.
            Each world can have its own service credentials.
          </span>
        </div>
      </div>

      <!-- Service Cards -->
      <%= if length(@services) == 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-8 text-center">
          <.icon name="hero-cloud" class="size-12 mx-auto text-base-content/30 mb-4" />
          <h3 class="font-semibold mb-2">No Services Available</h3>
          <p class="text-sm text-base-content/60">
            No external services are configured in this installation.
          </p>
        </div>
      <% else %>
        <div class="grid gap-4 md:grid-cols-2">
          <%= for service <- @services do %>
            <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
              <!-- Service Header -->
              <div class="flex items-start justify-between mb-4">
                <div>
                  <h4 class="font-semibold flex items-center gap-2">
                    <.icon name={service_icon(service.name)} class="size-5" />
                    {service.display_name}
                  </h4>
                  <p class="text-sm text-base-content/60 mt-1">{service.description}</p>
                </div>
                <.service_status_badge
                  configured={service.configured}
                  health={Map.get(@health_status, service.name)}
                />
              </div>

              <!-- Supported Intents -->
              <div class="mb-4">
                <div class="text-xs text-base-content/60 mb-1">Supports:</div>
                <div class="flex flex-wrap gap-1">
                  <%= for intent <- service.supported_intents do %>
                    <span class="badge badge-sm badge-ghost">{intent}</span>
                  <% end %>
                </div>
              </div>

              <!-- Credential Forms -->
              <div class="space-y-3">
                <%= for cred_key <- service.required_credentials do %>
                  <% has_cred = get_in(@credentials, [service.name, cred_key]) %>
                  <div class="form-control">
                    <label class="label py-1">
                      <span class="label-text text-sm">{humanize_credential(cred_key)}</span>
                      <%= if has_cred do %>
                        <span class="badge badge-success badge-xs gap-1">
                          <.icon name="hero-check" class="size-3" /> Set
                        </span>
                      <% end %>
                    </label>
                    <div class="flex gap-2">
                      <form
                        phx-submit="save_credential"
                        class="flex-1 flex gap-2"
                      >
                        <input type="hidden" name="service" value={service.name} />
                        <input type="hidden" name="key" value={cred_key} />
                        <input
                          type="password"
                          name="value"
                          placeholder={if has_cred, do: "••••••••", else: "Enter #{humanize_credential(cred_key)}..."}
                          class="input input-sm input-bordered flex-1"
                          autocomplete="off"
                        />
                        <button type="submit" class="btn btn-sm btn-primary">
                          <.icon name="hero-key" class="size-4" />
                          Save
                        </button>
                      </form>
                      <%= if has_cred do %>
                        <button
                          phx-click="delete_credential"
                          phx-value-service={service.name}
                          phx-value-key={cred_key}
                          class="btn btn-sm btn-ghost text-error"
                          title="Remove credential"
                        >
                          <.icon name="hero-trash" class="size-4" />
                        </button>
                      <% end %>
                    </div>
                  </div>
                <% end %>
              </div>

              <!-- Health Check -->
              <%= if service.configured do %>
                <div class="mt-4 pt-4 border-t border-base-300/50">
                  <button
                    phx-click="check_service_health"
                    phx-value-service={service.name}
                    class="btn btn-sm btn-outline w-full gap-1"
                    disabled={@checking == service.name}
                  >
                    <%= if @checking == service.name do %>
                      <span class="loading loading-spinner loading-xs"></span>
                      Checking...
                    <% else %>
                      <.icon name="hero-signal" class="size-4" />
                      Test Connection
                    <% end %>
                  </button>
                </div>
              <% end %>
            </div>
          <% end %>
        </div>
      <% end %>
    </div>
    """
  end

  defp service_status_badge(assigns) do
    ~H"""
    <div class={[
      "badge gap-1",
      cond do
        @health == :healthy -> "badge-success"
        @health in [:invalid_credentials, :missing_credentials] -> "badge-error"
        @health != nil -> "badge-warning"
        @configured -> "badge-info"
        true -> "badge-ghost"
      end
    ]}>
      <%= cond do %>
        <% @health == :healthy -> %>
          <.icon name="hero-check-circle" class="size-3" /> Healthy
        <% @health == :invalid_credentials -> %>
          <.icon name="hero-x-circle" class="size-3" /> Invalid Key
        <% @health == :missing_credentials -> %>
          <.icon name="hero-exclamation-circle" class="size-3" /> Missing
        <% @health != nil -> %>
          <.icon name="hero-exclamation-triangle" class="size-3" /> Error
        <% @configured -> %>
          <.icon name="hero-check" class="size-3" /> Configured
        <% true -> %>
          <.icon name="hero-minus-circle" class="size-3" /> Not Set
      <% end %>
    </div>
    """
  end

  defp service_icon(:weather), do: "hero-sun"
  defp service_icon(:news), do: "hero-newspaper"
  defp service_icon(:geocoding), do: "hero-map-pin"
  defp service_icon(_), do: "hero-cloud"

  defp humanize_credential(:api_key), do: "API Key"
  defp humanize_credential(:client_id), do: "Client ID"
  defp humanize_credential(:client_secret), do: "Client Secret"
  defp humanize_credential(key), do: key |> Atom.to_string() |> String.replace("_", " ") |> String.capitalize()

  defp filter_entities(entities, "") do
    entities
  end

  defp filter_entities(entities, search) do
    search = String.downcase(search)

    Enum.filter(entities, fn entity ->
      String.contains?(String.downcase(entity.key || ""), search) ||
        String.contains?(String.downcase(entity.value || ""), search)
    end)
  end

  defp filter_intents(intents, "") do
    intents
  end

  defp filter_intents(intents, search) do
    search = String.downcase(search)
    Enum.filter(intents, &String.contains?(String.downcase(&1), search))
  end

  defp is_overlay_entity(entity, overlay) do
    Enum.any?(overlay, fn {key, _info} ->
      String.downcase(key) == String.downcase(entity.key || "")
    end)
  end

  defp append_training_log(socket, message) do
    entry = {message, DateTime.utc_now()}
    log = Enum.take([entry | socket.assigns.ml_training_log], 50)
    assign(socket, :ml_training_log, log)
  end

  defp parse_integer(value, default) when is_binary(value) do
    case Integer.parse(value) do
      {n, _} when n > 0 -> n
      _ -> default
    end
  end

  defp parse_integer(_, default) do
    default
  end
end
