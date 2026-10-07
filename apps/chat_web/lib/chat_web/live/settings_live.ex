defmodule ChatWeb.SettingsLive do
  @moduledoc "Settings page for world management and entity administration.\n\nFeatures:\n- World management (create, delete, configure)\n- Gazetteer entity management\n- System configuration\n"

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
        "response_systems" -> :response_systems
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
      |> assign(:ml_selected_model, "tfidf")
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
      |> assign(:ha_discovered_entities, [])
      |> assign(:ha_discovering, false)
      |> assign(:response_domains, [])
      |> assign(:lattice_stats, %{})
      |> assign(:response_generating, false)
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
      :response_systems -> load_response_systems_data(socket)
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
    training_status = TrainingServer.get_status()
    schedules = TrainingServer.list_schedules()

    socket
    |> assign(:ml_model_statuses, %{})
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

  defp load_response_systems_data(socket) do
    domains =
      try do
        Brain.Response.ResponseSystemRouter.list_domains()
      rescue
        _ -> []
      catch
        :exit, _ -> []
      end

    lattice_stats =
      try do
        Brain.Response.PhraseInventory.stats()
      rescue
        _ -> %{status: :unavailable}
      catch
        :exit, _ -> %{status: :unavailable}
      end

    socket
    |> assign(:response_domains, domains)
    |> assign(:lattice_stats, lattice_stats)
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
        _ -> :tfidf
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
    result =
      try do
        Brain.ML.MicroClassifiers.reload()
      rescue
        e -> {:error, Exception.message(e)}
      catch
        :exit, reason -> {:error, inspect(reason)}
      end

    socket =
      case result do
        :ok ->
          socket
          |> load_ml_training_data()
          |> append_training_log("Reloaded all micro-classifiers")
          |> put_flash(:info, "Models reloaded successfully")

        {:error, reason} ->
          socket
          |> load_ml_training_data()
          |> append_training_log("Reload failed: #{inspect(reason)}")
          |> put_flash(:error, "Reload failed: #{inspect(reason)}")
      end

    {:noreply, socket}
  end

  def handle_event("update_ml_schedule_interval", %{"interval" => interval}, socket) do
    {:noreply, assign(socket, :ml_schedule_interval, interval)}
  end

  def handle_event("add_ml_schedule", _params, socket) do
    model_type =
      case socket.assigns.ml_selected_model do
        "tfidf" -> :tfidf
        _ -> :tfidf
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

  def handle_event("discover_ha_entities", _params, socket) do
    socket = assign(socket, :ha_discovering, true)

    Task.start(fn ->
      result =
        case fetch_ha_credentials() do
          {:ok, creds} ->
            Brain.Services.HomeAssistant.Discovery.discover(creds)

          {:error, _} ->
            {:error, :no_credentials}
        end

      send(socket.root_pid, {:ha_discovery_result, result})
    end)

    {:noreply, socket}
  end

  def handle_event("register_ha_entities", _params, socket) do
    case fetch_ha_credentials() do
      {:ok, creds} ->
        case Brain.Services.HomeAssistant.Discovery.discover_and_register(creds) do
          {:ok, %{total: count}} ->
            {:noreply, put_flash(socket, :info, "Registered #{count} entities in Gazetteer")}

          {:error, reason} ->
            {:noreply, put_flash(socket, :error, "Registration failed: #{inspect(reason)}")}
        end

      {:error, _} ->
        {:noreply, put_flash(socket, :error, "Home Assistant credentials not configured")}
    end
  end

  # ============================================================================
  # Response Systems Section Event Handlers
  # ============================================================================

  def handle_event("update_domain_system", %{"domain" => domain, "system" => system}, socket) do
    Brain.Response.ResponseSystemRouter.update_domain_config(domain, %{"system" => system})
    {:noreply, socket |> load_response_systems_data() |> put_flash(:info, "Updated #{domain} response system")}
  rescue
    _ -> {:noreply, put_flash(socket, :error, "Failed to update")}
  end

  def handle_event("update_domain_tone", %{"domain" => domain, "tone_bias" => tone}, socket) do
    Brain.Response.ResponseSystemRouter.update_domain_config(domain, %{"tone_bias" => tone})
    {:noreply, socket |> load_response_systems_data() |> put_flash(:info, "Updated #{domain} tone")}
  rescue
    _ -> {:noreply, put_flash(socket, :error, "Failed to update")}
  end

  def handle_event("update_domain_mirror", %{"domain" => domain, "mirror" => mirror_str}, socket) do
    {mirror, _} = Float.parse(mirror_str)
    mirror = max(0.0, min(1.0, mirror))
    Brain.Response.ResponseSystemRouter.update_domain_config(domain, %{"mirror_coefficient" => mirror})
    {:noreply, socket |> load_response_systems_data() |> put_flash(:info, "Updated #{domain} mirror coefficient")}
  rescue
    _ -> {:noreply, put_flash(socket, :error, "Failed to update")}
  end

  def handle_event("regenerate_lattice", _params, socket) do
    socket = assign(socket, :response_generating, true)

    case Brain.ML.TrainingServer.start_training(:lattice, []) do
      {:ok, _} ->
        {:noreply, put_flash(socket, :info, "Lattice regeneration started")}

      {:error, reason} ->
        {:noreply, socket |> assign(:response_generating, false) |> put_flash(:error, "Failed: #{inspect(reason)}")}
    end
  rescue
    _ -> {:noreply, socket |> assign(:response_generating, false) |> put_flash(:error, "Training server not available")}
  end

  @impl true
  def handle_info({:ha_discovery_result, result}, socket) do
    case result do
      {:ok, entities_by_domain} when is_map(entities_by_domain) ->
        flat_entities = Enum.flat_map(entities_by_domain, fn {_domain, entities} -> entities end)

        {:noreply,
         socket
         |> assign(:ha_discovered_entities, flat_entities)
         |> assign(:ha_discovering, false)}

      {:error, reason} ->
        {:noreply,
         socket
         |> assign(:ha_discovering, false)
         |> put_flash(:error, "Discovery failed: #{inspect(reason)}")}
    end
  end

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
            <h1 class="text-title text-ink">Settings</h1>
            <p class="text-body text-ink-muted">Manage worlds and entities</p>
          </div>
          <.btn phx-click="refresh" variant={:ghost} size={:sm}>
            <.icon name="hero-arrow-path" class="size-4" /> Refresh
          </.btn>
        </div>
      </:page_header>

      <div class="p-space-lg sm:p-space-xl">
        <!-- Section Tabs -->
        <.tabs class="mb-space-xl flex-wrap">
          <.tab
            phx-click="switch_section"
            phx-value-section="worlds"
            active={@section == :worlds}
            class="inline-flex items-center gap-space-xs"
          >
            <.icon name="hero-globe-alt" class="size-4" /> Worlds
          </.tab>
          <.tab
            phx-click="switch_section"
            phx-value-section="entities"
            active={@section == :entities}
            class="inline-flex items-center gap-space-xs"
          >
            <.icon name="hero-tag" class="size-4" /> Entities
          </.tab>
          <.tab
            phx-click="switch_section"
            phx-value-section="training"
            active={@section == :training}
            class="inline-flex items-center gap-space-xs"
          >
            <.icon name="hero-academic-cap" class="size-4" /> Training
          </.tab>
          <.tab
            phx-click="switch_section"
            phx-value-section="ml_training"
            active={@section == :ml_training}
            class="inline-flex items-center gap-space-xs"
          >
            <.icon name="hero-cpu-chip" class="size-4" /> ML Models
          </.tab>
          <.tab
            phx-click="switch_section"
            phx-value-section="templates"
            active={@section == :templates}
            class="inline-flex items-center gap-space-xs"
          >
            <.icon name="hero-chat-bubble-bottom-center-text" class="size-4" /> Templates
          </.tab>
          <.tab
            phx-click="switch_section"
            phx-value-section="services"
            active={@section == :services}
            class="inline-flex items-center gap-space-xs"
          >
            <.icon name="hero-cloud" class="size-4" /> Services
          </.tab>
          <.tab
            phx-click="switch_section"
            phx-value-section="response_systems"
            active={@section == :response_systems}
            class="inline-flex items-center gap-space-xs"
          >
            <.icon name="hero-sparkles" class="size-4" /> Response
          </.tab>
        </.tabs>

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
              ha_discovered_entities={@ha_discovered_entities}
              ha_discovering={@ha_discovering}
            />
          <% :response_systems -> %>
            <.response_systems_section
              domains={@response_domains}
              lattice_stats={@lattice_stats}
              generating={@response_generating}
            />
        <% end %>
      </div>
    </.app_shell>
    """
  end

  defp worlds_section(assigns) do
    ~H"""
    <div class="space-y-space-xl">
      <!-- Create World -->
      <.card class="p-space-lg">
        <h3 class="text-heading text-ink mb-space-lg">Create New World</h3>
        <form phx-change="update_new_world" phx-submit="create_world" class="flex flex-wrap gap-space-lg">
          <input
            type="text"
            name="name"
            value={@new_world_name}
            placeholder="World name"
            class="flex-1 min-w-[200px] h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
          />
          <select
            name="mode"
            class="h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink"
          >
            <option value="persistent" selected={@new_world_mode == "persistent"}>Persistent</option>
            <option value="ephemeral" selected={@new_world_mode == "ephemeral"}>Ephemeral</option>
          </select>
          <.btn type="submit" variant={:primary} disabled={@creating_world}>
            <%= if @creating_world do %>
              <.icon name="hero-arrow-path" class="size-4 animate-spin text-progress-fill" />
            <% else %>
              <.icon name="hero-plus" class="size-4" />
            <% end %>
            Create World
          </.btn>
        </form>
      </.card>

    <!-- Active Worlds -->
      <.card>
        <div class="p-space-lg border-b border-border">
          <h3 class="text-heading text-ink">Active Worlds</h3>
        </div>
        <%= if length(@worlds) == 0 do %>
          <div class="p-space-2xl text-center text-ink-muted">
            <.icon name="hero-globe-alt" class="size-12 mx-auto mb-space-lg text-ink-muted" />
            <p>No active worlds</p>
          </div>
        <% else %>
          <div class="divide-y divide-border">
            <%= for world <- @worlds do %>
              <div class="p-space-lg flex items-center justify-between hover:bg-surface-sunk">
                <div>
                  <div class="text-subheading text-ink">{world.name}</div>
                  <div class="text-ref text-ink-muted">{world.id}</div>
                </div>
                <div class="flex items-center gap-space-sm">
                  <.badge variant={world_mode_variant(world.mode)}>
                    {world.mode}
                  </.badge>
                  <%= if world.mode == :persistent do %>
                    <.icon_btn
                      phx-click="save_world"
                      phx-value-id={world.id}
                      variant={:primary}
                      size={:sm}
                      title="Save to disk"
                    >
                      <.icon name="hero-cloud-arrow-up" class="size-4" />
                    </.icon_btn>
                  <% end %>
                  <%= if world.id != "default" do %>
                    <.icon_btn
                      phx-click="delete_world"
                      phx-value-id={world.id}
                      variant={:ghost}
                      size={:sm}
                      title="Delete world"
                      data-confirm="Are you sure you want to delete this world?"
                    >
                      <.icon name="hero-trash" class="size-4" />
                    </.icon_btn>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </.card>

    <!-- Persisted Worlds (not loaded) -->
      <% not_loaded =
        Enum.filter(@persisted_worlds, fn pw -> not Enum.any?(@worlds, &(&1.id == pw.id)) end) %>
      <%= if length(not_loaded) > 0 do %>
        <.card>
          <div class="p-space-lg border-b border-border">
            <h3 class="text-heading text-ink">Persisted Worlds (Not Loaded)</h3>
          </div>
          <div class="divide-y divide-border">
            <%= for world <- not_loaded do %>
              <div class="p-space-lg flex items-center justify-between hover:bg-surface-sunk">
                <div>
                  <div class="text-subheading text-ink-muted">{world.name}</div>
                  <div class="text-ref text-ink-muted">{world.id}</div>
                </div>
                <.btn
                  phx-click="load_world"
                  phx-value-id={world.id}
                  variant={:primary}
                  size={:xs}
                >
                  <.icon name="hero-arrow-down-tray" class="size-4" /> Load
                </.btn>
              </div>
            <% end %>
          </div>
        </.card>
      <% end %>
    </div>
    """
  end

  defp entities_section(assigns) do
    ~H"""
    <div class="space-y-space-xl">
      <!-- Add Entity -->
      <.card class="p-space-lg">
        <h3 class="text-heading text-ink mb-space-lg">
          Add Entity to World: <span class="text-value-strong text-ink">{@current_world_id}</span>
        </h3>
        <form phx-change="update_new_entity" phx-submit="add_entity" class="flex flex-wrap gap-space-lg">
          <input
            type="text"
            name="key"
            value={@new_entity_key}
            placeholder="Lookup key (e.g., 'new york')"
            class="flex-1 min-w-[200px] h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
          />
          <input
            type="text"
            name="value"
            value={@new_entity_value}
            placeholder="Canonical value (optional)"
            class="flex-1 min-w-[200px] h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
          />
          <select
            name="type"
            class="h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink"
          >
            <%= for type <- @entity_types do %>
              <option value={type} selected={@new_entity_type == type}>{type}</option>
            <% end %>
          </select>
          <.btn type="submit" variant={:primary}>
            <.icon name="hero-plus" class="size-4" /> Add
          </.btn>
        </form>
      </.card>

      <div class="grid grid-cols-1 lg:grid-cols-4 gap-space-xl">
        <!-- Entity Type Sidebar -->
        <.card class="p-space-lg">
          <h3 class="text-heading text-ink mb-space-lg">Entity Types</h3>
          <ul class="space-y-space-xs">
            <%= for type <- @entity_types do %>
              <li>
                <button
                  phx-click="select_entity_type"
                  phx-value-type={type}
                  class={[
                    "w-full text-left px-space-md py-space-sm rounded-md text-body transition-colors",
                    if(type == @selected_entity_type,
                      do: "bg-accent-wash text-accent font-semibold",
                      else: "text-ink hover:bg-surface-sunk"
                    )
                  ]}
                >
                  {type}
                </button>
              </li>
            <% end %>
          </ul>
        </.card>

    <!-- Entities List -->
        <.card class="lg:col-span-3">
          <div class="p-space-lg border-b border-border flex items-center gap-space-lg">
            <h3 class="text-heading text-ink">{@selected_entity_type || "Select a type"}</h3>
            <input
              type="text"
              placeholder="Search entities..."
              value={@entity_search}
              phx-keyup="search_entities"
              name="query"
              phx-debounce="150"
              class="flex-1 max-w-xs h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink placeholder:text-ink-muted"
            />
          </div>
          <%= if length(@type_entities) == 0 do %>
            <div class="p-space-2xl text-center text-ink-muted">
              <.icon name="hero-tag" class="size-12 mx-auto mb-space-lg text-ink-muted" />
              <p>No entities of this type</p>
            </div>
          <% else %>
            <% filtered = filter_entities(@type_entities, @entity_search) %>
            <div class="max-h-96 overflow-y-auto">
              <table class="w-full text-left text-body-dense text-ink">
                <thead class="bg-surface-sunk sticky top-0">
                  <tr>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Key</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Value</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Source</th>
                    <th class="h-row-compact px-space-sm"></th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  <%= for entity <- Enum.take(filtered, 100) do %>
                    <% is_from_overlay = is_overlay_entity(entity, @world_overlay) %>
                    <tr class="hover:bg-surface-sunk">
                      <td class="h-row-compact px-space-sm font-semibold">{entity.key}</td>
                      <td class="h-row-compact px-space-sm">{entity.value || entity.key}</td>
                      <td class="h-row-compact px-space-sm">
                        <.badge variant={if is_from_overlay, do: :primary, else: :default} size={:xs}>
                          {if is_from_overlay, do: "world", else: "global"}
                        </.badge>
                      </td>
                      <td class="h-row-compact px-space-sm">
                        <%= if is_from_overlay do %>
                          <.icon_btn
                            phx-click="remove_entity"
                            phx-value-key={entity.key}
                            variant={:ghost}
                            size={:sm}
                            title="Remove from world"
                          >
                            <.icon name="hero-x-mark" class="size-4" />
                          </.icon_btn>
                        <% end %>
                      </td>
                    </tr>
                  <% end %>
                </tbody>
              </table>
              <%= if length(filtered) > 100 do %>
                <div class="p-space-lg text-center text-body text-ink-muted">
                  Showing 100 of {length(filtered)} entities
                </div>
              <% end %>
            </div>
          <% end %>
        </.card>
      </div>
    </div>
    """
  end

  defp training_section(assigns) do
    ~H"""
    <div class="space-y-space-xl">
      <!-- Stats Overview -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-space-lg">
        <.card class="p-space-lg">
          <div class="flex items-center gap-space-md">
            <div class="w-10 h-10 rounded-md bg-surface-sunk flex items-center justify-center">
              <.icon name="hero-academic-cap" class="size-5 text-ink-muted" />
            </div>
            <div>
              <div class="text-title text-ink tabular-nums">{@lc_stats[:total_sessions] || 0}</div>
              <div class="text-body text-ink-muted">Total Sessions</div>
            </div>
          </div>
        </.card>
        <.card class="p-space-lg">
          <div class="flex items-center gap-space-md">
            <div class="w-10 h-10 rounded-md bg-surface-sunk flex items-center justify-center">
              <.icon name="hero-cpu-chip" class="size-5 text-ink-muted" />
            </div>
            <div>
              <div class="text-title text-ink tabular-nums">{@lc_stats[:active_agents] || 0}</div>
              <div class="text-body text-ink-muted">Active Agents</div>
            </div>
          </div>
        </.card>
        <.card class="p-space-lg">
          <div class="flex items-center gap-space-md">
            <div class="w-10 h-10 rounded-md bg-surface-sunk flex items-center justify-center">
              <.icon name="hero-document-text" class="size-5 text-ink-muted" />
            </div>
            <div>
              <div class="text-title text-ink tabular-nums">{map_size(@available_tasks)}</div>
              <div class="text-body text-ink-muted">Task Categories</div>
            </div>
          </div>
        </.card>
      </div>

      <!-- Start Training -->
      <.card class="p-space-lg">
        <h3 class="text-heading text-ink mb-space-lg">Start Task-Based Training</h3>
        <p class="text-body text-ink-muted mb-space-lg">
          Train child agents using curated NLP benchmark tasks. Select a capability to focus the training.
        </p>
        <div class="flex flex-wrap gap-space-lg items-end">
          <div>
            <label class="block mb-space-xs text-label text-ink-muted">Capability</label>
            <select
              phx-change="select_capability"
              name="capability"
              class="h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink"
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
          <div class="flex items-center gap-space-sm">
            <.btn
              phx-click="start_task_training"
              variant={:primary}
              disabled={@starting_training}
              class="outline-mark outline-reach-shared focus-visible:outline-focus"
            >
              <%= if @starting_training do %>
                <.icon name="hero-arrow-path" class="size-4 animate-spin text-progress-fill" />
              <% else %>
                <.icon name="hero-play" class="size-4" />
              <% end %>
              Start Training
            </.btn>
            <.shared_reach_tag />
          </div>
        </div>
      </.card>

      <!-- Active Sessions -->
      <.card>
        <div class="p-space-lg border-b border-border flex items-center justify-between">
          <h3 class="text-heading text-ink">Training Sessions</h3>
          <.link navigate={~p"/sessions"} class="text-body text-accent hover:underline">
            View All Sessions
          </.link>
        </div>
        <%= if length(@training_sessions) == 0 do %>
          <div class="p-space-2xl text-center text-ink-muted">
            <.icon name="hero-academic-cap" class="size-12 mx-auto mb-space-lg text-ink-muted" />
            <p>No active training sessions</p>
            <p class="text-body mt-space-sm">Start a training session above to begin</p>
          </div>
        <% else %>
          <div class="divide-y divide-border">
            <%= for session <- @training_sessions do %>
              <% is_expanded = @expanded_session_id == session.id %>
              <% failed_goals = Enum.count(session.goals, &(&1.status == :failed)) %>
              <% completed_goals = Enum.count(session.goals, &(&1.status == :completed)) %>
              <% total_goals = length(session.goals) %>
              <div class={["transition-colors", if(is_expanded, do: "bg-surface-sunk", else: "hover:bg-surface-sunk")]}>
                <!-- Session Header (clickable) -->
                <div
                  class="p-space-lg cursor-pointer"
                  phx-click="toggle_session_detail"
                  phx-value-id={session.id}
                >
                  <div class="flex items-center justify-between mb-space-sm">
                    <div class="flex items-center gap-space-sm">
                      <.icon
                        name={if is_expanded, do: "hero-chevron-down", else: "hero-chevron-right"}
                        class="size-4 text-ink-muted"
                      />
                      <div>
                        <div class="text-subheading text-ink">{session.topic || "Untitled Session"}</div>
                        <div class="text-ref text-ink-muted">{session.id}</div>
                      </div>
                    </div>
                    <div class="flex items-center gap-space-sm">
                      <.badge variant={session_status_variant(session.status)}>
                        {session.status}
                      </.badge>
                      <%= if session.status == :active do %>
                        <.icon_btn
                          phx-click="cancel_session"
                          phx-value-id={session.id}
                          variant={:ghost}
                          size={:sm}
                          title="Cancel session"
                          class="outline-mark outline-reach-shared focus-visible:outline-focus"
                        >
                          <.icon name="hero-stop" class="size-4" />
                        </.icon_btn>
                        <.shared_reach_tag />
                      <% end %>
                    </div>
                  </div>

                  <!-- Summary stats row -->
                  <div class="flex flex-wrap gap-space-md ml-space-xl text-caption">
                    <span class="flex items-center gap-space-xs text-ink-muted">
                      <.icon name="hero-flag" class="size-3" />
                      {completed_goals}/{total_goals} goals
                    </span>
                    <%= if failed_goals > 0 do %>
                      <span class="flex items-center gap-space-xs text-red">
                        <.icon name="hero-exclamation-triangle" class="size-3" />
                        {failed_goals} failed
                      </span>
                    <% end %>
                    <%= if session.findings_count > 0 do %>
                      <span class="flex items-center gap-space-xs text-ink-muted">
                        <.icon name="hero-document-magnifying-glass" class="size-3" />
                        {session.findings_count} findings
                      </span>
                    <% end %>
                    <%= if session.approved_count > 0 do %>
                      <span class="flex items-center gap-space-xs text-ink">
                        <.icon name="hero-check-circle" class="size-3" />
                        {session.approved_count} approved
                      </span>
                    <% end %>
                    <%= if session.rejected_count > 0 do %>
                      <span class="flex items-center gap-space-xs text-red">
                        <.icon name="hero-x-circle" class="size-3" />
                        {session.rejected_count} rejected
                      </span>
                    <% end %>
                    <%= if session.hypotheses_tested > 0 do %>
                      <span class="flex items-center gap-space-xs text-ink-muted">
                        <.icon name="hero-beaker" class="size-3" />
                        {session.hypotheses_tested} hypotheses
                      </span>
                    <% end %>
                    <%= if session.started_at do %>
                      <span class="text-ink-muted">
                        Started {Calendar.strftime(session.started_at, "%H:%M:%S")}
                      </span>
                    <% end %>
                    <%= if session.completed_at do %>
                      <span class="text-ink-muted">
                        Completed {Calendar.strftime(session.completed_at, "%H:%M:%S")}
                      </span>
                    <% end %>
                  </div>
                </div>

                <!-- Expanded Detail Panel -->
                <%= if is_expanded do %>
                  <div class="px-space-lg pb-space-lg ml-space-xl space-y-space-lg">
                    <!-- Goals Detail -->
                    <.card>
                      <div class="px-space-md py-space-sm border-b border-border">
                        <h4 class="text-subheading text-ink">
                          Research Goals ({total_goals})
                        </h4>
                      </div>
                      <%= if total_goals == 0 do %>
                        <div class="p-space-md text-body text-ink-muted">No goals defined</div>
                      <% else %>
                        <div class="divide-y divide-border">
                          <%= for goal <- session.goals do %>
                            <div class="p-space-md">
                              <div class="flex items-start justify-between gap-space-sm">
                                <div class="flex-1 min-w-0">
                                  <div class="flex items-center gap-space-sm mb-space-xs">
                                    <.badge variant={goal_status_variant(goal.status)} size={:xs}>
                                      {goal.status}
                                    </.badge>
                                    <span class="text-subheading text-ink truncate">{goal.topic}</span>
                                    <%= if goal.priority != :normal do %>
                                      <.badge
                                        variant={priority_variant(goal.priority)}
                                        size={:xs}
                                        class="border border-border-strong"
                                      >
                                        {goal.priority}
                                      </.badge>
                                    <% end %>
                                  </div>
                                  <%= if length(goal.questions) > 0 do %>
                                    <div class="ml-space-sm mt-space-xs space-y-space-2xs">
                                      <%= for question <- goal.questions do %>
                                        <div class="text-caption text-ink-muted flex items-start gap-space-xs">
                                          <span class="text-ink-muted shrink-0">Q:</span>
                                          <span>{display_value(question)}</span>
                                        </div>
                                      <% end %>
                                    </div>
                                  <% end %>
                                  <%= if map_size(goal.constraints) > 0 do %>
                                    <div class="flex flex-wrap gap-space-xs mt-space-xs ml-space-sm">
                                      <%= for {key, val} <- goal.constraints do %>
                                        <.badge size={:xs}>{key}: {display_value(val)}</.badge>
                                      <% end %>
                                    </div>
                                  <% end %>
                                </div>
                                <div class="shrink-0">
                                  <%= case goal.status do %>
                                    <% :completed -> %>
                                      <.icon name="hero-check-circle" class="size-5 text-ink" />
                                    <% :failed -> %>
                                      <.icon name="hero-x-circle" class="size-5 text-red" />
                                    <% :in_progress -> %>
                                      <.icon name="hero-arrow-path" class="size-4 animate-spin text-progress-fill" />
                                    <% :pending -> %>
                                      <.icon name="hero-clock" class="size-5 text-ink-muted" />
                                  <% end %>
                                </div>
                              </div>
                            </div>
                          <% end %>
                        </div>
                      <% end %>
                    </.card>

                    <!-- Investigations Detail -->
                    <%= if length(session.investigations) > 0 do %>
                      <.card>
                        <div class="px-space-md py-space-sm border-b border-border">
                          <h4 class="text-subheading text-ink">
                            <.icon name="hero-beaker" class="size-4 inline-block mr-space-xs" />
                            Scientific Investigations ({length(session.investigations)})
                          </h4>
                        </div>
                        <div class="divide-y divide-border">
                          <%= for investigation <- session.investigations do %>
                            <div class="p-space-md">
                              <div class="flex items-center justify-between mb-space-sm">
                                <span class="text-subheading text-ink">{investigation.topic}</span>
                                <div class="flex items-center gap-space-sm">
                                  <.badge variant={investigation_status_variant(investigation.status)} size={:xs}>
                                    {investigation.status}
                                  </.badge>
                                  <%= if investigation.conclusion do %>
                                    <.badge variant={conclusion_variant(investigation.conclusion)} size={:xs}>
                                      {investigation.conclusion}
                                    </.badge>
                                  <% end %>
                                </div>
                              </div>
                              <!-- Hypotheses within investigation -->
                              <%= if length(investigation.hypotheses) > 0 do %>
                                <div class="ml-space-sm space-y-space-xs">
                                  <%= for hypothesis <- investigation.hypotheses do %>
                                    <div class="flex items-start gap-space-sm text-caption">
                                      <.badge
                                        variant={hypothesis_status_variant(hypothesis.status)}
                                        size={:xs}
                                        class="shrink-0 mt-space-2xs"
                                      >
                                        {hypothesis.status}
                                      </.badge>
                                      <div class="min-w-0">
                                        <span class="text-ink-muted">{hypothesis.claim}</span>
                                        <%= if hypothesis.confidence > 0 do %>
                                          <span class="text-ink-muted ml-space-xs">
                                            ({Float.round(hypothesis.confidence * 100, 1)}% confidence)
                                          </span>
                                        <% end %>
                                      </div>
                                    </div>
                                  <% end %>
                                </div>
                              <% end %>
                              <!-- Evidence counts -->
                              <div class="flex flex-wrap gap-space-sm mt-space-sm text-caption text-ink-muted">
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
                      </.card>
                    <% end %>

                    <!-- Session Metrics Summary -->
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-space-sm">
                      <.card class="p-space-md text-center">
                        <div class="text-heading text-ink tabular-nums">{session.findings_count}</div>
                        <div class="text-caption text-ink-muted">Findings</div>
                      </.card>
                      <.card class="p-space-md text-center">
                        <div class="text-heading text-ink tabular-nums">{session.approved_count}</div>
                        <div class="text-caption text-ink-muted">Approved</div>
                      </.card>
                      <.card class="p-space-md text-center">
                        <div class="text-heading text-red tabular-nums">{session.rejected_count}</div>
                        <div class="text-caption text-ink-muted">Rejected</div>
                      </.card>
                      <.card class="p-space-md text-center">
                        <div class="text-heading text-ink tabular-nums">
                          <%= if session.hypotheses_tested > 0 do %>
                            {Float.round(session.hypotheses_supported / session.hypotheses_tested * 100, 1)}%
                          <% else %>
                            N/A
                          <% end %>
                        </div>
                        <div class="text-caption text-ink-muted">Support Rate</div>
                      </.card>
                    </div>
                  </div>
                <% end %>
              </div>
            <% end %>
          </div>
        <% end %>
      </.card>

      <!-- Available Task Categories -->
      <.card>
        <div class="p-space-lg border-b border-border">
          <h3 class="text-heading text-ink">Available Task Categories</h3>
          <p class="text-body text-ink-muted">Domain-specific NLP tasks from benchmarks</p>
        </div>
        <%= if @tasks_loading do %>
          <div class="p-space-2xl text-center text-ink-muted">
            <.icon name="hero-arrow-path" class="size-8 mx-auto animate-spin text-progress-fill" />
            <p class="mt-space-lg">Scanning task files...</p>
            <p class="text-body mt-space-sm">This may take a moment on first load</p>
          </div>
        <% else %>
          <%= if map_size(@available_tasks) == 0 do %>
            <div class="p-space-2xl text-center text-ink-muted">
              <.icon name="hero-document-text" class="size-12 mx-auto mb-space-lg text-ink-muted" />
              <p>No tasks available</p>
              <p class="text-body mt-space-sm">Check that domain task files are in data/domain_specific_tasks/</p>
            </div>
          <% else %>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-space-lg p-space-lg">
              <%= for {category, tasks} <- @available_tasks do %>
                <div class="bg-surface-sunk rounded-md p-space-md">
                  <div class="flex items-center justify-between mb-space-sm">
                    <span class="text-subheading text-ink">{category}</span>
                    <.badge>{length(tasks)} tasks</.badge>
                  </div>
                  <div class="text-caption text-ink-muted">
                    <%= for task <- Enum.take(tasks, 3) do %>
                      <div class="truncate">{task.task_id}</div>
                    <% end %>
                    <%= if length(tasks) > 3 do %>
                      <div class="text-ink-muted">+{length(tasks) - 3} more</div>
                    <% end %>
                  </div>
                </div>
              <% end %>
            </div>
          <% end %>
        <% end %>
      </.card>
    </div>
    """
  end

  defp ml_training_section(assigns) do
    ~H"""
    <div class="space-y-space-xl">
      <!-- Training Form -->
      <.card class="p-space-lg">
        <h3 class="text-heading text-ink mb-space-lg">Train ML Model</h3>
        <p class="text-body text-ink-muted mb-space-lg">
          Start an async training job for a specific model. Training runs in the background.
        </p>
        <form phx-change="update_ml_training_form" phx-submit="start_ml_training">
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-space-lg mb-space-lg">
            <div>
              <label class="block mb-space-xs text-label text-ink-muted">Model Type</label>
              <select
                name="model_type"
                class="w-full h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink"
              >
                <option value="tfidf" selected={@selected_model == "tfidf"}>
                  TF-IDF Classifier
                </option>
              </select>
            </div>
            <div>
              <label class="block mb-space-xs text-label text-ink-muted">Encoder Epochs</label>
              <input
                type="number"
                name="epochs"
                value={@epochs}
                min="1"
                max="200"
                class="w-full h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
              />
            </div>
            <div>
              <label class="block mb-space-xs text-label text-ink-muted">Head Epochs</label>
              <input
                type="number"
                name="head_epochs"
                value={@head_epochs}
                min="1"
                max="200"
                class="w-full h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
                title="Epochs for task heads (sentiment, speech act) trained with frozen encoder"
              />
            </div>
            <div>
              <label class="block mb-space-xs text-label text-ink-muted">Batch Size</label>
              <input
                type="number"
                name="batch_size"
                value={@batch_size}
                min="1"
                max="256"
                class="w-full h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
              />
            </div>
            <div>
              <label class="block mb-space-xs text-label text-ink-muted">Experiment Name</label>
              <input
                type="text"
                name="experiment_name"
                value={@experiment_name}
                placeholder="Optional"
                class="w-full h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
              />
            </div>
          </div>
          <div class="flex items-center gap-space-lg">
            <%= case @training_status do %>
              <% :idle -> %>
                <.btn type="submit" variant={:primary}>
                  <.icon name="hero-play" class="size-4" /> Train
                </.btn>
              <% {:training, model_type, started_at} -> %>
                <div class="flex items-center gap-space-md">
                  <.icon name="hero-arrow-path" class="size-4 animate-spin text-progress-fill" />
                  <div>
                    <div class="text-subheading text-ink">Training {model_type}...</div>
                    <div class="text-caption text-ink-muted">
                      Started {Calendar.strftime(started_at, "%H:%M:%S")}
                    </div>
                  </div>
                  <.btn
                    type="button"
                    phx-click="cancel_ml_training"
                    variant={:danger}
                    size={:sm}
                  >
                    <.icon name="hero-stop" class="size-4" /> Cancel
                  </.btn>
                </div>
            <% end %>
          </div>
        </form>
      </.card>

      <!-- Training Progress Log -->
      <%= if length(@training_log) > 0 do %>
        <.card>
          <div class="p-space-lg border-b border-border">
            <h3 class="text-heading text-ink">Training Log</h3>
          </div>
          <div class="max-h-48 overflow-y-auto p-space-lg text-term space-y-space-xs">
            <%= for {message, timestamp} <- Enum.reverse(@training_log) do %>
              <div class="text-ink">
                <span class="text-ink-muted">[{Calendar.strftime(timestamp, "%H:%M:%S")}]</span>
                {message}
              </div>
            <% end %>
          </div>
        </.card>
      <% end %>

      <!-- Scheduling -->
      <.card class="p-space-lg">
        <h3 class="text-heading text-ink mb-space-lg">Training Schedules</h3>
        <p class="text-body text-ink-muted mb-space-lg">
          Schedule recurring training runs. Uses the model type and parameters from the form above.
        </p>
        <div class="flex flex-wrap gap-space-lg items-end mb-space-lg">
          <div>
            <label class="block mb-space-xs text-label text-ink-muted">Interval (hours)</label>
            <select
              name="interval"
              phx-change="update_ml_schedule_interval"
              class="h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink"
            >
              <option value="1" selected={@schedule_interval == "1"}>Every 1 hour</option>
              <option value="6" selected={@schedule_interval == "6"}>Every 6 hours</option>
              <option value="12" selected={@schedule_interval == "12"}>Every 12 hours</option>
              <option value="24" selected={@schedule_interval == "24"}>Every 24 hours</option>
              <option value="48" selected={@schedule_interval == "48"}>Every 48 hours</option>
              <option value="168" selected={@schedule_interval == "168"}>Every 7 days</option>
            </select>
          </div>
          <.btn phx-click="add_ml_schedule" variant={:primary}>
            <.icon name="hero-clock" class="size-4" /> Schedule
          </.btn>
        </div>

        <!-- Active Schedules -->
        <%= if length(@schedules) == 0 do %>
          <div class="text-body text-ink-muted p-space-lg text-center">
            No active schedules
          </div>
        <% else %>
          <div class="divide-y divide-border border border-border rounded-md">
            <%= for schedule <- @schedules do %>
              <div class="p-space-md flex items-center justify-between hover:bg-surface-sunk">
                <div>
                  <.badge variant={:primary} class="mr-space-sm">{schedule.model_type}</.badge>
                  <span class="text-body text-ink">Every {schedule.interval_hours} hour(s)</span>
                  <span class="text-ref text-ink-muted ml-space-sm">{schedule.id}</span>
                </div>
                <.icon_btn
                  phx-click="cancel_ml_schedule"
                  phx-value-id={schedule.id}
                  variant={:ghost}
                  size={:sm}
                  title="Cancel schedule"
                >
                  <.icon name="hero-x-mark" class="size-4" />
                </.icon_btn>
              </div>
            <% end %>
          </div>
        <% end %>
      </.card>
    </div>
    """
  end

  @session_status_variants %{active: :warning, completed: :success, cancelled: :error}

  @goal_status_variants %{
    completed: :success,
    failed: :error,
    in_progress: :warning,
    pending: :default
  }

  @priority_variants %{high: :error, low: :default}

  @investigation_status_variants %{
    concluded: :success,
    evaluating: :warning,
    gathering_evidence: :info,
    planning: :default
  }

  @conclusion_variants %{
    hypotheses_supported: :success,
    hypotheses_falsified: :error,
    inconclusive: :warning,
    mixed: :warning
  }

  @hypothesis_status_variants %{
    supported: :success,
    falsified: :error,
    inconclusive: :warning,
    testing: :info,
    untested: :default
  }

  @world_mode_variants %{persistent: :info, ephemeral: :default}

  @template_source_variants %{admin: :primary, file: :default}

  defp session_status_variant(status),
    do: fetch_variant!(@session_status_variants, status, "session status")

  defp goal_status_variant(status),
    do: fetch_variant!(@goal_status_variants, status, "goal status")

  defp priority_variant(priority),
    do: fetch_variant!(@priority_variants, priority, "goal priority")

  defp investigation_status_variant(status),
    do: fetch_variant!(@investigation_status_variants, status, "investigation status")

  defp conclusion_variant(conclusion),
    do: fetch_variant!(@conclusion_variants, conclusion, "investigation conclusion")

  defp hypothesis_status_variant(status),
    do: fetch_variant!(@hypothesis_status_variants, status, "hypothesis status")

  defp world_mode_variant(mode),
    do: fetch_variant!(@world_mode_variants, mode, "world mode")

  defp template_source_variant(source),
    do: fetch_variant!(@template_source_variants, source, "template source")

  defp fetch_variant!(variants, value, what) do
    case Map.fetch(variants, value) do
      {:ok, variant} ->
        variant

      :error ->
        raise ArgumentError,
              "ChatWeb.SettingsLive: no badge treatment for #{what} #{inspect(value)}. " <>
                "The #{what} values with a treatment are #{inspect(Map.keys(variants))}."
    end
  end

  defp shared_reach_tag(assigns) do
    ~H"""
    <span class="inline-flex items-center gap-space-xs rounded-sm border border-reach-shared px-space-xs text-caption font-semibold text-reach-shared whitespace-nowrap">
      <.icon name="hero-share-micro" class="size-3" /> writes shared
    </span>
    """
  end

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
    <div class="space-y-space-xl">
      <!-- Stats & Actions Bar -->
      <.card class="p-space-lg">
        <div class="flex flex-wrap items-center justify-between gap-space-lg">
          <div class="flex flex-wrap gap-space-lg">
            <.stat_kpi label="Intents" value={to_string(Map.get(@stats, :intent_count, 0))} />
            <.stat_kpi label="Templates" value={to_string(Map.get(@stats, :template_count, 0))} />
            <.stat_kpi
              label="Admin Added"
              value={to_string(Map.get(@stats, :admin_template_count, 0))}
            />
          </div>
          <div class="flex gap-space-sm">
            <%= if @has_unsaved do %>
              <.btn phx-click="sync_templates" variant={:primary} size={:sm}>
                <.icon name="hero-arrow-down-tray" class="size-4" />
                Save Changes
              </.btn>
            <% end %>
          </div>
        </div>
      </.card>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-space-xl">
        <!-- Intent List -->
        <.card class="p-space-lg">
          <h3 class="text-heading text-ink mb-space-lg">Intents</h3>
          <div class="mb-space-md">
            <input
              type="text"
              name="query"
              value={@search}
              placeholder="Search intents..."
              phx-debounce="300"
              phx-change="search_templates"
              class="w-full h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink placeholder:text-ink-muted"
            />
          </div>
          <div class="overflow-y-auto max-h-[400px] space-y-space-xs">
            <%= for intent <- filter_intents(@intents, @search) do %>
              <button
                phx-click="select_template_intent"
                phx-value-intent={intent}
                class={[
                  "w-full text-left px-space-md py-space-sm rounded-md text-body truncate transition-colors",
                  if(intent == @selected_intent,
                    do: "bg-accent-wash text-accent font-semibold",
                    else: "text-ink hover:bg-surface-sunk")
                ]}
                title={intent}
              >
                {intent}
              </button>
            <% end %>
          </div>
        </.card>

        <!-- Templates for Selected Intent -->
        <.card class="lg:col-span-2 p-space-lg">
          <h3 class="text-heading text-ink mb-space-lg">
            Templates
            <%= if @selected_intent do %>
              <span class="text-body text-ink-muted">for {@selected_intent}</span>
            <% end %>
          </h3>

          <%= if @selected_intent do %>
            <!-- Add Template Form -->
            <form phx-submit="add_template" class="mb-space-lg flex gap-space-sm">
              <input type="hidden" name="intent" value={@selected_intent} />
              <input
                type="text"
                name="text"
                value={@new_template_text}
                placeholder="Enter new template text..."
                phx-change="update_new_template"
                class="flex-1 h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
              />
              <.btn type="submit" variant={:primary}>
                <.icon name="hero-plus" class="size-4" /> Add
              </.btn>
            </form>

            <!-- Template List -->
            <div class="space-y-space-sm max-h-[400px] overflow-y-auto">
              <%= if length(@templates) == 0 do %>
                <div class="text-body text-ink-muted p-space-lg text-center">
                  No templates for this intent
                </div>
              <% else %>
                <%= for template <- @templates do %>
                  <div class="flex items-start gap-space-md p-space-md bg-surface-sunk rounded-md group">
                    <div class="flex-1 min-w-0">
                      <p class="text-body text-ink">{template.text}</p>
                      <div class="flex gap-space-sm mt-space-xs">
                        <.badge variant={template_source_variant(template.source)} size={:xs}>
                          {template.source}
                        </.badge>
                        <%= if template.condition do %>
                          <.badge variant={:info} size={:xs} title={template.condition}>
                            conditional
                          </.badge>
                        <% end %>
                      </div>
                    </div>
                    <.icon_btn
                      phx-click="remove_template"
                      phx-value-text={template.text}
                      variant={:ghost}
                      size={:sm}
                      class="opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Remove template"
                    >
                      <.icon name="hero-trash" class="size-4" />
                    </.icon_btn>
                  </div>
                <% end %>
              <% end %>
            </div>
          <% else %>
            <div class="text-body text-ink-muted p-space-2xl text-center">
              Select an intent to view its templates
            </div>
          <% end %>
        </.card>
      </div>
    </div>
    """
  end

  defp services_section(assigns) do
    ~H"""
    <div class="space-y-space-xl">
      <!-- Services Overview -->
      <.card class="p-space-lg">
        <div class="flex items-center justify-between mb-space-lg">
          <div>
            <h3 class="text-heading text-ink">External Services</h3>
            <p class="text-body text-ink-muted">
              Configure API credentials for live data enrichment
            </p>
          </div>
          <.badge class="border border-border-strong">
            <.icon name="hero-shield-check" class="size-3" />
            Credentials are encrypted
          </.badge>
        </div>

        <.alert variant={:info} class="mb-space-lg">
          API keys are stored encrypted and never exposed in responses or logs.
          Each world can have its own service credentials.
        </.alert>
      </.card>

      <!-- Service Cards -->
      <%= if length(@services) == 0 do %>
        <.card class="p-space-2xl text-center">
          <.icon name="hero-cloud" class="size-12 mx-auto text-ink-muted mb-space-lg" />
          <h3 class="text-heading text-ink mb-space-sm">No Services Available</h3>
          <p class="text-body text-ink-muted">
            No external services are configured in this installation.
          </p>
        </.card>
      <% else %>
        <div class="grid gap-space-lg md:grid-cols-2">
          <%= for service <- @services do %>
            <% {state_variant, state_status, state_label} =
              service_state(service.configured, Map.get(@health_status, service.name)) %>
            <.card class="p-space-lg">
              <!-- Service Header -->
              <div class="flex items-start justify-between mb-space-lg">
                <div>
                  <h4 class="text-subheading text-ink flex items-center gap-space-sm">
                    <.icon name={service_icon(service.name)} class="size-5" />
                    {service.display_name}
                  </h4>
                  <p class="text-body text-ink-muted mt-space-xs">{service.description}</p>
                </div>
                <.badge variant={state_variant}>
                  <.status_dot status={state_status} size={:sm} /> {state_label}
                </.badge>
              </div>

              <!-- Supported Intents -->
              <div class="mb-space-lg">
                <div class="text-caption text-ink-muted mb-space-xs">Supports:</div>
                <div class="flex flex-wrap gap-space-xs">
                  <%= for intent <- service.supported_intents do %>
                    <.badge>{intent}</.badge>
                  <% end %>
                </div>
              </div>

              <!-- Credential Forms -->
              <div class="space-y-space-md">
                <%= for cred_key <- service.required_credentials do %>
                  <% has_cred = get_in(@credentials, [service.name, cred_key]) %>
                  <div>
                    <label class="flex items-center justify-between py-space-xs mb-space-xs text-label text-ink-muted">
                      <span>{humanize_credential(cred_key)}</span>
                      <%= if has_cred do %>
                        <.badge variant={:success} size={:xs}>
                          <.icon name="hero-check" class="size-3" /> Set
                        </.badge>
                      <% end %>
                    </label>
                    <div class="flex items-center gap-space-sm">
                      <form
                        phx-submit="save_credential"
                        class="flex-1 flex items-center gap-space-sm"
                      >
                        <input type="hidden" name="service" value={service.name} />
                        <input type="hidden" name="key" value={cred_key} />
                        <input
                          type="password"
                          name="value"
                          placeholder={if has_cred, do: "••••••••", else: "Enter #{humanize_credential(cred_key)}..."}
                          class="flex-1 h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink placeholder:text-ink-muted"
                          autocomplete="off"
                        />
                        <.btn
                          type="submit"
                          variant={:primary}
                          size={:sm}
                          class="outline-mark outline-reach-shared focus-visible:outline-focus"
                        >
                          <.icon name="hero-key" class="size-4" />
                          Save
                        </.btn>
                      </form>
                      <%= if has_cred do %>
                        <.icon_btn
                          phx-click="delete_credential"
                          phx-value-service={service.name}
                          phx-value-key={cred_key}
                          variant={:ghost}
                          size={:sm}
                          title="Remove credential"
                          class="outline-mark outline-reach-shared focus-visible:outline-focus"
                        >
                          <.icon name="hero-trash" class="size-4" />
                        </.icon_btn>
                      <% end %>
                      <.shared_reach_tag />
                    </div>
                  </div>
                <% end %>
              </div>

              <!-- Health Check -->
              <%= if service.configured do %>
                <div class="mt-space-lg pt-space-lg border-t border-border">
                  <.btn
                    phx-click="check_service_health"
                    phx-value-service={service.name}
                    variant={:outline}
                    size={:sm}
                    class="w-full"
                    disabled={@checking == service.name}
                  >
                    <%= if @checking == service.name do %>
                      <.icon name="hero-arrow-path" class="size-4 animate-spin text-progress-fill" />
                      Checking...
                    <% else %>
                      <.icon name="hero-signal" class="size-4" />
                      Test Connection
                    <% end %>
                  </.btn>
                </div>
              <% end %>
            </.card>
          <% end %>
        </div>
      <% end %>

      <!-- Home Assistant Entity Discovery -->
      <%= if Enum.any?(@services, & &1.name == :home_assistant && &1.configured) do %>
        <.card class="p-space-lg">
          <div class="flex items-center justify-between mb-space-lg">
            <div>
              <h3 class="text-heading text-ink flex items-center gap-space-sm">
                <.icon name="hero-home" class="size-5" />
                Home Assistant Entity Discovery
              </h3>
              <p class="text-body text-ink-muted">
                Discover HA devices and register them in the Gazetteer
              </p>
            </div>
            <div class="flex gap-space-sm">
              <.btn
                phx-click="discover_ha_entities"
                variant={:outline}
                size={:sm}
                disabled={@ha_discovering}
              >
                <%= if @ha_discovering do %>
                  <.icon name="hero-arrow-path" class="size-4 animate-spin text-progress-fill" />
                  Discovering...
                <% else %>
                  <.icon name="hero-magnifying-glass" class="size-4" />
                  Discover
                <% end %>
              </.btn>
              <.btn
                phx-click="register_ha_entities"
                variant={:primary}
                size={:sm}
                disabled={@ha_discovering}
              >
                <.icon name="hero-plus-circle" class="size-4" />
                Register All
              </.btn>
            </div>
          </div>

          <%= if length(@ha_discovered_entities) > 0 do %>
            <div class="overflow-x-auto max-h-96">
              <table class="w-full text-left text-body-dense text-ink">
                <thead class="bg-surface-sunk">
                  <tr>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Entity ID</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Name</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Domain</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">State</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  <%= for entity <- @ha_discovered_entities do %>
                    <tr class="even:bg-surface-sunk">
                      <td class="h-row-compact px-space-sm text-ref">{entity.ha_entity_id}</td>
                      <td class="h-row-compact px-space-sm">{entity.name}</td>
                      <td class="h-row-compact px-space-sm">
                        <.badge>
                          {entity.ha_domain}
                        </.badge>
                      </td>
                      <td class="h-row-compact px-space-sm">{entity.state}</td>
                    </tr>
                  <% end %>
                </tbody>
              </table>
            </div>
            <div class="mt-space-sm text-body text-ink-muted">
              {length(@ha_discovered_entities)} entities found
            </div>
          <% end %>
        </.card>
      <% end %>
    </div>
    """
  end

  defp response_systems_section(assigns) do
    ~H"""
    <div class="space-y-space-xl">
      <!-- Lattice Stats -->
      <.card class="p-space-lg">
        <div class="flex items-center justify-between mb-space-lg">
          <div>
            <h3 class="text-heading text-ink">Phrase Lattice</h3>
            <p class="text-body text-ink-muted">Fragment inventory for response generation</p>
          </div>
          <.btn
            phx-click="regenerate_lattice"
            variant={:primary}
            size={:sm}
            disabled={@generating}
          >
            <%= if @generating do %>
              <.icon name="hero-arrow-path" class="size-4 animate-spin text-progress-fill" />
              Generating...
            <% else %>
              <.icon name="hero-arrow-path" class="size-4" />
              Regenerate
            <% end %>
          </.btn>
        </div>

        <%= if @lattice_stats[:status] == :ok do %>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-space-lg">
            <div class="bg-surface-sunk rounded-md p-space-md text-center">
              <div class="text-title text-ink tabular-nums">{Map.get(@lattice_stats, :total_fragments, 0)}</div>
              <div class="text-caption text-ink-muted">Total Fragments</div>
            </div>
            <%= for {chunk_type, count} <- Map.get(@lattice_stats, :by_chunk_type, %{}) do %>
              <div class="bg-surface-sunk rounded-md p-space-md text-center">
                <div class="text-heading text-ink tabular-nums">{count}</div>
                <div class="text-caption text-ink-muted">{chunk_type}</div>
              </div>
            <% end %>
          </div>

          <%= if map_size(Map.get(@lattice_stats, :tone_distribution, %{})) > 0 do %>
            <div class="mt-space-lg">
              <h4 class="text-subheading text-ink mb-space-sm">Tone Distribution</h4>
              <div class="flex flex-wrap gap-space-sm">
                <%= for {tone, count} <- Enum.sort_by(Map.get(@lattice_stats, :tone_distribution, %{}), fn {_, c} -> c end, :desc) do %>
                  <.badge class="border border-border-strong">
                    {tone}: {count}
                  </.badge>
                <% end %>
              </div>
            </div>
          <% end %>
        <% else %>
          <div class="text-body text-ink-muted p-space-lg text-center">
            <.icon name="hero-exclamation-circle" class="size-8 mx-auto mb-space-sm text-ink-muted" />
            <p>No phrase inventory loaded</p>
            <p class="mt-space-xs">Run "Regenerate" or <code class="text-term">mix gen_lattice_data</code> to build the inventory</p>
          </div>
        <% end %>
      </.card>

      <!-- Per-Domain Configuration -->
      <.card>
        <div class="p-space-lg border-b border-border">
          <h3 class="text-heading text-ink">Per-Domain Response System</h3>
          <p class="text-body text-ink-muted">Configure which system handles each domain</p>
        </div>

        <%= if length(@domains) == 0 do %>
          <div class="p-space-2xl text-center text-ink-muted">
            No domain configurations found
          </div>
        <% else %>
          <div class="divide-y divide-border">
            <%= for domain <- @domains do %>
              <div class="p-space-lg">
                <div class="flex flex-wrap items-center gap-space-lg">
                  <div class="min-w-[120px]">
                    <span class="text-subheading text-ink">{domain.domain}</span>
                  </div>

                  <div>
                    <label class="block mb-space-xs text-label text-ink-muted">System</label>
                    <select
                      phx-change="update_domain_system"
                      name="system"
                      class="h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink"
                    >
                      <input type="hidden" name="domain" value={domain.domain} />
                      <option value="lattice" selected={domain.system == "lattice"}>Lattice</option>
                      <option value="ouro" selected={domain.system == "ouro"}>Ouro</option>
                      <option value="template" selected={domain.system == "template"}>Template</option>
                    </select>
                  </div>

                  <div>
                    <label class="block mb-space-xs text-label text-ink-muted">Tone Bias</label>
                    <select
                      phx-change="update_domain_tone"
                      name="tone_bias"
                      class="h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink"
                    >
                      <input type="hidden" name="domain" value={domain.domain} />
                      <%= for tone <- ~w(enthusiastic cheery playful warm encouraging calm neutral professional matter_of_fact dry deadpan sardonic empathetic gentle patient) do %>
                        <option value={tone} selected={domain.tone_bias == tone}>{tone}</option>
                      <% end %>
                    </select>
                  </div>

                  <div class="min-w-[200px]">
                    <label class="block mb-space-xs text-label text-ink-muted">
                      Mirror: {domain.mirror_coefficient}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={domain.mirror_coefficient}
                      phx-change="update_domain_mirror"
                      name="mirror"
                      class="w-full accent-primary"
                    />
                    <input type="hidden" name="domain" value={domain.domain} />
                  </div>

                  <.badge>{domain.fallback} fallback</.badge>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </.card>
    </div>
    """
  end

  defp service_state(_configured, :healthy), do: {:success, :healthy, "Healthy"}
  defp service_state(_configured, :invalid_credentials), do: {:error, :error, "Invalid Key"}
  defp service_state(_configured, :missing_credentials), do: {:error, :error, "Missing"}
  defp service_state(_configured, {:error, _reason}), do: {:warning, :degraded, "Error"}
  defp service_state(true, nil), do: {:info, :idle, "Configured"}
  defp service_state(false, nil), do: {:default, :not_started, "Not Set"}

  defp service_state(configured, health) do
    raise ArgumentError,
          "ChatWeb.SettingsLive: no service state treatment for configured=#{inspect(configured)}, " <>
            "health=#{inspect(health)}. Health is nil, :healthy, :missing_credentials, " <>
            ":invalid_credentials or {:error, reason}; configured is a boolean."
  end

  defp service_icon(:weather), do: "hero-sun"
  defp service_icon(:news), do: "hero-newspaper"
  defp service_icon(:geocoding), do: "hero-map-pin"
  defp service_icon(:home_assistant), do: "hero-home"
  defp service_icon(_), do: "hero-cloud"

  defp fetch_ha_credentials do
    vault = Brain.Services.CredentialVault

    with {:ok, url} <- vault.get(:home_assistant, :url),
         {:ok, token} <- vault.get(:home_assistant, :access_token) do
      {:ok, %{url: url, access_token: token}}
    else
      _ -> {:error, :missing_credentials}
    end
  end

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
