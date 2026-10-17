defmodule Brain.Services.Dispatcher do
  @moduledoc """
  Routes intents to appropriate service handlers for enrichment.

  The dispatcher maintains a registry of service modules and routes
  incoming enrichment requests to the appropriate handler based on
  intent matching.

  ## Usage

      # Dispatch an intent for enrichment
      case Dispatcher.dispatch("weather.query", %{location: "NYC"}, context) do
        {:ok, enrichment_data} -> # Use enriched data
        {:error, reason} -> # Handle error
        :no_handler -> # No service registered for this intent
      end

      # Check if a service is available in a world
      Dispatcher.service_available?(:weather, world: world_id)

      # List all registered services with their status in a world
      Dispatcher.list_services(world: world_id)

  ## Scheduling

  The alarm, timer, reminder and calendar domains
  (`Brain.Services.SchedulingDriver.domains/0`) are routed to the scheduling
  driver the world has chosen (`assign_scheduling_driver/2`, stored in Atlas
  by `Atlas.WorldServiceDrivers`), whatever other service declares an intent
  or domain of theirs. A world that has chosen none raises, naming the world
  and the capability; no driver is assumed.
  """

  require Logger

  alias Brain.Services.CredentialVault
  alias Brain.Services.SchedulingDriver

  # ============================================================================
  # Service Registry
  # ============================================================================

  # Register service modules here. The dispatcher will discover their
  # supported intents and route requests accordingly. The intents of the
  # scheduling domains go to the world's scheduling driver instead
  # (`find_service/2`).
  @registered_services [
    Brain.Services.Weather,
    Brain.Services.SystemStatus,
    Brain.Services.HomeAssistant,
    Brain.Services.LocalScheduler
  ]

  # The drivers a world can choose for its alarms, timers, reminders and
  # calendar items. Each implements `Brain.Services.SchedulingDriver` and is
  # chosen by its service `name/0`. A world whose stored driver name is not
  # one of these raises (`scheduling_driver_named!/1`).
  @scheduling_drivers [
    Brain.Services.LocalScheduler
  ]

  # Build intent -> service lookup dynamically to handle module compilation order
  defp intent_to_service_map do
    for service <- @registered_services,
        Code.ensure_loaded?(service),
        function_exported?(service, :supported_intents, 0),
        intent <- service.supported_intents() do
      {intent, service}
    end
    |> Map.new()
  end

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Dispatch an intent to the appropriate service for enrichment.

  ## Parameters
    - intent: The classified intent name
    - slots: Map of filled slot values (e.g., %{location: "NYC"})
    - context: Additional context (user_id, etc.). It must name the world
      whose credentials the service uses, as `:world_id` (or `:world`);
      a missing world raises. With `side_effects: false`, an intent the
      service says `writes?/1` (a device action, a stored alarm) is refused
      with `{:error, :side_effects_off}` and the service is not called;
      read-only intents still run.

  An intent of a scheduling domain goes to the world's scheduling driver, as
  the operation it names (`Brain.Services.SchedulingDriver.operation/1`); a
  world with no driver chosen raises. Its dates and times are read against
  the turn they were said in: the context must carry `:turn_time`, the
  moment of the turn (a `DateTime`), and `:time_zone`, the zone of the person
  who said it (a `Brain.Temporal.Zone`), which go to the driver as the slots
  of those names; a context without either raises. The context's
  `:web_user_id`, the web user who said it, goes as the `:web_user_id` slot.

  ## Returns
    - {:ok, enrichment_data} on success
    - {:error, reason} on failure
    - :no_handler if no service registered for this intent
  """
  @spec dispatch(String.t(), map(), map()) ::
          {:ok, map()} | {:error, term()} | :no_handler
  def dispatch(intent, slots, context) when is_binary(intent) and is_map(slots) do
    world = context_world!(context)

    case find_service(intent, world) do
      nil ->
        Logger.debug("Dispatcher: no handler for intent=#{intent}")
        :no_handler

      service ->
        if Map.get(context, :side_effects, true) == false and service.writes?(intent) do
          Logger.info("Dispatcher: refused intent=#{intent} for #{inspect(service)}: it writes and side effects are off")
          {:error, :side_effects_off}
        else
          Logger.info("Dispatcher: routing intent=#{intent} to #{inspect(service)}")
          dispatch_to_service(service, intent, with_turn(intent, slots, context), world)
        end
    end
  end

  def dispatch(_intent, _slots, _context), do: :no_handler

  @doc """
  Check if a service is available (has credentials configured).

  ## Parameters
    - service_name: Atom identifying the service (e.g., :weather)
    - opts: Keyword list with :world (required), the world whose credentials are checked

  ## Returns
    - true if all required credentials are present
    - false otherwise
  """
  @spec service_available?(atom(), keyword()) :: boolean()
  def service_available?(service_name, opts) do
    world = world_opt!(opts, "service_available?")

    case find_service_by_name(service_name) do
      nil ->
        false

      service ->
        has_all_credentials?(service, world)
    end
  end

  @doc """
  List all registered services with their status.

  ## Parameters
    - opts: Keyword list with :world (required), the world whose credentials
      decide each service's `configured` status

  ## Returns
    List of maps with service info and availability status.
  """
  @spec list_services(keyword()) :: [map()]
  def list_services(opts) do
    world = world_opt!(opts, "list_services")

    @registered_services
    |> Enum.filter(&Code.ensure_loaded?/1)
    |> Enum.map(fn service ->
      %{
        name: service.name(),
        display_name: service.display_name(),
        description: service.description(),
        module: service,
        supported_intents: service.supported_intents(),
        provides_fields: service.provides_fields(),
        required_credentials: service.required_credentials(),
        configured: has_all_credentials?(service, world),
        enabled: service_enabled?(service)
      }
    end)
  end

  @doc """
  Collects slot schemas from all registered services that implement `slot_schema/0`.

  Returns a map of intent/domain name to schema, suitable for merging into
  SlotDetector's runtime schema map. Services using `supported_domains/0` have
  their schema registered under each domain prefix. Services using
  `supported_intents/0` have their schema registered per-intent. A scheduling
  driver that implements `slot_schema/0` has its schema registered under each
  scheduling domain, since any world may choose it.
  """
  @spec service_schemas() :: %{String.t() => map()}
  def service_schemas do
    intent_entries =
      for service <- @registered_services,
          Code.ensure_loaded?(service),
          function_exported?(service, :slot_schema, 0),
          function_exported?(service, :supported_intents, 0),
          intent <- service.supported_intents() do
        {intent, service.slot_schema()}
      end

    domain_entries =
      for service <- @registered_services,
          Code.ensure_loaded?(service),
          function_exported?(service, :slot_schema, 0) do
        domains = cond do
          function_exported?(service, :supported_domains, 0) ->
            service.supported_domains()

          function_exported?(service, :supported_intents, 0) ->
            service.supported_intents()
            |> Enum.map(fn intent ->
              case String.split(intent, ".", parts: 2) do
                [domain, _] -> domain
                _ -> nil
              end
            end)
            |> Enum.uniq()
            |> Enum.reject(&is_nil/1)

          true ->
            []
        end

        Enum.map(domains, fn domain ->
          {domain, service.slot_schema()}
        end)
      end
      |> List.flatten()

    scheduling_entries =
      for driver <- @scheduling_drivers,
          Code.ensure_loaded?(driver),
          function_exported?(driver, :slot_schema, 0),
          domain <- SchedulingDriver.domains() do
        {domain, driver.slot_schema()}
      end

    Map.new(intent_entries ++ domain_entries ++ scheduling_entries)
  end

  @doc """
  Get the service module for an intent in a world.

  An intent of a scheduling domain (`SchedulingDriver.domains/0`) goes to the
  world's scheduling driver when it names an operation
  (`SchedulingDriver.operation/1`), and to no service when it does not. A
  world with no scheduling driver chosen raises. Every other intent is routed
  the same in every world:

  1. Exact intent match from `supported_intents/0`
  2. Domain match from `supported_domains/0` (e.g., "smarthome" matches "smarthome.switch")
  3. Domain-prefix fallback (service declares "weather.query", matches "weather.forecast")
  """
  @spec find_service(String.t() | nil, String.t()) :: module() | nil
  def find_service(nil, _world), do: nil

  def find_service(intent, world) when is_binary(intent) do
    world = world_id!(world, "find_service")

    if SchedulingDriver.domain?(intent) do
      driver = scheduling_driver!(world)

      case SchedulingDriver.operation(intent) do
        {:ok, _operation} -> driver
        :error -> nil
      end
    else
      case Map.get(intent_to_service_map(), intent) do
        nil -> find_service_by_domain(intent)
        service -> service
      end
    end
  end

  # ============================================================================
  # Scheduling drivers
  # ============================================================================

  @doc "The drivers a world can choose for its alarms, timers, reminders and calendar items."
  @spec scheduling_drivers() :: [module()]
  def scheduling_drivers, do: @scheduling_drivers

  @doc """
  The scheduling driver named `name`, its service `name/0` as a string, as
  stored for a world. Raises for a name no scheduling driver has.
  """
  @spec scheduling_driver_named!(String.t()) :: module()
  def scheduling_driver_named!(name) when is_binary(name) do
    Enum.find(@scheduling_drivers, &(Atom.to_string(&1.name()) == name)) ||
      raise ArgumentError,
            "Dispatcher: no scheduling driver is named #{inspect(name)}; the drivers are " <>
              inspect(Enum.map(@scheduling_drivers, &Atom.to_string(&1.name())))
  end

  @doc """
  The scheduling driver the world has chosen, or `:none` when it has chosen
  none. Raises when the stored driver name is not a scheduling driver's.
  """
  @spec scheduling_driver(String.t()) :: {:ok, module()} | :none
  def scheduling_driver(world) do
    world = world_id!(world, "scheduling_driver")

    case Atlas.WorldServiceDrivers.fetch(world, SchedulingDriver.capability()) do
      {:ok, name} -> {:ok, scheduling_driver_named!(name)}
      :error -> :none
    end
  end

  @doc """
  The scheduling driver the world has chosen. Raises, naming the world and the
  capability, when it has chosen none.
  """
  @spec scheduling_driver!(String.t()) :: module()
  def scheduling_driver!(world) do
    case scheduling_driver(world) do
      {:ok, driver} ->
        driver

      :none ->
        raise "Dispatcher: world #{inspect(world)} has no driver for the " <>
                "#{inspect(SchedulingDriver.capability())} capability, so its alarm, timer, reminder " <>
                "and calendar intents have nowhere to go; choose one for the world in Settings > Services"
    end
  end

  @doc """
  The credentials `driver` needs that are not set in the world. A driver can
  be chosen in a world only when this is empty.
  """
  @spec missing_credentials(module(), String.t()) :: [atom()]
  def missing_credentials(driver, world) when is_atom(driver) do
    world = world_id!(world, "missing_credentials")
    Enum.reject(driver.required_credentials(), &CredentialVault.has_credential?(driver.name(), &1, world: world))
  end

  @doc """
  Chooses `driver` as the world's scheduling driver, stored in Atlas, replacing
  the one chosen before. Refused with `{:error, {:missing_credentials, keys}}`
  while a credential the driver needs is not set in the world, and nothing is
  stored. Raises for a module that is not a scheduling driver.
  """
  @spec assign_scheduling_driver(String.t(), module()) :: :ok | {:error, {:missing_credentials, [atom()]}}
  def assign_scheduling_driver(world, driver) when is_atom(driver) do
    world = world_id!(world, "assign_scheduling_driver")

    unless driver in @scheduling_drivers do
      raise ArgumentError,
            "Dispatcher: #{inspect(driver)} is not a scheduling driver; the drivers are #{inspect(@scheduling_drivers)}"
    end

    case missing_credentials(driver, world) do
      [] ->
        Atlas.WorldServiceDrivers.put!(world, SchedulingDriver.capability(), Atom.to_string(driver.name()))
        Logger.info("Dispatcher: world #{inspect(world)} schedules with #{inspect(driver)}")
        :ok

      missing ->
        {:error, {:missing_credentials, missing}}
    end
  end

  defp find_service_by_domain(intent) do
    domain = case String.split(intent, ".", parts: 2) do
      [d, _] -> d
      _ -> nil
    end

    if domain do
      Enum.find(@registered_services, fn service ->
        Code.ensure_loaded?(service) and
          function_exported?(service, :supported_domains, 0) and
          domain in service.supported_domains()
      end) || find_service_by_intent_prefix(domain)
    else
      nil
    end
  end

  defp find_service_by_intent_prefix(domain) do
    service_map = intent_to_service_map()

    Enum.find_value(service_map, fn {registered_intent, service} ->
      if String.starts_with?(registered_intent, domain <> "."), do: service
    end)
  end

  @doc """
  Get a service module by name.
  """
  @spec find_service_by_name(atom()) :: module() | nil
  def find_service_by_name(service_name) when is_atom(service_name) do
    Enum.find(@registered_services, fn service ->
      Code.ensure_loaded?(service) and service.name() == service_name
    end)
  end

  @doc """
  Run a health check for a service.

  ## Parameters
    - service_name: Atom identifying the service
    - opts: Keyword list with :world (required), the world whose credentials are checked

  ## Returns
    - :ok if credentials are valid
    - {:error, reason} if check fails
  """
  @spec health_check(atom(), keyword()) :: :ok | {:error, term()}
  def health_check(service_name, opts) do
    world = world_opt!(opts, "health_check")

    case find_service_by_name(service_name) do
      nil ->
        {:error, :service_not_found}

      service ->
        case get_credentials(service, world) do
          {:ok, credentials} ->
            started = System.monotonic_time(:millisecond)
            status = service.health_check(credentials)
            Brain.Telemetry.emit_service_health_check(service_name, status, System.monotonic_time(:millisecond) - started)
            status

          {:error, _} = error ->
            error
        end
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  # Credentials are stored per world, so every lookup names the world; there is
  # no default.
  defp world_opt!(opts, function) do
    case Keyword.fetch(opts, :world) do
      {:ok, world} when is_binary(world) and world != "" ->
        world

      {:ok, other} ->
        raise ArgumentError, "Dispatcher.#{function} option :world must be a world id, got: #{inspect(other)}"

      :error ->
        raise ArgumentError, "Dispatcher.#{function} needs :world, the world whose credentials are used"
    end
  end

  defp world_id!(world, _function) when is_binary(world) and world != "", do: world

  defp world_id!(other, function) do
    raise ArgumentError, "Dispatcher.#{function} needs the world it acts in, got: #{inspect(other)}"
  end

  # A scheduling operation reads its dates and times against the turn they
  # were said in, so the turn's time and zone go with its slots as
  # `:turn_time` and `:time_zone`. The context must carry both; nothing stands
  # in for either. The web user who said it goes as `:web_user_id`, `nil`
  # when the context names none: the items they set are theirs, and the
  # driver refuses to set one for no one.
  defp with_turn(intent, slots, context) do
    if SchedulingDriver.domain?(intent) do
      turn_time =
        case Map.get(context, :turn_time) do
          %DateTime{} = turn_time ->
            turn_time

          other ->
            raise ArgumentError,
                  "Dispatcher.dispatch context needs :turn_time, the moment of the turn a #{intent} intent was said " <>
                    "in, got: #{inspect(other)}"
        end

      zone =
        case Map.get(context, :time_zone) do
          %Brain.Temporal.Zone{} = zone ->
            zone

          other ->
            raise ArgumentError,
                  "Dispatcher.dispatch context needs :time_zone, the Brain.Temporal.Zone a #{intent} intent's dates " <>
                    "and times are read in, got: #{inspect(other)}"
        end

      Map.merge(slots, %{turn_time: turn_time, time_zone: zone, web_user_id: Map.get(context, :web_user_id)})
    else
      slots
    end
  end

  defp context_world!(context) do
    case Map.get(context, :world_id) || Map.get(context, :world) do
      world when is_binary(world) and world != "" ->
        world

      nil ->
        raise ArgumentError, "Dispatcher.dispatch context needs :world_id, the world whose credentials are used"

      other ->
        raise ArgumentError, "Dispatcher.dispatch context :world_id must be a world id, got: #{inspect(other)}"
    end
  end

  defp dispatch_to_service(service, intent, slots, world) do
    # Check if service is enabled
    if service_enabled?(service) do
      case get_credentials(service, world) do
      {:ok, credentials} ->
        # Call the service
        start_time = System.monotonic_time(:millisecond)

        result =
          try do
            invoke(service, intent, slots, credentials, world)
          rescue
            e ->
              Logger.error("Service error",
                service: service.name(),
                intent: intent,
                error: Exception.message(e)
              )

              {:error, :service_error}
          end

        duration = System.monotonic_time(:millisecond) - start_time
        log_dispatch(service, intent, result, duration)
        result

      {:error, :missing_credentials} ->
        Logger.debug("Missing credentials for service",
          service: service.name(),
          intent: intent
        )

        {:error, :missing_credentials}
    end
    else
      Logger.debug("Service disabled", service: service.name())
      {:error, :service_disabled}
    end
  end

  # A scheduling intent is the driver's operation of that name; every other
  # intent is the service's `enrich/4`. `find_service/2` routes a scheduling
  # domain's intent here only when it names an operation.
  defp invoke(service, intent, slots, credentials, world) do
    if SchedulingDriver.domain?(intent) do
      {:ok, operation} = SchedulingDriver.operation(intent)
      apply(service, operation, [slots, credentials, world])
    else
      service.enrich(intent, slots, credentials, world)
    end
  end

  defp get_credentials(service, world) do
    required = service.required_credentials()

    credentials =
      Enum.reduce_while(required, {:ok, %{}}, fn key, {:ok, acc} ->
        case CredentialVault.get(service.name(), key, world: world) do
          {:ok, value} ->
            {:cont, {:ok, Map.put(acc, key, value)}}

          {:error, :not_found} ->
            {:halt, {:error, :missing_credentials}}
        end
      end)

    credentials
  end

  defp has_all_credentials?(service, world) do
    case get_credentials(service, world) do
      {:ok, _} -> true
      {:error, _} -> false
    end
  end

  defp service_enabled?(service) do
    if function_exported?(service, :enabled?, 0) do
      service.enabled?()
    else
      true
    end
  end

  defp log_dispatch(service, intent, result, duration) do
    status =
      case result do
        {:ok, _} -> :success
        {:error, reason} -> {:error, reason}
      end

    Logger.debug("Service dispatch",
      service: service.name(),
      intent: intent,
      status: status,
      duration_ms: duration
    )

    # Emit telemetry event using Brain.Telemetry
    Brain.Telemetry.emit_service_dispatch(service.name(), intent, status, duration)
  end
end
