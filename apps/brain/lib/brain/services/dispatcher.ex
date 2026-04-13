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

      # Check if a service is available
      Dispatcher.service_available?(:weather)

      # List all registered services
      Dispatcher.list_services()
  """

  require Logger

  alias Brain.Services.CredentialVault

  # ============================================================================
  # Service Registry
  # ============================================================================

  # Register service modules here. The dispatcher will discover their
  # supported intents and route requests accordingly.
  @registered_services [
    Brain.Services.Weather,
    Brain.Services.SystemStatus
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
    - context: Additional context (world_id, user_id, etc.)

  ## Returns
    - {:ok, enrichment_data} on success
    - {:error, reason} on failure
    - :no_handler if no service registered for this intent
  """
  @spec dispatch(String.t(), map(), map()) ::
          {:ok, map()} | {:error, term()} | :no_handler
  def dispatch(intent, slots, context \\ %{})

  def dispatch(intent, slots, context) when is_binary(intent) and is_map(slots) do
    case find_service(intent) do
      nil ->
        :no_handler

      service ->
        dispatch_to_service(service, intent, slots, context)
    end
  end

  def dispatch(_intent, _slots, _context), do: :no_handler

  @doc """
  Check if a service is available (has credentials configured).

  ## Parameters
    - service_name: Atom identifying the service (e.g., :weather)
    - opts: Optional keyword list with :world

  ## Returns
    - true if all required credentials are present
    - false otherwise
  """
  @spec service_available?(atom(), keyword()) :: boolean()
  def service_available?(service_name, opts \\ []) do
    case find_service_by_name(service_name) do
      nil ->
        false

      service ->
        world = Keyword.get(opts, :world, "default")
        has_all_credentials?(service, world)
    end
  end

  @doc """
  List all registered services with their status.

  ## Returns
    List of maps with service info and availability status.
  """
  @spec list_services(keyword()) :: [map()]
  def list_services(opts \\ []) do
    world = Keyword.get(opts, :world, "default")

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
  Get the service module for an intent.
  """
  @spec find_service(String.t()) :: module() | nil
  def find_service(intent) when is_binary(intent) do
    Map.get(intent_to_service_map(), intent)
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

  ## Returns
    - :ok if credentials are valid
    - {:error, reason} if check fails
  """
  @spec health_check(atom(), keyword()) :: :ok | {:error, term()}
  def health_check(service_name, opts \\ []) do
    case find_service_by_name(service_name) do
      nil ->
        {:error, :service_not_found}

      service ->
        world = Keyword.get(opts, :world, "default")

        case get_credentials(service, world) do
          {:ok, credentials} ->
            service.health_check(credentials)

          {:error, _} = error ->
            error
        end
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp dispatch_to_service(service, intent, slots, context) do
    world = Map.get(context, :world_id) || Map.get(context, :world, "default")

    # Check if service is enabled
    if service_enabled?(service) do
      case get_credentials(service, world) do
      {:ok, credentials} ->
        # Call the service
        start_time = System.monotonic_time(:millisecond)

        result =
          try do
            service.enrich(intent, slots, credentials)
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
