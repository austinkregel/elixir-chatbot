defmodule Brain.Services.Service do
  @moduledoc """
  Behaviour for external service integrations.

  Services are plugins that can enrich responses with live data from
  external APIs (weather, news, etc.). Each service implements this
  behaviour to integrate with the response enrichment system.

  ## Implementing a Service

      defmodule Brain.Services.Weather do
        @behaviour Brain.Services.Service

        @impl true
        def name, do: :weather

        @impl true
        def display_name, do: "Weather"

        @impl true
        def description, do: "Current weather and forecasts via OpenWeatherMap"

        @impl true
        def required_credentials, do: [:api_key]

        @impl true
        def supported_intents, do: ["weather.query", "weather.forecast"]

        @impl true
        def provides_fields, do: [:temperature, :conditions, :humidity, :forecast]

        @impl true
        def enrich(intent, slots, credentials) do
          # Fetch weather data and return enriched fields
          {:ok, %{temperature: "72°F", conditions: "sunny"}}
        end

        @impl true
        def health_check(credentials) do
          # Validate API key works
          :ok
        end
      end

  ## Enrichment Fields

  Services declare which fields they provide via `provides_fields/0`.
  These are used by templates with `enriched:field_name` conditions.
  """

  @typedoc "Slot values extracted from user query"
  @type slots :: %{atom() => any()}

  @typedoc "Credentials for the service"
  @type credentials :: %{atom() => String.t()}

  @typedoc "Enrichment data returned by the service"
  @type enrichment_data :: %{atom() => any()}

  @typedoc "Error reasons"
  @type error_reason ::
          :missing_credentials
          | :invalid_credentials
          | :rate_limited
          | :service_unavailable
          | :location_not_found
          | {:api_error, integer()}
          | term()

  @doc """
  Unique identifier for the service.
  Used in credential storage and dispatcher routing.
  """
  @callback name() :: atom()

  @doc """
  Human-readable name for display in UI.
  """
  @callback display_name() :: String.t()

  @doc """
  Description of what the service provides.
  """
  @callback description() :: String.t()

  @doc """
  List of credential keys this service requires.
  E.g., [:api_key] or [:client_id, :client_secret]
  """
  @callback required_credentials() :: [atom()]

  @doc """
  List of intent names this service can enrich.
  E.g., ["weather.query", "weather.forecast"]
  """
  @callback supported_intents() :: [String.t()]

  @doc """
  List of fields this service can provide in enrichment data.
  Used for template condition matching (enriched:field_name).
  """
  @callback provides_fields() :: [atom()]

  @doc """
  Enrich a response with data from this service.

  Called when the intent matches one of `supported_intents/0` and
  all required slots are filled.

  ## Parameters
    - intent: The classified intent name
    - slots: Map of slot names to values (e.g., %{location: "NYC"})
    - credentials: Map of credential keys to values

  ## Returns
    - {:ok, enrichment_data} on success
    - {:error, reason} on failure
  """
  @callback enrich(intent :: String.t(), slots(), credentials()) ::
              {:ok, enrichment_data()} | {:error, error_reason()}

  @doc """
  Validate that the provided credentials are working.

  Called when the user enters credentials in the settings UI.

  ## Returns
    - :ok if credentials are valid
    - {:error, reason} if validation fails
  """
  @callback health_check(credentials()) :: :ok | {:error, term()}

  # ============================================================================
  # Optional Callbacks with Defaults
  # ============================================================================

  @doc """
  Whether this service is enabled.
  Override to add feature flags or conditional availability.
  """
  @callback enabled?() :: boolean()

  @optional_callbacks enabled?: 0

  # ============================================================================
  # Helper Functions
  # ============================================================================

  @doc """
  Check if a module implements the Service behaviour.
  """
  def implements?(module) when is_atom(module) do
    behaviours = module.module_info(:attributes)[:behaviour] || []
    __MODULE__ in behaviours
  rescue
    _ -> false
  end
end
