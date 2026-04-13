defmodule Brain.Services.Weather do
  @moduledoc """
  Weather service integration using OpenWeatherMap API.

  Provides current weather conditions and forecasts for locations.
  Includes geocoding to convert location names to coordinates.

  ## Configuration

  Requires an OpenWeatherMap API key stored via CredentialVault:

      CredentialVault.store(:weather, :api_key, "your_api_key")

  ## Enrichment Fields

  This service provides the following fields for template enrichment:
  - `:temperature` - Current temperature (e.g., "72°F")
  - `:conditions` - Weather conditions (e.g., "sunny", "partly cloudy")
  - `:humidity` - Humidity percentage (e.g., "45%")
  - `:wind_speed` - Wind speed (e.g., "10 mph")
  - `:feels_like` - "Feels like" temperature
  - `:forecast` - Multi-day forecast summary
  - `:location_name` - Resolved location name
  """

  @behaviour Brain.Services.Service

  require Logger

  alias Brain.Services.Cache

  @base_url "https://api.openweathermap.org"

  # HTTP client - configurable for testing
  @http_client Application.compile_env(:brain, :http_client, Req)

  # ============================================================================
  # Service Behaviour Implementation
  # ============================================================================

  @impl true
  def name, do: :weather

  @impl true
  def display_name, do: "Weather"

  @impl true
  def description do
    "Current weather conditions and forecasts via OpenWeatherMap"
  end

  @impl true
  def required_credentials, do: [:api_key]

  @impl true
  def supported_intents do
    # Only specific weather.* intents are supported
    ["weather.query", "weather.forecast", "weather.current"]
  end

  @impl true
  def provides_fields do
    [:temperature, :conditions, :humidity, :wind_speed, :feels_like, :forecast, :location_name]
  end

  @impl true
  def enrich(intent, slots, credentials) do
    location = Map.get(slots, :location) || Map.get(slots, "location")

    if location do
      case intent do
        "weather.forecast" ->
          get_forecast(location, credentials)

        _ ->
          get_current_weather(location, credentials)
      end
    else
      {:error, :missing_location}
    end
  end

  @impl true
  def health_check(credentials) do
    api_key = Map.get(credentials, :api_key)

    if api_key do
      # Try a simple API call to verify the key
      case geocode("London", api_key) do
        {:ok, _} -> :ok
        {:error, :invalid_api_key} -> {:error, :invalid_credentials}
        {:error, reason} -> {:error, reason}
      end
    else
      {:error, :missing_api_key}
    end
  end

  # ============================================================================
  # Weather API
  # ============================================================================

  @doc """
  Get current weather for a location.
  """
  @spec get_current_weather(String.t(), map()) :: {:ok, map()} | {:error, term()}
  def get_current_weather(location, credentials) do
    api_key = Map.get(credentials, :api_key)

    # Early validation - don't hit API without valid credentials
    if is_nil(api_key) or api_key == "" do
      {:error, :missing_api_key}
    else
      cache_key = "current:#{normalize_location(location)}"

      # Check cache first
      case Cache.get(:weather, cache_key) do
        {:ok, cached} ->
          {:ok, cached}

        :miss ->
          with {:ok, coords} <- geocode_cached(location, api_key),
               {:ok, weather} <- fetch_current(coords, api_key) do
            enrichment = format_current_weather(weather, location)
            Cache.put(:weather, cache_key, enrichment)
            {:ok, enrichment}
          end
      end
    end
  end

  @doc """
  Get weather forecast for a location.
  """
  @spec get_forecast(String.t(), map()) :: {:ok, map()} | {:error, term()}
  def get_forecast(location, credentials) do
    api_key = Map.get(credentials, :api_key)

    # Early validation - don't hit API without valid credentials
    if is_nil(api_key) or api_key == "" do
      {:error, :missing_api_key}
    else
      cache_key = "forecast:#{normalize_location(location)}"

      case Cache.get(:weather, cache_key) do
        {:ok, cached} ->
          {:ok, cached}

        :miss ->
          with {:ok, coords} <- geocode_cached(location, api_key),
               {:ok, forecast} <- fetch_forecast(coords, api_key) do
            enrichment = format_forecast(forecast, location)
            Cache.put(:weather, cache_key, enrichment)
            {:ok, enrichment}
          end
      end
    end
  end

  # ============================================================================
  # Geocoding
  # ============================================================================

  defp geocode_cached(location, api_key) do
    cache_key = normalize_location(location)

    case Cache.get(:geocoding, cache_key) do
      {:ok, coords} ->
        {:ok, coords}

      :miss ->
        # Use normalized location for API call to improve consistency
        geocode_location = format_for_geocoding(location)
        
        case geocode(geocode_location, api_key) do
          {:ok, coords} = result ->
            Cache.put(:geocoding, cache_key, coords, ttl: :timer.hours(24))
            result

          error ->
            error
        end
    end
  end

  # Format location for the OpenWeatherMap geocoding API
  # The API prefers "City,State,Country" format without spaces after commas
  defp format_for_geocoding(location) when is_binary(location) do
    location
    |> String.trim()
    |> String.replace(", ", ",")  # Remove space after comma
    |> String.replace(" ", ",")   # Replace remaining spaces with commas
    |> then(fn loc ->
      # Append US if not specified
      if String.contains?(loc, "US") or String.contains?(loc, "us") do
        loc
      else
        "#{loc},US"
      end
    end)
  end

  defp format_for_geocoding(_), do: ""

  defp geocode(location, api_key) do
    encoded = URI.encode(location)
    url = "#{@base_url}/geo/1.0/direct?q=#{encoded}&limit=1&appid=#{api_key}"

    case http_get(url) do
      {:ok, %{status: 200, body: [first | _]}} ->
        {:ok,
         %{
           lat: first["lat"],
           lon: first["lon"],
           name: first["name"],
           country: first["country"],
           state: first["state"]
         }}

      {:ok, %{status: 200, body: []}} ->
        {:error, :location_not_found}

      {:ok, %{status: 401}} ->
        {:error, :invalid_api_key}

      {:ok, %{status: 429}} ->
        {:error, :rate_limited}

      {:ok, %{status: status}} ->
        {:error, {:api_error, status}}

      {:error, reason} ->
        Logger.error("Geocoding request failed", location: location, error: inspect(reason))
        {:error, :service_unavailable}
    end
  end

  # ============================================================================
  # Weather API Calls
  # ============================================================================

  defp fetch_current(%{lat: lat, lon: lon}, api_key) do
    url = "#{@base_url}/data/2.5/weather?lat=#{lat}&lon=#{lon}&units=imperial&appid=#{api_key}"

    case http_get(url) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: 401}} ->
        {:error, :invalid_api_key}

      {:ok, %{status: 429}} ->
        {:error, :rate_limited}

      {:ok, %{status: status}} ->
        {:error, {:api_error, status}}

      {:error, reason} ->
        Logger.error("Weather request failed", error: inspect(reason))
        {:error, :service_unavailable}
    end
  end

  defp fetch_forecast(%{lat: lat, lon: lon}, api_key) do
    # Get 5-day forecast (3-hour intervals)
    url =
      "#{@base_url}/data/2.5/forecast?lat=#{lat}&lon=#{lon}&units=imperial&cnt=40&appid=#{api_key}"

    case http_get(url) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: 401}} ->
        {:error, :invalid_api_key}

      {:ok, %{status: 429}} ->
        {:error, :rate_limited}

      {:ok, %{status: status}} ->
        {:error, {:api_error, status}}

      {:error, reason} ->
        Logger.error("Forecast request failed", error: inspect(reason))
        {:error, :service_unavailable}
    end
  end

  # ============================================================================
  # Response Formatting
  # ============================================================================

  defp format_current_weather(data, original_location) do
    main = data["main"] || %{}
    weather = List.first(data["weather"] || []) || %{}
    wind = data["wind"] || %{}

    temp = main["temp"]
    feels_like = main["feels_like"]
    humidity = main["humidity"]
    wind_speed = wind["speed"]
    conditions = weather["main"]
    description = weather["description"]
    location_name = data["name"] || original_location

    %{
      temperature: format_temp(temp),
      feels_like: format_temp(feels_like),
      humidity: format_humidity(humidity),
      wind_speed: format_wind(wind_speed),
      conditions: format_conditions(conditions, description),
      location_name: location_name,
      # Raw values for advanced use
      raw: %{
        temp: temp,
        feels_like: feels_like,
        humidity: humidity,
        wind_speed: wind_speed,
        conditions: conditions,
        description: description
      }
    }
  end

  defp format_forecast(data, original_location) do
    list = data["list"] || []
    city = data["city"] || %{}
    location_name = city["name"] || original_location

    # Group by day and get daily summary
    daily_forecasts =
      list
      |> Enum.group_by(fn entry ->
        entry["dt_txt"]
        |> String.split(" ")
        |> List.first()
      end)
      |> Enum.take(5)
      |> Enum.map(fn {date, entries} ->
        # Get min/max temps and most common condition
        temps = Enum.map(entries, & &1["main"]["temp"])
        conditions = Enum.map(entries, & &1["weather"] |> List.first() |> Map.get("main"))

        most_common_condition =
          conditions
          |> Enum.frequencies()
          |> Enum.max_by(fn {_, count} -> count end, fn -> {"Unknown", 0} end)
          |> elem(0)

        %{
          date: date,
          high: Enum.max(temps, fn -> 0 end),
          low: Enum.min(temps, fn -> 0 end),
          conditions: most_common_condition
        }
      end)

    forecast_text =
      daily_forecasts
      |> Enum.map(fn day ->
        "#{day.date}: #{format_temp(day.high)} high, #{format_temp(day.low)} low, #{day.conditions}"
      end)
      |> Enum.join("; ")

    # Also get current weather from first entry
    first = List.first(list)
    current = if first, do: format_current_weather(first, original_location), else: %{}

    Map.merge(current, %{
      forecast: forecast_text,
      location_name: location_name,
      daily_forecasts: daily_forecasts
    })
  end

  # ============================================================================
  # Formatting Helpers
  # ============================================================================

  defp format_temp(nil), do: "unknown"
  defp format_temp(temp) when is_number(temp), do: "#{round(temp)}°F"
  defp format_temp(_), do: "unknown"

  defp format_humidity(nil), do: "unknown"
  defp format_humidity(humidity), do: "#{humidity}%"

  defp format_wind(nil), do: "calm"
  defp format_wind(speed) when speed < 1, do: "calm"
  defp format_wind(speed) when is_number(speed), do: "#{round(speed)} mph"
  defp format_wind(_), do: "unknown"

  defp format_conditions(nil, nil), do: "unknown"
  defp format_conditions(main, nil), do: String.downcase(main || "unknown")
  defp format_conditions(_main, description), do: description

  defp normalize_location(location) when is_binary(location) do
    # Normalize location for cache key
    # Simple normalization: lowercase, trim, collapse whitespace, remove punctuation
    # The geocoding API handles location resolution, this just ensures
    # consistent cache keys for the same conceptual location
    location
    |> String.downcase()
    |> String.trim()
    |> String.replace(",", " ")
    |> String.split()
    |> Enum.join(" ")
  end

  defp normalize_location(_), do: ""

  # ============================================================================
  # HTTP Client
  # ============================================================================

  defp http_get(url) do
    @http_client.get(url)
  end
end
