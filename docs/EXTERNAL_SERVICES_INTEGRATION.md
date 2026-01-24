# External Services Integration Guide

This document explains how to integrate external services (weather, geocoding, etc.) into the ChatBot system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Input                                      │
│                      "What's the weather in NYC?"                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Analysis Pipeline                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │  Tokenizer  │→ │Intent Classif│→ │Entity Extr. │→ │  Slot Detector   │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────────┘  │
│                                                              │              │
│                   Intent: weather.query                      │              │
│                   Entities: [{location: "NYC"}]              │              │
│                   Slots: {location: "NYC"}                   │              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Response System                                   │
│  ┌─────────────────┐  ┌───────────────────┐  ┌──────────────────────────┐  │
│  │ Service Router  │→ │ External Service  │→ │  Response Synthesizer    │  │
│  │ (TO BE ADDED)   │  │ (weather, geo...) │  │  (formats final reply)   │  │
│  └─────────────────┘  └───────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Response                                        │
│         "It's currently 72°F and sunny in New York City."                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Current System Components

### 1. Intent Classification

The system recognizes these service-related intents (see `priv/analysis/slot_schemas.json`):

| Intent | Required Slots | Optional Slots | Description |
|--------|---------------|----------------|-------------|
| `weather.query` | `location` | `date`, `time`, `unit-temperature` | Weather information request |
| `news.query` | - | `topic`, `news-source`, `date`, `news-sort` | News information request |
| `device.control` | `device`, `action` | `room`, `color`, `value` | Smart device control |
| `music.play` | - | `song`, `artist`, `album`, `playlist` | Music playback |
| `reminder.create` | `content` | `date`, `time`, `recurrence` | Create a reminder |
| `account.query` | `account` | `category`, `merchant`, `date` | Account information |
| `search.web` | `query` | `search-engine` | Web search |

### 2. Entity Extraction

The Gazetteer (`ChatBot.ML.Gazetteer`) extracts entities like:
- Locations (cities, countries, regions)
- Devices (lights, thermostat, etc.)
- Times/Dates
- Music artists, songs, albums

### 3. Slot Detection

When an intent requires slots that weren't extracted, the system asks for clarification:
```
User: "What's the weather?"
Bot: "What location would you like the weather for?"
User: "New York"
Bot: "It's currently 72°F and sunny in New York City."
```

---

## Integration Points

### Option A: Service Dispatcher (Recommended)

Create a centralized dispatcher that routes fulfilled intents to appropriate services.

**File:** `lib/chat_bot/services/dispatcher.ex`

```elixir
defmodule ChatBot.Services.Dispatcher do
  @moduledoc """
  Routes fulfilled intents to external service handlers.
  """
  
  alias ChatBot.Services.{Weather, Geocoding, News}
  
  @doc """
  Dispatch an intent with filled slots to the appropriate service.
  Returns {:ok, response_data} or {:error, reason}
  """
  def dispatch(intent, slots, context \\ %{})
  
  def dispatch("weather.query", %{location: location} = slots, context) do
    # First geocode the location if needed
    with {:ok, coordinates} <- Geocoding.resolve(location),
         {:ok, weather} <- Weather.get_current(coordinates, slots) do
      {:ok, format_weather_response(weather, location)}
    end
  end
  
  def dispatch("news.query", slots, _context) do
    News.fetch(slots)
  end
  
  def dispatch(_intent, _slots, _context) do
    {:error, :no_service_handler}
  end
  
  defp format_weather_response(weather, location) do
    %{
      type: :weather,
      location: location,
      temperature: weather.temp,
      condition: weather.condition,
      humidity: weather.humidity,
      text: "It's currently #{weather.temp}°F and #{weather.condition} in #{location}."
    }
  end
end
```

### Option B: HTTP API Endpoints

Expose REST endpoints for external service integration.

**Router additions** (`lib/chat_bot_web/router.ex`):

```elixir
scope "/api/v1", ChatBotWeb.API do
  pipe_through :api
  
  # Conversation endpoints
  post "/conversations", ConversationController, :create
  post "/conversations/:id/messages", ConversationController, :message
  get "/conversations/:id", ConversationController, :show
  delete "/conversations/:id", ConversationController, :delete
  
  # Service endpoints (for external integrations)
  post "/services/weather", ServiceController, :weather
  post "/services/geocode", ServiceController, :geocode
  
  # Webhook endpoints (for receiving data from external services)
  post "/webhooks/weather", WebhookController, :weather_update
end
```

---

## Service Implementation Examples

### Weather Service

**File:** `lib/chat_bot/services/weather.ex`

```elixir
defmodule ChatBot.Services.Weather do
  @moduledoc """
  Weather service integration.
  Supports multiple providers: OpenWeatherMap, WeatherAPI, etc.
  """
  
  require Logger
  
  @provider Application.compile_env(:chat_bot, [:services, :weather, :provider], :openweathermap)
  @api_key Application.compile_env(:chat_bot, [:services, :weather, :api_key])
  @base_url "https://api.openweathermap.org/data/2.5"
  
  @type coordinates :: {float(), float()}
  @type weather_data :: %{
    temp: float(),
    feels_like: float(),
    humidity: integer(),
    condition: String.t(),
    description: String.t(),
    wind_speed: float(),
    icon: String.t()
  }
  
  @doc """
  Get current weather for coordinates.
  """
  @spec get_current(coordinates(), map()) :: {:ok, weather_data()} | {:error, term()}
  def get_current({lat, lon}, opts \\ %{}) do
    units = Map.get(opts, :units, "imperial")  # imperial, metric, kelvin
    
    url = "#{@base_url}/weather?lat=#{lat}&lon=#{lon}&units=#{units}&appid=#{@api_key}"
    
    case http_get(url) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, parse_weather_response(body)}
        
      {:ok, %{status: status}} ->
        Logger.warning("Weather API returned #{status}")
        {:error, {:api_error, status}}
        
      {:error, reason} ->
        Logger.error("Weather API request failed: #{inspect(reason)}")
        {:error, reason}
    end
  end
  
  @doc """
  Get weather forecast for coordinates.
  """
  @spec get_forecast(coordinates(), map()) :: {:ok, list(weather_data())} | {:error, term()}
  def get_forecast({lat, lon}, opts \\ %{}) do
    units = Map.get(opts, :units, "imperial")
    days = Map.get(opts, :days, 5)
    
    url = "#{@base_url}/forecast?lat=#{lat}&lon=#{lon}&units=#{units}&cnt=#{days * 8}&appid=#{@api_key}"
    
    case http_get(url) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, parse_forecast_response(body)}
        
      {:ok, %{status: status}} ->
        {:error, {:api_error, status}}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  # Private functions
  
  defp http_get(url) do
    # Using Req (add {:req, "~> 0.4"} to mix.exs dependencies)
    Req.get(url)
  end
  
  defp parse_weather_response(body) do
    %{
      temp: body["main"]["temp"],
      feels_like: body["main"]["feels_like"],
      humidity: body["main"]["humidity"],
      condition: body["weather"] |> List.first() |> Map.get("main"),
      description: body["weather"] |> List.first() |> Map.get("description"),
      wind_speed: body["wind"]["speed"],
      icon: body["weather"] |> List.first() |> Map.get("icon")
    }
  end
  
  defp parse_forecast_response(body) do
    body["list"]
    |> Enum.map(&parse_weather_response/1)
  end
end
```

### Geocoding Service

**File:** `lib/chat_bot/services/geocoding.ex`

```elixir
defmodule ChatBot.Services.Geocoding do
  @moduledoc """
  Geocoding service for resolving location names to coordinates.
  Supports caching to reduce API calls.
  """
  
  require Logger
  
  @cache_ttl :timer.hours(24)
  @api_key Application.compile_env(:chat_bot, [:services, :geocoding, :api_key])
  @base_url "https://api.openweathermap.org/geo/1.0"
  
  @type coordinates :: {float(), float()}
  @type location_info :: %{
    name: String.t(),
    country: String.t(),
    state: String.t() | nil,
    lat: float(),
    lon: float()
  }
  
  @doc """
  Resolve a location name to coordinates.
  Uses cache when available.
  """
  @spec resolve(String.t()) :: {:ok, coordinates()} | {:error, term()}
  def resolve(location) when is_binary(location) do
    cache_key = location_cache_key(location)
    
    case get_cached(cache_key) do
      {:ok, coords} ->
        {:ok, coords}
        
      :miss ->
        case geocode_location(location) do
          {:ok, info} ->
            coords = {info.lat, info.lon}
            cache_result(cache_key, coords)
            {:ok, coords}
            
          error ->
            error
        end
    end
  end
  
  @doc """
  Get full location information including country, state, etc.
  """
  @spec get_location_info(String.t()) :: {:ok, location_info()} | {:error, term()}
  def get_location_info(location) do
    geocode_location(location)
  end
  
  @doc """
  Reverse geocode: coordinates to location name.
  """
  @spec reverse({float(), float()}) :: {:ok, location_info()} | {:error, term()}
  def reverse({lat, lon}) do
    url = "#{@base_url}/reverse?lat=#{lat}&lon=#{lon}&limit=1&appid=#{@api_key}"
    
    case http_get(url) do
      {:ok, %{status: 200, body: [first | _]}} ->
        {:ok, parse_location(first)}
        
      {:ok, %{status: 200, body: []}} ->
        {:error, :location_not_found}
        
      {:ok, %{status: status}} ->
        {:error, {:api_error, status}}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  # Private functions
  
  defp geocode_location(location) do
    encoded = URI.encode(location)
    url = "#{@base_url}/direct?q=#{encoded}&limit=1&appid=#{@api_key}"
    
    case http_get(url) do
      {:ok, %{status: 200, body: [first | _]}} ->
        {:ok, parse_location(first)}
        
      {:ok, %{status: 200, body: []}} ->
        {:error, :location_not_found}
        
      {:ok, %{status: status}} ->
        {:error, {:api_error, status}}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp parse_location(data) do
    %{
      name: data["name"],
      country: data["country"],
      state: data["state"],
      lat: data["lat"],
      lon: data["lon"]
    }
  end
  
  defp http_get(url) do
    Req.get(url)
  end
  
  # Simple ETS-based cache (or use Cachex for production)
  
  defp location_cache_key(location) do
    {:geocode, String.downcase(String.trim(location))}
  end
  
  defp get_cached(key) do
    case :ets.lookup(:geocode_cache, key) do
      [{^key, value, expires_at}] when expires_at > System.system_time(:millisecond) ->
        {:ok, value}
      _ ->
        :miss
    end
  end
  
  defp cache_result(key, value) do
    expires_at = System.system_time(:millisecond) + @cache_ttl
    :ets.insert(:geocode_cache, {key, value, expires_at})
  end
end
```

---

## Configuration

Add to `config/config.exs`:

```elixir
config :chat_bot, :services,
  weather: [
    provider: :openweathermap,
    api_key: System.get_env("OPENWEATHERMAP_API_KEY"),
    default_units: "imperial"
  ],
  geocoding: [
    provider: :openweathermap,
    api_key: System.get_env("OPENWEATHERMAP_API_KEY"),
    cache_ttl: :timer.hours(24)
  ],
  news: [
    provider: :newsapi,
    api_key: System.get_env("NEWSAPI_KEY")
  ]
```

Add to `config/runtime.exs` for production:

```elixir
config :chat_bot, :services,
  weather: [
    api_key: System.fetch_env!("OPENWEATHERMAP_API_KEY")
  ],
  geocoding: [
    api_key: System.fetch_env!("OPENWEATHERMAP_API_KEY")
  ]
```

---

## Hooking Into the Brain

Modify the Brain's response generation to use the service dispatcher.

**In `lib/chat_bot/brain.ex`**, add service dispatch to the evaluation flow:

```elixir
defp generate_response_for_intent(intent, slots, analysis, state) do
  # Try external service first
  case ChatBot.Services.Dispatcher.dispatch(intent, slots, %{persona: state.persona}) do
    {:ok, service_response} ->
      # Service handled it - use the response
      {:ok, service_response.text}
      
    {:error, :no_service_handler} ->
      # No service for this intent - use template-based response
      generate_template_response(intent, slots, analysis, state)
      
    {:error, reason} ->
      # Service failed - generate fallback
      Logger.warning("Service dispatch failed: #{inspect(reason)}")
      {:ok, "I'm having trouble getting that information right now. Please try again later."}
  end
end
```

---

## API Reference

### Brain API (Internal Elixir)

```elixir
# Start a conversation
{:ok, conversation_id} = ChatBot.Brain.create_conversation()

# Send a message
{:ok, response} = ChatBot.Brain.evaluate(conversation_id, "What's the weather in NYC?")

# End conversation
:ok = ChatBot.Brain.end_conversation(conversation_id)

# Get conversation history
{:ok, conversation} = ChatBot.Brain.get_conversation(conversation_id)
```

### REST API (HTTP)

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/api/v1/conversations` | Create conversation | - | `{conversation_id}` |
| `POST` | `/api/v1/conversations/:id/messages` | Send message | `{input: "..."}` | `{response, intent, entities}` |
| `GET` | `/api/v1/conversations/:id` | Get conversation | - | `{messages, created_at}` |
| `DELETE` | `/api/v1/conversations/:id` | End conversation | - | `{status: "ok"}` |
| `POST` | `/api/v1/services/weather` | Get weather | `{location, units?}` | `{weather_data}` |
| `POST` | `/api/v1/services/geocode` | Geocode location | `{location}` | `{lat, lon, info}` |

---

## Adding a New Service

1. **Create service module** in `lib/chat_bot/services/`
2. **Add intent/slots** to `priv/analysis/slot_schemas.json` if needed
3. **Add training data** to `data/intents/` for the intent
4. **Add entity types** to `data/entities/` for new entity types
5. **Add dispatch handler** in `ChatBot.Services.Dispatcher`
6. **Add config** to `config/config.exs`
7. **Re-train models**: `mix train_models`

### Example: Adding a Stock Price Service

```elixir
# 1. lib/chat_bot/services/stocks.ex
defmodule ChatBot.Services.Stocks do
  def get_price(symbol) do
    # Implementation
  end
end

# 2. Add to slot_schemas.json
# "stocks.query": {
#   "required": ["symbol"],
#   "optional": ["exchange"],
#   "clarification_templates": {
#     "symbol": "Which stock symbol would you like to look up?"
#   }
# }

# 3. Add to dispatcher.ex
def dispatch("stocks.query", %{symbol: symbol}, _context) do
  Stocks.get_price(symbol)
end
```

---

## Error Handling

Services should return standardized errors:

```elixir
{:error, :location_not_found}      # Location couldn't be geocoded
{:error, :service_unavailable}     # External API is down
{:error, :rate_limited}            # API rate limit exceeded
{:error, :invalid_api_key}         # API key issue
{:error, {:api_error, status}}     # HTTP error with status code
```

The dispatcher handles these and provides user-friendly responses:

```elixir
defp handle_service_error(:location_not_found, context) do
  "I couldn't find that location. Could you be more specific?"
end

defp handle_service_error(:service_unavailable, _context) do
  "I'm having trouble reaching the weather service. Please try again in a moment."
end

defp handle_service_error(:rate_limited, _context) do
  "I've made too many requests. Please wait a minute and try again."
end
```

---

## Testing

```elixir
# test/chat_bot/services/weather_test.exs
defmodule ChatBot.Services.WeatherTest do
  use ExUnit.Case, async: true
  
  import Mox
  
  setup :verify_on_exit!
  
  describe "get_current/2" do
    test "returns weather data for valid coordinates" do
      expect(ChatBot.HTTPMock, :get, fn url ->
        assert url =~ "lat=40.7128"
        {:ok, %{status: 200, body: weather_fixture()}}
      end)
      
      assert {:ok, weather} = ChatBot.Services.Weather.get_current({40.7128, -74.0060})
      assert weather.temp == 72.5
      assert weather.condition == "Clear"
    end
  end
end
```

---

## Next Steps

1. [ ] Add `{:req, "~> 0.4"}` to `mix.exs` for HTTP client
2. [ ] Create `lib/chat_bot/services/` directory
3. [ ] Implement `Weather` and `Geocoding` services
4. [ ] Create `Dispatcher` module
5. [ ] Hook dispatcher into Brain
6. [ ] Add API controllers for REST endpoints
7. [ ] Add caching layer (ETS or Cachex)
8. [ ] Set up API keys in environment
9. [ ] Write tests with mocked HTTP responses
