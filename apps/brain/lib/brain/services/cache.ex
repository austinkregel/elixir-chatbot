defmodule Brain.Services.Cache do
  @moduledoc """
  ETS-based cache for external service responses.

  Provides TTL-based caching to reduce API calls and improve response times.
  Different cache durations are used for different data types:
  - Geocoding: 24 hours (locations don't change)
  - Weather: 10 minutes (frequently updated)
  - News: 5 minutes (breaking news)

  ## Usage

      # Store with default TTL
      Cache.put(:weather, "NYC:current", weather_data)

      # Store with custom TTL (in milliseconds)
      Cache.put(:geocoding, "NYC", coordinates, ttl: :timer.hours(24))

      # Retrieve
      case Cache.get(:weather, "NYC:current") do
        {:ok, data} -> data
        :miss -> fetch_fresh_data()
      end

      # Delete
      Cache.delete(:weather, "NYC:current")

      # Clear all entries for a service
      Cache.clear(:weather)
  """

  use GenServer
  require Logger

  @table_name :service_cache

  # Default TTLs per service type (in milliseconds)
  @default_ttls %{
    geocoding: :timer.hours(24),
    weather: :timer.minutes(10),
    news: :timer.minutes(5),
    default: :timer.minutes(15)
  }

  # Cleanup interval (every 5 minutes)
  @cleanup_interval :timer.minutes(5)

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Store a value in the cache.

  ## Parameters
    - service: Service identifier (atom)
    - key: Cache key (string or any term)
    - value: Value to cache
    - opts: Optional keyword list with :ttl (in milliseconds)
  """
  @spec put(atom(), term(), term(), keyword()) :: :ok
  def put(service, key, value, opts \\ []) do
    ttl = Keyword.get(opts, :ttl, get_default_ttl(service))
    expires_at = System.system_time(:millisecond) + ttl

    cache_key = {service, key}
    :ets.insert(@table_name, {cache_key, value, expires_at})
    :ok
  end

  @doc """
  Retrieve a value from the cache.

  ## Returns
    - {:ok, value} if found and not expired
    - :miss if not found or expired
  """
  @spec get(atom(), term()) :: {:ok, term()} | :miss
  def get(service, key) do
    cache_key = {service, key}
    now = System.system_time(:millisecond)

    case :ets.lookup(@table_name, cache_key) do
      [{^cache_key, value, expires_at}] when expires_at > now ->
        # Emit cache hit telemetry
        Brain.Telemetry.emit_service_cache_hit(service, key)
        {:ok, value}

      [{^cache_key, _value, _expires_at}] ->
        # Expired, delete it
        :ets.delete(@table_name, cache_key)
        # Emit cache miss telemetry
        Brain.Telemetry.emit_service_cache_miss(service, key)
        :miss

      [] ->
        # Emit cache miss telemetry
        Brain.Telemetry.emit_service_cache_miss(service, key)
        :miss
    end
  end

  @doc """
  Delete a specific entry.
  """
  @spec delete(atom(), term()) :: :ok
  def delete(service, key) do
    cache_key = {service, key}
    :ets.delete(@table_name, cache_key)
    :ok
  end

  @doc """
  Clear all entries for a service.
  """
  @spec clear(atom()) :: :ok
  def clear(service) do
    :ets.match_delete(@table_name, {{service, :_}, :_, :_})
    :ok
  end

  @doc """
  Clear all cached entries.
  """
  @spec clear_all() :: :ok
  def clear_all do
    :ets.delete_all_objects(@table_name)
    :ok
  end

  @doc """
  Get cache statistics.
  """
  @spec stats() :: map()
  def stats do
    now = System.system_time(:millisecond)

    entries = :ets.tab2list(@table_name)
    total = length(entries)

    {valid, expired} =
      Enum.reduce(entries, {0, 0}, fn {_, _, expires_at}, {v, e} ->
        if expires_at > now, do: {v + 1, e}, else: {v, e + 1}
      end)

    by_service =
      entries
      |> Enum.group_by(fn {{service, _}, _, _} -> service end)
      |> Enum.map(fn {service, items} -> {service, length(items)} end)
      |> Map.new()

    %{
      total: total,
      valid: valid,
      expired: expired,
      by_service: by_service
    }
  end

  @doc """
  Check if the cache is ready.
  """
  @spec ready?() :: boolean()
  def ready? do
    :ets.whereis(@table_name) != :undefined
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    table = :ets.new(@table_name, [:set, :public, :named_table, read_concurrency: true])

    # Schedule periodic cleanup
    schedule_cleanup()

    Logger.info("Service cache initialized")
    {:ok, %{table: table}}
  end

  @impl true
  def handle_info(:cleanup, state) do
    cleanup_expired()
    schedule_cleanup()
    {:noreply, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_default_ttl(service) do
    Map.get(@default_ttls, service, @default_ttls.default)
  end

  defp schedule_cleanup do
    Process.send_after(self(), :cleanup, @cleanup_interval)
  end

  defp cleanup_expired do
    now = System.system_time(:millisecond)

    # Find and delete expired entries
    # Using match_spec for efficient bulk deletion
    expired_count =
      :ets.select_delete(@table_name, [
        {{:_, :_, :"$1"}, [{:<, :"$1", now}], [true]}
      ])

    if expired_count > 0 do
      Logger.debug("Cache cleanup", expired_entries: expired_count)
    end

    expired_count
  end
end
