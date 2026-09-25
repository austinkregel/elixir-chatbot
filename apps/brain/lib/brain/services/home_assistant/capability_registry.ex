defmodule Brain.Services.HomeAssistant.CapabilityRegistry do
  @moduledoc """
  Runtime capability registry for Home Assistant.

  Discovers available services and entities from the HA REST API and
  caches them in ETS for fast lookup. Refreshes periodically.

  This replaces hardcoded intent/service mappings by learning what
  the connected HA instance actually supports at runtime.
  """

  use GenServer

  require Logger

  alias Brain.Services.CredentialVault

  @ets_table :ha_capability_registry
  @refresh_interval_ms :timer.minutes(5)
  @service_name :home_assistant

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Returns all discovered HA domains with their services."
  def available_services do
    case :ets.lookup(@ets_table, :services) do
      [{:services, services}] -> services
      _ -> %{}
    end
  end

  @doc "Returns all discovered entity IDs grouped by HA domain."
  def entities_by_domain do
    case :ets.lookup(@ets_table, :entities_by_domain) do
      [{:entities_by_domain, entities}] -> entities
      _ -> %{}
    end
  end

  @doc "Returns the full entity state list from the last sync."
  def entity_states do
    case :ets.lookup(@ets_table, :entity_states) do
      [{:entity_states, states}] -> states
      _ -> []
    end
  end

  @doc "Checks if a given HA domain/service combination is available."
  def can_handle?(ha_domain, ha_service) do
    services = available_services()
    domain_services = Map.get(services, ha_domain, %{})
    Map.has_key?(domain_services, ha_service)
  end

  @doc "Returns field names for a given HA service."
  def service_fields(ha_domain, ha_service) do
    services = available_services()
    domain_services = Map.get(services, ha_domain, %{})
    service_info = Map.get(domain_services, ha_service, %{})
    fields = Map.get(service_info, "fields", %{})
    Map.keys(fields)
  end

  @doc "Returns entity IDs for a given HA domain prefix."
  def domain_entities(ha_domain) do
    entities = entities_by_domain()
    Map.get(entities, ha_domain, [])
  end

  @doc "Finds the HA domain for a given entity_id."
  def entity_domain(entity_id) when is_binary(entity_id) do
    case String.split(entity_id, ".", parts: 2) do
      [domain, _] -> domain
      _ -> nil
    end
  end

  @doc """
  Finds HA entities whose friendly_name or object_id relates to the given area name.

  This is a registry query, not NLP -- the NER system has already identified
  the area name. We're asking HA's own data "what devices exist here?"

  Returns a list of `%{entity_id: ..., friendly_name: ..., ha_domain: ..., state: ...}`
  sorted with controllable domains (light, switch, fan, etc.) first.
  """
  def find_entities_for_area(area_name) when is_binary(area_name) do
    normalized = String.downcase(area_name)
    states = entity_states()

    states
    |> Enum.filter(fn state ->
      entity_id = Map.get(state, "entity_id", "")
      friendly = String.downcase(get_in(state, ["attributes", "friendly_name"]) || "")
      object_id = entity_id |> String.split(".", parts: 2) |> List.last() |> String.replace("_", " ")

      String.downcase(object_id) == normalized or
        friendly == normalized or
        String.starts_with?(String.downcase(object_id), normalized <> " ") or
        String.starts_with?(friendly, normalized <> " ")
    end)
    |> Enum.map(fn state ->
      entity_id = Map.get(state, "entity_id", "")
      [domain | _] = String.split(entity_id, ".", parts: 2)

      %{
        entity_id: entity_id,
        friendly_name: get_in(state, ["attributes", "friendly_name"]) || entity_id,
        ha_domain: domain,
        state: Map.get(state, "state", "unknown")
      }
    end)
    |> Enum.sort_by(fn %{ha_domain: domain} ->
      domain_priority(domain)
    end)
  end

  def find_entities_for_area(_), do: []

  defp domain_priority("light"), do: 0
  defp domain_priority("switch"), do: 1
  defp domain_priority("fan"), do: 2
  defp domain_priority("cover"), do: 3
  defp domain_priority("climate"), do: 4
  defp domain_priority("media_player"), do: 5
  defp domain_priority("lock"), do: 6
  defp domain_priority("vacuum"), do: 7
  defp domain_priority(_), do: 99

  @doc "Forces an immediate refresh of capabilities."
  def refresh do
    GenServer.cast(__MODULE__, :refresh)
  end

  # GenServer callbacks

  @impl true
  def init(_opts) do
    :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
    :ets.insert(@ets_table, {:ready, false})

    send(self(), :initial_sync)
    {:ok, %{refresh_timer: nil}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    ready = case :ets.lookup(@ets_table, :ready) do
      [{:ready, val}] -> val
      _ -> false
    end
    {:reply, ready, state}
  end

  @impl true
  def handle_info(:initial_sync, state) do
    state = do_sync(state)
    timer = Process.send_after(self(), :periodic_refresh, @refresh_interval_ms)
    {:noreply, %{state | refresh_timer: timer}}
  end

  @impl true
  def handle_info(:periodic_refresh, state) do
    state = do_sync(state)
    timer = Process.send_after(self(), :periodic_refresh, @refresh_interval_ms)
    {:noreply, %{state | refresh_timer: timer}}
  end

  @impl true
  def handle_cast(:refresh, state) do
    state = do_sync(state)
    {:noreply, state}
  end

  defp do_sync(state) do
    case fetch_credentials() do
      {:ok, url, token} ->
        sync_services(url, token)
        sync_entities(url, token)
        :ets.insert(@ets_table, {:ready, true})
        Logger.info("CapabilityRegistry: sync complete")

        spawn(fn ->
          wait_for_gazetteer()
          Brain.Services.HomeAssistant.Discovery.sync_to_gazetteer()
        end)

      :no_credentials ->
        Logger.debug("CapabilityRegistry: no HA credentials configured, skipping sync")
    end

    state
  end

  defp fetch_credentials do
    with {:ok, url} <- CredentialVault.get(@service_name, :url),
         {:ok, token} <- CredentialVault.get(@service_name, :access_token) do
      {:ok, url, token}
    else
      _ -> :no_credentials
    end
  end

  defp sync_services(url, token) do
    case http_get("#{url}/api/services", token) do
      {:ok, services_list} when is_list(services_list) ->
        services_map =
          Enum.reduce(services_list, %{}, fn entry, acc ->
            domain = Map.get(entry, "domain", "")
            domain_services = Map.get(entry, "services", %{})

            service_map =
              if is_map(domain_services) do
                domain_services
              else
                Map.new(List.wrap(domain_services), fn s -> {s, %{}} end)
              end

            Map.put(acc, domain, service_map)
          end)

        :ets.insert(@ets_table, {:services, services_map})
        Logger.debug("CapabilityRegistry: discovered #{map_size(services_map)} HA domains")

      {:ok, _} ->
        Logger.warning("CapabilityRegistry: unexpected format from /api/services")

      {:error, reason} ->
        Logger.warning("CapabilityRegistry: failed to fetch services: #{inspect(reason)}")
    end
  end

  defp sync_entities(url, token) do
    case http_get("#{url}/api/states", token) do
      {:ok, states} when is_list(states) ->
        :ets.insert(@ets_table, {:entity_states, states})

        by_domain =
          Enum.group_by(states, fn state ->
            entity_id = Map.get(state, "entity_id", "")
            case String.split(entity_id, ".", parts: 2) do
              [domain, _] -> domain
              _ -> "unknown"
            end
          end)
          |> Map.new(fn {domain, entities} ->
            {domain, Enum.map(entities, &Map.get(&1, "entity_id", ""))}
          end)

        :ets.insert(@ets_table, {:entities_by_domain, by_domain})
        Logger.debug("CapabilityRegistry: discovered #{length(states)} entities across #{map_size(by_domain)} domains")

      {:ok, _} ->
        Logger.warning("CapabilityRegistry: unexpected format from /api/states")

      {:error, reason} ->
        Logger.warning("CapabilityRegistry: failed to fetch states: #{inspect(reason)}")
    end
  end

  defp wait_for_gazetteer do
    wait_for_gazetteer(0)
  end

  defp wait_for_gazetteer(attempts) when attempts > 30 do
    Logger.warning("CapabilityRegistry: gave up waiting for Gazetteer after 30s")
  end

  defp wait_for_gazetteer(attempts) do
    if Brain.ML.Gazetteer.is_loaded?() do
      :ok
    else
      Process.sleep(1_000)
      wait_for_gazetteer(attempts + 1)
    end
  end

  defp http_get(url, token) do
    headers = [
      {~c"Authorization", String.to_charlist("Bearer #{token}")},
      {~c"Content-Type", ~c"application/json"}
    ]

    case :httpc.request(:get, {String.to_charlist(url), headers}, [{:timeout, 15_000}], []) do
      {:ok, {{_, 200, _}, _resp_headers, body}} ->
        Jason.decode(List.to_string(body))

      {:ok, {{_, status, _}, _, body}} ->
        {:error, {:http_error, status, List.to_string(body)}}

      {:error, reason} ->
        {:error, {:connection_error, reason}}
    end
  end
end
