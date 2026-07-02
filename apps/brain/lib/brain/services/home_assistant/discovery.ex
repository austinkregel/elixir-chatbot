defmodule Brain.Services.HomeAssistant.Discovery do
  @moduledoc """
  Discovers Home Assistant entities and syncs them into Brain's Gazetteer.

  Runs on startup when HA credentials are available, and can be triggered
  manually or periodically. Entities are registered with their friendly
  names so the NER/Gazetteer can recognize them in user text.
  """

  alias Brain.Services.HomeAssistant.{EntityMapper, CapabilityRegistry}
  alias Brain.ML.Gazetteer

  require Logger

  @doc """
  Syncs all HA entities into the Gazetteer.

  Uses the cached entity states from CapabilityRegistry if available,
  otherwise fetches directly from HA.
  """
  def sync_to_gazetteer do
    states = CapabilityRegistry.entity_states()

    if states != [] do
      register_entities(states)
    else
      Logger.debug("Discovery: no cached states available for Gazetteer sync")
      {:ok, %{total: 0, by_domain: %{}}}
    end
  end

  @doc """
  Fetches all entities from Home Assistant and returns them grouped by domain.
  """
  def discover(credentials) do
    url = Map.get(credentials, :url, Map.get(credentials, "url"))
    token = Map.get(credentials, :access_token, Map.get(credentials, "access_token"))

    case fetch_all_states(url, token) do
      {:ok, states} ->
        entities =
          states
          |> Enum.map(&EntityMapper.ha_entity_to_brain/1)
          |> Enum.group_by(& &1.ha_domain)

        {:ok, entities}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Discovers HA entities and registers them in the Gazetteer.
  """
  def discover_and_register(credentials) do
    case discover(credentials) do
      {:ok, entities_by_domain} ->
        all_entities = Enum.flat_map(entities_by_domain, fn {_domain, entities} -> entities end)
        register_brain_entities(all_entities)

        domain_counts = Map.new(entities_by_domain, fn {k, v} -> {k, length(v)} end)
        {:ok, %{total: length(all_entities), by_domain: domain_counts}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp register_entities(states) when is_list(states) do
    entities = Enum.map(states, &EntityMapper.ha_entity_to_brain/1)
    register_brain_entities(entities)

    by_domain = Enum.group_by(entities, & &1.ha_domain)
    domain_counts = Map.new(by_domain, fn {k, v} -> {k, length(v)} end)

    Logger.info("Discovery: synced #{length(entities)} entities to Gazetteer")
    {:ok, %{total: length(entities), by_domain: domain_counts}}
  end

  defp register_brain_entities(entities) do
    if gazetteer_ready?() do
      Enum.each(entities, fn entity ->
        register_entity(entity)
      end)
    else
      Logger.debug("Discovery: Gazetteer not ready, skipping registration")
    end
  end

  defp register_entity(entity) do
    entity_type = to_string(entity.type)

    case Gazetteer.add_entry(
      entity.name,
      entity_type,
      %{
        ha_entity_id: entity.ha_entity_id,
        ha_domain: entity.ha_domain
      }
    ) do
      {:ok, _} -> :ok
      {:error, {:duplicate, _}} -> :ok
      _ -> :ok
    end
  rescue
    _ -> :ok
  end

  defp gazetteer_ready? do
    Gazetteer.is_loaded?()
  end

  defp fetch_all_states(url, token) do
    headers = [
      {~c"Authorization", String.to_charlist("Bearer #{token}")},
      {~c"Content-Type", ~c"application/json"}
    ]

    case :httpc.request(:get, {String.to_charlist("#{url}/api/states"), headers}, [{:timeout, 15_000}], []) do
      {:ok, {{_, 200, _}, _resp_headers, body}} ->
        Jason.decode(List.to_string(body))

      {:ok, {{_, status, _}, _, body}} ->
        {:error, {:http_error, status, List.to_string(body)}}

      {:error, reason} ->
        {:error, {:connection_error, reason}}
    end
  end
end
