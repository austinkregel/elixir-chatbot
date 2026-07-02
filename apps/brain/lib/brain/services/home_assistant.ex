defmodule Brain.Services.HomeAssistant do
  @moduledoc """
  Home Assistant integration via REST API.

  Self-describing service that discovers capabilities from the connected
  HA instance at runtime. No hardcoded intent lists, entity types, or
  service mappings -- all resolved dynamically from:

  - `/api/services` -- available HA domains and service calls
  - `/api/states` -- entity states and attributes
  - `priv/services/ha_action_verbs.json` -- maps classifier suffixes to HA service names
  - `priv/services/ha_domain_types.json` -- maps HA domains to Brain entity types

  The CapabilityRegistry handles discovery and caching.
  """

  @behaviour Brain.Services.Service

  alias Brain.Services.Cache
  alias Brain.Services.HomeAssistant.{CapabilityRegistry, EntityMapper, StateFormatter}

  require Logger

  @required_credentials [:url, :access_token]
  @cache_ttl_ms 60_000

  @impl true
  def name, do: :home_assistant

  @impl true
  def display_name, do: "Home Assistant"

  @impl true
  def description, do: "Smart home device control and queries via Home Assistant REST API"

  @impl true
  def required_credentials, do: @required_credentials

  @impl true
  def supported_intents, do: []

  @impl true
  def supported_domains do
    ["smarthome", "device", "music", "alarm", "calendar", "timer", "reminder"]
  end

  @impl true
  def provides_fields do
    [:device, :device_status, :action, :summary]
  end

  @impl true
  def slot_schema do
    %{
      "required" => [],
      "optional" => ["device", "location", "brightness", "color", "temperature", "volume", "duration", "time"],
      "entity_mappings" => %{
        "device" => ["device", "appliance", "light", "switch"],
        "location" => ["room", "location", "area"],
        "brightness" => ["percentage", "level", "number"],
        "color" => ["color"],
        "temperature" => ["temperature", "number", "degree"],
        "volume" => ["percentage", "level", "number"],
        "duration" => ["duration", "time"],
        "time" => ["time", "date", "temporal"]
      },
      "clarification_templates" => %{
        "device" => "Which device would you like to control?"
      }
    }
  end

  @impl true
  def health_check(credentials) do
    url = Map.get(credentials, :url, Map.get(credentials, "url"))
    token = Map.get(credentials, :access_token, Map.get(credentials, "access_token"))

    case http_get("#{url}/api/config", token) do
      {:ok, _} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @impl true
  def enrich(intent, slots, credentials) do
    url = Map.get(credentials, :url, Map.get(credentials, "url"))
    token = Map.get(credentials, :access_token, Map.get(credentials, "access_token"))

    Logger.info("HomeAssistant.enrich called: intent=#{inspect(intent)} slots=#{inspect(slots)} url=#{is_binary(url) and url != ""} token=#{is_binary(token) and token != ""}")

    entity_id = EntityMapper.resolve_entity_id(slots)
    action_suffix = extract_action_suffix(intent)
    ha_services = resolve_ha_services(action_suffix)

    Logger.info("HomeAssistant.enrich resolved: entity_id=#{inspect(entity_id)} action_suffix=#{inspect(action_suffix)} ha_services=#{inspect(ha_services)}")

    result = if ha_services do
      handle_action(entity_id, ha_services, slots, url, token, intent, action_suffix)
    else
      handle_query(entity_id, url, token, intent)
    end

    Logger.info("HomeAssistant.enrich result: #{inspect(result)}")
    result
  end

  defp extract_action_suffix(intent) do
    intent_str = to_string(intent)
    action_verbs = load_action_verbs()

    parts = String.split(intent_str, ".", parts: 2)
    suffix = case parts do
      [_domain, rest] -> rest
      _ -> intent_str
    end

    cond do
      Map.has_key?(action_verbs, suffix) ->
        suffix

      Map.has_key?(action_verbs, last_segment(suffix)) ->
        last_segment(suffix)

      true ->
        suffix
    end
  end

  defp last_segment(s) do
    s |> String.split(".") |> List.last()
  end

  defp resolve_ha_services(action_suffix) do
    action_verbs = load_action_verbs()

    case Map.get(action_verbs, action_suffix) do
      nil -> nil
      services when is_list(services) -> services
    end
  end

  defp handle_action(entity_id, ha_services, slots, url, token, intent, action_suffix) do
    ha_domain = determine_ha_domain(entity_id, intent)
    ha_service = pick_service_with_polarity(ha_domain, ha_services, slots, action_suffix)

    service_data =
      slots
      |> EntityMapper.build_service_data()
      |> maybe_add_entity_id(entity_id)

    Logger.info("HomeAssistant: calling #{ha_domain}/#{ha_service} entity_id=#{inspect(entity_id)} service_data=#{inspect(service_data)}")

    case call_service(url, token, ha_domain, ha_service, service_data) do
      {:ok, result} ->
        device_name = extract_device_name(slots, entity_id)
        enrichment = StateFormatter.format_action_result(ha_service, device_name)
        Logger.info("HomeAssistant: action succeeded device=#{inspect(device_name)} result_size=#{if is_list(result), do: length(result), else: "map"}")
        {:ok, enrichment}

      {:error, reason} ->
        Logger.warning("HomeAssistant: action FAILED reason=#{inspect(reason)}")
        {:error, reason}
    end
  end

  defp handle_query(entity_id, url, token, intent) do
    if entity_id do
      cache_key = "state:#{entity_id}"

      state =
        case Cache.get(:home_assistant, cache_key) do
          {:ok, cached} -> cached
          :miss ->
            case get_entity_state(url, token, entity_id) do
              {:ok, state} ->
                Cache.put(:home_assistant, cache_key, state, ttl: @cache_ttl_ms)
                state
              {:error, _} -> nil
            end
        end

      if state do
        {:ok, StateFormatter.format_state(state)}
      else
        {:error, :entity_not_found}
      end
    else
      ha_domain = infer_ha_domain_from_intent(intent)

      case get_domain_states(url, token, ha_domain) do
        {:ok, states} -> {:ok, StateFormatter.format_domain_states(states)}
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp determine_ha_domain(entity_id, intent) when is_binary(entity_id) do
    case String.split(entity_id, ".", parts: 2) do
      [domain, _] -> domain
      _ -> infer_ha_domain_from_intent(intent)
    end
  end

  defp determine_ha_domain(_, intent), do: infer_ha_domain_from_intent(intent)

  defp infer_ha_domain_from_intent(intent) do
    intent_str = to_string(intent)

    cond do
      String.starts_with?(intent_str, "smarthome.") -> "homeassistant"
      String.starts_with?(intent_str, "device.") -> "homeassistant"
      String.starts_with?(intent_str, "music.") -> "media_player"
      String.starts_with?(intent_str, "alarm.") -> "automation"
      String.starts_with?(intent_str, "calendar.") -> "calendar"
      String.starts_with?(intent_str, "timer.") -> "timer"
      String.starts_with?(intent_str, "reminder.") -> "automation"
      true -> "homeassistant"
    end
  end

  defp pick_service_with_polarity(ha_domain, ha_services, slots, action_suffix) do
    polarity = infer_polarity(slots, action_suffix)

    preferred = case polarity do
      :off -> Enum.find(ha_services, fn s -> String.contains?(s, "off") or String.contains?(s, "stop") or String.contains?(s, "pause") end)
      :on -> Enum.find(ha_services, fn s -> String.contains?(s, "on") or String.contains?(s, "play") or String.contains?(s, "start") end)
      :toggle -> Enum.find(ha_services, fn s -> s == "toggle" end)
      _ -> nil
    end

    service = preferred || Enum.find(ha_services, List.first(ha_services), fn service ->
      CapabilityRegistry.can_handle?(ha_domain, service)
    end)

    if CapabilityRegistry.can_handle?(ha_domain, service), do: service, else: List.first(ha_services)
  end

  defp infer_polarity(slots, action_suffix) do
    action_val = Map.get(slots, :action) || Map.get(slots, "action") || ""
    query_text = Map.get(slots, :_query_text) || Map.get(slots, "_query_text") || ""
    text = String.downcase(to_string(action_val) <> " " <> to_string(query_text))

    cond do
      String.contains?(text, "turn off") or String.contains?(text, "shut off") or
        String.contains?(text, "switch off") or String.contains?(text, "disable") -> :off
      String.contains?(text, "turn on") or String.contains?(text, "switch on") or
        String.contains?(text, "enable") -> :on
      action_suffix in ["device_down", "pause", "stop"] -> :off
      action_suffix in ["device_up", "play", "device_set"] -> :on
      true -> :toggle
    end
  end

  defp extract_device_name(slots, entity_id) do
    Map.get(slots, :device) || Map.get(slots, "device") ||
      Map.get(slots, :location) || Map.get(slots, "location") ||
      entity_id || "device"
  end

  defp maybe_add_entity_id(data, nil), do: data
  defp maybe_add_entity_id(data, entity_id), do: Map.put(data, "entity_id", entity_id)

  defp call_service(url, token, domain, service, data) do
    endpoint = "#{url}/api/services/#{domain}/#{service}"
    http_post(endpoint, token, data)
  end

  defp get_entity_state(url, token, entity_id) do
    endpoint = "#{url}/api/states/#{entity_id}"
    http_get(endpoint, token)
  end

  defp get_domain_states(url, token, ha_domain) do
    case http_get("#{url}/api/states", token) do
      {:ok, states} when is_list(states) ->
        prefix = "#{ha_domain}."
        filtered = Enum.filter(states, fn s ->
          entity_id = Map.get(s, "entity_id", "")
          String.starts_with?(entity_id, prefix)
        end)
        {:ok, filtered}

      {:ok, _} -> {:error, :unexpected_response}
      {:error, reason} -> {:error, reason}
    end
  end

  defp load_action_verbs do
    path = brain_priv("priv/services/ha_action_verbs.json")

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, map} -> map
          _ -> %{}
        end
      {:error, _} -> %{}
    end
  end

  defp brain_priv(relative) do
    case :code.priv_dir(:brain) do
      {:error, _} -> Path.join("apps/brain", relative)
      priv_dir -> Path.join(priv_dir, Path.relative_to(relative, "priv"))
    end
  end

  defp http_get(url, token) do
    headers = build_headers(token)

    case :httpc.request(:get, {String.to_charlist(url), headers}, [{:timeout, 10_000}], []) do
      {:ok, {{_, 200, _}, _resp_headers, body}} ->
        Jason.decode(List.to_string(body))

      {:ok, {{_, status, _}, _, body}} ->
        {:error, {:http_error, status, List.to_string(body)}}

      {:error, reason} ->
        {:error, {:connection_error, reason}}
    end
  end

  defp http_post(url, token, body) do
    headers = build_headers(token)
    body_json = Jason.encode!(body)

    case :httpc.request(
           :post,
           {String.to_charlist(url), headers, ~c"application/json", body_json},
           [{:timeout, 10_000}],
           []
         ) do
      {:ok, {{_, status, _}, _resp_headers, resp_body}} when status in 200..299 ->
        case Jason.decode(List.to_string(resp_body)) do
          {:ok, decoded} -> {:ok, decoded}
          {:error, _} -> {:ok, %{}}
        end

      {:ok, {{_, status, _}, _, body}} ->
        {:error, {:http_error, status, List.to_string(body)}}

      {:error, reason} ->
        {:error, {:connection_error, reason}}
    end
  end

  defp build_headers(token) do
    [
      {~c"Authorization", String.to_charlist("Bearer #{token}")},
      {~c"Content-Type", ~c"application/json"}
    ]
  end
end
