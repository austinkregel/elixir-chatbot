defmodule Brain.Services.HomeAssistant.StateFormatter do
  @moduledoc """
  Generic state formatter for Home Assistant enrichment.

  Dynamically builds enrichment maps from entity attributes rather than
  using hardcoded category-based formatting. The response templates in
  domain knowledge JSON define which `$placeholder` tokens they expect;
  this module provides whatever attributes are available.
  """

  @doc """
  Formats an HA entity state into an enrichment map.

  Extracts `friendly_name` as device, `state` as device_status,
  and includes all attributes as enrichment fields.
  """
  def format_state(state) when is_map(state) do
    attrs = Map.get(state, "attributes", %{})
    entity_state = Map.get(state, "state", "unknown")
    friendly_name = Map.get(attrs, "friendly_name", "device")

    base = %{
      device: friendly_name,
      device_status: entity_state
    }

    attrs
    |> Map.delete("friendly_name")
    |> Enum.reduce(base, fn {key, value}, acc ->
      atom_key = safe_atom(key)
      formatted_value = format_attribute_value(key, value)
      Map.put(acc, atom_key, formatted_value)
    end)
  end

  @doc """
  Formats the result of a service call action.

  Uses the HA service name and device name to produce a generic
  enrichment map.
  """
  def format_action_result(ha_service, device_name) do
    action = service_to_action_description(ha_service)

    %{
      device: device_name,
      action: action,
      device_status: "updated"
    }
  end

  @doc """
  Formats a list of domain states into a summary enrichment map.
  """
  def format_domain_states(states) when is_list(states) do
    devices =
      Enum.map(states, fn s ->
        name = get_in(s, ["attributes", "friendly_name"]) || Map.get(s, "entity_id", "unknown")
        state = Map.get(s, "state", "unknown")
        "#{name}: #{state}"
      end)

    %{
      device_count: length(states),
      device_list: Enum.join(devices, ", "),
      summary: "Found #{length(states)} devices"
    }
  end

  defp service_to_action_description(service) when is_binary(service) do
    service
    |> String.replace("_", " ")
    |> String.replace("media ", "")
  end

  defp service_to_action_description(_), do: "done"

  defp format_attribute_value("brightness", val) when is_number(val) do
    "#{round(val / 255.0 * 100)}%"
  end

  defp format_attribute_value("volume_level", val) when is_number(val) do
    "#{round(val * 100)}%"
  end

  defp format_attribute_value("current_temperature", val) when is_number(val) do
    "#{val}°"
  end

  defp format_attribute_value("temperature", val) when is_number(val) do
    "#{val}°"
  end

  defp format_attribute_value(_key, val) when is_binary(val), do: val
  defp format_attribute_value(_key, val) when is_number(val), do: to_string(val)
  defp format_attribute_value(_key, val) when is_boolean(val), do: to_string(val)
  defp format_attribute_value(_key, val) when is_list(val), do: Enum.join(val, ", ")
  defp format_attribute_value(_key, nil), do: ""
  defp format_attribute_value(_key, val), do: inspect(val)

  defp safe_atom(key) when is_binary(key) do
    key
    |> String.downcase()
    |> String.replace("-", "_")
    |> String.to_atom()
  end
end
