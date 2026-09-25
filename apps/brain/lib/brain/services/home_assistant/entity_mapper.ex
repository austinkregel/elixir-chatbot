defmodule Brain.Services.HomeAssistant.EntityMapper do
  @moduledoc """
  Maps between Home Assistant entity_id format and Brain entity types.

  HA entities use a `domain.object_id` format (e.g., `light.kitchen_ceiling`).
  Brain entities use friendly names with types (e.g., type: :device, name: "kitchen light").

  Domain-to-type mappings are loaded from `priv/services/ha_domain_types.json`.

  Entity resolution uses the CapabilityRegistry to query HA's own entity data
  rather than hardcoded string matching.
  """

  alias Brain.Services.HomeAssistant.CapabilityRegistry

  @domain_types_path "priv/services/ha_domain_types.json"

  @doc """
  Returns the Brain entity type for a given HA domain.
  """
  def brain_type_for_domain(ha_domain) do
    types = load_domain_types()
    Map.get(types, ha_domain, "device")
  end

  @doc """
  Resolves a Home Assistant entity_id from slot data.

  Resolution strategy:
  1. If `_ha_entity_id` is directly available in slots (from Gazetteer metadata), use it
  2. If a device slot has a direct entity match, use it
  3. Query CapabilityRegistry.find_entities_for_area with location/room values
  """
  def resolve_entity_id(entities) when is_list(entities) do
    device = find_entity_by_type(entities, ["device", "appliance", "room", "light", "switch"])

    case device do
      %{ha_entity_id: id} when is_binary(id) -> id
      %{name: name} when is_binary(name) -> resolve_from_registry(name)
      _ -> nil
    end
  end

  def resolve_entity_id(%{} = slots) do
    ha_entity_id = Map.get(slots, :_ha_entity_id) || Map.get(slots, "_ha_entity_id")

    if ha_entity_id do
      ha_entity_id
    else
      device = Map.get(slots, :device) || Map.get(slots, "device")
      location = Map.get(slots, :location) || Map.get(slots, "location") ||
                 Map.get(slots, :room) || Map.get(slots, "room")

      resolve_from_slots(device, location)
    end
  end

  def resolve_entity_id(_), do: nil

  defp resolve_from_slots(device, location) when is_binary(device) and is_binary(location) do
    resolve_from_registry("#{location} #{device}") ||
      resolve_from_registry(device) ||
      resolve_from_registry(location)
  end

  defp resolve_from_slots(device, _location) when is_binary(device) do
    resolve_from_registry(device)
  end

  defp resolve_from_slots(_device, location) when is_binary(location) do
    resolve_from_registry(location)
  end

  defp resolve_from_slots(_, _), do: nil

  defp resolve_from_registry(name) when is_binary(name) do
    results = CapabilityRegistry.find_entities_for_area(name)

    case results do
      [first | _] -> first.entity_id
      [] -> nil
    end
  end

  @doc """
  Builds service_data map from slots for an HA service call.

  Inspects the slot types present to determine what parameters
  to include. Generic -- no intent-specific logic.
  """
  def build_service_data(slots) when is_map(slots) do
    data = %{}

    data = maybe_add(data, slots, "brightness_pct", [:brightness, :level, :percentage], :number)
    data = maybe_add(data, slots, "color_name", [:color], :string)
    data = maybe_add(data, slots, "temperature", [:temperature, :temp, :degree], :number)
    data = maybe_add_volume(data, slots)
    data = maybe_add(data, slots, "duration", [:duration, :time], :string)

    data
  end

  def build_service_data(_), do: %{}

  @doc """
  Converts an HA entity state map to a Brain-friendly entity map.
  """
  def ha_entity_to_brain(ha_entity) when is_map(ha_entity) do
    entity_id = Map.get(ha_entity, "entity_id", "")
    friendly_name = get_in(ha_entity, ["attributes", "friendly_name"]) || entity_id
    state = Map.get(ha_entity, "state", "unknown")
    [ha_domain | _] = String.split(entity_id, ".", parts: 2)

    brain_type = brain_type_for_domain(ha_domain)

    %{
      name: friendly_name,
      type: String.to_atom(brain_type),
      ha_entity_id: entity_id,
      ha_domain: ha_domain,
      state: state,
      attributes: Map.get(ha_entity, "attributes", %{})
    }
  end

  @doc """
  Converts a friendly name to a plausible HA entity_id format.

  NOTE: Uses regex for string sanitization (formatting conversion, not NLP).
  """
  def name_to_entity_id(name) when is_binary(name) do
    name
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9\s]/, "")
    |> String.replace(~r/\s+/, "_")
    |> String.trim("_")
  end

  def name_to_entity_id(_), do: nil

  defp find_entity_by_type(entities, types) when is_list(entities) do
    Enum.find(entities, fn entity ->
      entity_type = Map.get(entity, :type) || Map.get(entity, "type")
      to_string(entity_type) in types
    end)
  end

  defp maybe_add(data, slots, ha_key, slot_keys, :number) do
    case find_slot_value(slots, slot_keys) do
      nil -> data
      val ->
        case parse_number(val) do
          nil -> data
          num -> Map.put(data, ha_key, num)
        end
    end
  end

  defp maybe_add(data, slots, ha_key, slot_keys, :string) do
    case find_slot_value(slots, slot_keys) do
      nil -> data
      val -> Map.put(data, ha_key, to_string(val))
    end
  end

  defp maybe_add_volume(data, slots) do
    case find_slot_value(slots, [:volume]) do
      nil -> data
      val ->
        case parse_number(val) do
          nil -> data
          num -> Map.put(data, "volume_level", num / 100.0)
        end
    end
  end

  defp find_slot_value(slots, keys) do
    Enum.find_value(keys, fn key ->
      Map.get(slots, key) || Map.get(slots, to_string(key))
    end)
  end

  defp parse_number(val) when is_number(val), do: val
  defp parse_number(val) when is_binary(val) do
    case Float.parse(val) do
      {n, _} -> n
      :error -> nil
    end
  end
  defp parse_number(_), do: nil

  defp load_domain_types do
    path = brain_priv(@domain_types_path)

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
end
