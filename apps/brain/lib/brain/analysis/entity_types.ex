defmodule Brain.Analysis.EntityTypes do
  @moduledoc """
  Provides entity type groupings and synonyms loaded from priv/analysis/entity_types.json.

  This module centralizes entity type classification to avoid hardcoded lists
  scattered across SlotDetector, EntityDisambiguator, and FollowupDetector.

  ## Type Groups

  - `location_types` - Types representing locations (city, room, etc.)
  - `device_types` - Smart device types (lights, heating, etc.)
  - `music_types` - Music-related types (artist, song, album, etc.)
  - `person_types` - Person/name types
  - `date_types` - Date-related types
  - `time_types` - Time-related types

  ## Usage

      EntityTypes.is_location_type?("city")      # => true
      EntityTypes.is_device_type?("lights")      # => true
      EntityTypes.location_types()               # => ["location", "city", ...]
  """

  @entity_types_path "priv/analysis/entity_types.json"
  @external_resource @entity_types_path

  @data (case File.read(@entity_types_path) do
           {:ok, content} ->
             case Jason.decode(content) do
               {:ok, data} -> data
               {:error, _} -> %{}
             end

           {:error, _} ->
             %{}
         end)

  @type_groups Map.get(@data, "type_groups", %{})
  @type_synonyms Map.get(@data, "type_synonyms", %{})

  # Pre-compute MapSets for efficient membership checks
  @location_types_set MapSet.new(Map.get(@type_groups, "location_types", ["location", "city", "room"]))
  @device_types_set MapSet.new(Map.get(@type_groups, "device_types", ["device", "lights", "heating"]))
  @music_types_set MapSet.new(Map.get(@type_groups, "music_types", ["song", "music-artist", "music-album"]))
  @person_types_set MapSet.new(Map.get(@type_groups, "person_types", ["person", "name"]))
  @date_types_set MapSet.new(Map.get(@type_groups, "date_types", ["date", "relative_date", "sys-date"]))
  @time_types_set MapSet.new(Map.get(@type_groups, "time_types", ["time", "sys-time"]))

  # ============================================================================
  # Type Group Accessors
  # ============================================================================

  @doc "Returns the list of location-related entity types."
  def location_types, do: Map.get(@type_groups, "location_types", ["location", "city", "room"])

  @doc "Returns the list of device-related entity types."
  def device_types, do: Map.get(@type_groups, "device_types", ["device", "lights", "heating"])

  @doc "Returns the list of music-related entity types."
  def music_types, do: Map.get(@type_groups, "music_types", ["song", "music-artist", "music-album"])

  @doc "Returns the list of person-related entity types."
  def person_types, do: Map.get(@type_groups, "person_types", ["person", "name"])

  @doc "Returns the list of date-related entity types."
  def date_types, do: Map.get(@type_groups, "date_types", ["date", "relative_date", "sys-date"])

  @doc "Returns the list of time-related entity types."
  def time_types, do: Map.get(@type_groups, "time_types", ["time", "sys-time"])

  # ============================================================================
  # Type Predicates
  # ============================================================================

  @doc "Checks if the given entity type is a location type."
  def is_location_type?(type) when is_binary(type), do: MapSet.member?(@location_types_set, type)
  def is_location_type?(type) when is_atom(type), do: is_location_type?(to_string(type))
  def is_location_type?(_), do: false

  @doc "Checks if the given entity type is a device type."
  def is_device_type?(type) when is_binary(type), do: MapSet.member?(@device_types_set, type)
  def is_device_type?(type) when is_atom(type), do: is_device_type?(to_string(type))
  def is_device_type?(_), do: false

  @doc "Checks if the given entity type is a music type."
  def is_music_type?(type) when is_binary(type), do: MapSet.member?(@music_types_set, type)
  def is_music_type?(type) when is_atom(type), do: is_music_type?(to_string(type))
  def is_music_type?(_), do: false

  @doc "Checks if the given entity type is a person type."
  def is_person_type?(type) when is_binary(type), do: MapSet.member?(@person_types_set, type)
  def is_person_type?(type) when is_atom(type), do: is_person_type?(to_string(type))
  def is_person_type?(_), do: false

  @doc "Checks if the given entity type is a date type."
  def is_date_type?(type) when is_binary(type), do: MapSet.member?(@date_types_set, type)
  def is_date_type?(type) when is_atom(type), do: is_date_type?(to_string(type))
  def is_date_type?(_), do: false

  @doc "Checks if the given entity type is a time type."
  def is_time_type?(type) when is_binary(type), do: MapSet.member?(@time_types_set, type)
  def is_time_type?(type) when is_atom(type), do: is_time_type?(to_string(type))
  def is_time_type?(_), do: false

  # ============================================================================
  # Type Synonyms
  # ============================================================================

  @doc """
  Gets synonym types for a canonical type.

  For example, `synonyms_for("location")` returns all types that
  can be treated as locations for slot filling purposes.
  """
  def synonyms_for(canonical_type) when is_binary(canonical_type) do
    Map.get(@type_synonyms, canonical_type, [canonical_type])
  end

  def synonyms_for(canonical_type) when is_atom(canonical_type) do
    synonyms_for(to_string(canonical_type))
  end

  @doc """
  Checks if type1 is a synonym for type2.
  """
  def synonym?(type1, type2) when is_binary(type1) and is_binary(type2) do
    type1 in synonyms_for(type2) or type2 in synonyms_for(type1)
  end

  @doc """
  Returns the full type synonyms map.
  """
  def type_synonyms, do: @type_synonyms

  # ============================================================================
  # Bulk Checks
  # ============================================================================

  @doc """
  Checks if any of the given entity types are location types.
  """
  def has_location_type?(entity_types) when is_list(entity_types) do
    Enum.any?(entity_types, &is_location_type?/1)
  end

  @doc """
  Checks if any of the given entity types are device types.
  """
  def has_device_type?(entity_types) when is_list(entity_types) do
    Enum.any?(entity_types, &is_device_type?/1)
  end

  @doc """
  Checks if any of the given entity types are music types.
  """
  def has_music_type?(entity_types) when is_list(entity_types) do
    Enum.any?(entity_types, &is_music_type?/1)
  end
end
