defmodule Mix.Tasks.Registry.DeriveTest do
  @moduledoc """
  The slot types `mix registry.derive` writes for the export's parameters:
  runtime entity types `analysis/entity_types.json` declares, read from its
  hierarchy and its `config.export_system_types`.
  """
  use ExUnit.Case, async: true

  alias Mix.Tasks.Registry.Derive

  defp param(name, type), do: %{"name" => name, "dataType" => "@" <> type}

  setup do
    {:ok, types: Derive.load_entity_types!()}
  end

  test "a system entity is written as the runtime type the entity types map it to", %{types: types} do
    assert Derive.entity_mappings(
             [param("final_value", "sys.number"), param("date_time", "sys.date_time"), param("artist", "sys.music_artist")],
             types
           ) == %{"final_value" => ["number"], "date_time" => ["date_time"], "artist" => ["music_artist"]}
  end

  test "an export type the hierarchy declares is written as itself", %{types: types} do
    assert Derive.entity_mappings([param("room", "room"), param("device", "lights")], types) ==
             %{"room" => ["room"], "device" => ["lights"]}
  end

  test "a system entity with no runtime type raises", %{types: types} do
    assert_raise Mix.Error, ~r/@sys\.any.*export_system_types maps to no runtime type/, fn ->
      Derive.entity_mappings([param("q", "sys.any")], types)
    end
  end

  test "a slot that is a declared type unrelated to the system entity's raises", %{types: types} do
    # A start year is a year, not any number.
    assert_raise Mix.Error, ~r/the slot start_year is typed @sys\.number, which stands for number/, fn ->
      Derive.entity_mappings([param("start_year", "sys.number")], types)
    end
  end

  test "an export type the hierarchy does not declare raises", %{types: types} do
    assert_raise Mix.Error, ~r/@spaceship, which .* does not declare/, fn ->
      Derive.entity_mappings([param("vehicle", "spaceship")], types)
    end
  end
end
