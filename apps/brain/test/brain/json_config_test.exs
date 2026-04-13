defmodule Brain.JsonConfigTest do
  @moduledoc """
  Verifies that all JSON config files are valid and loadable.
  """
  use ExUnit.Case, async: false

  @config_files [
    {"analysis/context_preferences.json", :brain},
    {"analysis/related_slot_mappings.json", :brain},
    {"analysis/speech_act_intent_map.json", :brain},
    {"knowledge/entity_slot_mappings.json", :brain},
    {"response/frame_key_mappings.json", :brain}
  ]

  describe "JSON config file validity" do
    for {file, app} <- @config_files do
      test "#{file} is valid JSON" do
        path = Path.join(:code.priv_dir(unquote(app)), unquote(file))
        assert File.exists?(path), "Config file #{unquote(file)} does not exist at #{path}"

        content = File.read!(path)
        assert {:ok, data} = Jason.decode(content)
        assert is_map(data), "Expected #{unquote(file)} to decode to a map"
      end
    end
  end

  describe "context_preferences.json" do
    test "contains expected context keys" do
      path = Path.join(:code.priv_dir(:brain), "analysis/context_preferences.json")
      {:ok, data} = path |> File.read!() |> Jason.decode()

      assert Map.has_key?(data, "default"), "Expected 'default' key in context_preferences"
      assert is_map(data["default"]), "Expected 'default' to be a map"
    end
  end

  describe "related_slot_mappings.json" do
    test "maps slot names to lists of related types" do
      path = Path.join(:code.priv_dir(:brain), "analysis/related_slot_mappings.json")
      {:ok, data} = path |> File.read!() |> Jason.decode()

      assert Map.has_key?(data, "location"), "Expected 'location' slot mapping"
      assert is_list(data["location"]), "Expected 'location' to map to a list"
      assert "city" in data["location"]
    end
  end

  describe "speech_act_intent_map.json" do
    test "contains default key" do
      path = Path.join(:code.priv_dir(:brain), "analysis/speech_act_intent_map.json")
      {:ok, data} = path |> File.read!() |> Jason.decode()

      assert Map.has_key?(data, "default")
      assert is_binary(data["default"])
    end
  end

  describe "entity_slot_mappings.json" do
    test "contains entity_type_to_slot section" do
      path = Path.join(:code.priv_dir(:brain), "knowledge/entity_slot_mappings.json")
      {:ok, data} = path |> File.read!() |> Jason.decode()

      assert Map.has_key?(data, "entity_type_to_slot")
      assert is_map(data["entity_type_to_slot"])
      assert Map.has_key?(data["entity_type_to_slot"], "location")
    end

    test "contains entity_type_to_slot_names section" do
      path = Path.join(:code.priv_dir(:brain), "knowledge/entity_slot_mappings.json")
      {:ok, data} = path |> File.read!() |> Jason.decode()

      assert Map.has_key?(data, "entity_type_to_slot_names")
      assert is_map(data["entity_type_to_slot_names"])
    end
  end

  describe "frame_key_mappings.json" do
    test "contains slot_to_frame_key and default" do
      path = Path.join(:code.priv_dir(:brain), "response/frame_key_mappings.json")
      {:ok, data} = path |> File.read!() |> Jason.decode()

      assert Map.has_key?(data, "slot_to_frame_key")
      assert Map.has_key?(data, "default")
      assert is_map(data["slot_to_frame_key"])
      assert data["default"] == "general"
    end
  end
end
