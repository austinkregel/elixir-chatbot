defmodule Brain.Response.GeneratorTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Response.Generator
  import Brain.TestHelpers

  # The world the test suite runs in, seeded by `mix test.models`.
  @opts [world_id: "default"]

  setup do
    start_brain_services()
    :ok
  end

  describe "generate/4" do
    test "raises when opts name no world" do
      assert_raise ArgumentError, ~r/requires :world_id/, fn ->
        Generator.generate("weather.query", [], nil, [])
      end
    end

    test "raises when the world is not a world id" do
      assert_raise ArgumentError, ~r/requires :world_id/, fn ->
        Generator.generate("weather.query", [], nil, world_id: "")
      end
    end

    # The suite's world has no credentials for Weather or Home Assistant, so an
    # intent dispatched to either is answered with the reply planned for the
    # service's error (`{:service_reply, enrichment}`, generate/4's contract),
    # and the slots the service was asked with carry the turn's entities.
    test "generates weather response with location entity" do
      entities = [%{entity_type: "location", value: "New York"}]

      {:ok, response, type} = Generator.generate("weather.query", entities, nil, @opts)

      assert {:service_reply,
              %{enrichment_service: :weather, enrichment_error: :missing_credentials, slots: %{location: "New York"}}} = type

      assert response =~ "Weather"
    end

    test "generates weather clarification without location" do
      entities = []

      {:ok, response, type} = Generator.generate("weather.query", entities, nil, @opts)

      assert {:service_reply, %{enrichment_service: :weather, enrichment_error: :missing_credentials, slots: slots}} = type
      assert slots == %{}
      assert response =~ "Weather"
    end

    test "generates music response with artist entity" do
      entities = [%{entity_type: "music_artist", value: "Taylor Swift"}]

      {:ok, response, type} = Generator.generate("music.play", entities, nil, @opts)

      assert {:service_reply, %{enrichment_service: :home_assistant, enrichment_error: :missing_credentials, slots: slots}} =
               type

      assert "Taylor Swift" in Map.values(slots)
      assert response =~ "Home Assistant"
    end

    test "generates device control response" do
      entities = [
        %{entity_type: "device", value: "lights"},
        %{entity_type: "action", value: "turn on"}
      ]

      {:ok, response, type} = Generator.generate("smarthome.lights.switch.on", entities, nil, @opts)

      assert {:service_reply,
              %{enrichment_service: :home_assistant, enrichment_error: :missing_credentials, slots: %{device: "lights"}}} = type

      assert response =~ "Home Assistant"
    end

    test "generates fallback for unknown intent" do
      entities = []

      {:ok, response, type} = Generator.generate("unknown.intent", entities, nil, @opts)

      assert is_binary(response)
      assert type in [:fallback, :template, :synthesized]
    end

    test "generates response for nil intent" do
      entities = []

      {:ok, response, type} = Generator.generate(nil, entities, nil, @opts)

      assert is_binary(response)
      assert type in [:fallback, :synthesized]
    end
  end

  describe "generate_with_events/5" do
    test "raises when opts name no world" do
      assert_raise ArgumentError, ~r/Generator.generate_with_events requires :world_id/, fn ->
        Generator.generate_with_events("weather.query", [], nil, [], [])
      end
    end
  end

  describe "generate_with_path/4" do
    test "raises when opts name no world" do
      assert_raise ArgumentError, ~r/Generator.generate_with_path requires :world_id/, fn ->
        Generator.generate_with_path("weather.query", [], nil, [])
      end
    end
  end

  describe "generate_with_context/4" do
    test "raises when the context names no world" do
      assert_raise ArgumentError, ~r/Generator.generate_with_context requires :world_id/, fn ->
        Generator.generate_with_context("weather.query", [], nil, %{confidence: 0.9})
      end
    end
  end
end
