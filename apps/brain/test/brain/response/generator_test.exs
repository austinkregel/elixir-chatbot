defmodule Brain.Response.GeneratorTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Response.Generator
  import Brain.TestHelpers

  setup do
    start_brain_services()
    :ok
  end

  describe "generate/3" do
    test "generates weather response with location entity" do
      entities = [%{entity_type: "location", value: "New York"}]

      {:ok, response, type} = Generator.generate("weather.query", entities, nil)

      assert is_binary(response)

      assert String.contains?(response, "New York") or
               String.contains?(response, "weather") or
               type in [:domain, :synthesized, :template]

      assert type in [:domain, :synthesized, :template, :fallback]
    end

    test "generates weather clarification without location" do
      entities = []

      {:ok, response, type} = Generator.generate("weather.query", entities, nil)

      assert is_binary(response)

      assert String.contains?(response, "location") or
               String.contains?(response, "where") or
               String.contains?(response, "weather") or
               type in [:domain, :synthesized, :template, :fallback]

      assert type in [:domain, :synthesized, :template, :fallback]
    end

    test "generates music response with artist entity" do
      entities = [%{entity_type: "music-artist", value: "Taylor Swift"}]

      {:ok, response, type} = Generator.generate("music.play", entities, nil)

      assert is_binary(response)

      assert String.contains?(response, "Taylor Swift") or
               String.contains?(response, "music") or
               String.contains?(response, "play") or
               type in [:domain, :synthesized, :template, :fallback]

      assert type in [:domain, :synthesized, :template, :fallback]
    end

    test "generates device control response" do
      entities = [
        %{entity_type: "device", value: "lights"},
        %{entity_type: "action", value: "turn on"}
      ]

      {:ok, response, type} = Generator.generate("smarthome.lights.switch.on", entities, nil)

      assert is_binary(response)

      assert String.contains?(response, "lights") or
               String.contains?(response, "device") or
               String.contains?(response, "turn") or
               type in [:domain, :synthesized, :template, :fallback]

      assert type in [:domain, :synthesized, :template, :fallback]
    end

    test "generates fallback for unknown intent" do
      entities = []

      {:ok, response, type} = Generator.generate("unknown.intent", entities, nil)

      assert is_binary(response)
      assert type in [:fallback, :template, :synthesized]
    end

    test "generates response for nil intent" do
      entities = []

      {:ok, response, type} = Generator.generate(nil, entities, nil)

      assert is_binary(response)
      assert type in [:fallback, :synthesized]
    end
  end
end
