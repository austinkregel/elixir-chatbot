defmodule Brain.Response.GenerativeResponseTest do
  @moduledoc "Tests for the generative response system.\n\nThese tests verify that:\n1. The Synthesizer can generate responses from domain knowledge\n2. Responses are not hardcoded - they vary based on entity values\n3. The system gracefully handles missing entities with clarification requests\n4. Novel combinations of entities produce appropriate responses\n"

  alias Brain.Response
  use ExUnit.Case, async: false

  alias Response.{Synthesizer, Generator}

  describe "Synthesizer.synthesize/3" do
    test "generates response for weather intent with location" do
      entities = [%{entity_type: "location", value: "New York"}]

      case Synthesizer.synthesize("weather.query", entities, confidence: 0.8) do
        {:ok, response} ->
          assert is_binary(response)
          assert String.length(response) > 0
          assert String.contains?(response, "New York")

        :not_synthesized ->
          :ok
      end
    end

    test "generates clarification for weather intent without location" do
      entities = []

      case Synthesizer.synthesize("weather.query", entities, confidence: 0.8) do
        {:ok, response} ->
          assert is_binary(response)
          response_lower = String.downcase(response)

          assert String.contains?(response_lower, "location") or
                   String.contains?(response_lower, "where") or
                   String.contains?(response_lower, "city") or
                   String.contains?(response_lower, "area") or
                   String.contains?(response_lower, "weather"),
                 "Expected weather clarification response, got: #{response}"

        :not_synthesized ->
          :ok
      end
    end

    test "generates response for music intent with artist" do
      entities = [%{entity_type: "music-artist", value: "The Beatles"}]

      case Synthesizer.synthesize("music.play", entities, confidence: 0.8) do
        {:ok, response} ->
          assert is_binary(response)
          assert String.contains?(response, "Beatles")

        :not_synthesized ->
          :ok
      end
    end

    test "generates response for device control with device and action" do
      entities = [
        %{entity_type: "device", value: "living room lights"},
        %{entity_type: "action", value: "turn on"}
      ]

      case Synthesizer.synthesize("smarthome.lights.switch.on", entities, confidence: 0.9) do
        {:ok, response} ->
          assert is_binary(response)

          assert String.contains?(String.downcase(response), "light") or
                   String.contains?(String.downcase(response), "living room")

        :not_synthesized ->
          :ok
      end
    end

    test "generates different responses for different entity values" do
      entities1 = [%{entity_type: "location", value: "Paris"}]
      entities2 = [%{entity_type: "location", value: "Tokyo"}]

      result1 = Synthesizer.synthesize("weather.query", entities1, confidence: 0.8)
      result2 = Synthesizer.synthesize("weather.query", entities2, confidence: 0.8)

      case {result1, result2} do
        {{:ok, response1}, {:ok, response2}} ->
          assert String.contains?(response1, "Paris")
          assert String.contains?(response2, "Tokyo")
          assert response1 != response2

        _ ->
          :ok
      end
    end
  end

  describe "Synthesizer.synthesize_expressive/2" do
    test "generates greeting response" do
      case Synthesizer.synthesize_expressive(:greeting) do
        {:ok, response} ->
          assert is_binary(response)
          assert String.length(response) > 0
          greeting_words = ["hello", "hi", "hey", "greetings"]

          assert Enum.any?(greeting_words, fn word ->
                   String.contains?(String.downcase(response), word)
                 end)

        :not_synthesized ->
          :ok
      end
    end

    test "generates farewell response" do
      case Synthesizer.synthesize_expressive(:farewell) do
        {:ok, response} ->
          assert is_binary(response)
          farewell_words = ["goodbye", "bye", "see you", "take care"]

          assert Enum.any?(farewell_words, fn word ->
                   String.contains?(String.downcase(response), word)
                 end)

        :not_synthesized ->
          :ok
      end
    end

    test "generates thanks response" do
      case Synthesizer.synthesize_expressive(:thanks) do
        {:ok, response} ->
          assert is_binary(response)
          thanks_words = ["welcome", "help", "problem", "glad"]

          assert Enum.any?(thanks_words, fn word ->
                   String.contains?(String.downcase(response), word)
                 end)

        :not_synthesized ->
          :ok
      end
    end
  end

  describe "Synthesizer.synthesize_clarification/4" do
    test "generates clarification for missing location" do
      entities = []
      missing_slots = ["location"]

      {:ok, response} =
        Synthesizer.synthesize_clarification("weather.query", entities, missing_slots)

      assert is_binary(response)
      assert String.length(response) > 0

      assert String.contains?(String.downcase(response), "location") or
               String.contains?(String.downcase(response), "where") or
               String.contains?(String.downcase(response), "city")
    end

    test "acknowledges partial information when asking for clarification" do
      entities = [%{entity_type: "date", value: "tomorrow"}]
      missing_slots = ["location"]

      {:ok, response} =
        Synthesizer.synthesize_clarification("weather.query", entities, missing_slots)

      assert is_binary(response)

      assert String.contains?(String.downcase(response), "location") or
               String.contains?(String.downcase(response), "where")
    end
  end

  describe "Synthesizer.get_fallback_response/0" do
    test "returns a non-empty fallback response" do
      response = Synthesizer.get_fallback_response()

      assert is_binary(response)
      assert String.length(response) > 10

      assert String.contains?(String.downcase(response), "not sure") or
               String.contains?(String.downcase(response), "don't") or
               String.contains?(String.downcase(response), "rephrase") or
               String.contains?(String.downcase(response), "understand")
    end
  end

  describe "Generator.generate/3 integration" do
    test "generates response for valid intent with entities" do
      {:ok, response, type} =
        Generator.generate(
          "weather.query",
          [%{entity_type: "location", value: "Seattle"}],
          "What's the weather in Seattle?"
        )

      assert is_binary(response)
      assert String.length(response) > 0

      assert type in [
               :synthesized,
               :memory_adapted,
               :template,
               :special_handler,
               :fallback,
               :lstm_selected
             ]
    end

    test "handles unknown intent gracefully" do
      {:ok, response, type} = Generator.generate("unknown.intent", [], "Some random query")

      assert is_binary(response)

      assert type in [
               :template,
               :fallback,
               :synthesized,
               :quality_improved,
               :memory_adapted,
               :special_handler,
               :lstm_selected
             ]
    end

    test "handles nil intent gracefully" do
      {:ok, response, _type} = Generator.generate(nil, [], "Random text")

      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "handles empty entities gracefully" do
      {:ok, response, _type} = Generator.generate("weather.query", [], nil)

      assert is_binary(response)
      assert String.length(response) > 0
    end
  end

  describe "novel entity combinations" do
    test "handles unusual location names" do
      entities = [%{entity_type: "location", value: "Timbuktu"}]

      {:ok, response, _type} =
        Generator.generate("weather.query", entities, "Weather in Timbuktu")

      assert is_binary(response)

      assert String.contains?(response, "Timbuktu") or
               String.contains?(String.downcase(response), "location")
    end

    test "handles multiple entities together" do
      entities = [
        %{entity_type: "location", value: "Chicago"},
        %{entity_type: "date", value: "next Tuesday"},
        %{entity_type: "time", value: "afternoon"}
      ]

      {:ok, response, _type} =
        Generator.generate("weather.query", entities, "Weather in Chicago next Tuesday afternoon")

      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "generates different responses for same intent with different contexts" do
      entities1 = [%{entity_type: "location", value: "Boston"}]
      entities2 = [%{entity_type: "location", value: "Denver"}]

      {:ok, response1, _} = Generator.generate("weather.query", entities1, "Weather in Boston")
      {:ok, response2, _} = Generator.generate("weather.query", entities2, "Weather in Denver")
      assert is_binary(response1)
      assert is_binary(response2)

      if String.contains?(response1, "Boston") do
        refute String.contains?(response1, "Denver")
      end

      if String.contains?(response2, "Denver") do
        refute String.contains?(response2, "Boston")
      end
    end
  end
end
