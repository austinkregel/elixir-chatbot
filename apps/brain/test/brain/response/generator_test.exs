defmodule Brain.Response.GeneratorTest do
  use ExUnit.Case, async: false

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
      assert String.contains?(response, "New York")
      assert type == :domain
    end

    test "generates weather clarification without location" do
      entities = []
      {:ok, response, type} = Generator.generate("weather.query", entities, nil)

      assert is_binary(response)
      assert String.contains?(response, "location")
      assert type == :domain
    end

    test "generates music response with artist entity" do
      entities = [%{entity_type: "music-artist", value: "Taylor Swift"}]
      {:ok, response, type} = Generator.generate("music.play", entities, nil)

      assert is_binary(response)
      assert String.contains?(response, "Taylor Swift")
      assert type == :domain
    end

    test "generates device control response" do
      entities = [
        %{entity_type: "device", value: "lights"},
        %{entity_type: "action", value: "turn on"}
      ]

      {:ok, response, type} = Generator.generate("device.control", entities, nil)

      assert is_binary(response)
      assert String.contains?(response, "lights")
      assert String.contains?(response, "turn on")
      assert type == :domain
    end

    test "generates fallback for unknown intent" do
      entities = []
      {:ok, response, type} = Generator.generate("unknown.intent", entities, nil)

      assert is_binary(response)
      # Should be fallback or template type
      assert type in [:fallback, :template]
    end

    test "generates response for nil intent" do
      entities = []
      {:ok, response, type} = Generator.generate(nil, entities, nil)

      assert is_binary(response)
      assert type == :fallback
    end
  end

  describe "generate_expressive/1" do
    test "generates greeting response" do
      speech_act = %{sub_type: :greeting, category: :expressive}
      response = Generator.generate_expressive(speech_act)

      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "generates farewell response" do
      speech_act = %{sub_type: :farewell, category: :expressive}
      response = Generator.generate_expressive(speech_act)

      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "generates thanks response" do
      speech_act = %{sub_type: :thanks, category: :expressive}
      response = Generator.generate_expressive(speech_act)

      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "generates apology response" do
      speech_act = %{sub_type: :apology, category: :expressive}
      response = Generator.generate_expressive(speech_act)

      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "returns nil for unknown speech act" do
      speech_act = %{sub_type: :unknown_type, category: :expressive}
      response = Generator.generate_expressive(speech_act)

      assert response == nil
    end

    test "returns nil for non-map input" do
      response = Generator.generate_expressive("not a map")
      assert response == nil
    end
  end

  describe "generate_from_analysis/4" do
    test "handles analysis model with expressive speech act" do
      analysis_model = %{
        analyses: [
          %{
            speech_act: %{category: :expressive, sub_type: :greeting, is_question: false},
            intent: "smalltalk.greetings.hello",
            entities: [],
            confidence: 0.9
          }
        ]
      }

      {response, type} =
        Generator.generate_from_analysis(analysis_model, "smalltalk.greetings.hello", [], nil)

      assert is_binary(response)
      assert String.length(response) > 0
      # Generator may use various response strategies including memory-augmented
      assert type in [:expressive, :template, :fallback, :memory_augmented, :domain]
    end

    test "handles analysis model with directive speech act" do
      analysis_model = %{
        analyses: [
          %{
            speech_act: %{category: :directive, sub_type: :command, is_question: false},
            intent: "device.control",
            entities: [%{entity_type: "device", value: "lights"}],
            confidence: 0.9
          }
        ]
      }

      entities = [%{entity_type: "device", value: "lights"}]

      {response, type} =
        Generator.generate_from_analysis(analysis_model, "device.control", entities, nil)

      assert is_binary(response)
      assert type in [:domain, :template, :fallback]
    end

    test "combines expressive and directive responses" do
      analysis_model = %{
        analyses: [
          %{
            speech_act: %{category: :expressive, sub_type: :greeting, is_question: false},
            intent: "smalltalk.greetings.hello",
            entities: [],
            confidence: 0.8
          },
          %{
            speech_act: %{category: :directive, sub_type: :request_information, is_question: true},
            intent: "weather.query",
            entities: [%{entity_type: "location", value: "Boston"}],
            confidence: 0.9
          }
        ]
      }

      entities = [%{entity_type: "location", value: "Boston"}]

      {response, type} =
        Generator.generate_from_analysis(analysis_model, "weather.query", entities, nil)

      assert is_binary(response)
      # Should combine greeting with weather response
      assert String.length(response) > 0
      assert type in [:domain, :expressive, :template, :fallback]
    end
  end
end
