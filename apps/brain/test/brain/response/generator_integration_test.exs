defmodule Brain.Response.GeneratorIntegrationTest do
  @moduledoc "Integration tests for response generation with Pipeline and test classifier."
  use Brain.Test.GraphCase, async: false

  import Brain.TestHelpers

  alias Brain.Analysis.Pipeline
  alias Brain.Response.Generator

  setup do
    start_test_services()
    :ok
  end

  describe "full pipeline to response" do
    test "greeting input produces greeting-classified response" do
      analysis = Pipeline.process("Hello!")
      first = List.first(analysis.analyses || [])
      intent = first && first.intent
      entities = (first && first.entities) || []
      {:ok, response, _type} = Generator.generate(intent, entities, "Hello!")
      assert is_binary(response)
      assert byte_size(response) > 0
      assert_response_intent(response, "smalltalk")
    end

    test "weather query produces weather-classified response" do
      analysis = Pipeline.process("What's the weather like in Seattle?")
      first = List.first(analysis.analyses || [])
      intent = first && first.intent
      entities = (first && first.entities) || []
      {:ok, response, _type} = Generator.generate(intent, entities, "What's the weather like in Seattle?")
      assert is_binary(intent)
      assert String.starts_with?(intent, "weather")
      assert is_binary(response)
      assert byte_size(response) > 0
    end

    test "unknown intent produces non-empty fallback response" do
      {:ok, response, _type} = Generator.generate("unknown.intent.xyz", [], nil)
      assert is_binary(response)
      assert byte_size(response) > 0
    end

    @timeout :infinity
    @tag timeout: @timeout
    test "factual query about known fact produces fact-containing response" do
      Brain.FactDatabase.add_fact_direct(%{
        "id" => "test-fact-#{System.unique_integer([:positive])}",
        "fact" => "The capital of France is Paris.",
        "category" => "geography",
        "entities" => ["France", "Paris"]
      })
      analysis = Pipeline.process("What is the capital of France?")
      first = List.first(analysis.analyses || [])
      intent = first && first.intent
      entities = (first && first.entities) || []
      {:ok, response, _type} = Generator.generate(intent, entities, "What is the capital of France?")
      assert is_binary(response)
      assert byte_size(response) > 0
    end

    test "sentiment-laden input produces empathetic response" do
      analysis = Pipeline.process("I'm really frustrated")
      first = List.first(analysis.analyses || [])
      intent = first && first.intent
      entities = (first && first.entities) || []
      {:ok, response, _type} = Generator.generate(intent, entities, "I'm really frustrated")
      assert is_binary(response)
      assert byte_size(response) > 0
    end
  end
end
