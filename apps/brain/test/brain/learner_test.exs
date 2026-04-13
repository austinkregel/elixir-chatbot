defmodule Brain.LearnerDataTest do
  @moduledoc """
  Data-driven tests for the Learner module covering learning from user inputs.
  """
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Learner

  @test_persona "LearnerTestPersona"

  setup _context do
    ensure_pubsub_started()
    ensure_started(Brain.KnowledgeStore)
    ensure_started(Brain.MemoryStore)
    :ok
  end

  # Test data for learn_from_input/3
  @learning_input_cases [
    # {persona, input, description}
    {@test_persona, "My cat is named Whiskers", "learn about pet"},
    {@test_persona, "I live in Seattle", "learn about location"},
    {@test_persona, "John is my friend", "learn about person"},
    {@test_persona, "The living room light is on", "learn about device"},
    {@test_persona, "Hello there", "simple greeting"},
    {@test_persona, "", "empty input"},
    {@test_persona, "a", "single character"},
  ]

  describe "learn_from_input/3 - data driven" do
    for {persona, input, description} <- @learning_input_cases do
      @persona persona
      @input input
      @description description

      test "#{description}" do
        {:ok, data} = Learner.learn_from_input(@persona, @input)
        assert is_map(data)
      end
    end
  end

  # Test with options
  @learning_with_opts_cases [
    {@test_persona, "Turn on the kitchen light", [discourse: nil], "with nil discourse"},
    {@test_persona, "Play some jazz music", [speech_act: nil], "with nil speech_act"},
    {@test_persona, "Weather in Paris", [], "with empty options"},
  ]

  describe "learn_from_input/3 with options - data driven" do
    for {persona, input, opts, description} <- @learning_with_opts_cases do
      @persona persona
      @input input
      @opts opts
      @description description

      test "#{description}" do
        assert {:ok, _data} = Learner.learn_from_input(@persona, @input, @opts)
      end
    end
  end

  # Test learn_from_classical_extraction/3
  @classical_extraction_cases [
    # {persona, entities, input, description}
    {@test_persona, [], "empty entities", "empty entity list"},
    {@test_persona, [%{value: "Paris", entity: "location"}], "Paris is nice", "single location entity"},
    {@test_persona, [%{value: "John", entity: "person"}, %{value: "Seattle", entity: "location"}],
      "John lives in Seattle", "multiple entities"},
    {@test_persona, [%{"value" => "Test", "entity" => "device"}], "Test device", "string key entity"},
  ]

  describe "learn_from_classical_extraction/3 - data driven" do
    for {persona, entities, input, description} <- @classical_extraction_cases do
      @persona persona
      @entities entities
      @input input
      @description description

      test "#{description}" do
        {:ok, data} = Learner.learn_from_classical_extraction(@persona, @entities, @input)
        assert is_map(data)
      end
    end
  end

  # Edge cases
  describe "edge cases" do
    test "handles unicode input" do
      result = Learner.learn_from_input(@test_persona, "日本語テスト")
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "handles very long input" do
      long_input = String.duplicate("This is a test sentence. ", 100)
      result = Learner.learn_from_input(@test_persona, long_input)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "handles special characters" do
      result = Learner.learn_from_input(@test_persona, "Test's & query <script>")
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end
end
