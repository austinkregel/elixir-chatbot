defmodule Brain.LearnerDataTest do
  @moduledoc """
  Data-driven tests for the Learner module covering learning from user inputs.
  """
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Learner

  @test_persona "LearnerTestPersona"

  # The world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

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
        {:ok, data} = Learner.learn_from_input(@persona, @input, world_id: @world_id)
        assert is_map(data)
      end
    end
  end

  # Test with options
  @learning_with_opts_cases [
    {@test_persona, "Turn on the kitchen light", [world_id: @world_id, discourse: nil], "with nil discourse"},
    {@test_persona, "Play some jazz music", [world_id: @world_id, speech_act: nil], "with nil speech_act"},
    {@test_persona, "Weather in Paris", [world_id: @world_id], "with no option but the world"},
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

  describe "learn_from_input/3 without a world" do
    test "raises when :world_id is absent" do
      assert_raise ArgumentError, ~r/Learner.learn_from_input needs :world_id/, fn ->
        Learner.learn_from_input(@test_persona, "My cat is named Whiskers", [])
      end
    end

    test "raises when :world_id is empty" do
      assert_raise ArgumentError, ~r/Learner.learn_from_input needs :world_id/, fn ->
        Learner.learn_from_input(@test_persona, "My cat is named Whiskers", world_id: "")
      end
    end
  end

  # Test learn_from_classical_extraction/3. Entities are in the analysis
  # shape a conversation turn passes: the type is under :entity_type.
  @classical_extraction_cases [
    # {persona, entities, input, description}
    {@test_persona, [], "empty entities", "empty entity list"},
    {@test_persona, [%{value: "Paris", entity_type: "location"}], "Paris is nice", "single location entity"},
    {@test_persona, [%{value: "John", entity_type: "person"}, %{value: "Seattle", entity_type: "location"}],
      "John lives in Seattle", "multiple entities"},
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

    test "each entity's type is read from :entity_type" do
      entities = [%{value: "John", entity_type: "person"}, %{value: "Seattle", entity_type: "location"}]

      {:ok, data} = Learner.learn_from_classical_extraction(@test_persona, entities, "John lives in Seattle")

      assert Enum.map(data["entities"], &{&1["name"], &1["type"]}) ==
               [{"John", "person"}, {"Seattle", "location"}]
    end

    test "an entity without :entity_type raises" do
      assert_raise KeyError, ~r/:entity_type/, fn ->
        Learner.learn_from_classical_extraction(@test_persona, [%{value: "Paris", entity: "location"}], "Paris")
      end
    end
  end

  describe "spans entity extraction could not decide" do
    test "learn_from_input/3 learns no entity from one" do
      for {text, undecided} <- undecided_span_cases(@world_id, 3) do
        {:ok, data} = Learner.learn_from_input(@test_persona, text, world_id: @world_id)

        learned = data |> Map.get("entities", []) |> MapSet.new(&String.downcase(&1["name"]))

        for span <- undecided do
          refute MapSet.member?(learned, String.downcase(span.value)),
                 "#{inspect(span.value)} of #{inspect(text)} was learned as an entity: #{inspect(data["entities"])}"
        end
      end
    end
  end

  # Edge cases
  describe "edge cases" do
    test "handles unicode input" do
      result = Learner.learn_from_input(@test_persona, "日本語テスト", world_id: @world_id)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "handles very long input" do
      long_input = String.duplicate("This is a test sentence. ", 100)
      result = Learner.learn_from_input(@test_persona, long_input, world_id: @world_id)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "handles special characters" do
      result = Learner.learn_from_input(@test_persona, "Test's & query <script>", world_id: @world_id)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end
end
