defmodule ChatBot.Analysis.AnaphoraResolverDataTest do
  @moduledoc """
  Data-driven tests for AnaphoraResolver covering pronoun and demonstrative resolution.
  """
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  alias ChatBot.Analysis.AnaphoraResolver

  setup do
    ensure_pubsub_started()
    ensure_started(ChatBot.ML.Gazetteer)

    # Load gazetteer data for anaphora type lookups
    try do
      ChatBot.ML.Gazetteer.load_all()
    catch
      _, _ -> :ok
    end

    :ok
  end

  # Test data for anaphora resolution - testing that function runs and returns valid types
  @anaphora_test_cases [
    # {input_text, conversation_history, description}
    {"it is working", [], "no history"},
    {"play it", [], "pronoun with empty history"},
    {"turn it off", [%{entities: [%{value: "lamp", type: "device"}], timestamp: DateTime.utc_now()}], "device pronoun with history"},
    {"she said hello", [%{entities: [%{value: "Alice", type: "person"}], timestamp: DateTime.utc_now()}], "person pronoun with history"},
    {"he is coming", [%{entities: [%{value: "Bob", type: "person"}], timestamp: DateTime.utc_now()}], "person pronoun he"},
    {"what about that", [%{entities: [%{value: "weather", type: "topic"}], timestamp: DateTime.utc_now()}], "demonstrative that"},
    {"this one please", [%{entities: [%{value: "song", type: "song"}], timestamp: DateTime.utc_now()}], "demonstrative this"},
    {"Paris is beautiful", [], "proper noun not anaphoric"},
    {"I like coffee", [], "first person pronoun"},
    {"we should go", [], "first person plural"},
  ]

  describe "resolve/2 - data driven" do
    for {input, history, description} <- @anaphora_test_cases do
      @input input
      @history history
      @description description

      test "returns valid result for: #{description}" do
        result = AnaphoraResolver.resolve(@input, @history)

        # Should return a tuple with either :resolved or :no_anaphora
        assert match?({:resolved, _}, result) or match?({:no_anaphora, _}, result)

        case result do
          {:resolved, resolutions} ->
            assert is_list(resolutions)

          {:no_anaphora, list} ->
            assert list == []
        end
      end
    end
  end

  # Test substitution function
  @substitution_test_cases [
    {"play it", [%{entities: [%{value: "Jazz", type: "song"}], timestamp: DateTime.utc_now()}], "substitutes pronoun"},
    {"turn it on", [%{entities: [%{value: "lamp", type: "device"}], timestamp: DateTime.utc_now()}], "substitutes device pronoun"},
    {"hello world", [], "no substitution needed"},
  ]

  describe "resolve_and_substitute/2 - data driven" do
    for {input, history, description} <- @substitution_test_cases do
      @input input
      @history history
      @description description

      test "returns valid result for: #{description}" do
        result = AnaphoraResolver.resolve_and_substitute(@input, @history)

        # Should return {:ok, substituted_text, resolved_entities}
        assert is_tuple(result)
        assert tuple_size(result) == 3

        {status, substituted, entities} = result
        assert status == :ok
        assert is_binary(substituted)
        assert is_list(entities)
      end
    end
  end

  # Edge cases
  @edge_cases [
    {"", [], "empty input"},
    {"   ", [], "whitespace only"},
    {"a b c", [], "short tokens"},
    {String.duplicate("word ", 50), [], "long input"},
    {"日本語テスト", [], "unicode input"},
  ]

  describe "edge cases - data driven" do
    for {input, history, description} <- @edge_cases do
      @input input
      @history history
      @description description

      test "handles #{description} without crashing" do
        result = AnaphoraResolver.resolve(@input, @history)
        assert match?({_, _}, result)
      end
    end
  end
end
