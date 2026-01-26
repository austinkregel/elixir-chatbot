defmodule ChatBot.Response.TemplateStoreTest do
  use ExUnit.Case, async: false

  alias ChatBot.Response.TemplateStore
  import ChatBot.TestHelpers

  setup do
    start_brain_services()
    # Give TemplateStore time to load
    wait_for_template_store()
    :ok
  end

  defp wait_for_template_store do
    Enum.reduce_while(1..50, false, fn _, _ ->
      if TemplateStore.ready?() do
        {:halt, true}
      else
        Process.sleep(100)
        {:cont, false}
      end
    end)
  end

  describe "ready?/0" do
    test "returns true when loaded" do
      assert TemplateStore.ready?() == true
    end
  end

  describe "get_templates/1" do
    test "returns list of templates for known intent" do
      # This depends on data being loaded
      templates = TemplateStore.get_templates("smalltalk.greetings.hello")
      assert is_list(templates)
    end

    test "returns empty list for unknown intent" do
      templates = TemplateStore.get_templates("completely.unknown.intent")
      assert templates == []
    end
  end

  describe "get_random_template/1" do
    test "returns nil for unknown intent" do
      template = TemplateStore.get_random_template("completely.unknown.intent")
      assert template == nil
    end
  end

  describe "intent_for_speech_act/1" do
    test "returns intent name for greeting" do
      assert TemplateStore.intent_for_speech_act(:greeting) == "smalltalk.greetings.hello"
    end

    test "returns intent name for farewell" do
      assert TemplateStore.intent_for_speech_act(:farewell) == "smalltalk.greetings.bye"
    end

    test "returns intent name for thanks" do
      assert TemplateStore.intent_for_speech_act(:thanks) == "smalltalk.appraisal.thank_you"
    end

    test "returns intent name for apology" do
      assert TemplateStore.intent_for_speech_act(:apology) == "smalltalk.dialog.sorry"
    end

    test "returns nil for unknown speech act" do
      assert TemplateStore.intent_for_speech_act(:unknown_type) == nil
    end

    test "returns nil for non-atom input" do
      assert TemplateStore.intent_for_speech_act("greeting") == nil
    end
  end

  describe "get_expressive_response/1" do
    test "returns response for greeting" do
      response = TemplateStore.get_expressive_response(:greeting)
      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "returns response for farewell" do
      response = TemplateStore.get_expressive_response(:farewell)
      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "returns response for thanks" do
      response = TemplateStore.get_expressive_response(:thanks)
      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "returns nil for unknown speech act" do
      response = TemplateStore.get_expressive_response(:unknown_type)
      assert response == nil
    end
  end

  describe "get_expressive_fallback/1" do
    test "returns fallback for greeting" do
      response = TemplateStore.get_expressive_fallback(:greeting)
      assert is_binary(response)
      assert response in ["Hello!", "Hi there!", "Hey!"]
    end

    test "returns fallback for farewell" do
      response = TemplateStore.get_expressive_fallback(:farewell)
      assert is_binary(response)
      assert response in ["Goodbye!", "See you!", "Take care!"]
    end

    test "returns nil for unknown speech act" do
      response = TemplateStore.get_expressive_fallback(:unknown_type)
      assert response == nil
    end
  end

  describe "substitute_slots/2" do
    test "substitutes entity placeholders in template" do
      template = "The weather in $location is sunny."
      entities = [%{entity_type: "location", value: "New York"}]

      result = TemplateStore.substitute_slots(template, entities)
      assert result == "The weather in New York is sunny."
    end

    test "handles multiple entities" do
      template = "Playing $artist on $device."

      entities = [
        %{entity_type: "music-artist", value: "Mozart"},
        %{entity_type: "device", value: "speaker"}
      ]

      result = TemplateStore.substitute_slots(template, entities)
      # music-artist maps to artist
      assert String.contains?(result, "Mozart")
    end

    test "leaves unmatched placeholders unchanged" do
      template = "Weather in $location at $time."
      entities = [%{entity_type: "location", value: "Paris"}]

      result = TemplateStore.substitute_slots(template, entities)
      assert String.contains?(result, "Paris")
      assert String.contains?(result, "$time")
    end

    test "handles empty entities list" do
      template = "Hello $name!"
      entities = []

      result = TemplateStore.substitute_slots(template, entities)
      assert result == "Hello $name!"
    end
  end

  describe "stats/0" do
    test "returns statistics about loaded templates" do
      stats = TemplateStore.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :intent_count)
      assert Map.has_key?(stats, :template_count)
      assert Map.has_key?(stats, :ready)

      assert stats.ready == true
      assert is_integer(stats.intent_count)
      assert is_integer(stats.template_count)
    end
  end
end
