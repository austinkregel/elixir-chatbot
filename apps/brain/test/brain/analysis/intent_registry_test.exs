defmodule Brain.Analysis.IntentRegistryTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.IntentRegistry

  describe "humanize/1" do
    test "converts dotted intent to readable format" do
      assert IntentRegistry.humanize("weather.query") == "weather query"
    end

    test "removes smalltalk prefix" do
      assert IntentRegistry.humanize("smalltalk.greetings.hello") == "greetings hello"
    end

    test "converts underscores to spaces" do
      assert IntentRegistry.humanize("device_control.turn_on") == "device control turn on"
    end

    test "handles nil input" do
      assert IntentRegistry.humanize(nil) == "something"
    end

    test "handles empty string" do
      assert IntentRegistry.humanize("") == "something"
    end

    test "handles non-string input" do
      assert IntentRegistry.humanize(123) == "something"
    end

    test "trims whitespace" do
      # After removing "smalltalk " from "smalltalk.foo", we get " foo" which should be trimmed
      assert IntentRegistry.humanize("smalltalk.foo") == "foo"
    end
  end

  describe "clarification_templates/1" do
    test "returns templates for weather.query" do
      templates = IntentRegistry.clarification_templates("weather.query")

      assert Map.get(templates, "location") == "What location would you like the weather for?"
    end

    test "returns templates for smarthome device intent" do
      templates = IntentRegistry.clarification_templates("smarthome.lights.switch.on")

      # Should return a map (possibly with clarification templates from registry)
      assert is_map(templates)
    end

    test "returns empty map for unknown intent" do
      templates = IntentRegistry.clarification_templates("unknown.intent")

      assert templates == %{}
    end
  end

  describe "intent_for_speech_act/1" do
    test "maps greeting to appropriate intent" do
      intent = IntentRegistry.intent_for_speech_act(:greeting)

      assert intent == "smalltalk.greetings.hello"
    end

    test "maps farewell to appropriate intent" do
      intent = IntentRegistry.intent_for_speech_act(:farewell)

      assert intent == "smalltalk.greetings.bye"
    end

    test "maps thanks to appropriate intent" do
      intent = IntentRegistry.intent_for_speech_act(:thanks)

      # Actual mapping in the registry
      assert intent == "smalltalk.appraisal.thank_you"
    end

    test "returns nil for unknown speech act" do
      intent = IntentRegistry.intent_for_speech_act(:unknown_act)

      assert intent == nil
    end
  end

  describe "domain/1" do
    test "returns domain for weather intent" do
      domain = IntentRegistry.domain("weather.query")

      # domain returns an atom
      assert domain == :weather
    end

    test "returns nil for unknown intent" do
      domain = IntentRegistry.domain("unknown.intent")

      assert domain == nil
    end
  end

  describe "category/1" do
    test "returns category for known intent" do
      category = IntentRegistry.category("weather.query")

      assert category != nil
    end
  end

  describe "speech_act/1" do
    test "returns speech act for greeting intent" do
      speech_act = IntentRegistry.speech_act("smalltalk.greetings.hello")

      # speech_act returns the speech_act field value, not the category
      assert speech_act == :greeting
    end

    test "returns nil for unknown intent" do
      speech_act = IntentRegistry.speech_act("unknown.intent")

      assert speech_act == nil
    end
  end
end
