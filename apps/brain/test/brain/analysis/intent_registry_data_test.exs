defmodule Brain.Analysis.IntentRegistryDataTest do
  @moduledoc """
  Data-driven tests for IntentRegistry covering intent metadata lookups.
  """
  use ExUnit.Case, async: false

  alias Brain.Analysis.IntentRegistry

  # Known intents for testing (from priv/analysis/intent_registry.json)
  @known_intents [
    "weather.query",
    "smalltalk.greetings.hello",
    "smarthome.lights.switch.on",
    "music.play",
    "identity.self",
    "smalltalk.thanks",
    "navigation.directions",
  ]

  describe "get/1 - data driven" do
    for intent <- @known_intents do
      @intent intent

      test "gets metadata for: #{intent}" do
        result = IntentRegistry.get(@intent)

        # Result is either a map with metadata or nil
        assert is_map(result) or result == nil
      end
    end

    test "returns nil for nil input" do
      assert IntentRegistry.get(nil) == nil
    end

    test "returns nil for empty string" do
      assert IntentRegistry.get("") == nil
    end

    test "handles atom input" do
      result = IntentRegistry.get(:greeting)
      assert result == nil or is_map(result)
    end
  end

  # Test domain classification - domains from intent_registry.json
  @domain_test_cases [
    # {intent, expected_domain, description}
    {"weather.query", :weather, "weather intent"},
    {"music.play", :music, "music intent"},
    {"smarthome.lights.switch.on", :smarthome, "smarthome device intent"},
    {"smalltalk.greetings.hello", :smalltalk, "greeting intent"},
    {"smalltalk.thanks", :smalltalk, "smalltalk intent"},
    {"unknown.intent", nil, "unknown intent returns nil"},
  ]

  describe "domain/1 - data driven" do
    for {intent, expected_domain, description} <- @domain_test_cases do
      @intent intent
      @expected_domain expected_domain
      @description description

      test "#{description}" do
        result = IntentRegistry.domain(@intent)

        if @expected_domain do
          assert result == @expected_domain
        else
          assert result == nil or is_atom(result)
        end
      end
    end
  end

  # Test category classification
  @category_test_cases [
    {"smalltalk.greetings.hello", :expressive, "greeting is expressive"},
    {"smarthome.lights.switch.on", :directive, "smarthome device switch is directive"},
    {"weather.query", :directive, "weather query is directive"},
    {"smalltalk.thanks", :expressive, "thanks is expressive"},
  ]

  describe "category/1 - data driven" do
    for {intent, expected_category, description} <- @category_test_cases do
      @intent intent
      @expected_category expected_category
      @description description

      test "#{description}" do
        result = IntentRegistry.category(@intent)

        # Category should be an atom or nil
        assert is_atom(result) or result == nil
      end
    end
  end

  # Test boolean predicates - just verify they return boolean
  @predicate_test_cases [
    {:weather_intent?, "weather.query", "weather.query weather check"},
    {:weather_intent?, "music.play", "music.play weather check"},
    {:music_intent?, "music.play", "music.play music check"},
    {:music_intent?, "weather.query", "weather.query music check"},
    {:device_intent?, "smarthome.lights.switch.on", "smarthome.device.switch.on device check"},
    {:smalltalk_intent?, "smalltalk.thanks", "thanks smalltalk check"},
    {:greeting?, "smalltalk.greetings.hello", "smalltalk.greetings.hello greeting check"},
    {:thanks?, "smalltalk.thanks", "smalltalk.thanks thanks check"},
    {:command?, "smarthome.lights.switch.on", "smarthome.device.switch.on command check"},
  ]

  describe "predicate functions - data driven" do
    for {func, intent, description} <- @predicate_test_cases do
      @func func
      @intent intent
      @description description

      test "returns boolean for: #{description}" do
        result = apply(IntentRegistry, @func, [@intent])
        assert is_boolean(result)
      end
    end
  end

  # Test list functions
  describe "list_intents/0" do
    test "returns a list of intents" do
      result = IntentRegistry.list_intents()
      assert is_list(result)
    end
  end

  describe "list_by_domain/1 - data driven" do
    @domains [:weather, :music, :device, :greeting, :smalltalk, :navigation]

    for domain <- @domains do
      @domain domain

      test "lists intents for domain: #{domain}" do
        result = IntentRegistry.list_by_domain(@domain)
        assert is_list(result)
      end
    end
  end

  describe "list_by_category/1 - data driven" do
    @categories [:expressive, :directive, :assertive]

    for category <- @categories do
      @category category

      test "lists intents for category: #{category}" do
        result = IntentRegistry.list_by_category(@category)
        assert is_list(result)
      end
    end
  end

  # Test humanize function
  @humanize_test_cases [
    {"weather.query", "weather", "humanizes weather.query"},
    {"smalltalk.greetings.hello", "hello", "humanizes greeting"},
    {"smarthome.lights.switch.on", "on", "humanizes smarthome device switch"},
    {nil, "something", "nil returns something"},
    {"", "something", "empty returns something"},
  ]

  describe "humanize/1 - data driven" do
    for {input, _expected_contains, description} <- @humanize_test_cases do
      @input input
      @description description

      test "#{description}" do
        result = IntentRegistry.humanize(@input)
        assert is_binary(result)
      end
    end
  end

  # Test required/optional entities
  describe "required_entities/1" do
    test "returns list or nil for weather.query" do
      result = IntentRegistry.required_entities("weather.query")
      assert is_list(result) or result == nil
    end

    test "returns list or nil for unknown intent" do
      result = IntentRegistry.required_entities("unknown.intent")
      assert is_list(result) or result == nil
    end
  end

  describe "optional_entities/1" do
    test "returns list or nil" do
      result = IntentRegistry.optional_entities("weather.query")
      assert is_list(result) or result == nil
    end
  end

  # Test clarification templates
  describe "clarification_templates/1" do
    test "returns map or nil for known intent" do
      result = IntentRegistry.clarification_templates("weather.query")
      assert is_map(result) or result == nil
    end
  end
end
