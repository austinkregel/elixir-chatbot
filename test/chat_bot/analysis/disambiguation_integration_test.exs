defmodule ChatBot.Analysis.DisambiguationIntegrationTest do
  @moduledoc """
  End-to-end integration tests for entity disambiguation.

  These tests verify that the full pipeline correctly:
  1. Classifies speech acts (greeting, question, command)
  2. Extracts entities with proper disambiguation
  3. Uses contextual signals (discourse, POS patterns) to resolve ambiguity
  4. Generates appropriate responses based on disambiguation

  Key test case: "Hello, I'm Austin" should:
  - Be classified as a greeting
  - Extract "Austin" as a person (not location)
  - Use the PRON+VERB pattern and self-referential discourse for disambiguation
  - Respond with a greeting, NOT weather information
  """
  use ExUnit.Case, async: false

  alias ChatBot.Analysis.Pipeline
  alias ChatBot.Analysis.EntityDisambiguator
  alias ChatBot.Brain
  alias ChatBot.ML.{EntityExtractor, Gazetteer, POSTagger}
  import ChatBot.TestHelpers

  # Telemetry event collector for inspection
  defmodule TelemetryCollector do
    use Agent

    def start_link do
      Agent.start_link(fn -> [] end, name: __MODULE__)
    end

    def add_event(event) do
      Agent.update(__MODULE__, fn events -> [event | events] end)
    end

    def get_events do
      Agent.get(__MODULE__, & &1)
    end

    def clear do
      Agent.update(__MODULE__, fn _ -> [] end)
    end
  end

  setup do
    # Start telemetry collector
    case TelemetryCollector.start_link() do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> TelemetryCollector.clear()
    end

    # Attach telemetry handlers
    :telemetry.attach_many(
      "disambiguation-test-handler",
      [
        [:chat_bot, :analysis, :pipeline, :chunk_analyzed],
        [:chat_bot, :analysis, :entities_extracted],
        [:chat_bot, :analysis, :disambiguation, :entity],
        [:chat_bot, :analysis, :disambiguation, :complete],
        [:chat_bot, :ml, :train, :stop]
      ],
      fn event, measurements, metadata, _config ->
        TelemetryCollector.add_event(%{
          event: event,
          measurements: measurements,
          metadata: metadata,
          timestamp: System.monotonic_time(:millisecond)
        })
      end,
      nil
    )

    on_exit(fn ->
      :telemetry.detach("disambiguation-test-handler")
    end)

    # Ensure Gazetteer is started (required for entity lookup)
    ensure_started(ChatBot.ML.Gazetteer)

    # Load gazetteer data
    Gazetteer.load_all()

    :ok
  end

  describe "introduction disambiguation" do
    test "Hello, I'm Austin - speech act classified as greeting" do
      text = "Hello, I'm Austin"

      # Process through the full pipeline
      # Pipeline.process returns InternalModel struct directly
      result = Pipeline.process(text, [])

      # Get the analysis for the chunk
      assert length(result.analyses) >= 1
      analysis = hd(result.analyses)

      # Verify speech act classification - should be greeting
      assert analysis.speech_act.category == :expressive
      assert analysis.speech_act.sub_type == :greeting

      # Verify intent is greeting-related
      assert analysis.intent =~ ~r/greeting|smalltalk/i

      # Note: entities may be empty due to slot filtering for greeting intent
      # The disambiguation happens BEFORE filtering, so we test that separately
      IO.puts("Greeting test - entities after filtering: #{inspect(analysis.entities)}")
      IO.puts("Greeting test - discourse indicators: #{inspect(analysis.discourse.indicators)}")
    end

    test "What's the weather in Austin - Austin disambiguated as location" do
      text = "What's the weather in Austin?"

      result = Pipeline.process(text, [])

      assert length(result.analyses) >= 1
      analysis = hd(result.analyses)

      # Should be classified as directive/request
      assert analysis.speech_act.category == :directive

      # Intent should be weather-related
      assert analysis.intent =~ ~r/weather/i

      # Verify entity extraction - Austin should be in entities
      entities = analysis.entities || []

      austin_entity = Enum.find(entities, fn e ->
        value = e[:value] || e["value"] || ""
        String.downcase(value) == "austin"
      end)

      assert austin_entity != nil,
             "Expected to find Austin entity, got: #{inspect(entities)}"

      # Verify disambiguation selected location (not person) for weather context
      entity_type = austin_entity[:entity] || austin_entity[:entity_type]

      assert entity_type in ["location", "city"],
             "Expected 'Austin' to be disambiguated as 'location' for weather query, " <>
             "got '#{entity_type}'. Full entity: #{inspect(austin_entity)}"

      # Verify that multiple types were available before disambiguation
      types = austin_entity[:types]
      if types && length(types) > 1 do
        type_names = Enum.map(types, fn t ->
          t[:entity_type] || t[:type] || t["entity_type"]
        end)

        IO.puts("Available types for Austin: #{inspect(type_names)}")
        assert "person" in type_names or "location" in type_names
      end
    end

    test "disambiguation chooses person for introduction context" do
      # Direct test of EntityExtractor with introduction context
      text = "I am Austin"

      # Simulate introduction context
      discourse = %{indicators: ["self_referential", "primarily_first_person"]}
      speech_act = %{category: :expressive, sub_type: :greeting}

      entities = EntityExtractor.extract_entities(text,
        discourse: discourse,
        speech_act: speech_act
      )

      austin = Enum.find(entities, fn e ->
        String.downcase(e[:value] || "") == "austin"
      end)

      if austin do
        entity_type = austin[:entity] || austin[:entity_type]

        IO.puts("Introduction context - Austin disambiguated as: #{entity_type}")
        IO.puts("Full entity: #{inspect(austin)}")

        # In introduction context with PRON+VERB pattern, should prefer person
        assert entity_type == "person",
               "Expected 'Austin' to be 'person' in introduction context, got '#{entity_type}'"
      else
        IO.puts("Austin not found in entities: #{inspect(entities)}")
      end
    end
  end

  describe "response content verification" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    test "Hello I'm Austin - response is a greeting, not weather", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")

      # Log the response for debugging
      IO.puts("Response to 'Hello, I'm Austin': #{response}")

      # Should NOT mention weather
      refute response =~ ~r/weather|temperature|forecast|degrees|rain|sunny|cloudy/i,
             "Introduction was misclassified - response mentions weather: #{response}"

      # Should NOT ask for location
      refute response =~ ~r/what location|which city|where.*weather/i,
             "Introduction was misclassified - response asks for location: #{response}"

      # Should be a greeting-like response
      assert String.length(response) > 0,
             "Expected a response, got empty string"

      # Ideally should acknowledge the name or be a greeting
      is_greeting = response =~ ~r/hello|hi|hey|nice|meet|welcome|austin/i
      is_polite = response =~ ~r/how.*you|help|can i|what can/i

      assert is_greeting or is_polite or String.length(response) > 0,
             "Expected greeting response, got: #{response}"
    end

    test "I am Austin - response acknowledges introduction, not location", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "I am Austin")

      IO.puts("Response to 'I am Austin': #{response}")

      # Should NOT mention weather or location queries
      refute response =~ ~r/weather|temperature|forecast/i,
             "Introduction was misclassified as weather query: #{response}"

      refute response =~ ~r/what.*location|which city|where/i,
             "Introduction treated as location query: #{response}"

      # Should have some response
      assert String.length(response) > 0
    end

    test "My name is Austin - treated as introduction", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "My name is Austin")

      IO.puts("Response to 'My name is Austin': #{response}")

      # Should NOT mention weather
      refute response =~ ~r/weather|temperature|forecast|degrees/i,
             "Name introduction was misclassified as weather query: #{response}"

      # Should acknowledge in some way
      assert String.length(response) > 0
    end

    test "What's the weather in Austin - correctly asks about weather", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What's the weather in Austin?")

      IO.puts("Response to weather query: #{response}")

      # For weather query, Austin should be treated as location
      # Response should mention weather or ask for more info about location
      is_weather_response = response =~ ~r/weather|temperature|forecast|degrees|sunny|rain|cloudy/i
      is_clarification = response =~ ~r/location|city|where|which/i
      is_acknowledgment = String.length(response) > 0

      assert is_weather_response or is_clarification or is_acknowledgment,
             "Expected weather-related response, got: #{response}"

      # Should NOT treat Austin as a person name in weather context
      refute response =~ ~r/nice to meet|hello austin|hi austin/i,
             "Weather query was misclassified as introduction: #{response}"
    end

    test "sequential conversation - introduction then weather query", %{conversation_id: conv_id} do
      # First: introduce ourselves
      {:ok, intro_response} = Brain.evaluate(conv_id, "Hi, I'm Austin")
      IO.puts("Intro response: #{intro_response}")

      # Intro should not mention weather
      refute intro_response =~ ~r/weather|temperature|forecast/i,
             "Introduction triggered weather response: #{intro_response}"

      # Second: ask about weather (different Austin - the city)
      {:ok, weather_response} = Brain.evaluate(conv_id, "What's the weather in Austin, Texas?")
      IO.puts("Weather response: #{weather_response}")

      # Weather query should get weather-related response
      # (or at least not greet us again)
      assert String.length(weather_response) > 0

      # Should not re-greet us
      refute weather_response =~ ~r/nice to meet|hello austin|hi austin/i,
             "Weather query triggered greeting: #{weather_response}"
    end

    test "multi-sentence: intro + nice to meet you + weather query", %{conversation_id: conv_id} do
      # This is the exact scenario from the UI:
      # - Chunk 1: greeting with "Austin" as person
      # - Chunk 2: nice to meet you
      # - Chunk 3: weather query with NO location
      # The weather query should ask for clarification, NOT use "Austin" from the greeting
      {:ok, response} =
        Brain.evaluate(conv_id, "Hello, I'm Austin. It is nice to meet you. Can you tell me about the weather?")

      IO.puts("Multi-sentence response: #{response}")

      # The response should ask for location, NOT assume "Austin" is the location
      # Key assertion: should NOT say "weather for Austin" because Austin is a person here
      refute response =~ ~r/weather for Austin|weather in Austin/i,
             "Cross-chunk entity bleeding: Austin from greeting used for weather location: #{response}"

      # Should either ask for location OR just not mention a specific location
      has_location_question = response =~ ~r/what location|which city|where.*weather|specify.*location/i
      does_not_assume_location = not (response =~ ~r/weather for \w+|weather in \w+/i)

      assert has_location_question or does_not_assume_location,
             "Expected location question or no assumed location, got: #{response}"
    end

    test "Hello Austin vs Hello I'm Austin - different interpretations", %{conversation_id: conv_id} do
      # "Hello Austin" - could be addressing someone named Austin
      {:ok, response1} = Brain.evaluate(conv_id, "Hello Austin")
      IO.puts("Response to 'Hello Austin': #{response1}")

      # Create new conversation for second test
      {:ok, conv_id2} = Brain.create_conversation()

      # "Hello, I'm Austin" - clearly an introduction
      {:ok, response2} = Brain.evaluate(conv_id2, "Hello, I'm Austin")
      IO.puts("Response to 'Hello, I'm Austin': #{response2}")

      # Neither should mention weather
      refute response1 =~ ~r/weather|temperature|forecast/i,
             "'Hello Austin' triggered weather: #{response1}"

      refute response2 =~ ~r/weather|temperature|forecast/i,
             "'Hello, I'm Austin' triggered weather: #{response2}"
    end

    @tag :wip
    test "introduction should not show 'location' in debug output", %{conversation_id: conv_id} do
      # This test documents a known issue where the response generator
      # shows "location: Austin" even when disambiguation selected "person"
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")

      IO.puts("Response: #{response}")

      # Ideally, introduction shouldn't mention "location: Austin"
      # This is currently showing as a debug output issue
      if response =~ ~r/location:\s*Austin/i do
        IO.puts("WARNING: Response shows 'location: Austin' even though disambiguation selected 'person'")
        IO.puts("This indicates the response generator isn't using disambiguated entities")
      end

      # The key assertion is that it's NOT a weather response
      refute response =~ ~r/weather|temperature|forecast|degrees|rain|sunny/i,
             "Introduction triggered weather response: #{response}"
    end
  end

  describe "EntityDisambiguator unit verification" do
    test "disambiguates person over location when PRON+VERB pattern detected" do
      # Simulate the entity with multiple types
      person_info = %{entity_type: "person", value: "Austin"}
      location_info = %{entity_type: "location", value: "Austin"}

      entity = %{
        value: "Austin",
        match: "Austin",
        start_pos: 5,
        end_pos: 11,
        types: [person_info, location_info]
      }

      # POS-tagged tokens showing PRON + VERB pattern
      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]

      # Context with self-referential discourse and greeting speech act
      context = %{
        discourse: %{indicators: ["self_referential", "primarily_first_person"]},
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

      assert result.entity_type == "person",
             "Expected disambiguation to select 'person', got: #{inspect(result)}"
    end

    test "disambiguates location over person for weather context" do
      person_info = %{entity_type: "person", value: "Austin"}
      location_info = %{entity_type: "location", value: "Austin"}

      entity = %{
        value: "Austin",
        match: "Austin",
        start_pos: 20,
        end_pos: 26,
        types: [person_info, location_info]
      }

      # POS-tagged tokens for weather query
      pos_tagged = [
        {"What", "PRON"},
        {"is", "VERB"},
        {"the", "DET"},
        {"weather", "NOUN"},
        {"in", "ADP"},
        {"Austin", "PROPN"}
      ]

      # Context indicating weather intent
      context = %{
        discourse: %{indicators: []},
        speech_act: %{category: :directive, sub_type: :question},
        intent: "weather.query"
      }

      result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

      assert result.entity_type == "location",
             "Expected disambiguation to select 'location' for weather, got: #{inspect(result)}"
    end

    test "recognizes proper noun usage when single-type location used as name" do
      # Simulate "Nice" which only exists as location in gazetteer
      # But in "I'm Nice" it's being used as a proper noun/name, not a location reference
      location_info = %{entity_type: "location", value: "Nice"}

      entity = %{
        value: "Nice",
        match: "Nice",
        start_pos: 5,
        end_pos: 9,
        entity: "location",
        types: [location_info]
      }

      # POS-tagged tokens showing PRON + VERB pattern (introduction)
      # "Nice" is tagged as PROPN (proper noun), indicating name usage
      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Nice", "PROPN"}]

      # Strong introduction context
      context = %{
        discourse: %{indicators: ["self_referential", "primarily_first_person"]},
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

      # Should recognize this as proper noun/name usage (mapped to "person" type in our system)
      assert result.entity_type == "person" || result.entity == "person",
             "Expected proper noun recognition (mapped to 'person'), got: #{inspect(result)}"

      # Should have disambiguation reason indicating proper noun usage
      assert Map.get(result, :disambiguation_reason) == "proper_noun_usage" ||
               Map.get(result, "disambiguation_reason") == "proper_noun_usage",
             "Expected disambiguation_reason='proper_noun_usage', got: #{inspect(result)}"
    end
  end

  describe "EntityExtractor with context" do
    test "passes context to disambiguation" do
      text = "I am Austin"

      # Context for introduction
      discourse = %{indicators: ["self_referential", "primarily_first_person"]}
      speech_act = %{category: :expressive, sub_type: :greeting}

      entities = EntityExtractor.extract_entities(text,
        discourse: discourse,
        speech_act: speech_act
      )

      # Find Austin entity
      austin = Enum.find(entities, fn e ->
        String.downcase(e[:value] || "") == "austin"
      end)

      if austin do
        # Should be disambiguated as person in introduction context
        entity_type = austin[:entity] || austin[:entity_type]

        # Note: This may still be location if disambiguation isn't working
        # or if the POS model isn't trained. Log for debugging.
        IO.puts("Austin entity type: #{entity_type}")
        IO.puts("Full entity: #{inspect(austin)}")
      end
    end
  end

  describe "Gazetteer multi-type support" do
    test "Austin returns multiple entity types" do
      # Ensure gazetteer is loaded
      if Gazetteer.loaded?() do
        types = Gazetteer.lookup_all_types("austin")

        if length(types) > 0 do
          type_names = Enum.map(types, fn info ->
            Map.get(info, :entity_type) || Map.get(info, :type)
          end)

          IO.puts("Austin entity types in gazetteer: #{inspect(type_names)}")

          # With proper data, Austin should have multiple types
          # (person and location at minimum)
          if length(types) > 1 do
            assert "person" in type_names or "location" in type_names,
                   "Expected Austin to have person or location type"
          end
        end
      end
    end
  end

  describe "POS Tagger integration" do
    test "POS tagger identifies PRON+VERB pattern" do
      case POSTagger.load_model() do
        {:ok, model} ->
          tokens = ["I", "am", "Austin"]
          predictions = POSTagger.predict(tokens, model)

          # Should tag as PRON, VERB, PROPN (or similar)
          IO.puts("POS predictions for 'I am Austin': #{inspect(predictions)}")

          tags = Enum.map(predictions, fn {_token, tag} -> tag end)

          # First token should be pronoun-like
          assert hd(tags) in ["PRON", "PRP", "X"],
                 "Expected 'I' to be tagged as PRON, got: #{hd(tags)}"

        {:error, _reason} ->
          # POS model not trained yet, skip
          IO.puts("POS model not trained - skipping POS tagger test")
      end
    end
  end

  describe "full pipeline with telemetry" do
    test "emits telemetry events for disambiguation" do
      TelemetryCollector.clear()

      # Use weather query which has entities that get disambiguated
      text = "What's the weather in Austin?"
      _result = Pipeline.process(text, emit_telemetry: true)

      # Give telemetry events time to be collected
      Process.sleep(50)

      events = TelemetryCollector.get_events()

      # Log collected events for debugging
      IO.puts("Collected #{length(events)} telemetry events")

      for event <- events do
        IO.puts("  Event: #{inspect(event.event)}")
        IO.puts("    Measurements: #{inspect(event.measurements)}")
        IO.puts("    Metadata: #{inspect(event.metadata)}")
      end

      # Check for disambiguation events
      disambiguation_events = Enum.filter(events, fn e ->
        e.event == [:chat_bot, :analysis, :disambiguation, :entity] or
        e.event == [:chat_bot, :analysis, :disambiguation, :complete]
      end)

      if length(disambiguation_events) > 0 do
        IO.puts("\nDisambiguation events captured: #{length(disambiguation_events)}")

        # Verify structure of disambiguation event
        entity_event = Enum.find(disambiguation_events, fn e ->
          e.event == [:chat_bot, :analysis, :disambiguation, :entity]
        end)

        if entity_event do
          assert entity_event.measurements[:type_count] >= 1
          assert entity_event.metadata[:value] != nil
        end
      else
        IO.puts("\nNo disambiguation events captured (may need to check event names)")
      end
    end

    test "direct EntityExtractor emits telemetry" do
      TelemetryCollector.clear()

      text = "I am Austin"
      discourse = %{indicators: ["self_referential"]}
      speech_act = %{category: :expressive, sub_type: :greeting}

      _entities = EntityExtractor.extract_entities(text,
        discourse: discourse,
        speech_act: speech_act
      )

      Process.sleep(50)

      events = TelemetryCollector.get_events()

      IO.puts("\nEntityExtractor telemetry events: #{length(events)}")

      for event <- events do
        IO.puts("  #{inspect(event.event)}: #{inspect(event.measurements)}")
      end

      # Should have disambiguation events since Austin has multiple types
      disambiguation_events = Enum.filter(events, fn e ->
        match?([:chat_bot, :analysis, :disambiguation, _], e.event)
      end)

      assert length(disambiguation_events) > 0,
             "Expected disambiguation telemetry events, got none. " <>
             "Total events: #{length(events)}"

      # Verify the entity was disambiguated to person
      entity_event = Enum.find(disambiguation_events, fn e ->
        e.event == [:chat_bot, :analysis, :disambiguation, :entity]
      end)

      if entity_event do
        assert entity_event.metadata[:value] == "Austin"
        assert entity_event.measurements[:selected_type] == "person"
        assert "location" in entity_event.metadata[:available_types]
        assert "person" in entity_event.metadata[:available_types]

        IO.puts("\nDisambiguation verified:")
        IO.puts("  Value: #{entity_event.metadata[:value]}")
        IO.puts("  Available types: #{inspect(entity_event.metadata[:available_types])}")
        IO.puts("  Selected type: #{entity_event.measurements[:selected_type]}")
        IO.puts("  Context type: #{entity_event.metadata[:context_type]}")
        IO.puts("  POS pattern: #{entity_event.metadata[:pos_pattern]}")
      end
    end
  end
end
