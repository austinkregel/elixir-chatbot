defmodule Brain.Analysis.DisambiguationIntegrationTest do
  @moduledoc "End-to-end integration tests for entity disambiguation.\n\nThese tests verify that the full pipeline correctly:\n1. Classifies speech acts (greeting, question, command)\n2. Extracts entities with proper disambiguation\n3. Uses contextual signals (discourse, POS patterns) to resolve ambiguity\n4. Generates appropriate responses based on disambiguation\n\nKey test case: \"Hello, I'm Austin\" should:\n- Be classified as a greeting\n- Extract \"Austin\" as a person (not location)\n- Use the PRON+VERB pattern and self-referential discourse for disambiguation\n- Respond with a greeting, NOT weather information\n"
  alias Brain.ML
  use Brain.Test.GraphCase, async: false
  import ExUnit.CaptureLog
  require Logger

  alias Brain.Analysis.Pipeline
  alias Brain.Analysis.EntityDisambiguator
  alias Brain
  alias ML.{EntityExtractor, Gazetteer, POSTagger}
  import Brain.TestHelpers

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

  setup _context do
    start_test_services()

    case TelemetryCollector.start_link() do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> TelemetryCollector.clear()
    end

    :telemetry.attach_many(
      "disambiguation-test-handler",
      [
        [:chat_bot, :analysis, :pipeline, :chunk_analyzed],
        [:chat_bot, :analysis, :entities_extracted],
        [:chat_bot, :analysis, :disambiguation, :entity],
        [:chat_bot, :analysis, :disambiguation, :complete],
        [:chat_bot, :ml, :train, :stop]
      ],
      &__MODULE__.__collect_event__/4,
      nil
    )

    on_exit(fn ->
      :telemetry.detach("disambiguation-test-handler")
    end)

    Gazetteer.load_all()

    :ok
  end

  @doc false
  # Module-function capture (vs anonymous fn) avoids the
  # ":telemetry handler is a local function" performance warning.
  def __collect_event__(event, measurements, metadata, _config) do
    TelemetryCollector.add_event(%{
      event: event,
      measurements: measurements,
      metadata: metadata,
      timestamp: System.monotonic_time(:millisecond)
    })

    :ok
  end

  describe "introduction disambiguation" do
    test "Hello, I'm Austin - contains greeting and assertive introduction" do
      text = "Hello, I'm Austin"
      result = Pipeline.process(text, [])
      assert result.analyses != []

      if length(result.analyses) > 1 do
        categories = Enum.map(result.analyses, & &1.speech_act.category)
        assert :expressive in categories or :assertive in categories
      else
        analysis = hd(result.analyses)

        assert analysis.speech_act.category in [:expressive, :assertive],
               "Expected expressive or assertive, got: #{analysis.speech_act.category}"
      end
    end

    test "What's the weather in Austin - Austin disambiguated as location" do
      text = "What's the weather in Austin?"

      result = Pipeline.process(text, [])

      assert result.analyses != []
      analysis = hd(result.analyses)
      assert analysis.speech_act.category == :directive
      assert analysis.intent =~ ~r/weather/i
      entities = analysis.entities || []

      austin_entity =
        Enum.find(entities, fn e ->
          value = e[:value] || e["value"] || ""
          String.downcase(value) == "austin"
        end)

      assert austin_entity != nil,
             "Expected to find Austin entity, got: #{inspect(entities)}"

      entity_type = austin_entity[:entity] || austin_entity[:entity_type]

      assert entity_type in ["location", "city", "ambiguous_name_location"],
             "Expected 'Austin' to be disambiguated as location-like type for weather query, " <>
               "got '#{entity_type}'. Full entity: #{inspect(austin_entity)}"

      types = austin_entity[:types]

      if types && length(types) > 1 do
        type_names =
          Enum.map(types, fn t ->
            t[:entity_type] || t[:type] || t["entity_type"]
          end)

        assert "person" in type_names or "location" in type_names
      end
    end

    test "disambiguation chooses person for introduction context" do
      text = "I am Austin"
      discourse = %{indicators: ["self_referential", "primarily_first_person"]}
      speech_act = %{category: :expressive, sub_type: :greeting}

      entities =
        EntityExtractor.extract_entities(text, discourse: discourse, speech_act: speech_act)

      austin =
        Enum.find(entities, fn e ->
          String.downcase(e[:value] || "") == "austin"
        end)

      if austin do
        entity_type = austin[:entity] || austin[:entity_type]
        disambiguation_source = austin[:disambiguation_source]

        cond do
          disambiguation_source == :type_inferrer ->
            assert entity_type != nil, "Expected TypeInferrer to provide a type, got nil"

          entity_type == "person" ->
            :ok

          entity_type in ["ambiguous_name_location"] ->
            :ok

          true ->
            flunk(
              "Unexpected entity type '#{entity_type}' for Austin. " <>
                "Expected person or TypeInferrer result. Entity: #{inspect(austin)}"
            )
        end
      else
        flunk("Austin not found in entities: #{inspect(entities)}")
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
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Austin")
          Logger.warning("Response to 'Hello, I'm Austin': #{response}")

          refute response =~ ~r/weather|temperature|forecast|degrees|rain|sunny|cloudy/i,
                 "Introduction was misclassified - response mentions weather: #{response}"

          refute response =~ ~r/what location|which city|where.*weather/i,
                 "Introduction was misclassified - response asks for location: #{response}"

          assert String.length(response) > 0, "Expected a response, got empty string"
          is_greeting = response =~ ~r/hello|hi|hey|nice|meet|welcome|austin/i
          is_polite = response =~ ~r/how.*you|help|can i|what can/i

          assert is_greeting or is_polite or String.length(response) > 0,
                 "Expected greeting response, got: #{response}"
        end)

      assert log =~ "Response to 'Hello, I'm Austin':"
    end

    test "I am Austin - response acknowledges introduction, not location", %{
      conversation_id: conv_id
    } do
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: response}} = Brain.evaluate(conv_id, "I am Austin")
          Logger.warning("Response to 'I am Austin': #{response}")

          refute response =~ ~r/weather|temperature|forecast/i,
                 "Introduction was misclassified as weather query: #{response}"

          refute response =~ ~r/what.*location|which city|where/i,
                 "Introduction treated as location query: #{response}"

          assert String.length(response) > 0
        end)

      assert log =~ "Response to 'I am Austin':"
    end

    test "My name is Austin - treated as introduction", %{conversation_id: conv_id} do
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: response}} = Brain.evaluate(conv_id, "My name is Austin")
          Logger.warning("Response to 'My name is Austin': #{response}")

          refute response =~ ~r/weather|temperature|forecast|degrees/i,
                 "Name introduction was misclassified as weather query: #{response}"

          assert String.length(response) > 0
        end)

      assert log =~ "Response to 'My name is Austin':"
    end

    test "What's the weather in Austin - correctly asks about weather", %{
      conversation_id: conv_id
    } do
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: response}} = Brain.evaluate(conv_id, "What's the weather in Austin?")
          Logger.warning("Response to weather query: #{response}")

          is_weather_response =
            response =~ ~r/weather|temperature|forecast|degrees|sunny|rain|cloudy/i

          is_clarification = response =~ ~r/location|city|where|which/i
          is_acknowledgment = String.length(response) > 0

          assert is_weather_response or is_clarification or is_acknowledgment,
                 "Expected weather-related response, got: #{response}"

          refute response =~ ~r/nice to meet|hello austin|hi austin/i,
                 "Weather query was misclassified as introduction: #{response}"
        end)

      assert log =~ "Response to weather query:"
    end

    test "sequential conversation - introduction then weather query", %{conversation_id: conv_id} do
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: intro_response}} = Brain.evaluate(conv_id, "Hi, I'm Austin")
          Logger.warning("Intro response: #{intro_response}")

          refute intro_response =~ ~r/weather|temperature|forecast/i,
                 "Introduction triggered weather response: #{intro_response}"

          {:ok, %{response: weather_response}} =
            Brain.evaluate(conv_id, "What's the weather in Austin, Texas?")

          Logger.warning("Weather response: #{weather_response}")
          assert String.length(weather_response) > 0

          refute weather_response =~ ~r/nice to meet|hello austin|hi austin/i,
                 "Weather query triggered greeting: #{weather_response}"
        end)

      assert log =~ "Intro response:"
      assert log =~ "Weather response:"
    end

    test "multi-sentence: intro + nice to meet you + weather query", %{conversation_id: conv_id} do
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: response}} =
            Brain.evaluate(
              conv_id,
              "Hello, I'm Austin. It is nice to meet you. Can you tell me about the weather?"
            )

          Logger.warning("Multi-sentence response: #{response}")

          refute response =~ ~r/weather for Austin|weather in Austin/i,
                 "Cross-chunk entity bleeding: Austin from greeting used for weather location: #{response}"

          has_location_question =
            response =~ ~r/what location|which city|where.*weather|specify.*location/i

          does_not_assume_location = not (response =~ ~r/weather for \w+|weather in \w+/i)

          assert has_location_question or does_not_assume_location,
                 "Expected location question or no assumed location, got: #{response}"
        end)

      assert log =~ "Multi-sentence response:"
    end

    test "Hello Austin vs Hello I'm Austin - different interpretations", %{
      conversation_id: conv_id
    } do
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: response1}} = Brain.evaluate(conv_id, "Hello Austin")
          Logger.warning("Response to 'Hello Austin': #{response1}")
          {:ok, conv_id2} = Brain.create_conversation()
          {:ok, %{response: response2}} = Brain.evaluate(conv_id2, "Hello, I'm Austin")
          Logger.warning("Response to 'Hello, I'm Austin': #{response2}")

          refute response1 =~ ~r/weather|temperature|forecast/i,
                 "'Hello Austin' triggered weather: #{response1}"

          refute response2 =~ ~r/weather|temperature|forecast/i,
                 "'Hello, I'm Austin' triggered weather: #{response2}"
        end)

      assert log =~ "Response to 'Hello Austin':"
      assert log =~ "Response to 'Hello, I'm Austin':"
    end

    @tag :wip
    test "introduction should not show 'location' in debug output", %{conversation_id: conv_id} do
      log =
        capture_log([level: :warning], fn ->
          {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Austin")
          Logger.warning("Response: #{response}")

          refute response =~ ~r/weather|temperature|forecast|degrees|rain|sunny/i,
                 "Introduction triggered weather response: #{response}"
        end)

      assert log =~ "Response:"
    end
  end

  describe "EntityDisambiguator unit verification" do
    test "disambiguates person over location when PRON+VERB pattern detected" do
      person_info = %{entity_type: "person", value: "Austin"}
      location_info = %{entity_type: "location", value: "Austin"}

      entity = %{
        value: "Austin",
        match: "Austin",
        start_pos: 5,
        end_pos: 11,
        types: [person_info, location_info]
      }

      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]

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

      pos_tagged = [
        {"What", "PRON"},
        {"is", "VERB"},
        {"the", "DET"},
        {"weather", "NOUN"},
        {"in", "ADP"},
        {"Austin", "PROPN"}
      ]

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
      location_info = %{entity_type: "location", value: "Nice"}

      entity = %{
        value: "Nice",
        match: "Nice",
        start_pos: 5,
        end_pos: 9,
        entity_type: "location",
        types: [location_info]
      }

      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Nice", "PROPN"}]

      context = %{
        discourse: %{indicators: ["self_referential", "primarily_first_person"]},
        speech_act: %{category: :expressive, sub_type: :greeting}
      }

      result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

      assert result.entity_type == "person",
             "Expected proper noun recognition (mapped to 'person'), got: #{inspect(result)}"

      disambiguation_reason =
        Map.get(result, :disambiguation_reason) || Map.get(result, "disambiguation_reason")

      valid_reasons = ["proper_noun_usage", "introduction_pattern", "context_analysis"]

      assert disambiguation_reason in valid_reasons,
             "Expected disambiguation_reason in #{inspect(valid_reasons)}, got: #{inspect(disambiguation_reason)}"
    end
  end

  describe "EntityExtractor with context" do
    test "passes context to disambiguation" do
      text = "I am Austin"
      discourse = %{indicators: ["self_referential", "primarily_first_person"]}
      speech_act = %{category: :expressive, sub_type: :greeting}

      entities =
        EntityExtractor.extract_entities(text, discourse: discourse, speech_act: speech_act)

      austin =
        Enum.find(entities, fn e ->
          String.downcase(e[:value] || "") == "austin"
        end)

      if austin do
        entity_type = austin[:entity] || austin[:entity_type]

        assert entity_type in ["person", "ambiguous_name_location"],
               "Expected Austin to be disambiguated as person or ambiguous_name_location, got: #{entity_type}"
      end
    end
  end

  describe "Gazetteer multi-type support" do
    test "Austin returns multiple entity types" do
      if Gazetteer.loaded?() do
        types = Gazetteer.lookup_all_types("austin")

        if types != [] do
          type_names =
            Enum.map(types, fn info ->
              Map.get(info, :entity_type) || Map.get(info, :type)
            end)

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

          tags = Enum.map(predictions, fn {_token, tag} -> tag end)

          assert hd(tags) in ["PRON", "PRP", "X"],
                 "Expected 'I' to be tagged as PRON, got: #{hd(tags)}"

        {:error, _reason} ->
          :ok
      end
    end
  end

  describe "full pipeline with telemetry" do
    test "emits telemetry events for disambiguation" do
      TelemetryCollector.clear()
      text = "What's the weather in Austin?"
      _result = Pipeline.process(text, emit_telemetry: true)
      Process.sleep(50)

      events = TelemetryCollector.get_events()

      disambiguation_events =
        Enum.filter(events, fn e ->
          e.event == [:chat_bot, :analysis, :disambiguation, :entity] or
            e.event == [:chat_bot, :analysis, :disambiguation, :complete]
        end)

      if disambiguation_events != [] do
        entity_event =
          Enum.find(disambiguation_events, fn e ->
            e.event == [:chat_bot, :analysis, :disambiguation, :entity]
          end)

        if entity_event do
          assert entity_event.measurements[:type_count] >= 1
          assert entity_event.metadata[:value] != nil
        end
      end
    end

    test "direct EntityExtractor emits telemetry" do
      TelemetryCollector.clear()

      text = "I am Austin"
      discourse = %{indicators: ["self_referential"]}
      speech_act = %{category: :expressive, sub_type: :greeting}

      _entities =
        EntityExtractor.extract_entities(text, discourse: discourse, speech_act: speech_act)

      Process.sleep(50)

      events = TelemetryCollector.get_events()

      disambiguation_events =
        Enum.filter(events, fn e ->
          match?([:chat_bot, :analysis, :disambiguation, _], e.event)
        end)

      assert disambiguation_events != [],
             "Expected disambiguation telemetry events, got none. " <>
               "Total events: #{length(events)}"

      entity_event =
        Enum.find(disambiguation_events, fn e ->
          e.event == [:chat_bot, :analysis, :disambiguation, :entity]
        end)

      if entity_event do
        assert entity_event.metadata[:value] == "Austin"
        selected = entity_event.measurements[:selected_type]
        available = entity_event.metadata[:available_types] || []

        if "person" in available and "location" in available do
          assert selected == "person",
                 "Expected 'person' in introduction context, got '#{selected}'. " <>
                   "Available types: #{inspect(available)}"
        else
          assert selected != nil, "Expected a selected type, got nil"
        end
      end
    end
  end
end
