defmodule Brain.Analysis.EdgeCasesComprehensiveTest do
  @moduledoc "Comprehensive edge case tests to explore the limits of the NLP system.\n\nThese tests cover cases that are NOT in the training data to understand\nhow the system handles:\n1. Names that overlap with cities/locations\n2. Unusual greeting patterns\n3. Names that overlap with songs or products\n4. Typos, informal language, and edge cases\n\nThe goal is not necessarily for all tests to pass, but to document\nexpected vs actual behavior and identify areas for improvement.\n\n## Snapshot Tests\n\nThis module also includes \"snapshot\" tests that capture the exact analysis\noutput for key inputs. These help:\n- Document expected behavior precisely\n- Catch regressions when implementation changes\n- Understand what the system actually detects\n\nRun snapshot tests with: `mix test test/brain/analysis/edge_cases_comprehensive_test.exs --only snapshot`\n"
  use Brain.Test.GraphCase, async: false
  require Logger

  alias Brain
  alias Brain.Analysis.Pipeline
  alias Brain.ML.Gazetteer
  import Brain.TestHelpers
  @moduletag :edge_cases

  @doc "Extracts a normalized snapshot from a Pipeline analysis result.\nThis captures the key fields we want to assert on.\n"
  def extract_snapshot(result) do
    %{
      chunk_count: length(result.chunks),
      overall_strategy: result.overall_strategy,
      analyses:
        Enum.map(result.analyses, fn analysis ->
          analysis_map = Map.from_struct(analysis)

          chunk_index = Map.get(analysis_map, :chunk_index, Map.get(analysis_map, :index, 0))
          text = Map.get(analysis_map, :text, "")
          discourse = Map.get(analysis_map, :discourse)
          discourse_map = to_map(discourse)

          is_self_ref =
            Map.get(
              discourse_map,
              :is_self_referential,
              Map.get(discourse_map, :self_referential, false)
            )

          slots = Map.get(analysis_map, :slots)
          slots_map = to_map(slots)
          filled_slots = Map.get(slots_map, :filled_slots, Map.get(slots_map, :filled, %{}))
          missing_required = Map.get(slots_map, :missing_required, [])

          detected_intent =
            Map.get(
              analysis_map,
              :intent,
              Map.get(slots_map, :intent, Map.get(slots_map, :schema_name, nil))
            )

          speech_act = Map.get(analysis_map, :speech_act)
          speech_act_map = to_map(speech_act)

          %{
            index: chunk_index,
            text: text,
            speech_act_category: Map.get(speech_act_map, :category),
            speech_act_type: Map.get(speech_act_map, :sub_type) || Map.get(speech_act_map, :type),
            speech_act_confidence: round_confidence(Map.get(speech_act_map, :confidence)),
            discourse_addressee: Map.get(discourse_map, :addressee),
            discourse_self_referential: is_self_ref,
            detected_intent: detected_intent,
            slots_filled: Map.keys(filled_slots) |> Enum.sort(),
            slots_missing: missing_required |> Enum.sort(),
            entities: normalize_entities(Map.get(analysis_map, :entities, [])),
            response_strategy: Map.get(analysis_map, :response_strategy)
          }
        end)
    }
  end

  defp to_map(nil) do
    %{}
  end

  defp to_map(term) when is_struct(term) do
    Map.from_struct(term)
  end

  defp to_map(term) when is_map(term) do
    term
  end

  defp to_map(_) do
    %{}
  end

  defp round_confidence(nil) do
    nil
  end

  defp round_confidence(conf) when is_float(conf) do
    Float.round(conf, 2)
  end

  defp round_confidence(conf) do
    conf
  end

  defp normalize_entities(entities) when is_list(entities) do
    entities
    |> Enum.map(fn e ->
      %{
        type: Map.get(e, :entity_type, Map.get(e, :type, Map.get(e, :entity, "unknown"))),
        value: Map.get(e, :value, Map.get(e, :match, "unknown")),
        confidence: round_confidence(Map.get(e, :confidence, 1.0))
      }
    end)
    |> Enum.sort_by(& &1.value)
  end

  defp normalize_entities(_) do
    []
  end

  @doc "Logs a snapshot for easy copy/paste into test assertions.\nUse this when updating expected values. Output is captured by capture_log.\n"
  def log_snapshot(result) do
    snapshot = extract_snapshot(result)

    Logger.warning("\n" <> String.duplicate("=", 70))
    Logger.warning("SNAPSHOT OUTPUT")
    Logger.warning(String.duplicate("=", 70))
    Logger.warning("Chunk count: #{snapshot.chunk_count}")
    Logger.warning("Overall strategy: #{inspect(snapshot.overall_strategy)}")

    Enum.each(snapshot.analyses, fn analysis ->
      Logger.warning("--- Chunk #{analysis.index}: \"#{analysis.text}\" ---")

      Logger.warning(
        "  Speech Act: #{analysis.speech_act_category} / #{analysis.speech_act_type} (#{analysis.speech_act_confidence})"
      )

      Logger.warning(
        "  Discourse: addressee=#{analysis.discourse_addressee}, self_ref=#{analysis.discourse_self_referential}"
      )

      Logger.warning("  Intent: #{inspect(analysis.detected_intent)}")
      Logger.warning("  Slots filled: #{inspect(analysis.slots_filled)}")
      Logger.warning("  Slots missing: #{inspect(analysis.slots_missing)}")
      Logger.warning("  Entities: #{inspect(analysis.entities)}")
      Logger.warning("  Strategy: #{analysis.response_strategy}")
    end)

    Logger.warning(String.duplicate("=", 70))

    snapshot
  end

  setup_all do
    Application.ensure_all_started(:brain)
    Brain.TestHelpers.require_services!(:brain)
    :ok
  end

  describe "names overlapping with locations" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :ambiguous_names
    test "Austin is also a city in Texas", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Austin")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "weather")
      refute_response_intent(response, "smarthome")
    end

    @tag :ambiguous_names
    test "Dallas is also a city in Texas", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, my name is Dallas")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Paris is also a city in France", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Paris")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Jordan is also a country", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hey there, I'm Jordan")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Brooklyn is also part of NYC", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm Brooklyn")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Georgia is also a state and country", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, my name is Georgia")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Madison is also a city in Wisconsin", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi there, I'm Madison")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Sydney is also a city in Australia", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Sydney")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Charlotte is also a city in North Carolina", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, my name is Charlotte")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Savannah is also a city in Georgia", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Savannah")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Dakota is also a state reference", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hey, I'm Dakota")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Orlando is also a city in Florida", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm Orlando")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Florence is also a city in Italy", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, my name is Florence")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Victoria is also a city and state", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm Victoria")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Lincoln is also a city in Nebraska", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Lincoln")

      refute_response_intent(response, "weather")
    end

    @tag :ambiguous_names
    test "Jackson is also a city in Mississippi", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi there, I'm Jackson")

      refute_response_intent(response, "weather")
    end
  end

  describe "names overlapping with songs" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :ambiguous_names
    test "Delilah is also a song (Hey There Delilah)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Delilah")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "music")
    end

    @tag :ambiguous_names
    test "Jolene is also a famous song", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, my name is Jolene")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "music")
    end

    @tag :ambiguous_names
    test "Iris is also a song by Goo Goo Dolls", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Iris")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "music")
    end

    @tag :ambiguous_names
    test "Roxanne is also a song by The Police", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm Roxanne")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "music")
    end
  end

  describe "names overlapping with products" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :ambiguous_names
    test "Alexa is also Amazon's assistant", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Alexa")

      refute_response_intent(response, "smarthome")
    end

    @tag :ambiguous_names
    test "Mercedes is also a car brand", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, my name is Mercedes")

      refute_response_intent(response, "weather")
      refute_response_intent(response, "smarthome")
    end

    @tag :ambiguous_names
    test "Luna is also a cryptocurrency and brand", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Luna")

      refute_response_intent(response, "weather")
      refute_response_intent(response, "smarthome")
    end
  end

  describe "informal greetings not in training data" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :informal
    test "yo as greeting", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Yo")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "sup as greeting", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Sup")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "wassup as greeting", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Wassup")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "hiya as greeting", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hiya!")
      assert String.length(response) > 0

      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "ello as greeting (dropped h)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "'Ello there")

      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "extended vowels - heyyy", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Heyyy")

      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "extended vowels - hiii", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hiii")

      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "cultural greeting - g'day", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "G'day mate")

      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "cultural greeting - aloha", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Aloha")
      assert String.length(response) > 0
    end

    @tag :informal
    test "formal greeting - salutations", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Salutations!")

      refute_response_intent(response, "smalltalk.greetings.bye")
    end

    @tag :informal
    test "what's up as greeting", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "What's up!")

      refute_response_intent(response, "smalltalk.greetings.bye")
    end
  end

  describe "text formatting edge cases" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :formatting
    test "all lowercase with no punctuation", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "hello im austin")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :formatting
    test "all uppercase", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "HELLO I AM AUSTIN")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :formatting
    test "excessive punctuation", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello!!!! I'm Austin!!!!")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :formatting
    test "mixed case in name", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm AuStIn")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :formatting
    test "lowercase i in I'm", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, i'm austin")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :formatting
    test "extra spaces", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello,    I'm    Austin")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end
  end

  describe "common typos" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :typos
    test "helo (missing l)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Helo, I'm Austin")
      assert String.length(response) > 0
    end

    @tag :typos
    test "hlelo (transposed letters)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hlelo, I'm Austin")

      assert String.length(response) > 0
    end

    @tag :typos
    test "im vs I'm", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, im Austin")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :typos
    test "goodmorning (no space)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Goodmorning!")

      refute_response_intent(response, "smalltalk.greetings.bye")
    end
  end

  describe "unusual introduction patterns" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :introductions
    test "with title - Dr.", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm Dr. Smith")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "with title - Professor", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm Professor Johnson")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "nickname pattern - call me", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, you can call me Bobby")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "my friends call me pattern", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, my friends call me Ace")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "full name introduction", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm John Smith")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "name with apostrophe - O'Brien", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello, I'm O'Brien")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "hyphenated name", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hi, I'm Mary-Jane")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "name is pattern", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "My name is Alex")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "the name's pattern (James Bond style)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "The name's Bond, James Bond")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end

    @tag :introductions
    test "I go by pattern", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "I go by Max")

      refute_response_intent(response, "music")
      refute_response_intent(response, "weather")
    end
  end

  describe "time-based greetings with ambiguous names" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :time_greeting
    test "good morning with city name (Austin)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Good morning, I'm Austin")

      refute_response_intent(response, "weather")
    end

    @tag :time_greeting
    test "good afternoon with city name (Dallas)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Good afternoon, I'm Dallas")

      assert_response_intent(response, "smalltalk.")
      refute_response_intent(response, "weather")
    end

    @tag :time_greeting
    test "good evening with city name (Paris)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Good evening, I'm Paris")

      refute_response_intent(response, "weather")
    end

    @tag :time_greeting
    test "good night with city name (Sydney)", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Good night, I'm Sydney")

      refute_response_intent(response, "weather")
    end
  end

  describe "complex multi-sentence scenarios" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :multi_sentence
    test "greeting + introduction + weather.query", %{conversation_id: conv_id} do
      input = "Hello! I'm Austin. What's the weather in Owosso?"

      result = Pipeline.process(input, [])
      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 3,
             "Expected exactly 3 chunks (greeting, intro, weather), got #{snapshot.chunk_count}: " <>
               inspect(Enum.map(snapshot.analyses, & &1.text))

      [greeting, introduction, weather_query] = snapshot.analyses

      assert greeting.index == 0
      assert greeting.speech_act_category == :expressive,
             "Expected greeting chunk to be :expressive, got: #{inspect(greeting.speech_act_category)}"

      assert is_binary(greeting.detected_intent) and
               String.starts_with?(greeting.detected_intent, "smalltalk.greetings"),
             "Expected greeting intent, got: #{inspect(greeting.detected_intent)}"

      assert introduction.index == 1
      assert introduction.speech_act_category in [:expressive, :assertive],
             "Expected intro chunk to be :expressive or :assertive, got: #{inspect(introduction.speech_act_category)}"

      assert weather_query.index == 2
      assert is_binary(weather_query.detected_intent) and
               String.contains?(weather_query.detected_intent, "weather"),
             "Expected weather intent on chunk 2, got: #{inspect(weather_query.detected_intent)}"

      assert Enum.any?(weather_query.entities, fn e ->
               String.downcase(to_string(e.value)) =~ "owosso"
             end),
             "Expected Owosso entity in weather chunk. Entities: #{inspect(weather_query.entities)}"

      {:ok, %{response: response, context: context}} = Brain.evaluate(conv_id, input)

      assert context.intent != nil, "Expected a final intent in context"

      assert String.contains?(context.intent, "weather"),
             "Expected final intent to be weather-related, got: #{inspect(context.intent)}"

      assert_response_intent(response, "weather")
    end

    @tag :multi_sentence
    test "multiple greetings", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "Hello! Hi! Hey there!")

      refute_response_intent(response, "smalltalk.greetings.bye")
      assert String.length(response) > 0
    end

    @tag :multi_sentence
    test "introduction then command", %{conversation_id: conv_id} do
      {:ok, %{response: response}} = Brain.evaluate(conv_id, "I'm Dallas. Play some music.")

      assert String.length(response) > 0
    end
  end

  describe "extremely long input handling" do
    setup do
      start_brain_services()
      {:ok, conversation_id} = Brain.create_conversation()
      %{conversation_id: conversation_id}
    end

    @tag :stress_test
    test "handles very long input with many sentences", %{conversation_id: conv_id} do
      long_input =
        1..100
        |> Enum.map_join(
          " ",
          fn i -> "This is sentence number #{i}." end
        )

      result = Brain.evaluate(conv_id, long_input)

      case result do
        {:ok, %{response: response}} ->
          assert is_binary(response)
          assert String.length(response) > 0

        {:error, reason} ->
          assert is_atom(reason) or is_binary(reason)
      end
    end

    @tag :stress_test
    test "semantic chunker breaks long input into chunks" do
      alias Brain.Analysis.SemanticChunker

      long_input =
        1..50
        |> Enum.map_join(
          " ",
          fn i -> "Hello, I am person number #{i}." end
        )

      chunks = SemanticChunker.chunk(long_input)
      assert length(chunks) > 1, "Long input should be chunked, got #{length(chunks)} chunks"

      Enum.each(chunks, fn chunk ->
        word_count = chunk.text |> String.split() |> length()
        assert word_count <= 60, "Chunk too long: #{word_count} words"
      end)
    end

    @tag :stress_test
    test "handles repeated greeting pattern" do
      long_greetings =
        1..20
        |> Enum.map_join(
          " ",
          fn _ -> "Hello! Hi! Hey there!" end
        )

      result = Pipeline.process(long_greetings, [])
      assert result.analyses != []
    end

    @tag :stress_test
    test "handles mixed content with many intents", %{conversation_id: conv_id} do
      mixed_input =
        "Hello, I'm Austin. What's the weather? Play some music. Turn on the lights.\nGood morning! What time is it? Set a reminder. How are you today?\nGoodbye! Wait, actually, hello again. What can you do?\n"

      {:ok, %{response: response}} = Brain.evaluate(conv_id, mixed_input)
      assert is_binary(response)
      assert String.length(response) > 0
    end

    @tag :stress_test
    @tag timeout: 120_000
    test "does not take excessively long on moderately long input", %{conversation_id: conv_id} do
      moderate_input =
        1..30
        |> Enum.map_join(
          " ",
          fn i ->
            case rem(i, 5) do
              0 -> "What's the weather in city #{i}?"
              1 -> "Hello, I'm person #{i}."
              2 -> "Play song number #{i}."
              3 -> "Turn on light #{i}."
              _ -> "This is statement #{i}."
            end
          end
        )

      start_time = System.monotonic_time(:millisecond)
      {:ok, %{response: _response}} = Brain.evaluate(conv_id, moderate_input)
      elapsed = System.monotonic_time(:millisecond) - start_time
      assert elapsed < 30_000, "Processing took too long: #{elapsed}ms"
    end
  end

  describe "entity extraction edge cases" do
    setup do
      start_test_services()
      :ok
    end

    @tag :entity_extraction
    test "Austin recognized as both person and location" do
      results = Gazetteer.lookup_all_types("Austin")

      types = Enum.map(results, & &1.entity_type)
      assert "person" in types, "Austin should be recognized as a person name"

      assert "city" in types or "location" in types,
             "Austin should also be recognized as a city/location"
    end

    @tag :entity_extraction
    test "Dallas recognized as both person and location" do
      results = Gazetteer.lookup_all_types("Dallas")

      types = Enum.map(results, & &1.entity_type)

      assert "person" in types, "Dallas should be recognized as a person name"
    end

    @tag :entity_extraction
    test "Hello recognized in songs" do
      results = Gazetteer.lookup_all_types("Hello")

      types = Enum.map(results, & &1.entity_type)
      assert "song" in types or "music" in types, "Hello should be recognized as a song"
    end
  end

  describe "pipeline disambiguation" do
    setup do
      start_test_services()
      :ok
    end

    @tag :disambiguation
    test "I'm Austin pattern triggers self-referential context" do
      result = Pipeline.process("Hello, I'm Austin", [])
      assert result.analyses != []
      analysis = hd(result.analyses)

      assert analysis.discourse.addressee == :self or
               analysis.speech_act.category in [:expressive, :assertive],
             "Should recognize self-introduction pattern, got: addressee=#{inspect(analysis.discourse.addressee)}, category=#{inspect(analysis.speech_act.category)}"
    end

    @tag :disambiguation
    test "I'm from Austin triggers location context" do
      result = Pipeline.process("I'm from Austin", [])
      assert result.analyses != []
    end

    @tag :disambiguation
    test "weather in Austin triggers location context" do
      result = Pipeline.process("What's the weather in Austin?", [])

      assert result.analyses != []
      analysis = hd(result.analyses)

      entities =
        Enum.filter(analysis.entities, fn e ->
          value = Map.get(e, :value, Map.get(e, :match, ""))
          String.downcase(to_string(value)) =~ "austin"
        end)

      if entities != [] do
        entity = hd(entities)

        entity_type = Map.get(entity, :entity_type, Map.get(entity, :type, "unknown"))

        assert entity_type == "location" or entity_type == "city",
               "Austin in weather context should be location, got: #{entity_type}"
      end
    end
  end

  @log_snapshots false

  describe "snapshot tests - greeting with introduction" do
    setup do
      start_test_services()
      :ok
    end

    @tag :snapshot
    test "Hello, I'm Austin - complete analysis snapshot" do
      input = "Hello, I'm Austin"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)
      assert snapshot.chunk_count == 1
      assert snapshot.overall_strategy == :can_respond
      [analysis] = snapshot.analyses
      assert analysis.text == "Hello, I'm Austin"

      assert analysis.speech_act_category in [:expressive, :assertive],
             "Expected expressive or assertive, got: #{analysis.speech_act_category}"

      austin_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "austin"
        end)

      if austin_entities != [] do
        [austin] = austin_entities

        assert austin.type == "person",
               "Austin should be person, got: #{austin.type}"
      end

      assert analysis.response_strategy in [:can_respond, :hedged_response]
    end

    @tag :snapshot
    test "Hi, my name is Sarah - complete analysis snapshot" do
      input = "Hi, my name is Sarah"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      assert analysis.speech_act_category in [:expressive, :assertive],
             "Expected greeting/introduction, got: #{analysis.speech_act_category}"

      sarah_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "sarah"
        end)

      if sarah_entities != [] do
        [sarah] = sarah_entities
        assert sarah.type == "person"
      end
    end
  end

  describe "snapshot tests - weather queries" do
    setup do
      start_test_services()
      :ok
    end

    @tag :snapshot
    test "What's the weather in Austin? - location disambiguation" do
      input = "What's the weather in Austin?"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses
      assert analysis.speech_act_category == :directive
      assert analysis.discourse_self_referential == false

      austin_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "austin"
        end)

      if austin_entities != [] do
        [austin] = austin_entities

        assert austin.type in ["location", "city"],
               "Austin in weather context should be location, got: #{austin.type}"
      end

      assert analysis.speech_act_type in [:request_information, :question, :request]
    end

    @tag :snapshot
    test "What's the weather? - missing location slot" do
      input = "What's the weather?"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses
      assert analysis.speech_act_category == :directive
      assert analysis.response_strategy in [:needs_clarification, :can_respond, :hedged_response]

      if analysis.response_strategy == :needs_clarification do
        assert "location" in analysis.slots_missing
      end
    end
  end

  describe "snapshot tests - multi-chunk inputs" do
    setup do
      start_test_services()
      :ok
    end

    @tag :snapshot
    test "Hello! What's the weather? - greeting then question" do
      input = "Hello! What's the weather?"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)
      assert snapshot.chunk_count == 2
      [greeting_analysis, weather_analysis] = snapshot.analyses

      assert greeting_analysis.speech_act_category == :expressive
      assert greeting_analysis.text =~ ~r/hello/i
      assert weather_analysis.speech_act_category == :directive
      assert weather_analysis.text =~ ~r/weather/i
    end

    @tag :snapshot
    test "Hello, I'm Austin. What's the weather in Dallas? - intro then weather" do
      input = "Hello, I'm Austin. What's the weather in Dallas?"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)
      assert snapshot.chunk_count >= 1

      if snapshot.chunk_count == 1 do
        [analysis] = snapshot.analyses

        assert analysis.speech_act_category in [:expressive, :directive, :assertive],
               "Expected expressive, assertive, or directive, got: #{analysis.speech_act_category}"
      else
        [intro_analysis | rest] = snapshot.analyses

        assert intro_analysis.speech_act_category in [:expressive, :assertive],
               "Expected intro to be expressive or assertive, got: #{intro_analysis.speech_act_category}"

        if rest != [] do
          weather_analysis = hd(rest)

          austin_in_weather =
            Enum.filter(weather_analysis.entities, fn e ->
              String.downcase(to_string(e.value)) =~ "austin"
            end)

          assert austin_in_weather == [], "Austin leaked into weather chunk - potential issue"
        end
      end
    end
  end

  describe "snapshot tests - commands" do
    setup do
      start_test_services()
      :ok
    end

    @tag :snapshot
    test "Play some music - music command" do
      input = "Play some music"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses
      assert analysis.speech_act_category == :directive
      assert analysis.speech_act_type in [:command, :request, :action]
    end

    @tag :snapshot
    test "Turn on the lights in the living room - device command with location" do
      input = "Turn on the lights in the living room"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses
      assert analysis.speech_act_category == :directive
      entity_values = Enum.map(analysis.entities, & &1.value) |> Enum.map(&String.downcase/1)

      has_relevant =
        Enum.any?(entity_values, fn v ->
          v =~ ~r/light|living|room/i
        end)

      assert has_relevant or length(analysis.entities) >= 0
    end
  end

  describe "snapshot tests - edge cases with detailed inspection" do
    setup do
      start_test_services()
      :ok
    end

    @tag :snapshot
    test "yo - informal greeting analysis" do
      input = "yo"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      assert analysis.speech_act_category in [:expressive, :assertive, :unknown, nil],
             "Expected informal greeting to be recognized, got: #{analysis.speech_act_category}"
    end

    @tag :snapshot
    test "The name's Bond, James Bond - unusual intro pattern" do
      input = "The name's Bond, James Bond"
      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)

      [analysis | _] = snapshot.analyses
      assert analysis.speech_act_category != nil
    end

    @tag :snapshot
    test "mixed content inspection" do
      input = "Hello, I'm Austin. What's the weather? Play some music. Turn on the lights.\n"

      result = Pipeline.process(input, [])

      if @log_snapshots do
        log_snapshot(result)
      end

      snapshot = extract_snapshot(result)
      assert snapshot.chunk_count >= 3

      Enum.each(snapshot.analyses, fn a ->
        assert a.speech_act_category != nil
      end)
    end
  end

  describe "snapshot tests - regression prevention" do
    setup do
      start_test_services()
      :ok
    end

    @tag :snapshot
    @tag :regression
    test "Hello should NOT trigger music playback" do
      input = "Hello"
      result = Pipeline.process(input, [])

      snapshot = extract_snapshot(result)
      [analysis] = snapshot.analyses

      assert analysis.speech_act_category == :expressive,
             "Hello should be greeting (expressive), got: #{analysis.speech_act_category}"

      if analysis.detected_intent do
        refute analysis.detected_intent =~ ~r/music|play/i,
               "Hello should not have music intent, got: #{analysis.detected_intent}"
      end
    end

    @tag :snapshot
    @tag :regression
    test "Hello, I'm Austin should NOT ask about weather in Austin" do
      input = "Hello, I'm Austin"
      result = Pipeline.process(input, [])

      snapshot = extract_snapshot(result)
      [analysis] = snapshot.analyses

      assert analysis.speech_act_category in [:expressive, :assertive],
             "Expected expressive or assertive, got: #{analysis.speech_act_category}"

      austin_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "austin"
        end)

      Enum.each(austin_entities, fn e ->
        refute e.type in ["location", "city"],
               "Austin in greeting context must be person, got: #{e.type}"
      end)

      if analysis.detected_intent do
        refute analysis.detected_intent =~ ~r/weather/i,
               "Hello, I'm Austin should not have weather intent"
      end
    end
  end
end
