defmodule Brain.Response.ChunkSegmenterTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Response.ChunkSegmenter
  alias Brain.Response.ChunkSegmenter.Chunk

  # "default" is the world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  describe "split_into_sentences/1" do
    test "splits on period" do
      text = "Hello. How are you."
      result = ChunkSegmenter.split_into_sentences(text)
      assert result == ["Hello.", "How are you."]
    end

    test "splits on exclamation mark" do
      text = "Hello! How are you!"
      result = ChunkSegmenter.split_into_sentences(text)
      assert result == ["Hello!", "How are you!"]
    end

    test "splits on question mark" do
      text = "Hello? How are you?"
      result = ChunkSegmenter.split_into_sentences(text)
      assert result == ["Hello?", "How are you?"]
    end

    test "handles mixed punctuation" do
      text = "Hello! How are you? I'm fine."
      result = ChunkSegmenter.split_into_sentences(text)
      assert result == ["Hello!", "How are you?", "I'm fine."]
    end

    test "trims whitespace" do
      text = "  Hello!   How are you?  "
      result = ChunkSegmenter.split_into_sentences(text)
      assert result == ["Hello!", "How are you?"]
    end

    test "filters empty strings" do
      text = "Hello!  "
      result = ChunkSegmenter.split_into_sentences(text)
      assert result == ["Hello!"]
    end
  end

  describe "segment/2" do
    test "returns list of Chunk structs" do
      text = "Hello! The weather is nice."
      result = ChunkSegmenter.segment(@world_id, text)

      assert is_list(result)
      assert Enum.all?(result, fn chunk -> is_struct(chunk, Chunk) end)
    end

    test "chunks have text, type, and embedding fields" do
      text = "Hello!"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)

      assert is_binary(chunk.text)
      assert is_atom(chunk.type)
    end

    test "a chunk's embedding is the world's embedding of its sentence" do
      [chunk] = ChunkSegmenter.segment(@world_id, "The weather forecast shows sunny skies.")

      assert {:ok, embedding} = World.Embedder.embed(@world_id, "The weather forecast shows sunny skies.")
      assert chunk.embedding == embedding
    end

    test "embeds in the named world's embedder" do
      ensure_started(World.Manager)
      setup_world_sandbox()
      {:ok, world} = create_test_world("chunk_segmenter_embedder")

      :ok =
        World.Embedder.load_model(world.id, %{
          vocabulary: %{"weather" => 0, "sunny" => 1},
          idf_weights: %{"weather" => 1.0, "sunny" => 1.0}
        })

      [chunk] = ChunkSegmenter.segment(world.id, "The weather forecast shows sunny skies.")

      # Two terms in the world's vocabulary, each seen once, normalized.
      assert chunk.embedding == [1.0 / :math.sqrt(2), 1.0 / :math.sqrt(2)]
    end

    test "raises for a world that holds no embedder" do
      # World.Manager holds the table world embedders are loaded into.
      ensure_started(World.Manager)
      world_id = "test_chunk_segmenter_no_embedder_#{System.unique_integer([:positive])}"

      assert_raise RuntimeError, ~r/world #{inspect(world_id)} has no embedder vocabulary/, fn ->
        ChunkSegmenter.segment(world_id, "Hello!")
      end
    end

    test "classifies greeting chunk" do
      text = "Hello!"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :greeting
    end

    test "classifies closing chunk" do
      text = "Goodbye!"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)

      assert chunk.type == :closing
    end

    test "classifies offer chunk" do
      text = "Can I help you with anything else?"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)

      assert chunk.type == :offer
    end

    test "classifies acknowledgment chunk" do
      text = "I understand."
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)

      assert chunk.type == :acknowledgment
    end

    test "defaults to body for substantive content" do
      text = "The weather forecast shows sunny skies."
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)

      assert chunk.type == :body
    end
  end

  describe "segment/3 with source_intent" do
    test "sets source_intent on chunks" do
      text = "Hello! How can I help?"
      result = ChunkSegmenter.segment(@world_id, text, "smalltalk.greetings.hello")

      assert Enum.all?(result, fn chunk ->
               chunk.source_intent == "smalltalk.greetings.hello"
             end)
    end
  end

  describe "segment_all/2" do
    test "segments all templates by intent" do
      templates_by_intent = %{
        "smalltalk.greetings.hello" => ["Hello!", "Hi there!"],
        "weather.query" => ["The weather is nice."]
      }

      result = ChunkSegmenter.segment_all(@world_id, templates_by_intent)

      assert is_list(result)
      assert length(result) >= 3
      greeting_chunks = Enum.filter(result, fn c -> c.source_intent == "smalltalk.greetings.hello" end)
      weather_chunks = Enum.filter(result, fn c -> c.source_intent == "weather.query" end)

      assert length(greeting_chunks) >= 2
      assert weather_chunks != []
    end

    test "handles empty templates" do
      templates_by_intent = %{
        "empty" => []
      }

      result = ChunkSegmenter.segment_all(@world_id, templates_by_intent)
      assert result == []
    end
  end

  describe "get_type_seeds/0" do
    test "returns map of chunk type seeds" do
      seeds = ChunkSegmenter.get_type_seeds()

      assert is_map(seeds)
      assert Map.has_key?(seeds, :greeting)
      assert Map.has_key?(seeds, :acknowledgment)
      assert Map.has_key?(seeds, :body)
      assert Map.has_key?(seeds, :offer)
      assert Map.has_key?(seeds, :clarification)
      assert Map.has_key?(seeds, :closing)
    end

    test "seeds contain example phrases" do
      seeds = ChunkSegmenter.get_type_seeds()

      assert "Hello!" in seeds[:greeting]
      assert "I understand." in seeds[:acknowledgment]
      assert "Goodbye!" in seeds[:closing]
    end
  end

  describe "heuristic classification" do
    test "classifies 'Welcome!' as greeting" do
      text = "Welcome!"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :greeting
    end

    test "classifies 'Nice to meet you!' as greeting" do
      text = "Nice to meet you!"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :greeting
    end

    test "classifies 'Take care!' as closing" do
      text = "Take care!"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :closing
    end

    test "classifies 'See you later!' as closing" do
      text = "See you later!"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :closing
    end

    test "classifies 'What would you like?' as offer" do
      text = "What would you like to do?"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type in [:offer, :clarification]
    end

    test "classifies 'Got it.' as acknowledgment" do
      text = "Got it."
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :acknowledgment
    end

    test "classifies 'Sure thing.' as acknowledgment" do
      text = "Sure thing."
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :acknowledgment
    end

    test "classifies short questions as clarification" do
      text = "Which one?"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :clarification
    end

    test "classifies 'What time?' as clarification" do
      text = "What time would you like?"
      [chunk | _] = ChunkSegmenter.segment(@world_id, text)
      assert chunk.type == :clarification
    end
  end
end
