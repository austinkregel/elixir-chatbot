defmodule Brain.Analysis.SemanticChunkerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.SemanticChunker
  alias Brain.Analysis.Chunk

  describe "chunk/1" do
    test "handles simple single sentence" do
      chunks = SemanticChunker.chunk("Hello, how are you?")

      assert match?([_], chunks)
      assert %Chunk{text: "Hello, how are you?", index: 0} = hd(chunks)
    end

    test "splits on sentence boundaries" do
      chunks = SemanticChunker.chunk("Hello there. How are you? I am fine.")

      assert match?([_, _, _], chunks)
      assert Enum.at(chunks, 0).text == "Hello there."
      assert Enum.at(chunks, 1).text == "How are you?"
      assert Enum.at(chunks, 2).text == "I am fine."
    end

    test "handles discourse markers" do
      chunks = SemanticChunker.chunk("I want to check the weather. Also, what's the news?")

      assert match?([_, _], chunks)
      second_chunk = Enum.at(chunks, 1)
      assert String.contains?(second_chunk.text, "Also")
      assert "also" in second_chunk.discourse_markers
    end

    test "preserves quoted content" do
      chunks = SemanticChunker.chunk("She said \"Hello, how are you?\" to me.")

      assert match?([_], chunks)
      chunk = hd(chunks)
      assert chunk.is_quoted == true
      assert String.contains?(chunk.text, "\"Hello, how are you?\"")
    end

    test "handles multi-sentence input with various punctuation" do
      input = "Turn on the lights! What's the weather? It's cold today."
      chunks = SemanticChunker.chunk(input)

      # May merge short sentences, so at least 2 chunks
      assert Enum.count_until(chunks, 2) >= 2
      # First chunk should end with ! or contain it
      assert String.contains?(Enum.at(chunks, 0).text, "!")
    end

    test "handles empty input" do
      chunks = SemanticChunker.chunk("")
      assert chunks == []
    end

    test "normalizes whitespace" do
      chunks = SemanticChunker.chunk("Hello   there.    How   are   you?")

      assert match?([_, _], chunks)
      assert Enum.at(chunks, 0).text == "Hello there."
    end

    test "assigns correct indices to chunks" do
      chunks = SemanticChunker.chunk("First. Second. Third.")

      Enum.each(Enum.with_index(chunks), fn {chunk, idx} ->
        assert chunk.index == idx
      end)
    end
  end

  describe "analyze/1" do
    test "returns chunk statistics" do
      result = SemanticChunker.analyze("Hello there. How are you?")

      assert %{
               chunks: chunks,
               total_length: _,
               chunk_count: 2,
               avg_chunk_length: _
             } = result

      assert match?([_, _], chunks)
    end

    test "detects quoted content" do
      result = SemanticChunker.analyze("He said \"test\" to me.")
      assert result.has_quoted == true
    end

    test "reports no quoted content when absent" do
      result = SemanticChunker.analyze("Hello there.")
      assert result.has_quoted == false
    end
  end
end
