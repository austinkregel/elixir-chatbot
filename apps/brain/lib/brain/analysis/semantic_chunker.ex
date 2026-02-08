defmodule Brain.Analysis.SemanticChunker do
  @moduledoc "Breaks longer user inputs into manageable semantic chunks (utterances).\n\nThis module handles:\n- Sentence boundary detection using punctuation patterns\n- Discourse marker detection (\"but\", \"however\", \"also\", \"and then\")\n- Quoted speech handling (keeps quoted content together)\n- Learnable parameters for chunk size thresholds\n"

  alias Brain.Analysis.Chunk
  alias Brain.Analysis.LearningStore
  alias Brain.ML.Tokenizer

  require Logger

  @default_discourse_markers [
    "also",
    "and",
    "and also",
    "and then",
    "plus",
    "moreover",
    "furthermore",
    "in addition",
    "additionally",
    "but",
    "however",
    "although",
    "though",
    "yet",
    "on the other hand",
    "nevertheless",
    "nonetheless",
    "because",
    "since",
    "so",
    "therefore",
    "thus",
    "consequently",
    "as a result",
    "then",
    "after that",
    "before that",
    "first",
    "next",
    "finally",
    "meanwhile",
    "anyway",
    "by the way",
    "speaking of",
    "oh",
    "well"
  ]

  @doc "Chunks the input text into semantic units.\n\nReturns a list of Chunk structs.\n"
  def chunk(text) when is_binary(text) do
    params = get_chunking_params()

    text
    |> normalize_whitespace()
    |> extract_quoted_sections()
    |> split_into_sentences()
    |> merge_short_sentences(params.min_chunk_words)
    |> detect_discourse_markers()
    |> apply_max_chunk_size(params.max_chunk_words)
    |> build_chunks(text)
  end

  @doc "Returns chunking statistics for the given text.\n"
  def analyze(text) when is_binary(text) do
    chunks = chunk(text)

    %{
      chunks: chunks,
      total_length: String.length(text),
      chunk_count: length(chunks),
      avg_chunk_length: calculate_avg_length(chunks),
      has_quoted: Enum.any?(chunks, & &1.is_quoted)
    }
  end

  defp get_chunking_params do
    case LearningStore.get_params("chunker") do
      {:ok, params} ->
        %{
          max_chunk_words: Map.get(params, "max_chunk_words", 50),
          min_chunk_words: Map.get(params, "min_chunk_words", 3),
          split_on_conjunctions: Map.get(params, "split_on_conjunctions", true),
          discourse_markers: Map.get(params, "discourse_markers", @default_discourse_markers)
        }

      {:error, _} ->
        %{
          max_chunk_words: 50,
          min_chunk_words: 3,
          split_on_conjunctions: true,
          discourse_markers: @default_discourse_markers
        }
    end
  end

  defp normalize_whitespace(text) do
    Tokenizer.collapse_whitespace_public(text)
    |> String.trim()
  end

  defp extract_quoted_sections(text) do
    quoted_sections = Tokenizer.extract_quoted_sections(text)

    {processed, quoted_map} =
      quoted_sections
      |> Enum.with_index()
      |> Enum.reduce({text, %{}}, fn {{quoted_text, _start, _end}, idx}, {acc_text, acc_map} ->
        full_quoted = find_quoted_in_text(acc_text, quoted_text)
        marker = "<<QUOTED_#{idx}>>"

        new_text =
          if full_quoted do
            String.replace(acc_text, full_quoted, marker, global: false)
          else
            acc_text
          end

        {new_text, Map.put(acc_map, marker, full_quoted || quoted_text)}
      end)

    {processed, quoted_map}
  end

  defp find_quoted_in_text(text, inner_text) do
    patterns = [
      "\"" <> inner_text <> "\"",
      "'" <> inner_text <> "'",
      "“" <> inner_text <> "”",
      "‘" <> inner_text <> "’"
    ]

    Enum.find(patterns, fn pattern ->
      String.contains?(text, pattern)
    end)
  end

  defp split_into_sentences({text, quoted_map}) do
    sentences =
      Tokenizer.split_sentences(text)
      |> Enum.map(fn sentence_map -> sentence_map.text end)
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    restored =
      Enum.map(sentences, fn sentence ->
        Enum.reduce(quoted_map, sentence, fn {marker, quoted}, acc ->
          String.replace(acc, marker, quoted)
        end)
      end)

    {restored, quoted_map}
  end

  defp merge_short_sentences({sentences, quoted_map}, min_words) do
    merged =
      Enum.reduce(sentences, [], fn sentence, acc ->
        word_count = count_words(sentence)

        case acc do
          [] ->
            [sentence]

          [prev | rest] when word_count < min_words ->
            [prev <> " " <> sentence | rest]

          _ ->
            [sentence | acc]
        end
      end)
      |> Enum.reverse()

    {merged, quoted_map}
  end

  defp detect_discourse_markers({sentences, quoted_map}) do
    marked =
      Enum.map(sentences, fn sentence ->
        lower = String.downcase(sentence)
        markers = find_leading_markers(lower)
        {sentence, markers}
      end)

    {marked, quoted_map}
  end

  defp find_leading_markers(text) do
    @default_discourse_markers
    |> Enum.filter(fn marker ->
      Tokenizer.starts_with_word?(text, marker)
    end)
  end

  defp apply_max_chunk_size({marked_sentences, quoted_map}, max_words) do
    split_sentences =
      Enum.flat_map(marked_sentences, fn {sentence, markers} ->
        word_count = count_words(sentence)

        if word_count > max_words do
          split_long_sentence(sentence, max_words)
          |> Enum.with_index()
          |> Enum.map(fn {part, idx} ->
            if idx == 0 do
              {part, markers}
            else
              {part, []}
            end
          end)
        else
          [{sentence, markers}]
        end
      end)

    {split_sentences, quoted_map}
  end

  defp split_long_sentence(sentence, max_words) do
    parts =
      sentence
      |> split_on_clause_boundaries()
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    Enum.flat_map(parts, fn part ->
      if count_words(part) > max_words do
        split_by_word_count(part, max_words)
      else
        [part]
      end
    end)
  end

  defp split_by_word_count(text, max_words) do
    words = Tokenizer.split_words(text)

    words
    |> Enum.chunk_every(max_words)
    |> Enum.map(&Enum.join(&1, " "))
  end

  defp split_on_clause_boundaries(text) do
    text
    |> String.graphemes()
    |> split_on_graphemes([",", ";"], [], "")
  end

  defp split_on_graphemes([], _delimiters, acc, current) do
    if current != "" do
      Enum.reverse([current | acc])
    else
      Enum.reverse(acc)
    end
  end

  defp split_on_graphemes([g | rest], delimiters, acc, current) do
    if g in delimiters do
      if current != "" do
        split_on_graphemes(rest, delimiters, [current | acc], "")
      else
        split_on_graphemes(rest, delimiters, acc, "")
      end
    else
      split_on_graphemes(rest, delimiters, acc, current <> g)
    end
  end

  defp build_chunks({marked_sentences, quoted_map}, original_text) do
    marked_sentences
    |> Enum.with_index()
    |> Enum.map(fn {{sentence, markers}, index} ->
      restored =
        Enum.reduce(quoted_map, sentence, fn {marker, quoted}, acc ->
          String.replace(acc, marker, quoted)
        end)

      {start_pos, end_pos} = find_position(original_text, restored, index)
      is_quoted = contains_quoted_text?(restored)

      Chunk.new(restored, index, start_pos, end_pos,
        is_quoted: is_quoted,
        discourse_markers: markers
      )
    end)
  end

  defp find_position(original, chunk_text, _index) do
    case :binary.match(original, chunk_text) do
      {start, length} ->
        {start, start + length - 1}

      :nomatch ->
        normalized_chunk = normalize_whitespace(chunk_text)

        case :binary.match(normalize_whitespace(original), normalized_chunk) do
          {start, length} -> {start, start + length - 1}
          :nomatch -> {0, String.length(chunk_text) - 1}
        end
    end
  end

  defp count_words(text) do
    text
    |> Tokenizer.expand_contractions()
    |> Tokenizer.split_words()
    |> Enum.count()
  end

  defp contains_quoted_text?(text) do
    quoted_sections = Tokenizer.extract_quoted_sections(text)
    quoted_sections != []
  end

  defp calculate_avg_length(chunks) do
    if chunks == [] do
      0
    else
      total = Enum.reduce(chunks, 0, fn chunk, acc -> acc + String.length(chunk.text) end)
      total / length(chunks)
    end
  end
end