defmodule Brain.Summarization.TurnSegmenter do
  @moduledoc """
  Parses dialogue text into structured turns.

  Handles various dialogue formats:
  - "Name: text, Name: text" (comma-separated)
  - "Name: text\\nName: text" (newline-separated)
  - Mixed formats with special tokens like <file_photo>

  Uses tokenization rather than regex for NLP operations.
  """

  alias Brain.Summarization.Types.Turn
  alias Brain.ML.Tokenizer

  @doc """
  Segments dialogue into a list of Turn structs.

  ## Examples

      iex> segment("Lucas: Hey!, Demi: Hi there!")
      {:ok, [%Turn{speaker: "Lucas", text: "Hey!", index: 0},
             %Turn{speaker: "Demi", text: "Hi there!", index: 1}]}
  """
  @spec segment(String.t()) :: {:ok, [Turn.t()]} | {:error, term()}
  def segment(dialogue) when is_binary(dialogue) do
    turns = 
      dialogue
      |> split_into_raw_turns()
      |> Enum.with_index()
      |> Enum.map(fn {{speaker, text}, index} ->
        Turn.new(speaker, String.trim(text), index)
      end)
      |> Enum.reject(fn turn -> turn.text == "" end)

    {:ok, turns}
  end

  @doc """
  Extract list of unique participants from dialogue.
  """
  @spec extract_participants(String.t()) :: [String.t()]
  def extract_participants(dialogue) do
    dialogue
    |> split_into_raw_turns()
    |> Enum.map(fn {speaker, _} -> speaker end)
    |> Enum.uniq()
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp split_into_raw_turns(dialogue) do
    # First, normalize the dialogue
    normalized = normalize_dialogue(dialogue)

    # Find all speaker patterns using tokenization approach
    # Pattern: CapitalizedName followed by colon
    find_turns(normalized)
  end

  defp normalize_dialogue(dialogue) do
    dialogue
    |> String.replace(~r/\s+/, " ")  # Normalize whitespace
    |> String.trim()
  end

  defp find_turns(text) do
    # Tokenize to find speaker boundaries
    tokens = Tokenizer.tokenize(text)

    # Find speaker markers (capitalized word followed by colon)
    speaker_positions = find_speaker_positions(tokens, text)

    # Extract turns based on positions
    extract_turns_from_positions(text, speaker_positions)
  end

  defp find_speaker_positions(tokens, text) do
    # Look for patterns like "Name:" in tokens
    tokens
    |> Enum.with_index()
    |> Enum.filter(fn {token, idx} ->
      token_text = get_token_text(token)
      next_char = get_next_char(text, token, tokens, idx)

      # Check if this looks like a speaker name
      is_capitalized?(token_text) and
      String.length(token_text) >= 2 and
      next_char == ":"
    end)
    |> Enum.map(fn {token, _idx} ->
      token_text = get_token_text(token)
      # Find position in original text
      case find_speaker_in_text(text, token_text) do
        nil -> nil
        positions -> positions
      end
    end)
    |> List.flatten()
    |> Enum.reject(&is_nil/1)
    |> Enum.sort_by(fn {pos, _} -> pos end)
    |> Enum.uniq_by(fn {pos, _} -> pos end)
  end

  defp get_token_text(%{text: text}), do: text
  defp get_token_text(text) when is_binary(text), do: text
  defp get_token_text(_), do: ""

  defp get_next_char(text, token, tokens, idx) do
    token_text = get_token_text(token)

    # Try to find where this token ends in the text
    case :binary.match(text, token_text) do
      {start, len} ->
        end_pos = start + len
        if end_pos < byte_size(text) do
          # Skip whitespace
          remaining = String.slice(text, end_pos, 10)
          remaining = String.trim_leading(remaining)
          if String.length(remaining) > 0, do: String.at(remaining, 0), else: nil
        else
          nil
        end
      :nomatch ->
        # Fallback: check next token
        if idx + 1 < length(tokens) do
          next_token = Enum.at(tokens, idx + 1)
          next_text = get_token_text(next_token)
          if String.starts_with?(next_text, ":"), do: ":", else: nil
        else
          nil
        end
    end
  end

  defp is_capitalized?(text) when is_binary(text) and byte_size(text) > 0 do
    first_char = String.at(text, 0)
    first_char == String.upcase(first_char) and
    first_char =~ ~r/[A-Z]/
  end
  defp is_capitalized?(_), do: false

  defp find_speaker_in_text(text, speaker_name) do
    # Find all occurrences of "SpeakerName:" pattern
    pattern_str = speaker_name <> ":"

    find_all_occurrences(text, pattern_str, 0, [])
    |> Enum.map(fn pos -> {pos, speaker_name} end)
  end

  defp find_all_occurrences(text, pattern, offset, acc) do
    case :binary.match(text, pattern, [{:scope, {offset, byte_size(text) - offset}}]) do
      {pos, _len} ->
        # Verify this is a word boundary (not mid-word)
        if pos == 0 or not letter_before?(text, pos) do
          find_all_occurrences(text, pattern, pos + 1, [pos | acc])
        else
          find_all_occurrences(text, pattern, pos + 1, acc)
        end
      :nomatch ->
        Enum.reverse(acc)
    end
  end

  defp letter_before?(text, pos) when pos > 0 do
    char = String.at(text, pos - 1)
    char && char =~ ~r/[a-zA-Z]/
  end
  defp letter_before?(_, _), do: false

  defp extract_turns_from_positions(text, []) do
    # No speakers found - treat as single anonymous turn
    [{extract_speaker_fallback(text), text}]
  end

  defp extract_turns_from_positions(text, positions) do
    # Add end position
    positions_with_end = positions ++ [{byte_size(text), nil}]

    positions
    |> Enum.with_index()
    |> Enum.map(fn {{start_pos, speaker}, idx} ->
      # Get end position from next speaker
      {end_pos, _} = Enum.at(positions_with_end, idx + 1)

      # Extract text between this speaker and next
      speaker_prefix_len = String.length(speaker) + 1  # +1 for colon
      text_start = start_pos + speaker_prefix_len

      turn_text = if text_start < end_pos do
        String.slice(text, text_start, end_pos - text_start)
        |> String.trim()
        |> String.trim_leading(",")
        |> String.trim()
      else
        ""
      end

      {speaker, turn_text}
    end)
    |> Enum.reject(fn {_, text} -> text == "" end)
  end

  defp extract_speaker_fallback(text) do
    # Try to extract first capitalized word as speaker
    tokens = Tokenizer.tokenize(text)

    case Enum.find(tokens, fn t ->
      token_text = get_token_text(t)
      is_capitalized?(token_text) and String.length(token_text) >= 2
    end) do
      nil -> "Unknown"
      token -> get_token_text(token)
    end
  end
end
