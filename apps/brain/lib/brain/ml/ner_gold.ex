defmodule Brain.ML.NerGold do
  @moduledoc """
  Turns rows of the NER gold standard (`priv/evaluation/ner/gold_standard.json`)
  into token-aligned spans and BIO tag sequences.

  A gold row gives its entities as `{type, value}` pairs in the order they occur,
  with no positions. A value is placed where it matches the text
  case-insensitively and starts and ends on a token boundary of
  `Brain.ML.Tokenizer.tokenize/1`, each entity after the previous one. A row whose
  entities have no such placement, or more than one, raises naming the row:
  choosing between placements would be a guess. Every row in the corpus had
  exactly one placement when this was written (2026-10-08, 2,110 rows).
  """

  alias Brain.ML.Tokenizer

  @type span :: %{type: String.t(), value: String.t(), start_pos: non_neg_integer(), end_pos: non_neg_integer()}

  @doc "The row's entities with grapheme positions (`end_pos` inclusive), in text order."
  @spec spans!(map()) :: [span()]
  def spans!(%{"text" => text, "expected" => expected} = row) when is_binary(text) and is_list(expected) do
    tokens = Tokenizer.tokenize(text)

    case placements(String.downcase(text), tokens, expected, 0) do
      [placement] ->
        Enum.zip_with(expected, placement, fn %{"type" => type, "value" => value}, {start_pos, end_pos} ->
          %{type: type, value: value, start_pos: start_pos, end_pos: end_pos}
        end)

      [] ->
        raise ArgumentError, "NerGold: no token-aligned placement of #{inspect(values(expected))} in #{inspect(row["text"])}"

      many ->
        raise ArgumentError,
              "NerGold: #{length(many)} token-aligned placements of #{inspect(values(expected))} in #{inspect(row["text"])}"
    end
  end

  @doc """
  The row's tokens and BIO tags: `B-<type>` on an entity's first token, `I-<type>`
  on the rest, `O` elsewhere. Returns `%{tokens: [token text], tags: [tag]}`.
  """
  @spec sequence!(map()) :: %{tokens: [String.t()], tags: [String.t()]}
  def sequence!(%{"text" => text} = row) do
    spans = spans!(row)
    tokens = Tokenizer.tokenize(text)

    tags =
      Enum.map(tokens, fn token ->
        case Enum.find(spans, &(token.start_pos >= &1.start_pos and token.end_pos <= &1.end_pos)) do
          nil -> "O"
          %{start_pos: start, type: type} when token.start_pos == start -> "B-" <> type
          %{type: type} -> "I-" <> type
        end
      end)

    %{tokens: Enum.map(tokens, & &1.text), tags: tags}
  end

  # Every ordered, non-overlapping placement of `entities` at or after `from`.
  defp placements(_text, _tokens, [], _from), do: [[]]

  defp placements(text, tokens, [%{"value" => value} | rest], from) do
    text
    |> occurrences(tokens, String.downcase(value))
    |> Enum.filter(fn {start_pos, _end_pos} -> start_pos >= from end)
    |> Enum.flat_map(fn {_start_pos, end_pos} = here ->
      text |> placements(tokens, rest, end_pos + 1) |> Enum.map(&[here | &1])
    end)
  end

  # Where `value` matches starting at a token start and ending at a token end.
  defp occurrences(text, tokens, value) do
    length = String.length(value)
    ends = MapSet.new(tokens, & &1.end_pos)

    for %{start_pos: start_pos} <- tokens,
        end_pos = start_pos + length - 1,
        MapSet.member?(ends, end_pos),
        String.slice(text, start_pos, length) == value,
        do: {start_pos, end_pos}
  end

  defp values(expected), do: Enum.map(expected, & &1["value"])
end
