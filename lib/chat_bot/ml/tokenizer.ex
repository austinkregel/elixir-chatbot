defmodule ChatBot.ML.Tokenizer do
  @moduledoc """
  Unicode-aware tokenization module without regex dependency.

  Provides:
  - Word tokenization (unicode-aware)
  - Sentence boundary detection
  - Quoted string preservation
  - Contraction handling
  - Punctuation handling
  - Token position tracking
  """

  @type token :: %{
          text: String.t(),
          normalized: String.t(),
          start_pos: non_neg_integer(),
          end_pos: non_neg_integer(),
          type: :word | :number | :punctuation | :emoji | :contraction | :unknown
        }

  @type sentence :: %{
          text: String.t(),
          tokens: [token()],
          start_pos: non_neg_integer(),
          end_pos: non_neg_integer()
        }

  # Sentence-ending punctuation
  @sentence_enders [?., ?!, ??]

  # Word-breaking punctuation (but not part of words)
  @punctuation [
    ?,,
    ?;,
    ?:,
    ?",
    ?',
    ?(,
    ?),
    ?[,
    ?],
    ?{,
    ?},
    ?<,
    ?>,
    ?/,
    ?\\,
    ?|,
    ?@,
    ?#,
    ?$,
    ?%,
    ?^,
    ?&,
    ?*,
    ?+,
    ?=,
    ?~,
    ?`
  ]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Tokenize text into words, preserving position information.
  Returns a list of token maps with text, normalized form, and positions.
  """
  def tokenize(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> tokenize_graphemes([], "", 0, 0)
    |> Enum.reverse()
    |> Enum.filter(fn token -> token.text != "" end)
  end

  @doc """
  Tokenize text into words, returning only the text values.
  Useful for simpler use cases that don't need position tracking.
  """
  def tokenize_words(text) when is_binary(text) do
    text
    |> tokenize()
    |> Enum.map(& &1.text)
  end

  @doc """
  Tokenize text into normalized lowercase words.
  Filters out punctuation and short tokens.

  Options:
    - :min_length - minimum token length (default: 1)
    - :include_numbers - include number tokens (default: true)
    - :expand_contractions - expand contractions before tokenizing (default: false)
  """
  def tokenize_normalized(text, opts \\ []) when is_binary(text) do
    min_length = Keyword.get(opts, :min_length, 1)
    include_numbers = Keyword.get(opts, :include_numbers, true)
    expand = Keyword.get(opts, :expand_contractions, false)

    processed_text = if expand, do: expand_contractions(text), else: text

    processed_text
    |> tokenize()
    |> Enum.filter(fn token ->
      case token.type do
        :word -> String.length(token.normalized) >= min_length
        :number -> include_numbers
        :contraction -> true
        _ -> false
      end
    end)
    |> Enum.map(& &1.normalized)
  end

  @doc """
  Expand contractions in text to their full forms using heuristics.

  This uses pattern-based rules rather than a lookup table, so it can
  handle contractions it hasn't seen before by recognizing the suffix patterns:

    - X'm → X am (I'm → I am)
    - X're → X are (you're → you are, they're → they are)
    - X'll → X will (I'll → I will, she'll → she will)
    - X've → X have (I've → I have, could've → could have)
    - X'd → X would (I'd → I would, he'd → he would)
    - X's → X is (it's → it is, what's → what is)
    - Xn't → X not (don't → do not, can't → can not)

  Special cases like "won't" → "will not" are handled separately.

  This is useful as a preprocessing step before pattern matching,
  so you only need to match against the canonical forms.
  """
  def expand_contractions(text) when is_binary(text) do
    # Split into words, expand each, rejoin
    # This preserves spacing and punctuation
    text
    |> split_preserving_delimiters()
    |> Enum.map(&expand_token/1)
    |> Enum.join()
  end

  # Split text into tokens while preserving delimiters (spaces, punctuation)
  defp split_preserving_delimiters(text) do
    # Split on word boundaries but keep the delimiters
    text
    |> String.graphemes()
    |> chunk_by_word_boundary([])
    |> Enum.reverse()
  end

  defp chunk_by_word_boundary([], acc), do: acc

  defp chunk_by_word_boundary(graphemes, acc) do
    {token, rest} = take_next_token(graphemes)
    chunk_by_word_boundary(rest, [token | acc])
  end

  defp take_next_token([]), do: {"", []}

  defp take_next_token([first | rest] = graphemes) do
    cond do
      # Whitespace - take all consecutive whitespace
      is_whitespace?(first) ->
        {spaces, remaining} = Enum.split_while(graphemes, &is_whitespace?/1)
        {Enum.join(spaces), remaining}

      # Word character - take the whole word (including apostrophes for contractions)
      is_word_char?(first) ->
        take_word(graphemes, [])

      # Punctuation or other - take single char
      true ->
        {first, rest}
    end
  end

  defp take_word([], acc), do: {Enum.join(Enum.reverse(acc)), []}

  defp take_word([char | rest] = graphemes, acc) do
    cond do
      is_word_char?(char) ->
        take_word(rest, [char | acc])

      # Apostrophe followed by word chars is part of contraction
      char == "'" and rest != [] and is_word_char?(hd(rest)) ->
        take_word(rest, [char | acc])

      true ->
        {Enum.join(Enum.reverse(acc)), graphemes}
    end
  end

  defp is_whitespace?(char), do: char in [" ", "\t", "\n", "\r"]

  defp is_word_char?(char) do
    # Check if it's a letter or digit
    case char do
      <<c::utf8>> when c in ?a..?z or c in ?A..?Z or c in ?0..?9 -> true
      _ -> false
    end
  end

  # Expand a single token if it's a contraction/informal form
  # Uses the data-driven InformalExpansions module
  defp expand_token(token) do
    alias ChatBot.ML.InformalExpansions

    case InformalExpansions.expand(token) do
      {:ok, expansion} ->
        # Case is already preserved by InformalExpansions
        expansion

      :not_found ->
        token
    end
  end

  @doc """
  Split text into sentences with position information.
  """
  def split_sentences(text) when is_binary(text) do
    graphemes = String.graphemes(text)

    split_sentences_impl(graphemes, [], "", 0, 0)
    |> Enum.reverse()
    |> Enum.filter(fn sent -> String.trim(sent.text) != "" end)
  end

  @doc """
  Normalize text for comparison (lowercase, trim, collapse whitespace).
  Does not use regex.
  """
  def normalize(text) when is_binary(text) do
    text
    |> String.downcase()
    |> String.trim()
    |> collapse_whitespace()
  end

  @doc """
  Check if a character is a word character (letter or digit).
  Unicode-aware.
  """
  def word_char?(grapheme) when is_binary(grapheme) do
    case String.to_charlist(grapheme) do
      [codepoint] ->
        letter?(codepoint) or digit?(codepoint)

      _ ->
        # Multi-codepoint grapheme (e.g., emoji with modifier)
        false
    end
  end

  @doc """
  Check if a grapheme is whitespace.
  """
  def whitespace?(grapheme) when is_binary(grapheme) do
    case grapheme do
      " " -> true
      "\t" -> true
      "\n" -> true
      "\r" -> true
      "\v" -> true
      "\f" -> true
      # Non-breaking space
      <<194, 160>> -> true
      _ -> false
    end
  end

  @doc """
  Check if a grapheme is punctuation.
  """
  def punctuation?(grapheme) when is_binary(grapheme) do
    case String.to_charlist(grapheme) do
      [codepoint] -> codepoint in @punctuation or codepoint in @sentence_enders
      _ -> false
    end
  end

  @doc """
  Check if a grapheme is an emoji.
  """
  def emoji?(grapheme) when is_binary(grapheme) do
    case String.to_charlist(grapheme) do
      [codepoint] ->
        # Common emoji ranges
        # Emoticons
        # Misc Symbols and Pictographs
        # Transport and Map
        # Flags
        # Misc symbols
        # Dingbats
        # Variation Selectors
        # Supplemental Symbols
        (codepoint >= 0x1F600 and codepoint <= 0x1F64F) or
          (codepoint >= 0x1F300 and codepoint <= 0x1F5FF) or
          (codepoint >= 0x1F680 and codepoint <= 0x1F6FF) or
          (codepoint >= 0x1F1E0 and codepoint <= 0x1F1FF) or
          (codepoint >= 0x2600 and codepoint <= 0x26FF) or
          (codepoint >= 0x2700 and codepoint <= 0x27BF) or
          (codepoint >= 0xFE00 and codepoint <= 0xFE0F) or
          (codepoint >= 0x1F900 and codepoint <= 0x1F9FF)

      _ ->
        # Multi-codepoint - likely emoji sequence
        byte_size(grapheme) > 4
    end
  end

  @doc """
  Extract numbers from text without regex.
  Returns list of {number_string, start_pos, end_pos}.
  """
  def extract_numbers(text) when is_binary(text) do
    graphemes = String.graphemes(text)

    extract_numbers_impl(graphemes, [], "", 0, nil)
    |> Enum.reverse()
  end

  @doc """
  Extract date-like patterns from tokens without regex.
  Returns list of {date_type, value, start_pos, end_pos}.
  """
  def extract_dates(tokens) when is_list(tokens) do
    # Look for relative date words
    relative_dates = ~w(today tomorrow yesterday)
    day_names = ~w(monday tuesday wednesday thursday friday saturday sunday)

    month_names =
      ~w(january february march april may june july august september october november december jan feb mar apr may jun jul aug sep oct nov dec)

    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, _idx} ->
      lower = String.downcase(token.text)

      cond do
        lower in relative_dates ->
          [{:relative_date, token.text, token.start_pos, token.end_pos}]

        lower in day_names ->
          [{:day_name, token.text, token.start_pos, token.end_pos}]

        lower in month_names ->
          [{:month_name, token.text, token.start_pos, token.end_pos}]

        is_date_number?(token.text) ->
          [{:date_number, token.text, token.start_pos, token.end_pos}]

        true ->
          []
      end
    end)
  end

  # ============================================================================
  # Text Cleaning Functions (Regex-Free)
  # ============================================================================

  @doc """
  Strip punctuation from text, returning only words and whitespace.
  Unicode-aware, does not use regex.
  """
  def strip_punctuation(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> Enum.map(fn g ->
      if punctuation?(g), do: " ", else: g
    end)
    |> Enum.join()
    |> collapse_whitespace_public()
    |> String.trim()
  end

  @doc """
  Collapse multiple whitespace characters into single spaces.
  Does not use regex.
  """
  def collapse_whitespace_public(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> collapse_whitespace_impl([], false)
    |> Enum.reverse()
    |> Enum.join()
  end

  @doc """
  Extract quoted sections from text.
  Returns list of {quoted_text, start_pos, end_pos}.
  Does not use regex.
  """
  def extract_quoted_sections(text) when is_binary(text) do
    graphemes = String.graphemes(text)
    extract_quoted_impl(graphemes, [], nil, "", 0, 0)
  end

  @doc """
  Split text on whitespace without using regex.
  Returns list of word strings.
  """
  def split_words(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> split_words_impl([], "")
    |> Enum.reverse()
    |> Enum.filter(&(&1 != ""))
  end

  @doc """
  Check if text starts with a given prefix (case-insensitive, word boundary).
  Does not use regex.
  """
  def starts_with_word?(text, prefix) when is_binary(text) and is_binary(prefix) do
    lower_text = String.downcase(text)
    lower_prefix = String.downcase(prefix)

    if String.starts_with?(lower_text, lower_prefix) do
      # Check word boundary after prefix
      rest = String.slice(lower_text, String.length(lower_prefix)..-1//1)

      case String.graphemes(rest) do
        [] -> true
        [first | _] -> whitespace?(first) or punctuation?(first)
      end
    else
      false
    end
  end

  # Private helpers for new functions

  defp extract_quoted_impl([], acc, nil, _current, _current_start, _pos), do: Enum.reverse(acc)

  defp extract_quoted_impl([], acc, quote_char, current, current_start, pos) do
    # Unclosed quote - ignore it
    _ = {quote_char, current, current_start, pos}
    Enum.reverse(acc)
  end

  defp extract_quoted_impl([g | rest], acc, nil, _current, _current_start, pos) do
    if quote_char?(g) do
      # Starting a quoted section
      extract_quoted_impl(rest, acc, g, "", pos, pos + 1)
    else
      extract_quoted_impl(rest, acc, nil, "", pos, pos + 1)
    end
  end

  defp extract_quoted_impl([g | rest], acc, quote_char, current, current_start, pos) do
    if g == quote_char or matching_quote?(quote_char, g) do
      # End of quoted section
      quoted = {current, current_start, pos - 1}
      extract_quoted_impl(rest, [quoted | acc], nil, "", pos, pos + 1)
    else
      extract_quoted_impl(rest, acc, quote_char, current <> g, current_start, pos + 1)
    end
  end

  @quote_chars [
    "\"",
    "'",
    "\u201C",
    "\u201D",
    "\u2018",
    "\u2019"
  ]

  defp quote_char?(g), do: g in @quote_chars

  defp matching_quote?(open, close) do
    case {open, close} do
      {"\"", "\""} -> true
      {"'", "'"} -> true
      {"\u201C", "\u201D"} -> true
      {"\u2018", "\u2019"} -> true
      _ -> false
    end
  end

  defp split_words_impl([], acc, current) do
    if current != "", do: [current | acc], else: acc
  end

  defp split_words_impl([g | rest], acc, current) do
    if whitespace?(g) do
      if current != "" do
        split_words_impl(rest, [current | acc], "")
      else
        split_words_impl(rest, acc, "")
      end
    else
      split_words_impl(rest, acc, current <> g)
    end
  end

  # ============================================================================
  # Private Implementation
  # ============================================================================

  # Main tokenizer state machine
  defp tokenize_graphemes([], acc, current, current_start, pos) do
    if current != "" do
      token = make_token(current, current_start, pos - 1)
      [token | acc]
    else
      acc
    end
  end

  defp tokenize_graphemes([g | rest], acc, current, current_start, pos) do
    cond do
      whitespace?(g) ->
        # End current token if any
        if current != "" do
          token = make_token(current, current_start, pos - 1)
          tokenize_graphemes(rest, [token | acc], "", pos + 1, pos + 1)
        else
          tokenize_graphemes(rest, acc, "", pos + 1, pos + 1)
        end

      punctuation?(g) ->
        # End current token and add punctuation as separate token
        acc2 =
          if current != "" do
            token = make_token(current, current_start, pos - 1)
            [token | acc]
          else
            acc
          end

        punct_token = make_token(g, pos, pos)
        tokenize_graphemes(rest, [punct_token | acc2], "", pos + 1, pos + 1)

      emoji?(g) ->
        # End current token and add emoji as separate token
        acc2 =
          if current != "" do
            token = make_token(current, current_start, pos - 1)
            [token | acc]
          else
            acc
          end

        emoji_token = %{
          text: g,
          normalized: g,
          start_pos: pos,
          end_pos: pos,
          type: :emoji
        }

        tokenize_graphemes(rest, [emoji_token | acc2], "", pos + 1, pos + 1)

      # Apostrophe handling for contractions
      g == "'" and current != "" ->
        # Look ahead for contraction patterns
        case check_contraction(rest) do
          {:contraction, suffix, consumed} ->
            # Include the contraction in current token
            full_token = current <> "'" <> suffix
            token_end = pos + String.length(suffix)

            token = %{
              text: full_token,
              normalized: String.downcase(full_token),
              start_pos: current_start,
              end_pos: token_end,
              type: :contraction
            }

            remaining = Enum.drop(rest, consumed)
            tokenize_graphemes(remaining, [token | acc], "", token_end + 1, token_end + 1)

          :not_contraction ->
            # Just add apostrophe to current
            tokenize_graphemes(rest, acc, current <> g, current_start, pos + 1)
        end

      # Hyphen in compound words
      g == "-" and current != "" ->
        # Check if it's a compound word (letter-letter)
        case rest do
          [next | _] when next != "" ->
            if word_char?(next) do
              # Part of compound word
              tokenize_graphemes(rest, acc, current <> g, current_start, pos + 1)
            else
              # End of word
              token = make_token(current, current_start, pos - 1)
              tokenize_graphemes(rest, [token | acc], "", pos + 1, pos + 1)
            end

          _ ->
            token = make_token(current, current_start, pos - 1)
            tokenize_graphemes(rest, [token | acc], "", pos + 1, pos + 1)
        end

      true ->
        # Regular character - add to current token
        start = if current == "", do: pos, else: current_start
        tokenize_graphemes(rest, acc, current <> g, start, pos + 1)
    end
  end

  defp check_contraction(graphemes) do
    # Common contraction suffixes
    suffixes = ["t", "re", "ve", "ll", "d", "m", "s"]

    remaining_str = Enum.join(graphemes)

    Enum.find_value(suffixes, :not_contraction, fn suffix ->
      if String.starts_with?(String.downcase(remaining_str), suffix) do
        # Check if followed by non-word char or end
        rest_after = String.slice(remaining_str, String.length(suffix)..-1//1)

        if rest_after == "" or not word_char?(String.first(rest_after) || "") do
          {:contraction, String.slice(remaining_str, 0, String.length(suffix)),
           String.length(suffix)}
        else
          nil
        end
      else
        nil
      end
    end)
  end

  defp make_token(text, start_pos, end_pos) do
    normalized = String.downcase(text)
    type = classify_token(text)

    %{
      text: text,
      normalized: normalized,
      start_pos: start_pos,
      end_pos: end_pos,
      type: type
    }
  end

  defp classify_token(text) do
    cond do
      all_digits?(text) -> :number
      all_punctuation?(text) -> :punctuation
      String.contains?(text, "'") -> :contraction
      true -> :word
    end
  end

  defp all_digits?(text) do
    text
    |> String.graphemes()
    |> Enum.all?(fn g ->
      case String.to_charlist(g) do
        [c] -> digit?(c) or c == ?. or c == ?,
        _ -> false
      end
    end)
  end

  defp all_punctuation?(text) do
    text
    |> String.graphemes()
    |> Enum.all?(&punctuation?/1)
  end

  # Sentence splitting
  defp split_sentences_impl([], acc, current, current_start, pos) do
    if String.trim(current) != "" do
      sent = %{
        text: current,
        tokens: tokenize(current),
        start_pos: current_start,
        end_pos: pos - 1
      }

      [sent | acc]
    else
      acc
    end
  end

  defp split_sentences_impl([g | rest], acc, current, current_start, pos) do
    case String.to_charlist(g) do
      [c] when c in @sentence_enders ->
        # Check if followed by space and capital letter (real sentence end)
        if is_sentence_boundary?(rest) do
          sent = %{
            text: current <> g,
            tokens: tokenize(current <> g),
            start_pos: current_start,
            end_pos: pos
          }

          # Skip whitespace before next sentence
          {remaining, new_pos} = skip_whitespace(rest, pos + 1)
          split_sentences_impl(remaining, [sent | acc], "", new_pos, new_pos)
        else
          split_sentences_impl(rest, acc, current <> g, current_start, pos + 1)
        end

      _ ->
        start = if current == "", do: pos, else: current_start
        split_sentences_impl(rest, acc, current <> g, start, pos + 1)
    end
  end

  defp is_sentence_boundary?([]) do
    true
  end

  defp is_sentence_boundary?([g | rest]) do
    if whitespace?(g) do
      # Look for capital letter after whitespace
      case rest do
        [] ->
          true

        [next | _] ->
          case String.to_charlist(next) do
            [c] -> c >= ?A and c <= ?Z
            _ -> false
          end
      end
    else
      false
    end
  end

  defp skip_whitespace([], pos), do: {[], pos}

  defp skip_whitespace([g | rest] = graphemes, pos) do
    if whitespace?(g) do
      skip_whitespace(rest, pos + 1)
    else
      {graphemes, pos}
    end
  end

  # Number extraction
  defp extract_numbers_impl([], acc, current, _pos, current_start) do
    if current != "" and current_start != nil do
      [{current, current_start, current_start + String.length(current) - 1} | acc]
    else
      acc
    end
  end

  defp extract_numbers_impl([g | rest], acc, current, pos, current_start) do
    case String.to_charlist(g) do
      [c] when c >= ?0 and c <= ?9 ->
        # Start or continue number
        start = if current == "", do: pos, else: current_start
        extract_numbers_impl(rest, acc, current <> g, pos + 1, start)

      [c] when c == ?. or c == ?, ->
        # Decimal or thousands separator - only if in number
        if current != "" do
          extract_numbers_impl(rest, acc, current <> g, pos + 1, current_start)
        else
          extract_numbers_impl(rest, acc, "", pos + 1, nil)
        end

      _ ->
        # End of number
        if current != "" and current_start != nil do
          entry = {current, current_start, pos - 1}
          extract_numbers_impl(rest, [entry | acc], "", pos + 1, nil)
        else
          extract_numbers_impl(rest, acc, "", pos + 1, nil)
        end
    end
  end

  defp is_date_number?(text) do
    # Check if looks like a day (1-31) or year (1900-2100)
    case Integer.parse(text) do
      {n, ""} -> (n >= 1 and n <= 31) or (n >= 1900 and n <= 2100)
      _ -> false
    end
  end

  # Utility functions
  defp collapse_whitespace(text) do
    text
    |> String.graphemes()
    |> collapse_whitespace_impl([], false)
    |> Enum.reverse()
    |> Enum.join()
  end

  defp collapse_whitespace_impl([], acc, _in_ws), do: acc

  defp collapse_whitespace_impl([g | rest], acc, in_ws) do
    if whitespace?(g) do
      if in_ws do
        # Skip consecutive whitespace
        collapse_whitespace_impl(rest, acc, true)
      else
        # First whitespace - add single space
        collapse_whitespace_impl(rest, [" " | acc], true)
      end
    else
      collapse_whitespace_impl(rest, [g | acc], false)
    end
  end

  defp letter?(codepoint) do
    # Basic Latin letters
    # Extended Latin
    # Greek
    # Cyrillic
    # General category check for other scripts
    (codepoint >= ?a and codepoint <= ?z) or
      (codepoint >= ?A and codepoint <= ?Z) or
      (codepoint >= 0x00C0 and codepoint <= 0x00FF) or
      (codepoint >= 0x0100 and codepoint <= 0x017F) or
      (codepoint >= 0x0370 and codepoint <= 0x03FF) or
      (codepoint >= 0x0400 and codepoint <= 0x04FF) or
      codepoint >= 0x0500
  end

  defp digit?(codepoint) do
    codepoint >= ?0 and codepoint <= ?9
  end
end
