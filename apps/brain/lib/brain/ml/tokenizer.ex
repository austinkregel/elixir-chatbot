defmodule Brain.ML.Tokenizer do
  @moduledoc "Unicode-aware tokenization module without regex dependency.\n\nProvides:\n- Word tokenization (unicode-aware)\n- Sentence boundary detection\n- Quoted string preservation\n- Contraction handling\n- Punctuation handling\n- Token position tracking\n"

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
  @sentence_enders ~c".!?"
  @punctuation ~c",;:\"'()[]{}<>/\\|@#$%^&*+=~`"

  @doc "Tokenize text into words, preserving position information.\nReturns a list of token maps with text, normalized form, and positions.\n"
  def tokenize(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> tokenize_graphemes([], "", 0, 0)
    |> Enum.reverse()
    |> Enum.filter(fn token -> token.text != "" end)
  end

  @doc "Tokenize text into words, returning only the text values.\nUseful for simpler use cases that don't need position tracking.\n"
  def tokenize_words(text) when is_binary(text) do
    text
    |> tokenize()
    |> Enum.map(& &1.text)
  end

  @doc """
  Tokenize text into lemmatized (base-form) words using WordNet morphological data.

  Applies the Lexicon's lemma/1 to each normalized token, reducing
  inflected forms ("running" -> "run", "cities" -> "city") and improving
  vocabulary overlap with training data.

  Falls back to the original token if no lemma is found.

  Options:
    - :min_length - minimum token length (default: 1)
    - :include_numbers - include number tokens (default: true)
    - :expand_contractions - expand contractions before tokenizing (default: false)
  """
  def tokenize_lemmatized(text, opts \\ []) when is_binary(text) do
    text
    |> tokenize_normalized(opts)
    |> Enum.map(&Brain.ML.Lexicon.lemma/1)
  end

  @doc "Tokenize text into normalized lowercase words.\nFilters out punctuation and short tokens.\n\nOptions:\n  - :min_length - minimum token length (default: 1)\n  - :include_numbers - include number tokens (default: true)\n  - :expand_contractions - expand contractions before tokenizing (default: false)\n"
  def tokenize_normalized(text, opts \\ []) when is_binary(text) do
    min_length = Keyword.get(opts, :min_length, 1)
    include_numbers = Keyword.get(opts, :include_numbers, true)
    expand = Keyword.get(opts, :expand_contractions, false)

    processed_text =
      if expand do
        expand_contractions(text)
      else
        text
      end

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

  @doc "Expand contractions in text to their full forms using heuristics.\n\nThis uses pattern-based rules rather than a lookup table, so it can\nhandle contractions it hasn't seen before by recognizing the suffix patterns:\n\n  - X'm → X am (I'm → I am)\n  - X're → X are (you're → you are, they're → they are)\n  - X'll → X will (I'll → I will, she'll → she will)\n  - X've → X have (I've → I have, could've → could have)\n  - X'd → X would (I'd → I would, he'd → he would)\n  - X's → X is (it's → it is, what's → what is)\n  - Xn't → X not (don't → do not, can't → can not)\n\nSpecial cases like \"won't\" → \"will not\" are handled separately.\n\nThis is useful as a preprocessing step before pattern matching,\nso you only need to match against the canonical forms.\n"
  def expand_contractions(text) when is_binary(text) do
    text
    |> split_preserving_delimiters()
    |> Enum.map_join(
      "",
      &expand_token/1
    )
  end

  defp split_preserving_delimiters(text) do
    text
    |> String.graphemes()
    |> chunk_by_word_boundary([])
    |> Enum.reverse()
  end

  defp chunk_by_word_boundary([], acc) do
    acc
  end

  defp chunk_by_word_boundary(graphemes, acc) do
    {token, rest} = take_next_token(graphemes)
    chunk_by_word_boundary(rest, [token | acc])
  end

  defp take_next_token([]) do
    {"", []}
  end

  defp take_next_token([first | rest] = graphemes) do
    cond do
      is_whitespace?(first) ->
        {spaces, remaining} = Enum.split_while(graphemes, &is_whitespace?/1)
        {Enum.join(spaces), remaining}

      is_word_char?(first) ->
        take_word(graphemes, [])

      true ->
        {first, rest}
    end
  end

  defp take_word([], acc) do
    {Enum.join(Enum.reverse(acc)), []}
  end

  defp take_word([char | rest] = graphemes, acc) do
    cond do
      is_word_char?(char) ->
        take_word(rest, [char | acc])

      char == "'" and rest != [] and is_word_char?(hd(rest)) ->
        take_word(rest, [char | acc])

      true ->
        {Enum.join(Enum.reverse(acc)), graphemes}
    end
  end

  defp is_whitespace?(char) do
    char in [" ", "\t", "\n", "\r"]
  end

  defp is_word_char?(char) do
    case char do
      <<c::utf8>> when c in 97..122 or c in 65..90 or c in 48..57 -> true
      _ -> false
    end
  end

  defp expand_token(token) do
    alias Brain.ML.InformalExpansions

    case InformalExpansions.expand(token) do
      {:ok, expansion} ->
        expansion

      :not_found ->
        token
    end
  end

  @doc "Split text into sentences with position information.\n"
  def split_sentences(text) when is_binary(text) do
    graphemes = String.graphemes(text)

    split_sentences_impl(graphemes, [], "", 0, 0)
    |> Enum.reverse()
    |> Enum.filter(fn sent -> String.trim(sent.text) != "" end)
  end

  @doc "Normalize text for comparison (lowercase, trim, collapse whitespace).\nDoes not use regex.\n"
  def normalize(text) when is_binary(text) do
    text
    |> String.downcase()
    |> String.trim()
    |> collapse_whitespace()
  end

  @doc "Check if a character is a word character (letter or digit).\nUnicode-aware.\n"
  def word_char?(grapheme) when is_binary(grapheme) do
    case String.to_charlist(grapheme) do
      [codepoint] ->
        letter?(codepoint) or digit?(codepoint)

      _ ->
        false
    end
  end

  @doc "Check if a grapheme is whitespace.\n"
  def whitespace?(grapheme) when is_binary(grapheme) do
    case grapheme do
      " " -> true
      "\t" -> true
      "\n" -> true
      "\r" -> true
      "\v" -> true
      "\f" -> true
      <<194, 160>> -> true
      _ -> false
    end
  end

  @doc "Check if a grapheme is punctuation.\n"
  def punctuation?(grapheme) when is_binary(grapheme) do
    case String.to_charlist(grapheme) do
      [codepoint] -> codepoint in @punctuation or codepoint in @sentence_enders
      _ -> false
    end
  end

  @doc "Check if a grapheme is an emoji.\n"
  def emoji?(grapheme) when is_binary(grapheme) do
    case String.to_charlist(grapheme) do
      [codepoint] ->
        (codepoint >= 128_512 and codepoint <= 128_591) or
          (codepoint >= 127_744 and codepoint <= 128_511) or
          (codepoint >= 128_640 and codepoint <= 128_767) or
          (codepoint >= 127_456 and codepoint <= 127_487) or
          (codepoint >= 9728 and codepoint <= 9983) or
          (codepoint >= 9984 and codepoint <= 10_175) or
          (codepoint >= 65_024 and codepoint <= 65_039) or
          (codepoint >= 129_280 and codepoint <= 129_535)

      _ ->
        byte_size(grapheme) > 4
    end
  end

  @doc "Extract numbers from text without regex.\nReturns list of {number_string, start_pos, end_pos}.\n"
  def extract_numbers(text) when is_binary(text) do
    graphemes = String.graphemes(text)

    extract_numbers_impl(graphemes, [], "", 0, nil)
    |> Enum.reverse()
  end

  @doc "Extract date-like patterns from tokens without regex.\nReturns list of {date_type, value, start_pos, end_pos}.\n"
  def extract_dates(tokens) when is_list(tokens) do
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

  @doc "Strip punctuation from text, returning only words and whitespace.\nUnicode-aware, does not use regex.\n"
  def strip_punctuation(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> Enum.map_join(
      "",
      fn g ->
        if punctuation?(g) do
          " "
        else
          g
        end
      end
    )
    |> collapse_whitespace_public()
    |> String.trim()
  end

  @doc "Collapse multiple whitespace characters into single spaces.\nDoes not use regex.\n"
  def collapse_whitespace_public(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> collapse_whitespace_impl([], false)
    |> Enum.reverse()
    |> Enum.join()
  end

  @doc "Extract quoted sections from text.\nReturns list of {quoted_text, start_pos, end_pos}.\nDoes not use regex.\n"
  def extract_quoted_sections(text) when is_binary(text) do
    graphemes = String.graphemes(text)
    extract_quoted_impl(graphemes, [], nil, "", 0, 0, nil)
  end

  @doc "Split text on whitespace without using regex.\nReturns list of word strings.\n"
  def split_words(text) when is_binary(text) do
    text
    |> String.graphemes()
    |> split_words_impl([], "")
    |> Enum.reverse()
    |> Enum.filter(&(&1 != ""))
  end

  @doc "Check if text starts with a given prefix (case-insensitive, word boundary).\nDoes not use regex.\n"
  def starts_with_word?(text, prefix) when is_binary(text) and is_binary(prefix) do
    lower_text = String.downcase(text)
    lower_prefix = String.downcase(prefix)

    if String.starts_with?(lower_text, lower_prefix) do
      rest = String.slice(lower_text, String.length(lower_prefix)..-1//1)

      case String.graphemes(rest) do
        [] -> true
        [first | _] -> whitespace?(first) or punctuation?(first)
      end
    else
      false
    end
  end

  @question_words ~w(who what when where why how which whom whose)

  @doc "Returns the set of English interrogative words (question words)."
  def question_words, do: @question_words

  @doc """
  Checks whether two strings share any tokens after normalization.

  Tokenizes both strings into words, lowercases them, and checks for
  set intersection. Returns true if any tokens overlap.
  """
  def tokens_overlap?(a, b) when is_binary(a) and is_binary(b) do
    a_tokens = a |> tokenize_words() |> MapSet.new(&String.downcase/1)
    b_tokens = b |> tokenize_words() |> MapSet.new(&String.downcase/1)
    not MapSet.disjoint?(a_tokens, b_tokens)
  end

  def tokens_overlap?(_, _), do: false

  @doc "Check if text ends with a question mark.\nTrims whitespace before checking.\nAccepts both binary strings and tokenized maps with a :text key.\n"
  def ends_with_question?(text) when is_binary(text) do
    text |> String.trim_trailing() |> do_ends_with_char(63)
  end

  def ends_with_question?(%{text: text}) when is_binary(text) do
    text |> String.trim_trailing() |> do_ends_with_char(63)
  end

  def ends_with_question?(_) do
    false
  end

  @doc "Check if text ends with an exclamation mark.\nTrims whitespace before checking.\n"
  def ends_with_exclamation?(text) when is_binary(text) do
    text |> String.trim_trailing() |> do_ends_with_char(33)
  end

  @doc "Check if text ends with a period.\nTrims whitespace before checking.\n"
  def ends_with_period?(text) when is_binary(text) do
    text |> String.trim_trailing() |> do_ends_with_char(46)
  end

  @doc "Check if text ends with terminal punctuation (. ! ?).\nTrims whitespace before checking.\n"
  def ends_with_terminal_punctuation?(text) when is_binary(text) do
    trimmed = String.trim_trailing(text)

    case String.last(trimmed) do
      nil -> false
      char -> char in [".", "!", "?"]
    end
  end

  @doc "Check if text ends with an ellipsis (...).\nTrims whitespace before checking.\n"
  def ends_with_ellipsis?(text) when is_binary(text) do
    trimmed = String.trim_trailing(text)
    String.ends_with?(trimmed, "...") or String.ends_with?(trimmed, "…")
  end

  @doc "Get the terminal punctuation character from text, if any.\nReturns the punctuation character or nil.\n"
  def terminal_punctuation(text) when is_binary(text) do
    trimmed = String.trim_trailing(text)

    case String.last(trimmed) do
      char when char in [".", "!", "?"] -> char
      _ -> nil
    end
  end

  defp do_ends_with_char(text, char) do
    case String.last(text) do
      nil -> false
      <<c::utf8>> -> c == char
      _ -> false
    end
  end

  defp extract_quoted_impl([], acc, nil, _current, _current_start, _pos, _prev) do
    Enum.reverse(acc)
  end

  defp extract_quoted_impl([], acc, quote_char, current, current_start, pos, _prev) do
    _ = {quote_char, current, current_start, pos}
    Enum.reverse(acc)
  end

  defp extract_quoted_impl([g | rest], acc, nil, _current, _current_start, pos, prev) do
    if quote_char?(g) and not contraction_apostrophe?(g, prev) do
      extract_quoted_impl(rest, acc, g, "", pos, pos + 1, g)
    else
      extract_quoted_impl(rest, acc, nil, "", pos, pos + 1, g)
    end
  end

  defp extract_quoted_impl([g | rest], acc, quote_char, current, current_start, pos, _prev) do
    if g == quote_char or matching_quote?(quote_char, g) do
      quoted = {current, current_start, pos - 1}
      extract_quoted_impl(rest, [quoted | acc], nil, "", pos, pos + 1, g)
    else
      extract_quoted_impl(rest, acc, quote_char, current <> g, current_start, pos + 1, g)
    end
  end

  defp contraction_apostrophe?("'", prev) when is_binary(prev), do: is_word_char?(prev)
  defp contraction_apostrophe?(_, _), do: false

  @quote_chars ["\"", "'", "“", "”", "‘", "’"]

  defp quote_char?(g) do
    g in @quote_chars
  end

  defp matching_quote?(open, close) do
    case {open, close} do
      {"\"", "\""} -> true
      {"'", "'"} -> true
      {"“", "”"} -> true
      {"‘", "’"} -> true
      _ -> false
    end
  end

  defp split_words_impl([], acc, current) do
    if current != "" do
      [current | acc]
    else
      acc
    end
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
        if current != "" do
          token = make_token(current, current_start, pos - 1)
          tokenize_graphemes(rest, [token | acc], "", pos + 1, pos + 1)
        else
          tokenize_graphemes(rest, acc, "", pos + 1, pos + 1)
        end

      punctuation?(g) ->
        cond do
          g in [".", ","] and current != "" and all_digits_or_currency?(current) and
              starts_with_digit?(rest) ->
            tokenize_graphemes(rest, acc, current <> g, current_start, pos + 1)

          currency_symbol?(g) and current == "" and starts_with_digit?(rest) ->
            tokenize_graphemes(rest, acc, g, pos, pos + 1)

          true ->
            acc2 =
              if current != "" do
                token = make_token(current, current_start, pos - 1)
                [token | acc]
              else
                acc
              end

            punct_token = make_token(g, pos, pos)
            tokenize_graphemes(rest, [punct_token | acc2], "", pos + 1, pos + 1)
        end

      emoji?(g) ->
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

      g == "'" and current != "" ->
        case check_contraction(rest) do
          {:contraction, suffix, consumed} ->
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
            tokenize_graphemes(rest, acc, current <> g, current_start, pos + 1)
        end

      g == "-" and current != "" ->
        case rest do
          [next | _] when next != "" ->
            if word_char?(next) do
              tokenize_graphemes(rest, acc, current <> g, current_start, pos + 1)
            else
              token = make_token(current, current_start, pos - 1)
              tokenize_graphemes(rest, [token | acc], "", pos + 1, pos + 1)
            end

          _ ->
            token = make_token(current, current_start, pos - 1)
            tokenize_graphemes(rest, [token | acc], "", pos + 1, pos + 1)
        end

      true ->
        start =
          if current == "" do
            pos
          else
            current_start
          end

        tokenize_graphemes(rest, acc, current <> g, start, pos + 1)
    end
  end

  defp check_contraction(graphemes) do
    suffixes = ["t", "re", "ve", "ll", "d", "m", "s"]

    remaining_str = Enum.join(graphemes)

    Enum.find_value(suffixes, :not_contraction, fn suffix ->
      if String.starts_with?(String.downcase(remaining_str), suffix) do
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
      is_currency_number?(text) -> :number
      all_punctuation?(text) -> :punctuation
      String.contains?(text, "'") -> :contraction
      true -> :word
    end
  end

  defp is_currency_number?(text) do
    case String.graphemes(text) do
      [first | rest] when rest != [] ->
        currency_symbol?(first) and has_digit?(rest)

      _ ->
        false
    end
  end

  defp has_digit?(graphemes) do
    Enum.any?(graphemes, fn g ->
      case String.to_charlist(g) do
        [c] -> digit?(c)
        _ -> false
      end
    end)
  end

  defp all_digits?(text) do
    graphemes = String.graphemes(text)

    has_digit =
      Enum.any?(graphemes, fn g ->
        case String.to_charlist(g) do
          [c] -> digit?(c)
          _ -> false
        end
      end)

    all_numeric_chars =
      Enum.all?(graphemes, fn g ->
        case String.to_charlist(g) do
          [c] -> digit?(c) or c == 46 or c == 44
          _ -> false
        end
      end)

    has_digit and all_numeric_chars
  end

  defp all_punctuation?(text) do
    text
    |> String.graphemes()
    |> Enum.all?(&punctuation?/1)
  end

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
        if is_sentence_boundary?(rest) do
          sent = %{
            text: current <> g,
            tokens: tokenize(current <> g),
            start_pos: current_start,
            end_pos: pos
          }

          {remaining, new_pos} = skip_whitespace(rest, pos + 1)
          split_sentences_impl(remaining, [sent | acc], "", new_pos, new_pos)
        else
          split_sentences_impl(rest, acc, current <> g, current_start, pos + 1)
        end

      _ ->
        start =
          if current == "" do
            pos
          else
            current_start
          end

        split_sentences_impl(rest, acc, current <> g, start, pos + 1)
    end
  end

  defp is_sentence_boundary?([]) do
    true
  end

  defp is_sentence_boundary?([g | rest]) do
    if whitespace?(g) do
      case skip_leading_whitespace(rest) do
        [] ->
          true

        [next | _] ->
          case String.to_charlist(next) do
            [c] -> c >= 65 and c <= 90
            _ -> false
          end
      end
    else
      false
    end
  end

  defp skip_leading_whitespace([]), do: []

  defp skip_leading_whitespace([g | rest] = graphemes) do
    if whitespace?(g), do: skip_leading_whitespace(rest), else: graphemes
  end

  defp skip_whitespace([], pos) do
    {[], pos}
  end

  defp skip_whitespace([g | rest] = graphemes, pos) do
    if whitespace?(g) do
      skip_whitespace(rest, pos + 1)
    else
      {graphemes, pos}
    end
  end

  defp extract_numbers_impl([], acc, current, _pos, current_start) do
    if current != "" and current_start != nil do
      [{current, current_start, current_start + String.length(current) - 1} | acc]
    else
      acc
    end
  end

  defp extract_numbers_impl([g | rest], acc, current, pos, current_start) do
    case String.to_charlist(g) do
      [c] when c >= 48 and c <= 57 ->
        start =
          if current == "" do
            pos
          else
            current_start
          end

        extract_numbers_impl(rest, acc, current <> g, pos + 1, start)

      [c] when c == 46 or c == 44 ->
        if current != "" do
          extract_numbers_impl(rest, acc, current <> g, pos + 1, current_start)
        else
          extract_numbers_impl(rest, acc, "", pos + 1, nil)
        end

      _ ->
        if current != "" and current_start != nil do
          entry = {current, current_start, pos - 1}
          extract_numbers_impl(rest, [entry | acc], "", pos + 1, nil)
        else
          extract_numbers_impl(rest, acc, "", pos + 1, nil)
        end
    end
  end

  defp is_date_number?(text) do
    case Integer.parse(text) do
      {n, ""} -> (n >= 1 and n <= 31) or (n >= 1900 and n <= 2100)
      _ -> false
    end
  end

  defp collapse_whitespace(text) do
    text
    |> String.graphemes()
    |> collapse_whitespace_impl([], false)
    |> Enum.reverse()
    |> Enum.join()
  end

  defp collapse_whitespace_impl([], acc, _in_ws) do
    acc
  end

  defp collapse_whitespace_impl([g | rest], acc, in_ws) do
    if whitespace?(g) do
      if in_ws do
        collapse_whitespace_impl(rest, acc, true)
      else
        collapse_whitespace_impl(rest, [" " | acc], true)
      end
    else
      collapse_whitespace_impl(rest, [g | acc], false)
    end
  end

  defp letter?(codepoint) do
    (codepoint >= 97 and codepoint <= 122) or
      (codepoint >= 65 and codepoint <= 90) or
      (codepoint >= 192 and codepoint <= 255) or
      (codepoint >= 256 and codepoint <= 383) or
      (codepoint >= 880 and codepoint <= 1023) or
      (codepoint >= 1024 and codepoint <= 1279) or
      codepoint >= 1280
  end

  defp digit?(codepoint) do
    codepoint >= 48 and codepoint <= 57
  end

  defp all_digits_or_currency?(text) do
    case String.graphemes(text) do
      [] ->
        false

      [first | rest] ->
        if currency_symbol?(first) do
          Enum.all?(rest, fn g ->
            case String.to_charlist(g) do
              [c] -> digit?(c) or c == 46 or c == 44
              _ -> false
            end
          end)
        else
          Enum.all?(String.graphemes(text), fn g ->
            case String.to_charlist(g) do
              [c] -> digit?(c) or c == 46 or c == 44
              _ -> false
            end
          end)
        end
    end
  end

  defp starts_with_digit?([]) do
    false
  end

  defp starts_with_digit?([first | _rest]) do
    case String.to_charlist(first) do
      [c] -> digit?(c)
      _ -> false
    end
  end

  @currency_symbols [
    "$",
    "€",
    "£",
    "¥",
    "₹",
    "₽",
    "₩",
    "฿",
    "₫",
    "₴",
    "₦",
    "₱",
    "₪",
    "₡",
    "₲",
    "₵"
  ]

  defp currency_symbol?(g) when is_binary(g) do
    g in @currency_symbols
  end

  defp currency_symbol?(_) do
    false
  end
end
