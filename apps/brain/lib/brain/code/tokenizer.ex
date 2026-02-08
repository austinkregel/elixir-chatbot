defmodule Brain.Code.Tokenizer do
  @moduledoc """
  Specialized tokenizer for programming languages.

  Unlike the NLP tokenizer in `Brain.ML.Tokenizer`, this module understands
  code-specific constructs like:

  - String literals (single, double, triple-quoted)
  - Comments (line and block)
  - Operators and punctuation
  - Numeric literals (integers, floats, hex, binary)
  - Identifiers and keywords

  ## Usage

      tokens = Brain.Code.Tokenizer.tokenize("def foo(x): return x * 2", :python)
      # => [%{type: :keyword, text: "def"}, %{type: :identifier, text: "foo"}, ...]
  """

  @type token :: %{
          type: atom(),
          text: String.t(),
          start: non_neg_integer(),
          end: non_neg_integer(),
          line: non_neg_integer(),
          column: non_neg_integer()
        }

  # Operator characters by language
  @operators %{
    common: ["+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~", "?", ":", ".", ",", ";", "(", ")", "[", "]", "{", "}"],
    elixir: ["|>", "->", "<-", "::", "@", "\\", "++", "--", "..", "<>", "&&", "||", "!=", "==", "===", "!==", "<=", ">=", "=~"],
    python: ["**", "//", "@", ":=", "->", "...", "//=", "**=", "@=", "<<=", ">>=", "&=", "^=", "|="],
    ruby: ["<<", ">>", "<=>", "=~", "!~", "**", "&&=", "||=", "**="],
    go: [":=", "<-", "...", "&^", "&^="],
    java: ["++", "--", "<<", ">>", ">>>"],
    cpp: ["::", "->", ".*", "->*", "<<", ">>", "<=>", "++", "--", "&&=", "||="],
    csharp: ["=>", "??", "?.", "?[", "??=", "<<=", ">>="],
    php: ["=>", "->", "::", "??", "??=", "<=>", "**", "**="]
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Tokenizes source code into a list of tokens.

  ## Parameters
    - `source` - The source code string
    - `language` - The programming language (atom)

  ## Returns
    A list of token maps
  """
  @spec tokenize(String.t(), atom()) :: [token()]
  def tokenize(source, language \\ :common) when is_binary(source) do
    do_tokenize(source, language, 0, 1, 0, [])
    |> Enum.reverse()
  end

  @doc """
  Tokenizes and returns only token texts.

  Useful for quick analysis where position info isn't needed.
  """
  @spec tokenize_texts(String.t(), atom()) :: [String.t()]
  def tokenize_texts(source, language \\ :common) do
    tokenize(source, language)
    |> Enum.map(& &1.text)
  end

  @doc """
  Checks if a string is a valid identifier in the given language.
  """
  @spec valid_identifier?(String.t(), atom()) :: boolean()
  def valid_identifier?(text, language \\ :common) do
    case language do
      :php ->
        String.match?(text, ~r/^(\$[a-zA-Z_][a-zA-Z0-9_]*|[a-zA-Z_][a-zA-Z0-9_]*)$/)

      :elixir ->
        String.match?(text, ~r/^[a-z_][a-zA-Z0-9_]*[!?]?$/)

      _ ->
        String.match?(text, ~r/^[a-zA-Z_][a-zA-Z0-9_]*$/)
    end
  end

  @doc """
  Checks if a token is a string literal.
  """
  @spec string_literal?(token()) :: boolean()
  def string_literal?(token) do
    token.type in [:string, :char, :heredoc, :template_string]
  end

  @doc """
  Checks if a token is a comment.
  """
  @spec comment?(token()) :: boolean()
  def comment?(token) do
    token.type in [:line_comment, :block_comment, :doc_comment]
  end

  # ============================================================================
  # Private Functions - Tokenization
  # ============================================================================

  defp do_tokenize("", _language, _pos, _line, _col, tokens), do: tokens

  defp do_tokenize(source, language, pos, line, col, tokens) do
    cond do
      # Newline
      String.starts_with?(source, "\n") ->
        rest = String.slice(source, 1..-1//1)
        do_tokenize(rest, language, pos + 1, line + 1, 0, tokens)

      # Whitespace
      String.starts_with?(source, " ") or String.starts_with?(source, "\t") or
          String.starts_with?(source, "\r") ->
        rest = String.slice(source, 1..-1//1)
        do_tokenize(rest, language, pos + 1, line, col + 1, tokens)

      # Line comment
      is_line_comment_start?(source, language) ->
        {comment_text, rest} = consume_line_comment(source)
        token = make_token(:line_comment, comment_text, pos, line, col)
        do_tokenize(rest, language, pos + String.length(comment_text), line, col + String.length(comment_text), [token | tokens])

      # Block comment
      is_block_comment_start?(source, language) ->
        {comment_text, rest, lines_consumed} = consume_block_comment(source, language)
        token = make_token(:block_comment, comment_text, pos, line, col)
        new_line = line + lines_consumed
        do_tokenize(rest, language, pos + String.length(comment_text), new_line, col + String.length(comment_text), [token | tokens])

      # String literal
      is_string_start?(source, language) ->
        {string_text, rest, lines_consumed} = consume_string(source, language)
        token = make_token(:string, string_text, pos, line, col)
        new_line = line + lines_consumed
        do_tokenize(rest, language, pos + String.length(string_text), new_line, col + String.length(string_text), [token | tokens])

      # Number
      is_number_start?(source) ->
        {num_text, rest} = consume_number(source)
        token = make_token(:number, num_text, pos, line, col)
        do_tokenize(rest, language, pos + String.length(num_text), line, col + String.length(num_text), [token | tokens])

      # Identifier or keyword
      is_identifier_start?(source, language) ->
        {id_text, rest} = consume_identifier(source, language)
        type = if keyword?(id_text, language), do: :keyword, else: :identifier
        token = make_token(type, id_text, pos, line, col)
        do_tokenize(rest, language, pos + String.length(id_text), line, col + String.length(id_text), [token | tokens])

      # Operator or punctuation
      true ->
        {op_text, rest} = consume_operator(source, language)
        if op_text != "" do
          token = make_token(:operator, op_text, pos, line, col)
          do_tokenize(rest, language, pos + String.length(op_text), line, col + String.length(op_text), [token | tokens])
        else
          # Skip unknown character
          rest = String.slice(source, 1..-1//1)
          do_tokenize(rest, language, pos + 1, line, col + 1, tokens)
        end
    end
  end

  defp make_token(type, text, pos, line, col) do
    %{
      type: type,
      text: text,
      start: pos,
      end: pos + String.length(text),
      line: line,
      column: col
    }
  end

  # ============================================================================
  # Comment Detection
  # ============================================================================

  defp is_line_comment_start?(source, language) do
    case language do
      :python -> String.starts_with?(source, "#")
      :ruby -> String.starts_with?(source, "#")
      :elixir -> String.starts_with?(source, "#")
      :php -> String.starts_with?(source, "//") or String.starts_with?(source, "#")
      _ -> String.starts_with?(source, "//")
    end
  end

  defp is_block_comment_start?(source, language) do
    case language do
      :python -> String.starts_with?(source, "\"\"\"") or String.starts_with?(source, "'''")
      :ruby -> String.starts_with?(source, "=begin")
      :elixir -> false  # Elixir doesn't have block comments
      _ -> String.starts_with?(source, "/*")
    end
  end

  defp consume_line_comment(source) do
    lines = String.split(source, "\n", parts: 2)
    comment = List.first(lines) || ""
    rest = if length(lines) > 1, do: "\n" <> Enum.at(lines, 1), else: ""
    {comment, rest}
  end

  defp consume_block_comment(source, language) do
    end_marker = case language do
      :python -> if String.starts_with?(source, "\"\"\""), do: "\"\"\"", else: "'''"
      :ruby -> "=end"
      _ -> "*/"
    end

    start_marker = case language do
      :python -> if String.starts_with?(source, "\"\"\""), do: "\"\"\"", else: "'''"
      :ruby -> "=begin"
      _ -> "/*"
    end

    source_after_start = String.slice(source, String.length(start_marker)..-1//1)

    case String.split(source_after_start, end_marker, parts: 2) do
      [content, rest] ->
        full_comment = start_marker <> content <> end_marker
        lines_consumed = String.graphemes(full_comment) |> Enum.count(&(&1 == "\n"))
        {full_comment, rest, lines_consumed}

      [_] ->
        # Unclosed comment - consume rest
        lines_consumed = String.graphemes(source) |> Enum.count(&(&1 == "\n"))
        {source, "", lines_consumed}
    end
  end

  # ============================================================================
  # String Detection
  # ============================================================================

  defp is_string_start?(source, language) do
    case language do
      :python ->
        String.starts_with?(source, "\"\"\"") or
          String.starts_with?(source, "'''") or
          String.starts_with?(source, "\"") or
          String.starts_with?(source, "'") or
          String.starts_with?(source, "f\"") or
          String.starts_with?(source, "r\"")

      :elixir ->
        String.starts_with?(source, "\"\"\"") or
          String.starts_with?(source, "\"") or
          String.starts_with?(source, "'") or
          String.starts_with?(source, "~")

      :ruby ->
        String.starts_with?(source, "\"") or
          String.starts_with?(source, "'") or
          String.starts_with?(source, "%q") or
          String.starts_with?(source, "%Q")

      :go ->
        String.starts_with?(source, "\"") or
          String.starts_with?(source, "'") or
          String.starts_with?(source, "`")

      _ ->
        String.starts_with?(source, "\"") or String.starts_with?(source, "'")
    end
  end

  defp consume_string(source, language) do
    # Detect quote type
    {quote_char, is_triple} = detect_quote(source, language)

    if is_triple do
      consume_triple_quoted(source, quote_char)
    else
      consume_quoted(source, quote_char)
    end
  end

  defp detect_quote(source, language) do
    cond do
      language in [:python, :elixir] and String.starts_with?(source, "\"\"\"") ->
        {"\"\"\"", true}

      language == :python and String.starts_with?(source, "'''") ->
        {"'''", true}

      language == :go and String.starts_with?(source, "`") ->
        {"`", false}

      String.starts_with?(source, "\"") ->
        {"\"", false}

      String.starts_with?(source, "'") ->
        {"'", false}

      true ->
        {"\"", false}
    end
  end

  defp consume_triple_quoted(source, quote) do
    quote_len = String.length(quote)
    rest = String.slice(source, quote_len..-1//1)

    case find_closing_quote(rest, quote, false) do
      {:found, content, remaining} ->
        full_string = quote <> content <> quote
        lines = String.graphemes(full_string) |> Enum.count(&(&1 == "\n"))
        {full_string, remaining, lines}

      :not_found ->
        lines = String.graphemes(source) |> Enum.count(&(&1 == "\n"))
        {source, "", lines}
    end
  end

  defp consume_quoted(source, quote) do
    rest = String.slice(source, String.length(quote)..-1//1)

    case find_closing_quote(rest, quote, true) do
      {:found, content, remaining} ->
        {quote <> content <> quote, remaining, 0}

      :not_found ->
        {source, "", 0}
    end
  end

  defp find_closing_quote(source, quote, handle_escapes) do
    find_closing_quote(source, quote, handle_escapes, "")
  end

  defp find_closing_quote("", _quote, _handle_escapes, _acc), do: :not_found

  defp find_closing_quote(source, quote, handle_escapes, acc) do
    if String.starts_with?(source, quote) do
      rest = String.slice(source, String.length(quote)..-1//1)
      {:found, acc, rest}
    else
      first = String.first(source)
      rest = String.slice(source, 1..-1//1)

      if handle_escapes and first == "\\" and String.length(rest) > 0 do
        escaped = String.first(rest)
        rest2 = String.slice(rest, 1..-1//1)
        find_closing_quote(rest2, quote, handle_escapes, acc <> first <> escaped)
      else
        find_closing_quote(rest, quote, handle_escapes, acc <> first)
      end
    end
  end

  # ============================================================================
  # Number Detection
  # ============================================================================

  defp is_number_start?(source) do
    first = String.first(source)
    first != nil and (first in ~w(0 1 2 3 4 5 6 7 8 9) or
      (first == "." and second_is_digit?(source)))
  end

  defp second_is_digit?(source) do
    second = String.at(source, 1)
    second != nil and second in ~w(0 1 2 3 4 5 6 7 8 9)
  end

  defp consume_number(source) do
    # Handle hex, binary, octal prefixes
    {prefix, rest} = consume_number_prefix(source)
    {digits, remaining} = consume_number_digits(rest, prefix)
    {prefix <> digits, remaining}
  end

  defp consume_number_prefix(source) do
    cond do
      String.starts_with?(source, "0x") or String.starts_with?(source, "0X") ->
        {String.slice(source, 0, 2), String.slice(source, 2..-1//1)}

      String.starts_with?(source, "0b") or String.starts_with?(source, "0B") ->
        {String.slice(source, 0, 2), String.slice(source, 2..-1//1)}

      String.starts_with?(source, "0o") or String.starts_with?(source, "0O") ->
        {String.slice(source, 0, 2), String.slice(source, 2..-1//1)}

      true ->
        {"", source}
    end
  end

  defp consume_number_digits(source, prefix) do
    valid_chars = case prefix do
      "0x" <> _ -> ~w(0 1 2 3 4 5 6 7 8 9 a b c d e f A B C D E F _)
      "0X" <> _ -> ~w(0 1 2 3 4 5 6 7 8 9 a b c d e f A B C D E F _)
      "0b" <> _ -> ~w(0 1 _)
      "0B" <> _ -> ~w(0 1 _)
      "0o" <> _ -> ~w(0 1 2 3 4 5 6 7 _)
      "0O" <> _ -> ~w(0 1 2 3 4 5 6 7 _)
      _ -> ~w(0 1 2 3 4 5 6 7 8 9 . e E + - _)
    end

    consume_while(source, valid_chars)
  end

  # ============================================================================
  # Identifier Detection
  # ============================================================================

  defp is_identifier_start?(source, language) do
    first = String.first(source)

    cond do
      first == nil -> false
      language == :php and first == "$" -> true
      first >= "a" and first <= "z" -> true
      first >= "A" and first <= "Z" -> true
      first == "_" -> true
      true -> false
    end
  end

  defp consume_identifier(source, language) do
    valid_chars = if language == :elixir do
      # Elixir allows ? and ! at end of identifiers
      ~w(a b c d e f g h i j k l m n o p q r s t u v w x y z
         A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
         0 1 2 3 4 5 6 7 8 9 _ ? !)
    else
      ~w(a b c d e f g h i j k l m n o p q r s t u v w x y z
         A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
         0 1 2 3 4 5 6 7 8 9 _ $)
    end

    consume_while(source, valid_chars)
  end

  # ============================================================================
  # Operator Detection
  # ============================================================================

  defp consume_operator(source, language) do
    # Try multi-char operators first, then single char
    lang_ops = Map.get(@operators, language, [])
    common_ops = Map.get(@operators, :common, [])
    all_ops = (lang_ops ++ common_ops)
      |> Enum.sort_by(&(-String.length(&1)))  # Longest first

    found = Enum.find(all_ops, fn op ->
      String.starts_with?(source, op)
    end)

    if found do
      {found, String.slice(source, String.length(found)..-1//1)}
    else
      {"", source}
    end
  end

  # ============================================================================
  # Keyword Detection
  # ============================================================================

  @keywords_by_language %{
    elixir: ~w(def defp defmodule do end fn case cond if else unless when with for receive after try catch rescue raise throw exit spawn import require use alias quote unquote and or not in true false nil),
    python: ~w(False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case),
    ruby: ~w(BEGIN END alias and begin break case class def defined? do else elsif end ensure false for if in module next nil not or redo rescue retry return self super then true undef unless until when while yield),
    go: ~w(break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var),
    java: ~w(abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for goto if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while true false null var yield record sealed permits),
    c: ~w(auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while),
    cpp: ~w(alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public register reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while xor xor_eq override final),
    csharp: ~w(abstract as base bool break byte case catch char checked class const continue decimal default delegate do double else enum event explicit extern false finally fixed float for foreach goto if implicit in int interface internal is lock long namespace new null object operator out override params private protected public readonly ref return sbyte sealed short sizeof stackalloc static string struct switch this throw true try typeof uint ulong unchecked unsafe ushort using virtual void volatile while add alias ascending async await by descending dynamic equals from get global group into join let nameof on orderby partial remove select set value var when where with yield record init required file scoped),
    php: ~w(abstract and array as break callable case catch class clone const continue declare default die do echo else elseif empty enddeclare endfor endforeach endif endswitch endwhile eval exit extends final finally fn for foreach function global goto if implements include include_once instanceof insteadof interface isset list match namespace new or print private protected public readonly require require_once return static switch throw trait try unset use var while xor yield true false null self parent enum)
  }

  defp keyword?(text, language) do
    keywords = Map.get(@keywords_by_language, language, [])
    text in keywords
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp consume_while(source, valid_chars) do
    consume_while(source, valid_chars, "")
  end

  defp consume_while("", _valid_chars, acc), do: {acc, ""}

  defp consume_while(source, valid_chars, acc) do
    first = String.first(source)

    if first in valid_chars do
      rest = String.slice(source, 1..-1//1)
      consume_while(rest, valid_chars, acc <> first)
    else
      {acc, source}
    end
  end
end
