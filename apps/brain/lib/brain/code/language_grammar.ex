defmodule Brain.Code.LanguageGrammar do
  @moduledoc """
  Manages tree-sitter language grammars.

  This module handles:
  - Loading and caching tree-sitter parsers
  - Grammar file management
  - Language detection from file content

  ## Architecture

  Parsers are loaded lazily and cached in ETS for performance.
  Each language has a corresponding tree-sitter grammar that must
  be compiled as a shared library (.so/.dylib).

  ## Current Status: Degraded Mode (Fallback Parser)

  The `TreeSitter` dependency is **not currently installed** in `mix.exs`.
  As a result, this module always operates in **fallback mode**, which
  provides basic line-based parsing and tokenization instead of real AST
  construction. This means the entire Code Analysis system (Parser,
  SymbolExtractor, RelationMapper, Summarizer, etc.) works on shallow
  heuristics rather than true abstract syntax trees.

  To enable full tree-sitter support:
  1. Add `{:tree_sitter, "~> x.x"}` to `apps/brain/mix.exs`
  2. Compile tree-sitter grammars as shared libraries
  3. Place them in `priv/code/grammars/`

  ## Grammar Sources

  Tree-sitter grammars are available from:
  - https://github.com/tree-sitter/tree-sitter-python
  - https://github.com/tree-sitter/tree-sitter-c
  - https://github.com/elixir-lang/tree-sitter-elixir
  - etc.
  """

  # TreeSitter is an optional dependency (not currently installed).
  # See mix.exs for installation instructions.
  @compile {:no_warn_undefined, TreeSitter}

  use GenServer
  require Logger

  @ets_table :code_language_grammars
  @grammars_dir "priv/code/grammars"

  # Language to grammar mapping
  @language_grammars %{
    c: "tree-sitter-c",
    cpp: "tree-sitter-cpp",
    java: "tree-sitter-java",
    csharp: "tree-sitter-c-sharp",
    php: "tree-sitter-php",
    python: "tree-sitter-python",
    ruby: "tree-sitter-ruby",
    elixir: "tree-sitter-elixir",
    go: "tree-sitter-go"
  }

  # Language display names
  @language_names %{
    c: "C",
    cpp: "C++",
    java: "Java",
    csharp: "C#",
    php: "PHP",
    python: "Python",
    ruby: "Ruby",
    elixir: "Elixir",
    go: "Go"
  }

  # Shebang patterns for language detection - defined as a function to avoid module attribute issues
  defp shebang_patterns do
    [
      {"python", :python},
      {"ruby", :ruby},
      {"php", :php},
      {"bash", :bash},
      {"sh", :shell}
    ]
  end

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the LanguageGrammar GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Gets a parser for the specified language.

  Returns a cached parser if available, otherwise loads and caches it.

  ## Returns
    - `{:ok, parser}` - The parser ready for use
    - `{:error, reason}` - If the grammar is not available
  """
  @spec get_parser(atom()) :: {:ok, term()} | {:error, term()}
  def get_parser(language) when is_atom(language) do
    case lookup_cached_parser(language) do
      {:ok, parser} ->
        {:ok, parser}

      :not_cached ->
        load_parser(language)
    end
  end

  @doc """
  Checks if the LanguageGrammar service is ready.
  """
  @spec ready?() :: boolean()
  def ready? do
    try do
      :ets.info(@ets_table) != :undefined and Process.whereis(__MODULE__) != nil
    rescue
      ArgumentError -> false
    end
  end

  @doc """
  Checks if a grammar is available for the language.
  """
  @spec grammar_available?(atom()) :: boolean()
  def grammar_available?(language) do
    case get_parser(language) do
      {:ok, _} -> true
      {:error, _} -> false
    end
  end

  @doc """
  Returns the display name for a language.

  ## Examples

      iex> Brain.Code.LanguageGrammar.language_name(:cpp)
      "C++"
  """
  @spec language_name(atom()) :: String.t()
  def language_name(language) do
    Map.get(@language_names, language, to_string(language))
  end

  @doc """
  Detects language from file content using shebang and content patterns.

  This is used when file extension is not available or ambiguous.

  ## Examples

      iex> Brain.Code.LanguageGrammar.detect_from_content("#!/usr/bin/env python3\\nprint('hi')")
      {:ok, :python}
  """
  @spec detect_from_content(String.t()) :: {:ok, atom()} | :unknown
  def detect_from_content(content) when is_binary(content) do
    # Check shebang first
    first_line = content |> String.split("\n", parts: 2) |> List.first() || ""

    shebang_result =
      if String.starts_with?(first_line, "#!") do
        Enum.find_value(shebang_patterns(), fn {pattern, lang} ->
          if String.contains?(first_line, pattern), do: lang
        end)
      end

    if shebang_result do
      {:ok, shebang_result}
    else
      # Try content-based detection
      detect_from_patterns(content)
    end
  end

  @doc """
  Returns all available languages and their status.

  ## Returns
    A list of maps with language info and availability status.
  """
  @spec list_languages() :: [map()]
  def list_languages do
    Map.keys(@language_grammars)
    |> Enum.map(fn lang ->
      %{
        language: lang,
        name: language_name(lang),
        grammar: Map.get(@language_grammars, lang),
        available: grammar_available?(lang)
      }
    end)
    |> Enum.sort_by(& &1.name)
  end

  @doc """
  Preloads all available grammars into cache.

  Useful at application startup to ensure grammars are ready.
  """
  @spec preload_all() :: :ok
  def preload_all do
    GenServer.cast(__MODULE__, :preload_all)
  end

  @doc """
  Returns grammar statistics.
  """
  @spec stats() :: map()
  def stats do
    try do
      cached_count =
        :ets.tab2list(@ets_table)
        |> Enum.count(fn {_lang, status, _} -> status == :loaded end)

      total = map_size(@language_grammars)

      %{
        total_languages: total,
        cached_parsers: cached_count,
        grammars_dir: grammars_path()
      }
    rescue
      ArgumentError ->
        %{total_languages: map_size(@language_grammars), cached_parsers: 0, grammars_dir: grammars_path()}
    end
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    create_ets_table()
    ensure_grammars_dir()

    Logger.info("LanguageGrammar started", %{
      languages: Map.keys(@language_grammars),
      grammars_dir: grammars_path()
    })

    {:ok, %{initialized: true}}
  end

  @impl true
  def handle_call({:load_parser, language}, _from, state) do
    result = do_load_parser(language)
    {:reply, result, state}
  end

  @impl true
  def handle_cast(:preload_all, state) do
    Logger.info("Preloading all language grammars...")

    Map.keys(@language_grammars)
    |> Enum.each(fn lang ->
      case do_load_parser(lang) do
        {:ok, _} -> Logger.debug("Loaded grammar for #{lang}")
        {:error, reason} -> Logger.debug("Could not load grammar for #{lang}: #{inspect(reason)}")
      end
    end)

    {:noreply, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp create_ets_table do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
    end
  end

  defp ensure_grammars_dir do
    path = grammars_path()

    unless File.exists?(path) do
      File.mkdir_p!(path)
    end
  end

  defp grammars_path do
    case :code.priv_dir(:brain) do
      {:error, _} ->
        Path.join(["apps", "brain", @grammars_dir])

      priv_dir ->
        Path.join(priv_dir, "code/grammars")
    end
  end

  defp lookup_cached_parser(language) do
    try do
      case :ets.lookup(@ets_table, language) do
        [{^language, :loaded, parser}] -> {:ok, parser}
        [{^language, :error, _reason}] -> :not_cached
        [] -> :not_cached
      end
    rescue
      ArgumentError -> :not_cached
    end
  end

  defp load_parser(language) do
    GenServer.call(__MODULE__, {:load_parser, language})
  catch
    :exit, {:noproc, _} ->
      # GenServer not started, load directly
      do_load_parser(language)
  end

  defp do_load_parser(language) do
    grammar_name = Map.get(@language_grammars, language)

    cond do
      is_nil(grammar_name) ->
        {:error, {:unknown_language, language}}

      not Code.ensure_loaded?(TreeSitter) ->
        fallback = create_fallback_parser(language)
        cache_parser(language, fallback)
        {:ok, fallback}

      true ->
        case load_grammar_file(language, grammar_name) do
          {:ok, parser} ->
            cache_parser(language, parser)
            {:ok, parser}

          {:error, reason} ->
            Logger.warning("Failed to load grammar for #{language}: #{inspect(reason)}, using fallback")
            fallback = create_fallback_parser(language)
            cache_parser(language, fallback)
            {:ok, fallback}
        end
    end
  end

  defp load_grammar_file(language, grammar_name) do
    # Try to load the tree-sitter grammar shared library
    grammar_path = Path.join(grammars_path(), "#{grammar_name}.so")

    if File.exists?(grammar_path) do
      try do
        # Load via tree-sitter NIF
        if Code.ensure_loaded?(TreeSitter) do
          TreeSitter.load_language(grammar_path, language)
        else
          {:error, :tree_sitter_not_available}
        end
      rescue
        e -> {:error, {:load_failed, Exception.message(e)}}
      end
    else
      {:error, {:grammar_not_found, grammar_path}}
    end
  end

  defp create_fallback_parser(language) do
    # Create a simple fallback parser that provides basic functionality
    # when tree-sitter grammars are not available
    %{
      type: :fallback,
      language: language,
      parse: fn source_code ->
        # Simple line-based parsing
        lines = String.split(source_code, "\n")

        children =
          lines
          |> Enum.with_index()
          |> Enum.map(fn {line, idx} ->
            %{
              type: "line",
              text: line,
              start_byte: 0,
              end_byte: byte_size(line),
              start_point: {idx, 0},
              end_point: {idx, String.length(line)},
              children: tokenize_line(line, language)
            }
          end)

        %{
          type: "source",
          text: source_code,
          start_byte: 0,
          end_byte: byte_size(source_code),
          start_point: {0, 0},
          end_point: {length(lines) - 1, 0},
          children: children,
          is_fallback: true
        }
      end
    }
  end

  # Simple tokenization for fallback parser
  defp tokenize_line(line, _language) do
    # Split on whitespace and punctuation, keeping tokens with position info
    line
    |> String.graphemes()
    |> Enum.reduce({[], "", 0}, fn char, {tokens, current, pos} ->
      cond do
        char in [" ", "\t"] ->
          if current == "" do
            {tokens, "", pos + 1}
          else
            token = make_token(current, pos - String.length(current), pos - 1)
            {[token | tokens], "", pos + 1}
          end

        char in ["(", ")", "{", "}", "[", "]", ",", ";", ":", "."] ->
          tokens =
            if current == "" do
              tokens
            else
              token = make_token(current, pos - String.length(current), pos - 1)
              [token | tokens]
            end

          punct_token = make_token(char, pos, pos)
          {[punct_token | tokens], "", pos + 1}

        true ->
          {tokens, current <> char, pos + 1}
      end
    end)
    |> then(fn {tokens, current, pos} ->
      if current == "" do
        Enum.reverse(tokens)
      else
        token = make_token(current, pos - String.length(current), pos - 1)
        Enum.reverse([token | tokens])
      end
    end)
  end

  defp make_token(text, start_pos, end_pos) do
    %{
      type: classify_token(text),
      text: text,
      start_byte: start_pos,
      end_byte: end_pos,
      start_point: {0, start_pos},
      end_point: {0, end_pos},
      children: []
    }
  end

  defp classify_token(text) do
    cond do
      String.match?(text, ~r/^[0-9]+$/) -> "number"
      String.match?(text, ~r/^[A-Z]/) -> "identifier_capitalized"
      String.match?(text, ~r/^[a-z_]/) -> "identifier"
      String.match?(text, ~r/^["']/) -> "string"
      true -> "token"
    end
  end

  defp cache_parser(language, parser) do
    try do
      :ets.insert(@ets_table, {language, :loaded, parser})
    rescue
      ArgumentError -> :ok
    end
  end

  # Content-based language detection patterns
  defp detect_from_patterns(content) do
    cond do
      # Elixir patterns
      String.contains?(content, "defmodule ") or String.contains?(content, "def ") ->
        {:ok, :elixir}

      # Python patterns
      String.contains?(content, "import ") and String.contains?(content, "def ") ->
        {:ok, :python}

      # Ruby patterns
      String.contains?(content, "require ") and String.contains?(content, "end") ->
        {:ok, :ruby}

      # Go patterns
      String.contains?(content, "package ") and String.contains?(content, "func ") ->
        {:ok, :go}

      # Java patterns
      String.contains?(content, "public class ") or String.contains?(content, "public static void main") ->
        {:ok, :java}

      # C# patterns
      String.contains?(content, "namespace ") and String.contains?(content, "class ") ->
        {:ok, :csharp}

      # PHP patterns
      String.starts_with?(content, "<?php") or String.contains?(content, "<?php") ->
        {:ok, :php}

      # C/C++ patterns
      String.contains?(content, "#include ") ->
        if String.contains?(content, "iostream") or String.contains?(content, "std::") do
          {:ok, :cpp}
        else
          {:ok, :c}
        end

      true ->
        :unknown
    end
  end
end
