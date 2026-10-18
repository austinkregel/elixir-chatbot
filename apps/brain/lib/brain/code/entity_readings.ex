defmodule Brain.Code.EntityReadings do
  @moduledoc """
  The code subsystem's readings of a sentence, one of the sources
  `Brain.ML.EntityExtractor` reads every sentence with.

  A span is any run of the sentence's tokens no longer, in graphemes, than
  the longest name it could be (`Brain.Code.CodeGazetteer.longest_name/1`,
  the longest file name, the longest language name). It is read when its text,
  lowercased as `Brain.Code.CodeGazetteer` compares names, is

  - the name or qualified name of a symbol of the world's analyzed code
    (`Brain.Code.CodeGazetteer.symbols_named/2`), read as the type its kind
    maps to. The keywords, builtins and types a language's grammar file
    names (`Brain.Code.CodeGazetteer.load_language_keywords/2`, recorded with
    `metadata.source` `:language_definition`) are part of the language, not
    definitions in the world's code, so their names are read as none of them;
  - the path of one of the world's analyzed files, or a run of its trailing
    path segments (`Brain.Code.CodeGazetteer.file_names/1`);
  - a language the code parser supports (`Brain.Code.Parser.supported_languages/0`),
    by its display name (`Brain.Code.LanguageGrammar.language_name/1`) or its
    own name.

  The types come from `config.code_entity_types` in
  `priv/analysis/entity_types.json`: `kinds` maps each kind the code gazetteer
  stores a symbol as to a type, or to `null` for a kind whose mentions are no
  entity; `file` and `language` are the types of a file's and a language's
  mention. A symbol of a kind the table does not list raises.

  Each reading has the span's extent and its text, and the gazetteer records
  behind it (`:code_symbols`, `:code_files` or `:code_language`). The code
  gazetteer matches names exactly and holds no confidence of its own, so a
  reading carries none. Like every source's, these readings are decided by
  `Brain.ML.SpanReliability`.

  ## Names that are ordinary words

  A name can also be an ordinary English word: a function `render`, a module
  `Session`, the language Go. Such a name is still read, but whether a
  mention of it is the code or the word is for the context to decide, so its
  readings come from a source of their own, `:code_gazetteer_ordinary_word`,
  whose reliability for each type `Brain.ML.SpanReliability` learns apart
  from that of every other name (`:code_gazetteer`), and a span it leaves
  undecided can be decided again once the turn's intent is known. A name is
  an ordinary word as the gazetteer's loaders tell one
  (`Brain.ML.DataLoaders.ordinary_word?/2`): the owned lexicon records more
  uses of it as an ordinary word than as any entity type.
  """

  alias Brain.Analysis.TypeHierarchy
  alias Brain.Code.{CodeGazetteer, LanguageGrammar, Parser}

  @source :code_gazetteer
  @ordinary_word_source :code_gazetteer_ordinary_word

  # Where a grammar file's keywords, builtins and types are recorded as from.
  @language_definition :language_definition

  @doc """
  Every reading of `text` in `world_id`, in text order. `tokens` are
  `Brain.ML.Tokenizer.tokenize/1`'s tokens of `text`, whose `:start_pos` and
  `:end_pos` index its graphemes. `ordinary_word?` tells whether a name, as
  written, is an ordinary word (`Brain.ML.DataLoaders.ordinary_word?/2` at
  runtime); it is asked once per name read.
  """
  @spec readings(String.t(), [map()], String.t(), (String.t() -> boolean())) :: [map()]
  def readings(text, tokens, world_id, ordinary_word?)
      when is_binary(text) and is_list(tokens) and is_binary(world_id) and is_function(ordinary_word?, 1) do
    types = types!()
    files = CodeGazetteer.file_names(world_id)
    languages = language_names()

    # No span longer than every name the world's code, files and languages
    # hold can be one of them, so none is cut: a sentence of n tokens has
    # n(n+1)/2 runs, and cutting and lowercasing each made a long turn's
    # extraction cubic in its length.
    longest =
      Enum.max([CodeGazetteer.longest_name(world_id) | Enum.map(Map.keys(files) ++ Map.keys(languages), &String.length/1)])

    spans = spans(text, tokens, longest)
    symbols = CodeGazetteer.symbols_named(world_id, Enum.map(spans, &elem(&1, 2)))

    read =
      Enum.flat_map(spans, fn {start_pos, end_pos, key, surface} ->
        extent = %{start_pos: start_pos, end_pos: end_pos, value: surface, match: surface}

        symbol_readings(symbols |> Map.get(key, []) |> Enum.reject(&language_definition?/1), types.kinds, extent) ++
          file_readings(Map.get(files, key), types.file, extent) ++
          language_readings(Map.get(languages, key), types.language, extent)
      end)

    ordinary = read |> Enum.map(& &1.value) |> Enum.uniq() |> Map.new(&{&1, ordinary_word!(ordinary_word?, &1)})

    Enum.map(read, fn reading ->
      Map.put(reading, :source, if(Map.fetch!(ordinary, reading.value), do: @ordinary_word_source, else: @source))
    end)
  end

  defp ordinary_word!(ordinary_word?, name) do
    case ordinary_word?.(name) do
      answer when is_boolean(answer) ->
        answer

      other ->
        raise ArgumentError, "Brain.Code.EntityReadings: ordinary_word? answered #{inspect(other)} for #{inspect(name)}; it answers true or false"
    end
  end

  defp language_definition?(symbol), do: match?(%{metadata: %{source: @language_definition}}, symbol)

  @doc """
  The types `config.code_entity_types` declares: `%{kinds: %{kind => type |
  nil}, file: type, language: type}`. Raises when entity_types.json does not
  declare them.
  """
  @spec types!() :: %{kinds: %{String.t() => String.t() | nil}, file: String.t(), language: String.t()}
  def types! do
    case TypeHierarchy.config("code_entity_types") do
      %{"kinds" => kinds, "file" => file, "language" => language}
      when is_map(kinds) and is_binary(file) and is_binary(language) ->
        %{kinds: kinds, file: file, language: language}

      other ->
        raise "Brain.Code.EntityReadings: config.code_entity_types in entity_types.json must hold \"kinds\", " <>
                "\"file\" and \"language\"; got #{inspect(other, limit: 5)}"
    end
  end

  # One reading per type the symbols' kinds map to; kinds mapped to nil are
  # no entity.
  defp symbol_readings([], _kinds, _extent), do: []

  defp symbol_readings(symbols, kinds, extent) do
    symbols
    |> Enum.group_by(&kind_type!(kinds, &1.entity_type))
    |> Map.delete(nil)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map(fn {type, of_type} -> reading(type, extent, %{code_symbols: of_type}) end)
  end

  defp file_readings(nil, _type, _extent), do: []
  defp file_readings(paths, type, extent), do: [reading(type, extent, %{code_files: Enum.sort(paths)})]

  defp language_readings(nil, _type, _extent), do: []
  defp language_readings(language, type, extent), do: [reading(type, extent, %{code_language: language})]

  defp reading(type, extent, records) do
    extent
    |> Map.put(:entity_type, type)
    |> Map.merge(records)
  end

  defp kind_type!(kinds, kind) do
    case Map.fetch(kinds, kind) do
      {:ok, type} when is_binary(type) or is_nil(type) ->
        type

      {:ok, other} ->
        raise "Brain.Code.EntityReadings: config.code_entity_types.kinds maps #{inspect(kind)} to #{inspect(other)}; " <>
                "a kind maps to a type name or to null"

      :error ->
        raise "Brain.Code.EntityReadings: config.code_entity_types.kinds in entity_types.json does not map " <>
                "#{inspect(kind)}, a kind Brain.Code.CodeGazetteer holds a symbol as"
    end
  end

  # Every language the parser supports, by each of its names, lowercased.
  defp language_names do
    for language <- Parser.supported_languages(),
        name <- [LanguageGrammar.language_name(language), Atom.to_string(language)],
        into: %{},
        do: {String.downcase(name), language}
  end

  # Every run of tokens at most `longest` graphemes long, as `{start_pos,
  # end_pos, key, surface}`: its extent, its text lowercased, and its text. A
  # run's text is the text of the run one token shorter with what follows it
  # up to the next token's end, so each stretch of the text is cut once. The
  # positions index graphemes, so a run's length is read from its extent.
  defp spans(_text, [], _longest), do: []

  defp spans(text, tokens, longest) do
    graphemes = text |> String.graphemes() |> List.to_tuple()
    cut = fn first, last -> Enum.map_join(first..last//1, &elem(graphemes, &1)) end
    tokens = List.to_tuple(tokens)
    last = tuple_size(tokens) - 1

    Enum.flat_map(0..last, fn i ->
      first = elem(tokens, i)

      i..last
      |> Enum.reduce_while({[], nil}, fn j, {runs, previous} ->
        token = elem(tokens, j)

        if token.end_pos - first.start_pos + 1 > longest do
          {:halt, {runs, previous}}
        else
          surface =
            case previous do
              nil -> cut.(first.start_pos, token.end_pos)
              text_so_far -> text_so_far <> cut.(elem(tokens, j - 1).end_pos + 1, token.end_pos)
            end

          {:cont, {[{first.start_pos, token.end_pos, String.downcase(surface), surface} | runs], surface}}
        end
      end)
      |> elem(0)
      |> Enum.reverse()
    end)
  end
end
