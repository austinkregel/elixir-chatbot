defmodule Brain.Corpus.Export do
  @moduledoc """
  The shape of the Dialogflow intent export in `data/intents/`.

  One `<label>_usersays_en.json` file per intent, holding a JSON array of
  utterance entries. An entry's text is split across `data` segments — a plain
  run, then an annotated entity, then another run — so the utterance is the
  concatenation of those segments rather than a single field.

  ## Editing the files in place

  The export cannot be rewritten by re-encoding it. Two escaping conventions coexist
  across these files — Dialogflow escapes an apostrophe as `\\u0027` and an ampersand
  as `\\u0026`, while the rows `scripts/materialize_orphan_intents.exs` writes back
  carry both raw — and key order varies between Dialogflow's and alphabetical. So
  `Jason.encode!/2` reproduces fewer than half of them byte-for-byte, and any edit
  made by decode-modify-encode shows up as a whole-file diff.

  `elements!/2` instead splits a file into the *text* of its top-level entries,
  which `rejoin/1` puts back together. Removing an entry then leaves every other
  byte of the file untouched, so a diff shows only what was removed.
  `round_trips?/2` checks that guarantee per file, and callers are expected to
  refuse to write when it does not hold rather than emit a reformatted file.

  Those three functions only assume a 2-space pretty-printed JSON array, so they
  also serve the derived corpus files — `gold_standard.json` and `held_out.json`
  — whose rows likewise should not be reshuffled to remove one of them.
  """

  @usersays_suffix "_usersays_en.json"
  @dir "data/intents"

  @doc "The export directory, relative to the umbrella root."
  @spec dir() :: Path.t()
  def dir, do: @dir

  @doc "The filename suffix that marks a usersays file, as opposed to its metadata pair."
  @spec usersays_suffix() :: String.t()
  def usersays_suffix, do: @usersays_suffix

  @doc "Every usersays file in the export, sorted."
  @spec usersays_files(Path.t()) :: [Path.t()]
  def usersays_files(dir \\ @dir) do
    dir |> Path.join("*#{@usersays_suffix}") |> Path.wildcard() |> Enum.sort()
  end

  @doc "The intent label a usersays file is filed under, taken from its name."
  @spec label_of(Path.t()) :: String.t()
  def label_of(path), do: path |> Path.basename(@usersays_suffix)

  @doc """
  An entry's utterance, concatenated from its `data` segments.
  """
  @spec phrase_text(map()) :: String.t()
  def phrase_text(entry) when is_map(entry) do
    entry
    |> Map.get("data", [])
    |> Enum.map_join(fn segment -> Map.get(segment, "text", "") end)
  end

  @doc """
  The identity of an utterance within the corpus.

  Two entries are the same utterance when this agrees, which is how the export's
  rows dedupe into the corpus.
  """
  @spec normalise(String.t()) :: String.t()
  def normalise(text) do
    text |> String.trim() |> String.downcase() |> String.replace(~r/\s+/, " ")
  end

  @doc """
  Splits a file's contents into the text of its top-level entries.

  Returns `%{prologue:, elements:, epilogue:}`. Each element is the entry's exact
  source text with no trailing comma; `rejoin/1` re-inserts the separators.

  A top-level entry begins on a line that is exactly `"  {"` and ends on `"  }"`
  or `"  },"`. Nested objects sit at indent 4 or deeper and a JSON string cannot
  contain a literal newline, so those two lines are unambiguous boundaries for
  this format. Anything else raises.
  """
  @spec elements!(String.t(), Path.t()) :: %{
          prologue: String.t(),
          elements: [String.t()],
          epilogue: String.t()
        }
  def elements!(content, path) do
    case String.split(content, "\n") do
      ["[" | rest] ->
        {elements, epilogue} = collect(rest, nil, [], path)
        %{prologue: "[", elements: elements, epilogue: epilogue}

      [first | _] ->
        raise """
        #{path} does not start with a bare `[`

          first line: #{inspect(first)}

        The export is 2-space pretty-printed JSON arrays. A file in another shape
        cannot be edited without reformatting it.
        """
    end
  end

  defp collect(["  {" | rest], nil, done, path), do: collect(rest, ["  {"], done, path)

  defp collect(lines, nil, done, _path), do: {Enum.reverse(done), Enum.join(lines, "\n")}

  defp collect([], _current, _done, path) do
    raise "#{path} ends inside an entry: no closing `  }` for the last `  {`"
  end

  defp collect([line | rest], current, done, path) when line in ["  }", "  },"] do
    element = ["  }" | current] |> Enum.reverse() |> Enum.join("\n")
    collect(rest, nil, [element | done], path)
  end

  defp collect([line | rest], current, done, path), do: collect(rest, [line | current], done, path)

  @doc "Reassembles what `elements!/2` split, or a subset of its elements."
  @spec rejoin(%{prologue: String.t(), elements: [String.t()], epilogue: String.t()}) ::
          String.t()
  def rejoin(%{prologue: prologue, elements: elements, epilogue: epilogue}) do
    prologue <> "\n" <> Enum.join(elements, ",\n") <> "\n" <> epilogue
  end

  @doc """
  True when splitting and reassembling `content` returns it unchanged, and every
  element parses as JSON with the count the whole file decodes to.

  Checked before any write: if it is false, this module's assumptions about the
  file do not hold and editing it would silently reformat or corrupt it.
  """
  @spec round_trips?(String.t(), Path.t()) :: boolean()
  def round_trips?(content, path) do
    split = elements!(content, path)

    rejoin(split) == content and
      length(split.elements) == length(Jason.decode!(content)) and
      Enum.all?(split.elements, &match?({:ok, _}, Jason.decode(&1)))
  end
end
