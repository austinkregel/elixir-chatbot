defmodule Mix.Tasks.RebuildNerGold do
  @shortdoc "Rebuild the authored rows of the NER gold standard from the authored entity annotations"

  @moduledoc """
  Rebuilds the rows of `priv/evaluation/ner/gold_standard.json` labeled
  `"authored"` from the hand-written entity annotations in
  `priv/training/authored_entities/` (`Brain.Corpus.Provenance.authored_entities_dir/0`),
  using those files as the sole authority for those rows.

      mix rebuild_ner_gold                 # dry run, prints the summary
      mix rebuild_ner_gold --save          # write it
      mix rebuild_ner_gold --save --output /tmp/gold_standard.json
      mix rebuild_ner_gold --check         # fail unless the files match

  `--check` rebuilds in memory and compares the result byte for byte with the
  committed `gold_standard.json`, and checks that every row of the held-out
  split (`held_out.json`) labeled `"authored"` is one of the rebuilt rows. It
  writes nothing, and raises naming what differs. With `--output`, it compares
  with the file at that path instead.

  ## The other rows

  Every other row is kept exactly as the committed file holds it, byte for
  byte and in its place. Those rows were first written from the Dialogflow
  export's entity segments and have been relabeled by hand since, and the
  export is not tracked, so no source rebuilds them; this task never rewrites
  them.

  ## The annotations

  One `<intent>_usersays_en.json` file per intent, in the export's usersays
  format: a JSON array of entries, each entry's text the concatenation of its
  `data` segments. A segment that is an entity carries `meta`, `"@"` and its
  entity type, and `alias`, the slot of the intent it fills, as Dialogflow
  marks one; any other segment carries only `text`. An entry carries no id: its
  provenance is the directory it is in. An entry whose text names nothing is
  an annotation too, of no entity.

  Each entry becomes the NER row `%{"text", "expected", "labeled_by" =>
  "authored"}`, `expected` its entity segments in text order as `{type,
  value}`. The run aborts, naming the entry, when:

    * its intent (the file's name) has no schema in the intent registry, or its
      `alias` is no slot of that schema, or the slot's `entity_mappings` does
      not list its type;
    * its text is not the text of an intent gold row filed under the file's
      intent, so every annotation is of a text the intent gold holds;
    * its text is already the text of an NER gold row that is not authored, or
      of another annotation;
    * its entities have no single token-aligned placement in the text
      (`Brain.ML.NerGold.spans!/1`).

  ## Output

  The authored rows follow every other row, sorted by normalized text
  (`Brain.Corpus.Export.normalise/1`). Each row's keys are written in the order
  `expected`, `labeled_by`, `text`, each entity's as `type`, `value`; a key
  outside those raises. The file is 2-space pretty-printed JSON with no final
  newline, as it has been since it was first written.
  """

  use Mix.Task

  alias Brain.Analysis.SlotDetector
  alias Brain.Corpus.{Export, Label, Provenance}
  alias Brain.ML.{EvaluationStore, NerGold}

  @requirements ["app.config"]

  @switches [save: :boolean, output: :string, check: :boolean]

  @task "ner"

  @row_keys ~w(expected labeled_by text)
  @entity_keys ~w(type value)
  @plain_segment_keys ~w(text)
  @entity_segment_keys ~w(alias meta text)

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("rebuild_ner_gold: unknown options #{inspect(invalid)}")
    if positional != [], do: Mix.raise("rebuild_ner_gold: unexpected arguments #{inspect(positional)}")

    save? = opts[:save] || false
    check? = opts[:check] || false

    if save? and check?,
      do: Mix.raise("rebuild_ner_gold: --check compares without writing; it cannot be combined with --save")

    output = output_path!(opts[:output], save? or check?)
    committed_path = EvaluationStore.gold_standard_path(@task)

    authored = authored_rows!()
    rebuilt = rebuild!(File.read!(committed_path), committed_path, authored)

    report(authored)

    cond do
      save? ->
        File.write!(output, rebuilt)
        Mix.shell().info("\n  Wrote #{output}\n")

      check? ->
        check!(rebuilt, output, authored)

      true ->
        Mix.shell().info("\n  Dry run. Pass --save to write.\n")
    end
  end

  # -- the annotations ----------------------------------------------------------

  @doc """
  The NER rows the authored entity annotations rebuild to, sorted by normalized
  text, each `%{"expected", "labeled_by", "text"}`. Raises as the moduledoc
  says.
  """
  @spec authored_rows!(Path.t()) :: [map()]
  def authored_rows!(dir \\ Provenance.authored_entities_dir()) do
    unless File.dir?(dir), do: Mix.raise("rebuild_ner_gold: no authored entity annotations at #{dir}")

    files = Export.usersays_files(dir)
    if files == [], do: Mix.raise("rebuild_ner_gold: no *#{Export.usersays_suffix()} files under #{dir}")

    registry = SlotDetector.registry()
    intent_texts = intent_texts()

    rows =
      Enum.flat_map(files, fn path ->
        intent = path |> Export.label_of() |> Label.canonical()
        schema = Map.get(registry, intent) || Mix.raise("rebuild_ner_gold: #{path}: the intent registry has no schema for #{inspect(intent)}")
        texts = Map.get(intent_texts, intent, MapSet.new())

        path
        |> read_entries!()
        |> Enum.map(&row!(&1, path, intent, schema, texts))
      end)

    case rows |> Enum.frequencies_by(&Export.normalise(&1["text"])) |> Enum.filter(fn {_norm, n} -> n > 1 end) do
      [] -> :ok
      repeated -> Mix.raise("rebuild_ner_gold: texts annotated more than once: #{inspect(Enum.map(repeated, &elem(&1, 0)))}")
    end

    Enum.sort_by(rows, &Export.normalise(&1["text"]))
  end

  # Intent => the normalized texts of the intent gold rows filed under it, or
  # listing it among their candidates.
  defp intent_texts do
    "intent"
    |> EvaluationStore.load_gold_standard(:all)
    |> Enum.flat_map(fn row -> Enum.map([row["intent"] | Map.get(row, "candidates", [])], &{&1, Export.normalise(row["text"])}) end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Map.new(fn {intent, norms} -> {intent, MapSet.new(norms)} end)
  end

  defp read_entries!(path) do
    case path |> File.read!() |> Jason.decode() do
      {:ok, [_ | _] = entries} -> entries
      other -> Mix.raise("rebuild_ner_gold: #{path} is not a non-empty JSON array: #{inspect(other, limit: 3)}")
    end
  end

  defp row!(%{"data" => [_ | _] = segments} = entry, path, intent, schema, texts) do
    if Map.keys(entry) != ["data"] do
      Mix.raise(
        "rebuild_ner_gold: #{path}: an entry holds #{inspect(Map.keys(entry) -- ["data"])} beside its data; " <>
          "an authored annotation carries its data alone, its provenance being the directory it is in: #{inspect(entry)}"
      )
    end

    text = Enum.map_join(segments, &segment_text!(&1, path))
    where = "#{path}: #{inspect(text)}"

    unless MapSet.member?(texts, Export.normalise(text)) do
      Mix.raise("rebuild_ner_gold: #{where} is not the text of an intent gold row filed under #{intent}")
    end

    expected =
      for %{"meta" => _} = segment <- segments do
        %{"type" => entity_type!(segment, schema, intent, where), "value" => segment["text"]}
      end

    row = %{"expected" => expected, "labeled_by" => Provenance.authored(), "text" => text}

    try do
      NerGold.spans!(row)
    rescue
      e in ArgumentError -> Mix.raise("rebuild_ner_gold: #{where}: #{Exception.message(e)}")
    end

    row
  end

  defp row!(entry, path, _intent, _schema, _texts) do
    Mix.raise("rebuild_ner_gold: #{path}: an entry needs a non-empty \"data\" list of segments: #{inspect(entry)}")
  end

  defp segment_text!(%{"text" => text} = segment, path) when is_binary(text) and text != "" do
    case Enum.sort(Map.keys(segment)) do
      @plain_segment_keys -> text
      @entity_segment_keys -> text
      keys -> Mix.raise("rebuild_ner_gold: #{path}: a segment holds #{inspect(keys)}; a segment is text alone, or text with meta and alias: #{inspect(segment)}")
    end
  end

  defp segment_text!(segment, path) do
    Mix.raise("rebuild_ner_gold: #{path}: a segment needs a non-empty \"text\": #{inspect(segment)}")
  end

  # The type `meta` names, which the slot `alias` names must list.
  defp entity_type!(%{"meta" => "@" <> type, "alias" => slot}, schema, intent, where) when type != "" do
    case schema |> Map.get("entity_mappings", %{}) |> Map.fetch(slot) do
      {:ok, types} ->
        if type in types,
          do: type,
          else: Mix.raise("rebuild_ner_gold: #{where}: slot #{inspect(slot)} of #{intent} lists #{inspect(types)}, not #{inspect(type)}")

      :error ->
        Mix.raise("rebuild_ner_gold: #{where}: #{inspect(slot)} is no slot of #{intent}'s entity_mappings")
    end
  end

  defp entity_type!(segment, _schema, _intent, where) do
    Mix.raise("rebuild_ner_gold: #{where}: an entity segment's meta is \"@\" and its type: #{inspect(segment)}")
  end

  # -- the rebuild --------------------------------------------------------------

  @doc """
  The bytes of the NER gold standard `content` (the file at `path`) with its
  authored rows replaced by `authored`: every other row kept exactly as it
  stands, in its place, and the authored rows after them. Raises when an
  authored text is the text of another row.
  """
  @spec rebuild!(String.t(), Path.t(), [map()]) :: String.t()
  def rebuild!(content, path, authored) do
    unless Export.round_trips?(content, path) do
      Mix.raise("rebuild_ner_gold: #{path} is not a 2-space pretty-printed JSON array of rows; editing it would reformat it")
    end

    split = Export.elements!(content, path)
    kept = Enum.reject(split.elements, &(Jason.decode!(&1)["labeled_by"] == Provenance.authored()))
    taken = MapSet.new(kept, &(&1 |> Jason.decode!() |> Map.fetch!("text") |> Export.normalise()))

    case Enum.filter(authored, &MapSet.member?(taken, Export.normalise(&1["text"]))) do
      [] -> :ok
      clashes -> Mix.raise("rebuild_ner_gold: annotated texts already in #{path} as rows of their own: #{inspect(Enum.map(clashes, & &1["text"]))}")
    end

    authored_elements =
      case authored do
        [] -> []
        rows -> (rows |> encode!() |> Export.elements!("the authored rows")).elements
      end

    Export.rejoin(%{split | elements: kept ++ authored_elements})
  end

  @doc "The bytes of `rows` as the NER gold standard writes them, each row's and entity's keys in the documented order."
  @spec encode!([map()]) :: String.t()
  def encode!(rows) do
    rows
    |> Enum.map(fn row ->
      entities = row |> Map.fetch!("expected") |> Enum.map(&ordered_object!(&1, @entity_keys))
      row |> Map.put("expected", entities) |> ordered_object!(@row_keys)
    end)
    |> Jason.encode!(pretty: true)
  end

  # A map's key order is the runtime's, not the file's, so each is written as
  # an ordered object. A key the order does not name raises rather than being
  # dropped or appended wherever it happens to land.
  defp ordered_object!(map, keys) do
    case Map.keys(map) -- keys do
      [] ->
        keys |> Enum.filter(&Map.has_key?(map, &1)) |> Enum.map(&{&1, Map.fetch!(map, &1)}) |> Jason.OrderedObject.new()

      unknown ->
        raise ArgumentError, "rebuild_ner_gold: #{inspect(map)} has keys #{inspect(unknown)} outside the written order #{inspect(keys)}"
    end
  end

  # -- output -------------------------------------------------------------------

  defp output_path!(nil, _writes_or_checks?), do: EvaluationStore.gold_standard_path(@task)
  defp output_path!(path, true), do: Path.expand(path)

  defp output_path!(path, false) do
    Mix.raise("rebuild_ner_gold: --output #{path} says where --save writes or what --check compares with; pass one of them")
  end

  defp report(authored) do
    types = authored |> Enum.flat_map(& &1["expected"]) |> Enum.frequencies_by(& &1["type"])

    Mix.shell().info("")
    Mix.shell().info("  authored rows        #{length(authored)}")
    Mix.shell().info("  of no entity         #{Enum.count(authored, &(&1["expected"] == []))}")

    for {type, n} <- Enum.sort(types), do: Mix.shell().info("  #{String.pad_trailing(type, 20)} #{n}")
  end

  defp check!(rebuilt, output, authored) do
    held_path = EvaluationStore.held_out_path(@task)
    rows = MapSet.new(authored)

    stray =
      held_path
      |> File.read!()
      |> Jason.decode!()
      |> Enum.filter(&(&1["labeled_by"] == Provenance.authored() and not MapSet.member?(rows, &1)))

    problems =
      [
        if(File.exists?(output) and File.read!(output) == rebuilt, do: nil, else: describe_mismatch(output, rebuilt)),
        if(stray == [], do: nil, else: "#{held_path} holds authored rows no annotation rebuilds:\n" <> Enum.map_join(stray, "\n", &"  #{Jason.encode!(&1)}"))
      ]
      |> Enum.reject(&is_nil/1)

    if problems != [] do
      Mix.raise("""
      rebuild_ner_gold: the committed files are not what the annotations rebuild to.

      #{Enum.join(problems, "\n\n")}

      Either the annotations or the committed files changed without the other.
      Run `mix rebuild_ner_gold --save --output PATH`, review the diff against
      the committed file, and update whichever side is wrong.
      """)
    end

    Mix.shell().info("\n  #{output} matches the rebuild byte for byte, and #{held_path}'s authored rows are rebuilt rows.\n")
  end

  defp describe_mismatch(path, rebuilt) do
    if File.exists?(path) do
      committed = path |> File.read!() |> Jason.decode!()
      ours = Jason.decode!(rebuilt)

      rows =
        Enum.map(committed -- ours, &"  - #{Jason.encode!(&1)}") ++
          Enum.map(ours -- committed, &"  + #{Jason.encode!(&1)}")

      detail =
        if rows == [],
          do: "  the same rows, in a different order or with different key order or formatting",
          else: Enum.join(rows, "\n")

      "#{path} (- committed only, + rebuild only):\n#{detail}"
    else
      "#{path} does not exist"
    end
  end
end
