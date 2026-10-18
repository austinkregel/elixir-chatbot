defmodule Mix.Tasks.RebuildNerGold do
  @shortdoc "Rebuild the authored and relabeled rows of the NER gold standard from the tracked entity annotations"

  @moduledoc """
  Rebuilds the rows of `priv/evaluation/ner/gold_standard.json` labeled
  `"authored"` from the hand-written entity annotations in
  `priv/training/authored_entities/` (`Brain.Corpus.Provenance.authored_entities_dir/0`),
  using those files as the sole authority for those rows, and relabels in
  place the existing rows the annotations in
  `priv/training/relabeled_entities/` (`Brain.Corpus.Provenance.relabeled_entities_dir/0`)
  add entities to, in the gold standard and in its held-out split
  (`held_out.json`) alike.

      mix rebuild_ner_gold                 # dry run, prints the summary
      mix rebuild_ner_gold --save          # write it
      mix rebuild_ner_gold --save --output /tmp/gold_standard.json --held-out-output /tmp/held_out.json
      mix rebuild_ner_gold --check         # fail unless the files match

  `--check` rebuilds in memory and compares the results byte for byte with the
  committed `gold_standard.json` and `held_out.json`, and checks that every row
  of the held-out split labeled `"authored"` is one of the rebuilt rows. It
  writes nothing, and raises naming what differs. With `--output` or
  `--held-out-output`, it compares with the file at that path instead.

  ## The other rows

  Every other row is kept exactly as the committed file holds it, byte for
  byte and in its place, unless a relabeled entity annotation names it. Those
  rows were first written from the Dialogflow export's entity segments and
  have been relabeled by hand since, and the export is not tracked, so no
  source rebuilds them; this task rewrites none of them but the relabeled
  ones.

  ## The annotations

  One `<intent>_usersays_en.json` file per intent, in the export's usersays
  format: a JSON array of entries, each entry's text the concatenation of its
  `data` segments. A segment that is an entity carries `meta`, `"@"` and its
  entity type, and `alias`, the slot of the intent it fills, as Dialogflow
  marks one; any other segment carries only `text`. A segment whose `meta` is
  `"@sys.any"` is a free-text span, the value of a free-text slot
  (`Brain.Analysis.FreeTextCapture` learns from it), and names no entity. An
  entry carries no id: its provenance is the directory it is in. An entry
  whose text names nothing is an annotation too, of no entity.

  Each entry becomes the NER row `%{"text", "expected", "labeled_by" =>
  "authored"}`, `expected` its entity segments in text order as `{type,
  value}`. The run aborts, naming the entry, when:

    * its intent (the file's name) has no schema in the intent registry, or an
      entity's `alias` is no slot of that schema, or its type is not one that
      fills the slot (`Brain.Analysis.SlotDetector.slot_entity_types!/2`), or a
      free-text span's `alias` is no slot the schema declares;
    * its text is not the text of an intent gold row filed under the file's
      intent, so every annotation is of a text the intent gold holds;
    * its text is already the text of an NER gold row that is not authored, or
      of another annotation;
    * its entities have no single token-aligned placement in the text
      (`Brain.ML.NerGold.spans!/1`).

  ## The relabeled rows

  A relabeled entity annotation is an entry of the same format whose text is
  exactly the text of an existing NER gold row that is not authored, filed
  under an intent the intent gold files that text under. Each of its entity
  segments is an entity added to the row, at the segment's place: its `alias`
  must be a slot of the intent that its `meta` type fills, as for an authored
  annotation, and its type in the row is that type's gold label through the
  NER type mappings (`Brain.ML.EntityExtractor.load_type_mappings/0`; `lights`
  is written `device`). The row keeps every entity it had, the added ones
  join them in text order, and it is labeled `"relabeled"`. The row stays in
  its place, and a held-out copy of it is relabeled the same way, so the
  held-out split keeps the same rows.

  The committed file holds a relabeled row with its added entities, so a
  rebuild takes them off again -- each at the segment's place, of its gold
  label -- before adding them back, and the result is the same bytes. The run
  aborts when an annotation adds no entity or a free-text span, when its text
  is no row's or is an authored row's, when one text is annotated twice, when
  an added entity overlaps one the row has, when the row and its added
  entities have no single token-aligned placement, when a committed relabeled
  row lacks an entity its annotation adds or has no annotation at all (its
  labels from before would be lost), and when the held-out split holds a row
  of a relabeled text that is not the gold standard's row.

  ## Output

  The authored rows follow every other row, sorted by normalized text
  (`Brain.Corpus.Export.normalise/1`). Each row's keys are written in the order
  `expected`, `labeled_by`, `text`, each entity's as `type`, `value`; a key
  outside those raises. The files are 2-space pretty-printed JSON with no
  final newline, as they have been since they were first written.
  """

  use Mix.Task

  alias Brain.Analysis.{FreeTextCapture, SlotDetector}
  alias Brain.Corpus.{Export, Label, Provenance}
  alias Brain.ML.{EntityExtractor, EvaluationStore, NerGold}

  @requirements ["app.config"]

  @switches [save: :boolean, output: :string, held_out_output: :string, check: :boolean]

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

    committed_path = EvaluationStore.gold_standard_path(@task)
    held_path = EvaluationStore.held_out_path(@task)
    output = output_path!(opts[:output], committed_path, "--output", save? or check?)
    held_output = output_path!(opts[:held_out_output], held_path, "--held-out-output", save? or check?)

    committed = File.read!(committed_path)
    authored = authored_rows!()
    relabels = relabels!()
    rebuilt = rebuild!(committed, committed_path, authored, relabels)
    held_rebuilt = rebuild_held_out!(File.read!(held_path), held_path, committed, committed_path, relabels)

    report(authored, relabels, held_rebuilt)

    cond do
      save? ->
        File.write!(output, rebuilt)
        File.write!(held_output, held_rebuilt)
        Mix.shell().info("\n  Wrote #{output} and #{held_output}\n")

      check? ->
        check!(rebuilt, output, held_rebuilt, held_output, authored)

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
    rows =
      dir
      |> annotation_entries!("authored entity annotations")
      |> Enum.map(fn {entry, path, intent, schema, texts} -> row!(entry, path, intent, schema, texts) end)

    case rows |> Enum.frequencies_by(&Export.normalise(&1["text"])) |> Enum.filter(fn {_norm, n} -> n > 1 end) do
      [] -> :ok
      repeated -> Mix.raise("rebuild_ner_gold: texts annotated more than once: #{inspect(Enum.map(repeated, &elem(&1, 0)))}")
    end

    Enum.sort_by(rows, &Export.normalise(&1["text"]))
  end

  @doc """
  The relabeled entity annotations in `dir` (see the moduledoc), as a map from
  each annotated text to `%{text, intent, path, added}`: `added` the entities
  the annotation adds, in text order, each `%{"type", "value", start_pos,
  end_pos}` with its gold label through `canonical` (runtime type => gold
  label, `Brain.ML.EntityExtractor.load_type_mappings/0`) and its grapheme
  positions (`end_pos` inclusive). Raises as the moduledoc says.
  """
  @spec relabels!(Path.t(), %{String.t() => String.t()}) :: %{String.t() => map()}
  def relabels!(dir \\ Provenance.relabeled_entities_dir(), canonical \\ EntityExtractor.load_type_mappings()) do
    relabels =
      dir
      |> annotation_entries!("relabeled entity annotations")
      |> Enum.map(fn {entry, path, intent, schema, texts} -> relabel!(entry, path, intent, schema, texts, canonical) end)

    case relabels |> Enum.frequencies_by(& &1.text) |> Enum.filter(fn {_text, n} -> n > 1 end) do
      [] -> :ok
      repeated -> Mix.raise("rebuild_ner_gold: texts relabeled more than once: #{inspect(Enum.map(repeated, &elem(&1, 0)))}")
    end

    Map.new(relabels, &{&1.text, &1})
  end

  # Every entry of every usersays file in `dir`, with its file, its intent, the
  # intent's schema and the normalized texts the intent gold files under it.
  defp annotation_entries!(dir, what) do
    unless File.dir?(dir), do: Mix.raise("rebuild_ner_gold: no #{what} at #{dir}")

    files = Export.usersays_files(dir)
    if files == [], do: Mix.raise("rebuild_ner_gold: no *#{Export.usersays_suffix()} files under #{dir}")

    registry = SlotDetector.registry()
    intent_texts = intent_texts()

    Enum.flat_map(files, fn path ->
      intent = path |> Export.label_of() |> Label.canonical()
      schema = Map.get(registry, intent) || Mix.raise("rebuild_ner_gold: #{path}: the intent registry has no schema for #{inspect(intent)}")
      texts = Map.get(intent_texts, intent, MapSet.new())

      path
      |> read_entries!()
      |> Enum.map(&{&1, path, intent, schema, texts})
    end)
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
    {text, where} = entry_text!(entry, segments, path, intent, texts)

    {free_text, entities} = segments |> Enum.filter(&Map.has_key?(&1, "meta")) |> Enum.split_with(&(&1["meta"] == FreeTextCapture.any_text()))

    for segment <- free_text, do: free_text_slot!(segment, schema, intent, where)

    expected =
      for segment <- entities do
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

  defp relabel!(%{"data" => [_ | _] = segments} = entry, path, intent, schema, texts, canonical) do
    {text, where} = entry_text!(entry, segments, path, intent, texts)
    any_text = FreeTextCapture.any_text()

    {added, _end} =
      Enum.reduce(segments, {[], 0}, fn segment, {added, pos} ->
        length = String.length(segment["text"])

        added =
          case segment do
            %{"meta" => ^any_text} ->
              Mix.raise("rebuild_ner_gold: #{where}: a relabeled annotation adds entities; a free-text span names none: #{inspect(segment)}")

            %{"meta" => _} ->
              type = entity_type!(segment, schema, intent, where)
              [%{"type" => Map.get(canonical, type, type), "value" => segment["text"], start_pos: pos, end_pos: pos + length - 1} | added]

            _ ->
              added
          end

        {added, pos + length}
      end)

    if added == [], do: Mix.raise("rebuild_ner_gold: #{where}: a relabeled annotation must add an entity, and this one marks none")

    %{text: text, intent: intent, path: path, added: Enum.reverse(added)}
  end

  defp relabel!(entry, path, _intent, _schema, _texts, _canonical) do
    Mix.raise("rebuild_ner_gold: #{path}: an entry needs a non-empty \"data\" list of segments: #{inspect(entry)}")
  end

  # The entry's text and where it is, after checking it carries its data alone
  # and is the text of an intent gold row filed under its file's intent.
  defp entry_text!(entry, segments, path, intent, texts) do
    if Map.keys(entry) != ["data"] do
      Mix.raise(
        "rebuild_ner_gold: #{path}: an entry holds #{inspect(Map.keys(entry) -- ["data"])} beside its data; " <>
          "an annotation carries its data alone, its provenance being the directory it is in: #{inspect(entry)}"
      )
    end

    text = Enum.map_join(segments, &segment_text!(&1, path))
    where = "#{path}: #{inspect(text)}"

    unless MapSet.member?(texts, Export.normalise(text)) do
      Mix.raise("rebuild_ner_gold: #{where} is not the text of an intent gold row filed under #{intent}")
    end

    {text, where}
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
    types =
      try do
        SlotDetector.slot_entity_types!(schema, slot)
      rescue
        ArgumentError -> Mix.raise("rebuild_ner_gold: #{where}: #{inspect(slot)} is no slot of #{intent}")
      end

    if type in types,
      do: type,
      else: Mix.raise("rebuild_ner_gold: #{where}: slot #{inspect(slot)} of #{intent} lists #{inspect(types)}, not #{inspect(type)}")
  end

  defp entity_type!(segment, _schema, _intent, where) do
    Mix.raise("rebuild_ner_gold: #{where}: an entity segment's meta is \"@\" and its type: #{inspect(segment)}")
  end

  # A free-text span names no entity; its slot must be one the intent declares.
  defp free_text_slot!(%{"alias" => slot}, schema, intent, where) do
    unless slot in Map.get(schema, "required", []) ++ Map.get(schema, "optional", []) do
      Mix.raise("rebuild_ner_gold: #{where}: the free-text span's slot #{inspect(slot)} is no slot #{intent} declares")
    end
  end

  # -- the rebuild --------------------------------------------------------------

  @doc """
  The bytes of the NER gold standard `content` (the file at `path`) with its
  authored rows replaced by `authored` and the rows `relabels` (`relabels!/2`)
  names relabeled in place: every other row kept exactly as it stands, in its
  place, and the authored rows after them. Raises when an authored text is the
  text of another row, and as the moduledoc says of the relabeled rows.
  """
  @spec rebuild!(String.t(), Path.t(), [map()], %{String.t() => map()}) :: String.t()
  def rebuild!(content, path, authored, relabels) do
    split = split!(content, path)
    kept = Enum.reject(split.elements, &(Jason.decode!(&1)["labeled_by"] == Provenance.authored()))
    taken = MapSet.new(kept, &(&1 |> Jason.decode!() |> Map.fetch!("text") |> Export.normalise()))

    case Enum.filter(authored, &MapSet.member?(taken, Export.normalise(&1["text"]))) do
      [] -> :ok
      clashes -> Mix.raise("rebuild_ner_gold: annotated texts already in #{path} as rows of their own: #{inspect(Enum.map(clashes, & &1["text"]))}")
    end

    relabeled = relabeled_rows!(kept, path, relabels)
    kept = Enum.map(kept, &relabeled_element(&1, relabeled))

    authored_elements =
      case authored do
        [] -> []
        rows -> (rows |> encode!() |> Export.elements!("the authored rows")).elements
      end

    Export.rejoin(%{split | elements: kept ++ authored_elements})
  end

  @doc """
  The bytes of the held-out split `held_content` (the file at `held_path`)
  with every row of the gold standard `gold_content` (the file at
  `gold_path`) that `relabels` names replaced, in its place, by the row
  `rebuild!/4` relabels it to; every other row kept exactly as it stands.
  Raises when a held-out row of a relabeled text is not the gold standard's
  row.
  """
  @spec rebuild_held_out!(String.t(), Path.t(), String.t(), Path.t(), %{String.t() => map()}) :: String.t()
  def rebuild_held_out!(held_content, held_path, gold_content, gold_path, relabels) do
    kept = gold_content |> split!(gold_path) |> Map.fetch!(:elements) |> Enum.reject(&(Jason.decode!(&1)["labeled_by"] == Provenance.authored()))
    relabeled = relabeled_rows!(kept, gold_path, relabels)
    split = split!(held_content, held_path)

    elements =
      Enum.map(split.elements, fn element ->
        row = Jason.decode!(element)

        cond do
          Map.has_key?(relabeled, row) ->
            relabeled_element(element, relabeled)

          Map.has_key?(relabels, row["text"]) ->
            Mix.raise(
              "rebuild_ner_gold: #{held_path} holds #{inspect(row)}, of a relabeled text, which is not the row #{gold_path} holds; " <>
                "the held-out copy and the gold standard's row must be the same row"
            )

          true ->
            element
        end
      end)

    Export.rejoin(%{split | elements: elements})
  end

  defp split!(content, path) do
    unless Export.round_trips?(content, path) do
      Mix.raise("rebuild_ner_gold: #{path} is not a 2-space pretty-printed JSON array of rows; editing it would reformat it")
    end

    Export.elements!(content, path)
  end

  # Each committed row (decoded) a relabel names => the row relabeled. Every
  # relabel must name exactly one of the `kept` elements, the rows that are
  # not authored, and every row already labeled relabeled must have a relabel.
  defp relabeled_rows!(kept, path, relabels) do
    rows = Enum.map(kept, &Jason.decode!/1)
    by_text = Enum.group_by(rows, & &1["text"])

    for {text, _relabel} <- relabels do
      case Map.get(by_text, text, []) do
        [_row] -> :ok
        [] -> Mix.raise("rebuild_ner_gold: #{inspect(text)} is relabeled, but no row of #{path} outside the authored ones has that text")
        many -> Mix.raise("rebuild_ner_gold: #{inspect(text)} is relabeled, and #{length(many)} rows of #{path} have that text")
      end
    end

    for row <- rows, row["labeled_by"] == Provenance.relabeled(), not Map.has_key?(relabels, row["text"]) do
      Mix.raise(
        "rebuild_ner_gold: #{path} holds #{inspect(row)}, labeled #{Provenance.relabeled()}, which no relabeled entity annotation names; " <>
          "without it the row's labels from before the relabel cannot be told apart"
      )
    end

    for row <- rows, relabel <- [Map.get(relabels, row["text"])], relabel != nil, into: %{}, do: {row, relabeled_row!(row, relabel)}
  end

  defp relabeled_element(element, relabeled) do
    case Map.fetch(relabeled, Jason.decode!(element)) do
      {:ok, row} -> [row] |> encode!() |> Export.elements!("the relabeled row") |> Map.fetch!(:elements) |> hd()
      :error -> element
    end
  end

  # The row with the relabel's entities added: those of a row already
  # relabeled taken off first, so the result does not depend on whether it was.
  defp relabeled_row!(row, relabel) do
    where = "#{relabel.path}: #{inspect(relabel.text)}"
    relabeled_by = Provenance.relabeled()

    original =
      case Map.get(row, "labeled_by") do
        nil -> row
        ^relabeled_by -> without_added!(row, relabel, where)
        other -> Mix.raise("rebuild_ner_gold: #{where}: the row is labeled #{inspect(other)}; only a row with no tracked source is relabeled")
      end

    spans = placed!(original, where)

    for added <- relabel.added, span <- spans, added.start_pos <= span.end_pos and span.start_pos <= added.end_pos do
      Mix.raise("rebuild_ner_gold: #{where}: the added #{added["type"]} #{inspect(added["value"])} overlaps the row's #{span.type} #{inspect(span.value)}")
    end

    merged =
      (Enum.map(spans, &%{"type" => &1.type, "value" => &1.value, start_pos: &1.start_pos, end_pos: &1.end_pos}) ++ relabel.added)
      |> Enum.sort_by(& &1.start_pos)

    relabeled = %{"expected" => Enum.map(merged, &Map.take(&1, ["type", "value"])), "labeled_by" => Provenance.relabeled(), "text" => row["text"]}

    if Enum.map(placed!(relabeled, where), &{&1.start_pos, &1.end_pos}) != Enum.map(merged, &{&1.start_pos, &1.end_pos}) do
      Mix.raise("rebuild_ner_gold: #{where}: the relabeled row's entities are placed elsewhere than the annotation marks them")
    end

    relabeled
  end

  # A relabeled row without the entities its relabel adds, each found at the
  # annotation's place with its gold label.
  defp without_added!(row, relabel, where) do
    spans = placed!(row, where)

    for added <- relabel.added, not Enum.any?(spans, &({&1.type, &1.start_pos, &1.end_pos} == {added["type"], added.start_pos, added.end_pos})) do
      Mix.raise("rebuild_ner_gold: #{where}: the committed row lacks the #{added["type"]} #{inspect(added["value"])} its annotation adds: #{inspect(row)}")
    end

    places = MapSet.new(relabel.added, &{&1["type"], &1.start_pos, &1.end_pos})
    kept = Enum.reject(spans, &MapSet.member?(places, {&1.type, &1.start_pos, &1.end_pos}))

    %{"expected" => Enum.map(kept, &%{"type" => &1.type, "value" => &1.value}), "text" => row["text"]}
  end

  defp placed!(row, where) do
    NerGold.spans!(row)
  rescue
    e in ArgumentError -> Mix.raise("rebuild_ner_gold: #{where}: #{Exception.message(e)}")
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

  defp output_path!(nil, committed, _flag, _writes_or_checks?), do: committed
  defp output_path!(path, _committed, _flag, true), do: Path.expand(path)

  defp output_path!(path, _committed, flag, false) do
    Mix.raise("rebuild_ner_gold: #{flag} #{path} says where --save writes or what --check compares with; pass one of them")
  end

  defp report(authored, relabels, held_rebuilt) do
    types = authored |> Enum.flat_map(& &1["expected"]) |> Enum.frequencies_by(& &1["type"])
    held_texts = held_rebuilt |> Jason.decode!() |> MapSet.new(& &1["text"])
    {held, train} = relabels |> Map.values() |> Enum.split_with(&MapSet.member?(held_texts, &1.text))

    Mix.shell().info("")
    Mix.shell().info("  authored rows        #{length(authored)}")
    Mix.shell().info("  of no entity         #{Enum.count(authored, &(&1["expected"] == []))}")

    for {type, n} <- Enum.sort(types), do: Mix.shell().info("  #{String.pad_trailing(type, 20)} #{n}")

    Mix.shell().info("  relabeled rows       #{map_size(relabels)} (#{length(train)} training, #{length(held)} held out)")

    for {type, n} <- relabels |> Map.values() |> Enum.flat_map(& &1.added) |> Enum.frequencies_by(& &1["type"]) |> Enum.sort() do
      Mix.shell().info("  #{String.pad_trailing(type <> " added", 20)} #{n}")
    end
  end

  defp check!(rebuilt, output, held_rebuilt, held_output, authored) do
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
        if(File.exists?(held_output) and File.read!(held_output) == held_rebuilt, do: nil, else: describe_mismatch(held_output, held_rebuilt)),
        if(stray == [], do: nil, else: "#{held_path} holds authored rows no annotation rebuilds:\n" <> Enum.map_join(stray, "\n", &"  #{Jason.encode!(&1)}"))
      ]
      |> Enum.reject(&is_nil/1)

    if problems != [] do
      Mix.raise("""
      rebuild_ner_gold: the committed files are not what the annotations rebuild to.

      #{Enum.join(problems, "\n\n")}

      Either the annotations or the committed files changed without the other.
      Run `mix rebuild_ner_gold --save --output PATH --held-out-output PATH`, review
      the diffs against the committed files, and update whichever side is wrong.
      """)
    end

    Mix.shell().info("\n  #{output} and #{held_output} match the rebuild byte for byte, and #{held_path}'s authored rows are rebuilt rows.\n")
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
