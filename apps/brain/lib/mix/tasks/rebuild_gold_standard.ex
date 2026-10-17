defmodule Mix.Tasks.RebuildGoldStandard do
  @shortdoc "Rebuild the intent gold standard from the export and the authored examples"

  @moduledoc """
  Rebuilds `priv/evaluation/intent/gold_standard.json` from `data/intents/`
  and the hand-written examples in `priv/training/authored_intents/`, using
  those files as the sole authority.

      mix rebuild_gold_standard                 # dry run, prints the summary
      mix rebuild_gold_standard --save          # write it
      mix rebuild_gold_standard --save --verbose
      mix rebuild_gold_standard --save --output /tmp/gold_standard.json
      mix rebuild_gold_standard --check         # fail unless the files match

  `--output` writes the gold standard to another path, with its
  `ambiguous_texts.json` beside it, so a rebuild can be compared with the
  committed file without replacing it.

  `--check` rebuilds in memory and compares the result byte for byte with the
  committed `gold_standard.json` and its `ambiguous_texts.json`. It writes
  nothing, and raises naming the rows that differ when either file is not
  exactly what the sources rebuild to. With `--output`, it compares with the
  files at that path instead.

  No model is consulted. Every label comes from the files, so the corpus can be
  regenerated from them alone. Not every row in the export came from
  Dialogflow — see `labeled_by` below.

  ## Where labels come from

  The sources are `data/intents/*_usersays_en.json` and
  `priv/training/authored_intents/*_usersays_en.json`
  (`Brain.Corpus.Provenance.authored_dir/0`), and the label is the filename
  stem — the same key `.claude/corpus/build_corpus.py` uses for the export. The
  export's paired metadata file's `"name"` field is not read, because a usersays
  file can exist without one. `data/intents/` is not tracked; the authored
  examples are.

  One transformation is applied to the stem: `smarthome.lights.X` becomes
  `smarthome.device.X`, lights being a kind of device. A stem containing `" - "`
  is a contextual variant that was not collapsed into its intent, and raises
  (`Brain.Corpus.Label.canonical/1`).

  ## Follow-ups

  A usersays row that came from a contextual variant carries `followupContext`
  (the context it continues from) and `followupVariant` (the variant it came
  from). They are written to the entry as `followup_context` and
  `followup_variant`, so an utterance that needs a preceding turn is never mixed
  with one that stands alone. When the same text occurs both ways under the
  row's intent, the follow-up occurrence is kept; ties go to the first variant
  name, then the first context name.

  Dialogflow intents marked `fallbackIntent: true` are excluded: that flag means
  "nothing matched", so the phrases are not examples of an intent. The `events`
  field is not used for this — `smalltalk.greetings.hello` carries a welcome
  event and is a real intent.

  ## Output

  Each entry carries `labeled_by` and `source_file`. A text that more than one
  intent claims, after the transformations above, is written with
  `status: "needs_review"` and a `candidates` list; it is never resolved by
  guessing. Its `intent` is the first candidate, and its `source_file` and
  follow-up fields describe that candidate's own occurrences, never another
  candidate's: a row filed under one intent that carried another intent's
  follow-up context would tell `Brain.Analysis.ActiveContexts` that the intent
  follows a turn it does not. Ambiguous texts are also written to
  `ambiguous_texts.json`.

  Source files are ordered by the intent name they are filed under, so an
  intent's file comes before its sub-intents' (`smarthome.device.volume`
  before `smarthome.device.volume.check`). Sorting the whole filename would let
  the `_usersays_en.json` suffix decide instead, since `.` sorts before `_`.
  `source_file` is the first of these.

  Rows are sorted by normalized text. The output is the same bytes on every
  run: each gold row's keys are written in the order `candidates`, `intent`,
  `labeled_by`, `followup_context`, `followup_variant`, `source_file`,
  `status`, `text` (absent keys omitted), and each ambiguous text's as `text`,
  `sources`, `candidates`. A key outside those lists raises.

  ## `labeled_by`

  The export is not purely Dialogflow. `scripts/materialize_orphan_intents.exs`
  writes gold-standard texts that have no source backing back into
  `data/intents/*_usersays_en.json`, tagging them `orphan-<hash>` or
  `augmented-<hash>`; augmented text is a mutation of a real utterance and can be
  ungrammatical.

  `labeled_by` is read per row from that id by `Brain.Corpus.Provenance`, which
  owns the rule:

    * `orphan-` -> `"materialized"`
    * `augmented-` -> `"augmented"`
    * a UUID, or no id at all -> `"dialogflow"`

  Every row of the authored examples is `"authored"`, read from the directory
  rather than an id; an authored row that carries an id aborts the run.

  Where the same normalised text occurs more than once the most genuine origin
  wins, so a text counts as synthetic only when it exists nowhere outside a
  synthetic writer. An id shaped like a synthetic marker with an unknown prefix
  aborts the run. The breakdown is printed on every run.

  A label that is not a dotted lowercase identifier of at least two parts aborts
  the run, naming the offenders and their files. Mapping such a name onto a real
  intent is a decision about the taxonomy and belongs in the export, not here.
  """

  use Mix.Task

  alias Brain.Corpus.Export
  alias Brain.Corpus.Label
  alias Brain.Corpus.Provenance
  alias Brain.ML.EvaluationStore

  @requirements ["app.config"]

  @switches [save: :boolean, verbose: :boolean, output: :string, check: :boolean]

  @usersays_suffix "_usersays_en.json"

  # The order each gold row's keys are written in, and each ambiguous text's.
  @entry_keys ~w(candidates intent labeled_by followup_context followup_variant source_file status text)
  @conflict_keys ~w(text sources candidates)

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("rebuild_gold_standard: unknown options #{inspect(invalid)}")

    if positional != [],
      do: Mix.raise("rebuild_gold_standard: unexpected arguments #{inspect(positional)}")

    save? = opts[:save] || false
    check? = opts[:check] || false
    verbose? = opts[:verbose] || false

    if save? and check?,
      do: Mix.raise("rebuild_gold_standard: --check compares without writing; it cannot be combined with --save")

    output = output_path!(opts[:output], save? or check?)

    banner("REBUILD INTENT GOLD STANDARD (from the export and the authored examples)")

    export = load_dialogflow!()
    Mix.shell().info("  #{length(export)} usersays phrases across #{count_files(export)} intent files")

    authored = load_authored!()
    Mix.shell().info("  #{length(authored)} authored phrases across #{count_files(authored)} intent files")

    phrases = export ++ authored
    validate_labels!(phrases)

    {entries, conflicts} = build_entries(phrases)

    report(entries, conflicts, verbose?)

    cond do
      save? ->
        write_output!(entries, output)
        write_conflict_report!(conflicts, output)

      check? ->
        check!(entries, conflicts, output)

      true ->
        Mix.shell().info("\n  Dry run. Pass --save to write.\n")
    end
  end

  # -- loading ----------------------------------------------------------------

  defp load_dialogflow! do
    dir = Brain.data_path("intents")

    unless File.dir?(dir) do
      Mix.raise("rebuild_gold_standard: no Dialogflow export at #{dir}")
    end

    files = Export.usersays_files(dir)

    if files == [] do
      Mix.raise("rebuild_gold_standard: no *#{@usersays_suffix} files under #{dir}")
    end

    fallbacks = fallback_intents(dir)

    files
    |> Enum.reject(fn path -> Path.basename(path, @usersays_suffix) in fallbacks end)
    |> Enum.flat_map(fn path -> phrases_of!(path, &Provenance.of_entry!(&1, path)) end)
  end

  # The hand-written intent examples (`Brain.Corpus.Provenance.authored_dir/0`),
  # in the export's usersays format, every one labeled "authored". An authored
  # row carries no id: its provenance is the directory it is in, and an id
  # could claim another.
  defp load_authored! do
    dir = Provenance.authored_dir()

    unless File.dir?(dir) do
      Mix.raise("rebuild_gold_standard: no authored intent examples at #{dir}")
    end

    files = Export.usersays_files(dir)

    if files == [] do
      Mix.raise("rebuild_gold_standard: no *#{@usersays_suffix} files under #{dir}")
    end

    Enum.flat_map(files, fn path -> phrases_of!(path, &authored_provenance!(&1, path)) end)
  end

  defp authored_provenance!(entry, path) do
    case Map.fetch(entry, "id") do
      :error ->
        Provenance.authored()

      {:ok, id} ->
        Mix.raise(
          "rebuild_gold_standard: #{path} has a row with id #{inspect(id)}; an authored example " <>
            "carries no id, since its provenance is the directory it is in"
        )
    end
  end

  # Every phrase of one usersays file, filed under the file's label, with the
  # provenance `provenance_of` reads from each entry.
  defp phrases_of!(path, provenance_of) do
    label = path |> Path.basename(@usersays_suffix) |> Label.canonical()
    source = Path.basename(path)

    path
    |> read_phrases!(provenance_of)
    |> Enum.map(fn phrase ->
      %{
        text: phrase.text,
        norm: Export.normalise(phrase.text),
        label: label,
        source_file: source,
        provenance: phrase.provenance,
        followup: phrase.followup
      }
    end)
  end

  # Dialogflow's own `fallbackIntent: true` marks the no-match case. It is the
  # absence of an intent by definition, so its phrases are not training data for
  # intent classification. This is read out of the export rather than matched
  # against a name: "Default Fallback Intent" is only its display name, and a
  # project can rename it.
  #
  # Note that `events` is deliberately *not* used as a discriminator, though it
  # looks like one. `smalltalk.greetings.hello` carries
  # `events: [{"name": "GOOGLE_ASSISTANT_WELCOME"}]` and is a real intent with
  # many phrases, so "is triggered by a platform event" and "is not a
  # classification target" are different properties.
  defp fallback_intents(dir) do
    Path.join(dir, "*.json")
    |> Path.wildcard()
    |> Enum.reject(&String.ends_with?(&1, @usersays_suffix))
    |> Enum.filter(fn path ->
      case path |> File.read!() |> Jason.decode() do
        {:ok, %{"fallbackIntent" => true}} -> true
        _ -> false
      end
    end)
    |> Enum.map(&Path.basename(&1, ".json"))
    |> MapSet.new()
  end

  # Aborts rather than mapping an unplaceable name onto a real intent, which
  # would be a taxonomy decision made in a loader.
  #
  # Depth is not capped: the export contains five-part intents such as
  # `smarthome.device.brightness.check.implicit`.
  defp validate_labels!(phrases) do
    offenders =
      phrases
      |> Enum.group_by(& &1.label)
      |> Enum.reject(fn {label, _} -> dotted_intent?(label) end)
      |> Enum.map(fn {label, rows} ->
        sources = rows |> Enum.map(& &1.source_file) |> Enum.uniq() |> Enum.sort()
        "  #{String.pad_trailing(inspect(label), 32)} #{length(rows)} phrases, from #{length(sources)} file(s): #{Enum.join(Enum.take(sources, 2), ", ")}"
      end)
      |> Enum.sort()

    if offenders != [] do
      Mix.raise("""
      rebuild_gold_standard: #{length(offenders)} label(s) are not dotted intent paths.

      #{Enum.join(offenders, "\n")}

      An intent label must be a dotted lowercase identifier of at least two
      parts: a domain and something it does.

      Rename the intent's file in data/intents/ or in #{Provenance.authored_dir()},
      or delete its files if they are leftovers, then run this again.
      """)
    end
  end

  defp dotted_intent?(label) do
    parts = String.split(label, ".")

    length(parts) >= 2 and
      Enum.all?(parts, fn part -> part != "" and Regex.match?(~r/^[a-z0-9_]+$/, part) end)
  end

  defp read_phrases!(path, provenance_of) do
    entries =
      case path |> File.read!() |> Jason.decode() do
        {:ok, list} when is_list(list) ->
          list

        other ->
          Mix.raise("rebuild_gold_standard: #{path} is not a JSON array: #{inspect(other) |> String.slice(0, 120)}")
      end

    entries
    |> Enum.map(fn entry ->
      text =
        entry
        |> Map.get("data", [])
        |> Enum.map_join(fn segment -> Map.get(segment, "text", "") end)

      %{text: text, provenance: provenance_of.(entry), followup: followup!(entry, path)}
    end)
    |> Enum.reject(&(String.trim(&1.text) == ""))
  end

  # A follow-up row carries both keys; one without the other is a broken merge.
  defp followup!(entry, path) do
    case {Map.get(entry, "followupContext"), Map.get(entry, "followupVariant")} do
      {nil, nil} ->
        nil

      {context, variant} when is_binary(context) and is_binary(variant) ->
        %{"followup_context" => context, "followup_variant" => variant}

      other ->
        Mix.raise("rebuild_gold_standard: #{path} row #{inspect(entry["id"])} has an incomplete follow-up #{inspect(other)}")
    end
  end

  # The follow-up occurrence of a text, if any; ties go to the first variant
  # name, then the first context name, so the choice never depends on the order
  # the files were read in.
  defp followup_of(occurrences) do
    occurrences
    |> Enum.map(& &1.followup)
    |> Enum.reject(&is_nil/1)
    |> Enum.sort_by(&{&1["followup_variant"], &1["followup_context"]})
    |> List.first(%{})
  end

  # The distinct files of `occurrences`, ordered by the intent name each is
  # filed under rather than by the whole filename, whose suffix would put
  # `x.check_usersays_en.json` before `x_usersays_en.json`.
  defp sources_of(occurrences) do
    occurrences
    |> Enum.map(& &1.source_file)
    |> Enum.uniq()
    |> Enum.sort_by(&Path.basename(&1, @usersays_suffix))
  end

  # -- building ---------------------------------------------------------------

  defp resolve_provenance(occurrences) do
    occurrences
    |> Enum.map(& &1.provenance)
    |> Provenance.most_genuine()
  end

  @doc """
  The gold rows and the ambiguous texts that `phrases` rebuild to.

  Each phrase is a map of `:text`, `:norm`, `:label`, `:source_file`,
  `:provenance` and `:followup` (`nil`, or the `followup_context` /
  `followup_variant` pair). Rows come back sorted by `:norm`, one per
  normalized text, as string-keyed maps; ambiguous texts as maps of `"text"`,
  `"sources"` and `"candidates"`, in the same order.
  """
  @spec build_entries([map()]) :: {[map()], [map()]}
  def build_entries(phrases) do
    grouped = Enum.group_by(phrases, & &1.norm)

    {entries, conflicts} =
      grouped
      |> Enum.sort_by(fn {norm, _} -> norm end)
      |> Enum.map_reduce([], fn {_norm, occurrences}, conflicts ->
        labels = occurrences |> Enum.map(& &1.label) |> Enum.uniq() |> Enum.sort()
        first = hd(occurrences)
        intent = hd(labels)

        # The row is filed under `intent`, so its file and follow-up are read
        # from that intent's occurrences only.
        own = Enum.filter(occurrences, &(&1.label == intent))

        entry =
          Map.merge(
            %{
              "text" => first.text,
              "intent" => intent,
              "labeled_by" => resolve_provenance(occurrences),
              "source_file" => hd(sources_of(own))
            },
            followup_of(own)
          )

        case labels do
          [_label] ->
            {entry, conflicts}

          _many ->
            # Genuinely ambiguous even after folding: the same sentence is filed
            # under two intents that are not a device/lights pair. Written as
            # needs_review with every candidate kept, never resolved by guessing
            # -- resolving it by asking a model is precisely what produced the
            # corpus this task is repairing.
            entry = Map.merge(entry, %{"status" => "needs_review", "candidates" => labels})

            conflict = %{"text" => first.text, "sources" => sources_of(occurrences), "candidates" => labels}

            {entry, [conflict | conflicts]}
        end
      end)

    {entries, Enum.reverse(conflicts)}
  end

  # -- reporting --------------------------------------------------------------

  defp report(entries, conflicts, verbose?) do
    labels = entries |> Enum.map(& &1["intent"]) |> Enum.uniq()
    review = Enum.count(entries, &Map.has_key?(&1, "status"))

    counts = entries |> Enum.frequencies_by(& &1["intent"])
    sorted = counts |> Map.values() |> Enum.sort()
    median = Enum.at(sorted, div(length(sorted), 2))

    Mix.shell().info("")
    Mix.shell().info("  entries              #{length(entries)}")
    Mix.shell().info("  distinct labels      #{length(labels)}")
    Mix.shell().info("  needs_review         #{review}")
    Mix.shell().info("  median per label     #{median}")
    Mix.shell().info("  labels with <10      #{Enum.count(sorted, &(&1 < 10))}")

    report_provenance(entries)
    compare_to_pre_rebuild(entries)

    if verbose? and conflicts != [] do
      Mix.shell().info("\n  Ambiguous after the lights -> device fold:")

      Enum.each(conflicts, fn c ->
        Mix.shell().info("    #{String.pad_trailing(String.slice(c["text"], 0, 44), 46)} #{inspect(c["candidates"])}")
      end)
    end
  end

  defp report_provenance(entries) do
    counts = Enum.frequencies_by(entries, & &1["labeled_by"])
    total = length(entries)

    Mix.shell().info("")
    Mix.shell().info("  provenance:")

    Provenance.kinds()
    |> Enum.each(fn kind ->
      n = Map.get(counts, kind, 0)

      Mix.shell().info(
        "    #{String.pad_trailing(kind, 14)} #{String.pad_leading(to_string(n), 5)}  #{Float.round(n / total * 100, 1)}%"
      )
    end)

    synthetic = Provenance.written_back() |> Enum.map(&Map.get(counts, &1, 0)) |> Enum.sum()

    if synthetic > 0 do
      Mix.shell().info("")

      Mix.shell().info(
        "    #{synthetic} of #{total} rows (#{Float.round(synthetic / total * 100, 1)}%) were written back into the export"
      )

      Mix.shell().info("    by scripts/materialize_orphan_intents.exs, not by Dialogflow.")
    end
  end

  # `gold_standard.pre-rebuild.json` is a separately derived copy of the same
  # corpus. Agreement with it is a check that this rebuild produces the labels
  # the export describes; a drop means one of the two has changed.
  defp compare_to_pre_rebuild(entries) do
    path = Path.join(Path.dirname(EvaluationStore.gold_standard_path("intent")), "gold_standard.pre-rebuild.json")

    if File.exists?(path) do
      pre =
        path
        |> File.read!()
        |> Jason.decode!()
        |> Map.new(fn row -> {Export.normalise(row["text"]), Label.fold_lights(row["intent"])} end)

      shared =
        entries
        |> Enum.map(fn e -> {Export.normalise(e["text"]), e["intent"]} end)
        |> Enum.filter(fn {norm, _} -> Map.has_key?(pre, norm) end)

      agree = Enum.count(shared, fn {norm, label} -> Map.fetch!(pre, norm) == label end)
      total = length(shared)

      Mix.shell().info("")
      Mix.shell().info("  vs gold_standard.pre-rebuild.json (folded the same way):")
      Mix.shell().info("    shared texts       #{total}")

      Mix.shell().info(
        "    agree              #{agree} (#{Float.round(agree * 100 / max(total, 1), 1)}%)"
      )
    else
      Mix.shell().info("\n  (no pre-rebuild snapshot at #{path} to cross-check against)")
    end
  end

  defp count_files(phrases), do: phrases |> Enum.map(& &1.source_file) |> Enum.uniq() |> length()

  # -- output -----------------------------------------------------------------

  # The gold standard's own path unless `--output` names another. `--output`
  # only says where `--save` writes or what `--check` compares with, so it is
  # refused without one of them.
  defp output_path!(nil, _writes_or_checks?), do: EvaluationStore.gold_standard_path("intent")

  defp output_path!(path, true), do: Path.expand(path)

  defp output_path!(path, false) do
    Mix.raise(
      "rebuild_gold_standard: --output #{path} says where --save writes or what --check compares with; pass one of them"
    )
  end

  defp write_output!(entries, path) do
    if File.exists?(path) do
      stamp = DateTime.utc_now() |> DateTime.to_iso8601() |> String.replace(":", "-")
      backup = "#{path}.#{stamp}.bak"
      File.cp!(path, backup)
      Mix.shell().info("\n  Backed up to #{backup}")
    end

    File.write!(path, encode_entries!(entries))
    Mix.shell().info("  Wrote #{length(entries)} entries to #{path}")
  end

  defp write_conflict_report!(conflicts, output) do
    path = conflict_path(output)

    File.write!(path, encode_conflicts!(conflicts))
    Mix.shell().info("  Wrote #{length(conflicts)} ambiguous texts to #{path}\n")
  end

  defp conflict_path(output), do: output |> Path.dirname() |> Path.join("ambiguous_texts.json")

  @doc "The bytes of `gold_standard.json` for `entries`, each row's keys in the documented order."
  @spec encode_entries!([map()]) :: String.t()
  def encode_entries!(entries), do: encode_ordered!(entries, @entry_keys)

  @doc "The bytes of `ambiguous_texts.json` for `conflicts`, each one's keys in the documented order."
  @spec encode_conflicts!([map()]) :: String.t()
  def encode_conflicts!(conflicts), do: encode_ordered!(conflicts, @conflict_keys)

  # A map's key order is the runtime's, not the file's, so every row is written
  # as an ordered object. A key the order does not name raises rather than being
  # dropped or appended wherever it happens to land.
  defp encode_ordered!(rows, keys) do
    rows
    |> Enum.map(fn row ->
      case Map.keys(row) -- keys do
        [] ->
          keys
          |> Enum.filter(&Map.has_key?(row, &1))
          |> Enum.map(&{&1, Map.fetch!(row, &1)})
          |> Jason.OrderedObject.new()

        unknown ->
          raise ArgumentError,
                "rebuild_gold_standard: row #{inspect(row)} has keys #{inspect(unknown)} outside the written order #{inspect(keys)}"
      end
    end)
    |> Jason.encode!(pretty: true)
    |> Kernel.<>("\n")
  end

  # Compares what the sources rebuild to with the files at `output`, byte for
  # byte, and raises naming the rows that differ.
  defp check!(entries, conflicts, output) do
    mismatches =
      [
        {output, encode_entries!(entries)},
        {conflict_path(output), encode_conflicts!(conflicts)}
      ]
      |> Enum.reject(fn {path, rebuilt} -> File.exists?(path) and File.read!(path) == rebuilt end)
      |> Enum.map(fn {path, rebuilt} -> describe_mismatch(path, rebuilt) end)

    if mismatches != [] do
      Mix.raise("""
      rebuild_gold_standard: the committed files are not what the sources rebuild to.

      #{Enum.join(mismatches, "\n\n")}

      Either the sources or the committed files changed without the other. Run
      `mix rebuild_gold_standard --save --output PATH`, review the diff against the
      committed file, and update whichever side is wrong.
      """)
    end

    Mix.shell().info("\n  #{output} and #{conflict_path(output)} match the rebuild byte for byte.\n")
  end

  defp describe_mismatch(path, rebuilt) do
    if File.exists?(path) do
      committed = path |> File.read!() |> Jason.decode!()
      ours = Jason.decode!(rebuilt)
      only_committed = committed -- ours
      only_rebuilt = ours -- committed

      rows =
        Enum.map(only_committed, &"  - #{Jason.encode!(&1)}") ++
          Enum.map(only_rebuilt, &"  + #{Jason.encode!(&1)}")

      detail =
        if rows == [],
          do: "  the same rows, in a different order or with different key order or formatting",
          else: Enum.join(rows, "\n")

      "#{path} (- committed only, + rebuild only):\n#{detail}"
    else
      "#{path} does not exist"
    end
  end

  defp banner(title) do
    Mix.shell().info("")
    Mix.shell().info(String.duplicate("=", 64))
    Mix.shell().info(title)
    Mix.shell().info(String.duplicate("=", 64))
  end
end
