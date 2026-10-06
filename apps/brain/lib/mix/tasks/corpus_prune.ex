defmodule Mix.Tasks.Corpus.Prune do
  @shortdoc "Remove every row of one provenance from the intent export and the held-out split"

  @moduledoc """
  Removes the rows of a single provenance from every copy of the intent corpus,
  after archiving them: the export in `data/intents/`, its POS-annotated mirror in
  `data/training/intents/`, and the held-out split.

      mix corpus.prune --provenance augmented          # dry run, prints the plan
      mix corpus.prune --provenance augmented --save   # archive, then write

  Removing a row needs nothing but its `id`, in every copy. Re-deriving the
  annotated mirror would need a POS tagger; that is a different operation and this
  is not it.

  ## Why all three

  The held-out split is applied at read time: `EvaluationStore.load_gold_standard/2`
  rejects a row from training when it is whole-map equal to a row in
  `held_out.json`. Removing rows from the export therefore does not remove them
  from the evaluation set — `:held_out` reads that file directly. Pruning the
  export alone would leave the split scoring the model against texts the corpus no
  longer contains, which is the opposite of the intent.

  The split is pruned rather than re-carved. `mix split.held_out` shuffles, so
  re-carving draws a different held-out set and confounds this removal with a
  reshuffle; pruning leaves every surviving row in place, so a score before and
  after is measured on the same rows.

  ## What it refuses to do

  * prune `dialogflow` rows — those are the corpus, not contamination in it
  * run when no row carries the requested provenance
  * empty a usersays file, which would silently delete an intent
  * write when a file does not survive `Brain.Corpus.Export.round_trips?/2`
  * write when a surviving held-out row's text would no longer exist in the export

  Every one of these aborts the run. Nothing is written until the archive has been
  written and read back.
  """

  use Mix.Task

  alias Brain.Corpus.Export
  alias Brain.Corpus.Provenance
  alias Brain.ML.EvaluationStore

  @requirements ["app.config"]

  @switches [provenance: :string, save: :boolean]

  @task "intent"

  @impl Mix.Task
  def run(args) do
    {opts, _positional, invalid} = OptionParser.parse(args, strict: @switches)

    unless invalid == [] do
      Mix.raise("corpus.prune: unknown option #{inspect(invalid)}")
    end

    kind = opts |> Keyword.get(:provenance) |> validate_kind!()
    save? = Keyword.get(opts, :save, false)

    export = plan_export!(kind)
    annotated = plan_annotated!(kind)
    held = plan_held_out!(kind)

    check!(export, annotated, held, kind)
    report(export, annotated, held, kind)

    if save? do
      apply!(export, annotated, held, kind)
    else
      Mix.shell().info("\n  Dry run. Pass --save to archive and write.\n")
    end
  end

  # -- validation -------------------------------------------------------------

  defp validate_kind!(nil) do
    Mix.raise("""
    corpus.prune: --provenance is required

      mix corpus.prune --provenance #{Enum.join(prunable(), " | ")}
    """)
  end

  defp validate_kind!(kind) do
    cond do
      not Provenance.kind?(kind) ->
        Mix.raise("""
        corpus.prune: #{inspect(kind)} is not a provenance this corpus records

          known: #{Enum.join(Provenance.kinds(), ", ")}
        """)

      kind not in prunable() ->
        Mix.raise("""
        corpus.prune: refusing to prune #{inspect(kind)}

        Those rows are the corpus. This task removes text that was written back
        into the export by something other than Dialogflow; removing the export's
        own utterances is not a pruning decision.
        """)

      true ->
        kind
    end
  end

  defp prunable, do: Provenance.kinds() -- ["dialogflow"]

  # -- planning ---------------------------------------------------------------

  defp plan_export!(kind) do
    dir = Brain.data_path("intents")

    unless File.dir?(dir) do
      Mix.raise("corpus.prune: no Dialogflow export at #{dir}")
    end

    files = Export.usersays_files(dir)

    if files == [] do
      Mix.raise("corpus.prune: no *#{Export.usersays_suffix()} files under #{dir}")
    end

    Enum.map(files, &plan_file!(&1, kind))
  end

  defp plan_file!(path, kind) do
    content = File.read!(path)

    unless Export.round_trips?(content, path) do
      Mix.raise("""
      corpus.prune: #{path} cannot be edited without reformatting it

      Splitting it into entries and reassembling them does not reproduce the file
      byte for byte, so removing one entry would rewrite the rest. Nothing was
      written.
      """)
    end

    split = Export.elements!(content, path)

    {removed, kept} =
      split.elements
      |> Enum.map(fn text ->
        entry = Jason.decode!(text)
        %{text: text, entry: entry, provenance: Provenance.of_entry!(entry, path)}
      end)
      |> Enum.split_with(&(&1.provenance == kind))

    %{path: path, split: split, kept: kept, removed: removed}
  end

  # `data/training/intents/` is the POS-annotated mirror of the export, one file
  # per intent, each row carrying the same `id` as the export row it came from. It
  # is derived output, so a row removed from the export has to be removed here too
  # -- and removing a row needs no tagger, only the id. Regenerating the directory
  # would need one; that is a different operation and not what this is.
  defp plan_annotated!(kind) do
    dir = Brain.data_path("training/intents")

    unless File.dir?(dir) do
      Mix.raise("""
      corpus.prune: no annotated corpus at #{dir}

      It mirrors the export row for row. If it is genuinely gone, delete the
      readers that expect it; this task will not prune one copy of the corpus and
      leave another holding #{kind} rows.
      """)
    end

    dir
    |> Path.join("*.json")
    |> Path.wildcard()
    |> Enum.sort()
    |> Enum.map(&plan_file!(&1, kind))
  end

  defp plan_held_out!(kind) do
    path = EvaluationStore.held_out_path(@task)

    unless File.exists?(path) do
      Mix.raise("""
      corpus.prune: no held-out split at #{path}

      Without one there is nothing to keep consistent with the export, and the
      rows removed here would still be scored against.
      """)
    end

    content = File.read!(path)

    unless Export.round_trips?(content, path) do
      Mix.raise("corpus.prune: #{path} cannot be edited without reformatting it")
    end

    split = Export.elements!(content, path)

    {removed, kept} =
      split.elements
      |> Enum.map(fn text -> %{text: text, row: Jason.decode!(text)} end)
      |> Enum.split_with(&(Map.get(&1.row, "labeled_by") == kind))

    %{path: path, split: split, kept: kept, removed: removed}
  end

  # -- guards -----------------------------------------------------------------

  # Nothing to do is an error, but only when *no* copy holds such a row. The copies
  # can legitimately disagree: pruning the export and the annotated mirror in
  # separate runs is how a half-finished prune gets finished.
  defp check!(export, annotated, held, kind) do
    if removed_count(export) + removed_count(annotated) + length(held.removed) == 0 do
      Mix.raise("""
      corpus.prune: no row in any copy of the corpus carries provenance #{inspect(kind)}

      Either it has already been pruned or the provenance rule no longer
      classifies these rows the way the caller expects.
      """)
    end

    emptied = Enum.filter(export ++ annotated, &(&1.removed != [] and &1.kept == []))

    unless emptied == [] do
      Mix.raise("""
      corpus.prune: #{length(emptied)} file(s) would be left with no utterances

      #{Enum.map_join(emptied, "\n", &("  " <> Path.basename(&1.path)))}

      An empty usersays file is an intent with no examples. Removing #{kind} rows
      cannot be the whole story for these intents.
      """)
    end

    stranded = stranded_held_out(export, held)

    unless stranded == [] do
      Mix.raise("""
      corpus.prune: #{length(stranded)} surviving held-out row(s) would reference text
      that no longer exists in the export

      #{Enum.map_join(Enum.take(stranded, 10), "\n", &("  " <> inspect(&1)))}

      The split is applied by matching rows against the corpus, so these rows
      would be scored against a corpus that cannot produce them.
      """)
    end
  end

  # A surviving held-out row must still be derivable from the pruned export, or
  # the evaluation set and the corpus have come apart.
  defp stranded_held_out(export, held) do
    surviving =
      export
      |> Enum.flat_map(fn file -> Enum.map(file.kept, & &1.entry) end)
      |> MapSet.new(&Export.normalise(Export.phrase_text(&1)))

    held.kept
    |> Enum.map(&Map.get(&1.row, "text"))
    |> Enum.reject(&MapSet.member?(surviving, Export.normalise(&1)))
  end

  defp removed_count(export), do: Enum.sum(Enum.map(export, &length(&1.removed)))

  # -- reporting --------------------------------------------------------------

  defp report(export, annotated, held, kind) do
    touched = Enum.filter(export, &(&1.removed != []))
    kept_rows = Enum.sum(Enum.map(export, &length(&1.kept)))
    all_rows = kept_rows + removed_count(export)

    ann_kept = Enum.sum(Enum.map(annotated, &length(&1.kept)))
    ann_removed = removed_count(annotated)

    Mix.shell().info("")
    Mix.shell().info(String.duplicate("=", 64))
    Mix.shell().info("PRUNE #{String.upcase(kind)} ROWS")
    Mix.shell().info(String.duplicate("=", 64))

    Mix.shell().info("")
    row("export files", length(export))
    row("...holding #{kind} rows", length(touched))
    row("export rows", "#{all_rows} -> #{kept_rows}")
    row("removed", removed_count(export))

    Mix.shell().info("")
    row("annotated files", length(annotated))
    row("annotated rows", "#{ann_kept + ann_removed} -> #{ann_kept}")
    row("removed", ann_removed)

    Mix.shell().info("")
    row("held-out rows", "#{length(held.kept) + length(held.removed)} -> #{length(held.kept)}")
    row("removed", length(held.removed))

    Mix.shell().info("")
    Mix.shell().info("  per file:")

    touched
    |> Enum.sort_by(&(-length(&1.removed)))
    |> Enum.each(fn file ->
      total = length(file.kept) + length(file.removed)

      Mix.shell().info(
        "    #{String.pad_trailing(Path.basename(file.path, Export.usersays_suffix()), 52)}" <>
          " #{String.pad_leading(to_string(length(file.removed)), 3)}/#{total}"
      )
    end)
  end

  defp row(label, value), do: Mix.shell().info("  #{String.pad_trailing(label, 29)}#{value}")

  # -- writing ----------------------------------------------------------------

  defp apply!(export, annotated, held, kind) do
    touched = Enum.filter(export, &(&1.removed != []))
    ann_touched = Enum.filter(annotated, &(&1.removed != []))
    archive = Brain.data_path(Path.join("archive", "#{kind}_rows"))
    ann_archive = Path.join(archive, "training_intents")

    File.mkdir_p!(ann_archive)
    write_archive!(archive, touched, held, kind)
    write_rows!(ann_archive, ann_touched)
    verify_archive!(archive, touched)
    verify_archive!(ann_archive, ann_touched)
    write_manifest!(archive, kind)

    Enum.each(touched ++ ann_touched, fn file ->
      File.write!(file.path, Export.rejoin(%{file.split | elements: Enum.map(file.kept, & &1.text)}))
    end)

    File.write!(held.path, Export.rejoin(%{held.split | elements: Enum.map(held.kept, & &1.text)}))

    Mix.shell().info("")
    Mix.shell().info("  archived to  #{archive}")
    Mix.shell().info("  rewrote      #{length(touched)} export files")
    Mix.shell().info("  rewrote      #{length(ann_touched)} annotated files")
    Mix.shell().info("  rewrote      #{held.path}")

    Mix.shell().info("")

    Mix.shell().info(
      "  Next: mix rebuild_gold_standard --save, then regenerate and retrain. " <>
        "The provenance gates will refuse any model still built against the old corpus."
    )

    Mix.shell().info("")
  end

  # The removed rows are kept verbatim, so what was taken out can be read back
  # exactly as it stood rather than as a re-encoding of it. `data/archive/` is the
  # one part of `data/` under version control, so this is also the point at which
  # the rows stop being unrecoverable.
  defp write_archive!(archive, touched, held, _kind) do
    write_rows!(archive, touched)

    if held.removed != [] do
      File.write!(
        Path.join(archive, Path.basename(held.path)),
        wrap(Enum.map(held.removed, & &1.text))
      )
    end
  end

  # The manifest counts what is *in* the archive, not what one run removed. The
  # copies of the corpus can be pruned in separate runs, and a manifest describing
  # the last run would report zero for the rows an earlier one had archived --
  # overwriting the only record of them with a more recent, emptier truth.
  defp write_manifest!(archive, kind) do
    per_file =
      archive
      |> Path.join("**/*.json")
      |> Path.wildcard()
      |> Enum.reject(&(Path.basename(&1) == "manifest.json"))
      |> Map.new(fn path ->
        {Path.relative_to(path, archive), length(Jason.decode!(File.read!(path)))}
      end)

    {held, corpus} = Map.split(per_file, ["held_out.json"])

    manifest = %{
      "provenance" => kind,
      "rows_archived" => Enum.sum(Map.values(per_file)),
      "from_corpus" => Enum.sum(Map.values(corpus)),
      "from_held_out" => Enum.sum(Map.values(held)),
      "files" => map_size(corpus),
      "per_file" => corpus
    }

    File.write!(Path.join(archive, "manifest.json"), Jason.encode!(manifest, pretty: true) <> "\n")
  end

  defp write_rows!(dir, files) do
    Enum.each(files, fn file ->
      File.write!(Path.join(dir, Path.basename(file.path)), wrap(Enum.map(file.removed, & &1.text)))
    end)
  end

  defp wrap([]), do: "[]\n"
  defp wrap(rows), do: "[\n" <> Enum.join(rows, ",\n") <> "\n]\n"

  # Reading the archive back before touching the export is the difference between
  # having archived the rows and having intended to.
  defp verify_archive!(archive, touched) do
    Enum.each(touched, fn file ->
      path = Path.join(archive, Path.basename(file.path))
      written = path |> File.read!() |> Jason.decode!()
      expected = Enum.map(file.removed, & &1.entry)

      unless written == expected do
        Mix.raise("""
        corpus.prune: #{path} does not read back as the rows it archived

        Nothing was removed from the export.
        """)
      end
    end)
  end
end
