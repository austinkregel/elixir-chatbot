defmodule Mix.Tasks.Axes.Snapshot do
  @shortdoc "Record the 18 ChunkProfile axes over a corpus, with run provenance"

  @moduledoc """
  Measures every `Brain.Analysis.ChunkProfile` axis over a corpus and stores the
  result as an `Atlas.Schemas.AxisRun` plus one observation per utterance per
  axis.

      mix axes.snapshot --corpus .claude/corpus/utterances.json
      mix axes.snapshot --corpus path/to/corpus.json --tag baseline --note "first honest measurement"
      mix axes.snapshot --corpus path/to/corpus.json --limit 200

  ## Tagged and untagged runs

  An untagged run is disposable and prunable. `--tag` promotes a run to a
  durable reference point and requires `--note` saying what it represents; a
  reference point nobody can interpret later is not one. Tagged runs also store
  the full 343-float feature vector per utterance, so a regression like task 072
  — three memory dimensions inverting while the vector length stayed 343 — is
  attributable to a feature rather than only visible as an axis moving.

  ## Why `--corpus` has no default

  Whether `.claude/corpus/` is the right home for this data is task 067, still
  open. Defaulting to a path inside `.claude/` would put an undecided location
  on a production code path and quietly bake the decision in, so the path is
  required and a missing file raises.

  ## Corpus format

  A JSON array of objects with at least a `"norm"` (normalised text) and
  `"text"` field, as `.claude/corpus/build_corpus.py` produces. `utterance_id`
  is derived from `"norm"` by SHA-256 rather than from array position, so a
  corpus rebuild that reorders rows still lines observations up across runs.
  """

  use Mix.Task

  alias Brain.Analysis.{ChunkProfile, FeatureExtractor, Pipeline, RunProvenance}

  @requirements ["app.start"]

  @switches [corpus: :string, tag: :string, note: :string, limit: :integer]

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("axes.snapshot: unknown options #{inspect(invalid)}")
    if positional != [], do: Mix.raise("axes.snapshot: unexpected arguments #{inspect(positional)}")

    corpus_path = opts[:corpus] || Mix.raise(usage("--corpus is required"))
    tag = opts[:tag]
    note = opts[:note]

    if tag && (is_nil(note) or String.trim(note) == "") do
      Mix.raise(usage("--tag requires --note saying what this run represents"))
    end

    if note && is_nil(tag) do
      Mix.raise(usage("--note only means something with --tag"))
    end

    unless File.regular?(corpus_path) do
      Mix.raise("axes.snapshot: no corpus at #{corpus_path}")
    end

    Mix.shell().info("Capturing run provenance ...")
    provenance = RunProvenance.capture!()

    if provenance.git.dirty do
      Mix.shell().info(
        "  note: the tree is dirty, so #{String.slice(provenance.git.sha, 0, 8)} does not " <>
          "identify this code exactly. Recorded as dirty: true."
      )
    end

    utterances = load_corpus!(corpus_path, opts[:limit])
    Mix.shell().info("Measuring #{length(utterances)} utterances over #{length(ChunkProfile.axes())} axes ...")

    observations = measure(utterances, store_vectors?: not is_nil(tag))

    run_attrs = %{
      provenance: provenance,
      schema_fingerprint: provenance.extractor.schema_fingerprint,
      corpus_sha256: RunProvenance.sha256_file!(corpus_path),
      utterance_count: length(utterances),
      tag: tag,
      tag_note: note
    }

    case Atlas.Axes.record_run(run_attrs, observations) do
      {:ok, %{run: run, observations: count}} ->
        report(run, count, observations)

      {:error, {:run, changeset}} ->
        Mix.raise("axes.snapshot: the run was rejected: #{inspect(changeset.errors)}")

      {:error, {:observation, index, changeset}} ->
        Mix.raise(
          "axes.snapshot: observation #{index} was rejected: #{inspect(changeset.errors)}. " <>
            "Nothing was written."
        )
    end
  end

  # -- measurement ------------------------------------------------------------

  defp measure(utterances, opts) do
    store_vectors? = Keyword.fetch!(opts, :store_vectors?)
    axes = ChunkProfile.axes()

    utterances
    |> Enum.with_index(1)
    |> Enum.flat_map(fn {utterance, position} ->
      if rem(position, 250) == 0 do
        Mix.shell().info("  #{position}/#{length(utterances)}")
      end

      observations_for(utterance, axes, store_vectors?)
    end)
  end

  defp observations_for(%{id: id, text: text}, axes, store_vectors?) do
    analysis = Pipeline.analyze_chunk(text)
    vector = FeatureExtractor.extract_vector(analysis)
    profile = ChunkProfile.materialize(analysis, vector)

    Enum.map(axes, fn axis ->
      entry =
        ChunkProfile.provenance(profile, axis) ||
          Mix.raise(
            "axes.snapshot: no provenance for #{axis} on #{inspect(text)}. " <>
              "An axis with no provenance cannot be told apart from a defaulted one, so the " <>
              "snapshot would record a measurement it cannot interpret."
          )

      %{
        utterance_id: id,
        axis: to_string(axis),
        value: serialize_value(Map.fetch!(profile, axis)),
        status: to_string(entry.status),
        reason: entry[:reason] && to_string(entry[:reason]),
        # Stored once per utterance, on the first axis only: repeating 343
        # floats 18 times would multiply a tagged run's storage by the axis
        # count for no extra information.
        feature_vector: if(store_vectors? and axis == hd(axes), do: vector)
      }
    end)
  end

  # Axis values are atoms for the 15 categorical axes and floats for the three
  # continuous ones. Both become strings, because the kind is declared in
  # ChunkProfile.axis_manifest/0 and a typed column here would be a second
  # declaration that can disagree with it.
  defp serialize_value(value) when is_atom(value), do: to_string(value)
  defp serialize_value(value) when is_float(value), do: Float.to_string(value)
  defp serialize_value(value) when is_integer(value), do: Integer.to_string(value)
  defp serialize_value(value) when is_binary(value), do: value

  defp serialize_value(value) do
    Mix.raise(
      "axes.snapshot: an axis holds #{inspect(value)}, which has no defined string form. " <>
        "Add one here rather than letting it round-trip as inspect/1 output."
    )
  end

  # -- corpus -----------------------------------------------------------------

  defp load_corpus!(path, limit) do
    rows =
      case path |> File.read!() |> Jason.decode!() do
        rows when is_list(rows) ->
          rows

        other ->
          Mix.raise(
            "axes.snapshot: expected #{path} to hold a JSON array, got #{inspect(other) |> String.slice(0, 80)}"
          )
      end

    rows
    |> Enum.with_index()
    |> Enum.map(fn {row, index} -> utterance!(row, index, path) end)
    |> dedupe!()
    |> then(fn rows -> if limit, do: Enum.take(rows, limit), else: rows end)
  end

  defp utterance!(row, index, path) do
    text = row["text"] || row["norm"]

    if is_nil(text) or String.trim(text) == "" do
      Mix.raise("axes.snapshot: row #{index} of #{path} has no usable text: #{inspect(row) |> String.slice(0, 120)}")
    end

    # Keyed on the normalised form so the id survives a corpus rebuild that
    # changes whitespace or ordering.
    norm = row["norm"] || String.downcase(String.trim(text))

    %{id: digest(norm), text: text, norm: norm}
  end

  defp dedupe!(utterances) do
    by_id = Enum.group_by(utterances, & &1.id)

    collisions =
      by_id
      |> Enum.filter(fn {_id, rows} -> length(Enum.uniq_by(rows, & &1.norm)) > 1 end)
      |> Enum.map(fn {id, rows} -> "  #{id}: #{inspect(Enum.map(rows, & &1.norm))}" end)

    if collisions != [] do
      Mix.raise(
        "axes.snapshot: distinct utterances hashed to the same id:\n" <>
          Enum.join(collisions, "\n")
      )
    end

    # A corpus may legitimately hold the same normalised text twice; measuring
    # it twice would double-count it in every per-axis census.
    Enum.uniq_by(utterances, & &1.id)
  end

  defp digest(text) do
    :crypto.hash(:sha256, text) |> Base.encode16(case: :lower) |> binary_part(0, 16)
  end

  # -- reporting --------------------------------------------------------------

  defp report(run, count, observations) do
    Mix.shell().info("")
    Mix.shell().info(String.duplicate("=", 72))
    Mix.shell().info("run              #{run.id}")
    Mix.shell().info("tag              #{run.tag || "(untagged -- prunable)"}")
    Mix.shell().info("fingerprint      #{run.schema_fingerprint}")
    Mix.shell().info("utterances       #{run.utterance_count}")
    Mix.shell().info("observations     #{count}")
    Mix.shell().info(String.duplicate("=", 72))
    Mix.shell().info("")

    census =
      observations
      |> Enum.group_by(& &1.axis)
      |> Enum.map(fn {axis, rows} ->
        computed = Enum.count(rows, &(&1.status == "computed"))
        {axis, computed, length(rows) - computed, rows |> Enum.map(& &1.value) |> Enum.uniq() |> length()}
      end)
      |> Enum.sort_by(fn {_axis, computed, _d, _v} -> -computed end)

    Mix.shell().info(
      "#{String.pad_trailing("axis", 24)}#{String.pad_leading("computed", 10)}" <>
        "#{String.pad_leading("dflt", 8)}#{String.pad_leading("distinct", 10)}"
    )

    Mix.shell().info(String.duplicate("-", 72))

    Enum.each(census, fn {axis, computed, defaulted, distinct} ->
      Mix.shell().info(
        "#{String.pad_trailing(axis, 24)}#{String.pad_leading(to_string(computed), 10)}" <>
          "#{String.pad_leading(to_string(defaulted), 8)}#{String.pad_leading(to_string(distinct), 10)}"
      )
    end)

    never = for {axis, 0, _d, _v} <- census, do: axis

    if never != [] do
      Mix.shell().info("")
      Mix.shell().info("#{length(never)} axes NEVER computed on this corpus: #{Enum.join(never, ", ")}")
    end

    Mix.shell().info("")

    if run.tag do
      Mix.shell().info("Tagged as #{inspect(run.tag)}; feature vectors stored.")
    else
      Mix.shell().info("Untagged. Promote with Atlas.Axes.tag_run/3, or leave it to be pruned.")
    end
  end

  defp usage(message) do
    """
    axes.snapshot: #{message}

        mix axes.snapshot --corpus PATH [--tag NAME --note "what this represents"] [--limit N]
    """
  end
end
