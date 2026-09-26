defmodule Mix.Tasks.RebuildGoldStandard do
  @shortdoc "Rebuild the intent gold standard from Dialogflow anchoring only"

  @moduledoc """
  Rebuilds `priv/evaluation/intent/gold_standard.json` from `data/intents/`,
  using Dialogflow as the sole authority.

      mix rebuild_gold_standard                 # dry run, prints the summary
      mix rebuild_gold_standard --save          # write it
      mix rebuild_gold_standard --save --verbose

  ## What changed, and why (task 079)

  The previous implementation had a three-layer strategy whose **layer 2 asked
  the intent classifier for a label and wrote the answer as gold** whenever the
  model was at least 70% confident, with no field distinguishing it from a
  human- or Dialogflow-anchored label. Measured afterwards: of 4,870 entries,
  exactly one carried a `status`. The rest were indistinguishable from ground
  truth whatever produced them.

  What that did to the corpus, measured against
  `gold_standard.pre-rebuild.json`:

    * 2,398 of 4,867 shared texts had their label changed
    * **83 labels were destroyed**, 79 of them real Dialogflow intents --
      `smalltalk.greetings.goodnight`, `.goodmorning`, `.goodevening`,
      `.nice_to_meet_you` all collapsed into `smalltalk.greet`;
      `smarthome.device.volume.mute` and `.unmute` into `smarthome.switch`
    * 405 texts were dropped entirely
    * the taxonomy fell from 208 labels to 137

  Layer 2 is gone. No model is consulted anywhere in this task. A text with no
  Dialogflow anchor is not written at all, because there is no longer any
  mechanism that could invent a label for it.

  ## It now iterates Dialogflow, not the old gold standard

  The previous version walked the *existing* gold standard and relabelled what
  it found there, so it could never recover a text an earlier run had dropped,
  and it inherited whatever that run had left behind. This walks
  `data/intents/*_usersays_en.json` -- 5,274 distinct normalised texts, which
  covers 100% of both the current and the pre-rebuild corpora.

  ## The label comes from the filename

  Two readers of `data/intents/` disagreed. This task read the `"name"` field
  of the paired metadata file and downcased it; `.claude/corpus/build_corpus.py`
  used the filename stem. They produce different strings for the same intent,
  and the metadata route silently dropped the two usersays files that have no
  paired metadata (`weather.query`, `test.intent.promotion`).

  The filename is now authoritative, because that is what `utterances.json`,
  `axis_sample.exs` and every task 081/082 measurement already key on. A
  ` - context:...` suffix is stripped: per Austin, a hyphenated name is a subset
  of the unhyphenated one. 273 Dialogflow names reduce to 230 roots this way.

  ## The lights -> device fold

  `smarthome.lights.X` is written as `smarthome.device.X`. Austin's ruling:
  lights are a specialization of device, and the text alone cannot separate them
  -- `"brightness up"` is filed under both in Dialogflow. 24 `lights.*` roots
  fold, 17 onto existing `device.*` roots and 7 creating
  `device.{hue,saturation}.*`.

  This is also what resolves most of the genuine ambiguities: of 108 texts that
  map to more than one intent file, the large majority are a `device`/`lights`
  pair that becomes one label after folding.

  ## Provenance per entry

  Every entry carries `labeled_by` and `source_file`. A gold standard that
  cannot say where a label came from cannot be trusted, and its absence is
  exactly what made the layer-2 damage invisible for months.
  """

  use Mix.Task

  alias Brain.ML.EvaluationStore

  @requirements ["app.config"]

  @switches [save: :boolean, verbose: :boolean]

  @usersays_suffix "_usersays_en.json"
  @lights_prefix "smarthome.lights."
  @device_prefix "smarthome.device."

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("rebuild_gold_standard: unknown options #{inspect(invalid)}")

    if positional != [],
      do: Mix.raise("rebuild_gold_standard: unexpected arguments #{inspect(positional)}")

    save? = opts[:save] || false
    verbose? = opts[:verbose] || false

    banner("REBUILD INTENT GOLD STANDARD (Dialogflow only)")

    phrases = load_dialogflow!()
    Mix.shell().info("  #{length(phrases)} usersays phrases across #{count_files(phrases)} intent files")

    {entries, conflicts} = build_entries(phrases)

    report(entries, conflicts, verbose?)

    if save? do
      write_output!(entries)
      write_conflict_report!(conflicts)
    else
      Mix.shell().info("\n  Dry run. Pass --save to write.\n")
    end
  end

  # -- loading ----------------------------------------------------------------

  defp load_dialogflow! do
    dir = Brain.data_path("intents")

    unless File.dir?(dir) do
      Mix.raise("rebuild_gold_standard: no Dialogflow export at #{dir}")
    end

    files = Path.wildcard(Path.join(dir, "*" <> @usersays_suffix))

    if files == [] do
      Mix.raise("rebuild_gold_standard: no *#{@usersays_suffix} files under #{dir}")
    end

    fallbacks = fallback_intents(dir)

    phrases =
      files
      |> Enum.reject(fn path -> Path.basename(path, @usersays_suffix) in fallbacks end)
      |> Enum.flat_map(fn path ->
        label = path |> Path.basename(@usersays_suffix) |> root_label() |> fold_lights()
        source = Path.basename(path)

        path
        |> read_phrases!()
        |> Enum.map(fn text -> %{text: text, norm: normalise(text), label: label, source_file: source} end)
      end)

    validate_labels!(phrases)

    phrases
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

  # A label must be a dotted lowercase identifier of at least two parts:
  # a domain and something it does. A single bare word names a domain with no
  # action, and a display name like "Default Welcome Intent" is not an intent
  # path at all -- neither can be placed in the taxonomy, and placing one anyway
  # means inventing a mapping. That is how `cleanup_gold_standard` accumulated
  # 123 hardcoded rename rules whose `@smarthome_consolidation` block reproduces
  # 96.2% of the collapse task 079 blamed on a classifier.
  #
  # No upper bound on depth. Austin described the taxonomy as two to four parts,
  # but the export contains legitimate five-part intents
  # (`smarthome.device.switch.schedule.off`,
  # `smarthome.device.brightness.check.implicit`), so enforcing four would reject
  # 16 real intents on a literal reading of a description. Whether the depth
  # should be capped is a question about the taxonomy, not something this loader
  # gets to decide.
  #
  # So this raises and names the offenders instead. The fix belongs in the
  # export, where the intent is actually named.
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

      An intent label must be a dotted lowercase identifier of at least two parts:
      a domain and something it does. A bare domain names no action, and a display
      name is not an intent path. Guessing where these belong is the
      hardcoded-rename-map pattern this rebuild exists to remove.

      Fix them in data/intents/ -- rename the intent, or delete the export files
      if they are leftovers -- then run this again.
      """)
    end
  end

  defp dotted_intent?(label) do
    parts = String.split(label, ".")

    length(parts) >= 2 and
      Enum.all?(parts, fn part -> part != "" and Regex.match?(~r/^[a-z0-9_]+$/, part) end)
  end

  defp read_phrases!(path) do
    entries =
      case path |> File.read!() |> Jason.decode() do
        {:ok, list} when is_list(list) ->
          list

        other ->
          Mix.raise("rebuild_gold_standard: #{path} is not a JSON array: #{inspect(other) |> String.slice(0, 120)}")
      end

    entries
    |> Enum.map(fn entry ->
      entry
      |> Map.get("data", [])
      |> Enum.map_join(fn segment -> Map.get(segment, "text", "") end)
    end)
    |> Enum.reject(&(String.trim(&1) == ""))
  end

  # `account.balance.check - context: account` -> `account.balance.check`.
  # Austin's rule: anything with a hyphenated suffix is a subset of the version
  # without it, so the suffix names a context, not a distinct intent.
  defp root_label(name), do: name |> String.split(" - ") |> List.first() |> String.trim()

  defp fold_lights(@lights_prefix <> rest), do: @device_prefix <> rest
  defp fold_lights(label), do: label

  defp normalise(text), do: text |> String.trim() |> String.downcase() |> String.replace(~r/\s+/, " ")

  # -- building ---------------------------------------------------------------

  defp build_entries(phrases) do
    grouped = Enum.group_by(phrases, & &1.norm)

    {entries, conflicts} =
      grouped
      |> Enum.sort_by(fn {norm, _} -> norm end)
      |> Enum.map_reduce([], fn {_norm, occurrences}, conflicts ->
        labels = occurrences |> Enum.map(& &1.label) |> Enum.uniq() |> Enum.sort()
        first = hd(occurrences)
        sources = occurrences |> Enum.map(& &1.source_file) |> Enum.uniq() |> Enum.sort()

        case labels do
          [label] ->
            entry = %{
              "text" => first.text,
              "intent" => label,
              "labeled_by" => "dialogflow",
              "source_file" => hd(sources)
            }

            {entry, conflicts}

          _many ->
            # Genuinely ambiguous even after folding: the same sentence is filed
            # under two intents that are not a device/lights pair. Written as
            # needs_review with every candidate kept, never resolved by guessing
            # -- resolving it by asking a model is precisely what produced the
            # corpus this task is repairing.
            entry = %{
              "text" => first.text,
              "intent" => hd(labels),
              "labeled_by" => "dialogflow",
              "source_file" => hd(sources),
              "status" => "needs_review",
              "candidates" => labels
            }

            {entry, [%{text: first.text, candidates: labels, sources: sources} | conflicts]}
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

    compare_to_pre_rebuild(entries)

    if verbose? and conflicts != [] do
      Mix.shell().info("\n  Ambiguous after the lights -> device fold:")

      Enum.each(conflicts, fn c ->
        Mix.shell().info("    #{String.pad_trailing(String.slice(c.text, 0, 44), 46)} #{inspect(c.candidates)}")
      end)
    end
  end

  # The pre-rebuild snapshot is an independent witness: it predates the layer-2
  # contamination and preserves the fine-grained taxonomy. Agreement with it is
  # the acceptance number task 079 could not specify, because it did not know
  # the file existed.
  defp compare_to_pre_rebuild(entries) do
    path = Path.join(Path.dirname(EvaluationStore.gold_standard_path("intent")), "gold_standard.pre-rebuild.json")

    if File.exists?(path) do
      pre =
        path
        |> File.read!()
        |> Jason.decode!()
        |> Map.new(fn row -> {normalise(row["text"]), fold_lights(row["intent"])} end)

      shared =
        entries
        |> Enum.map(fn e -> {normalise(e["text"]), e["intent"]} end)
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

  defp write_output!(entries) do
    path = EvaluationStore.gold_standard_path("intent")

    if File.exists?(path) do
      stamp = DateTime.utc_now() |> DateTime.to_iso8601() |> String.replace(":", "-")
      backup = "#{path}.#{stamp}.bak"
      File.cp!(path, backup)
      Mix.shell().info("\n  Backed up to #{backup}")
    end

    File.write!(path, Jason.encode!(entries, pretty: true) <> "\n")
    Mix.shell().info("  Wrote #{length(entries)} entries to #{path}")
  end

  defp write_conflict_report!(conflicts) do
    path =
      EvaluationStore.gold_standard_path("intent")
      |> Path.dirname()
      |> Path.join("ambiguous_texts.json")

    File.write!(path, Jason.encode!(conflicts, pretty: true) <> "\n")
    Mix.shell().info("  Wrote #{length(conflicts)} ambiguous texts to #{path}\n")
  end

  defp banner(title) do
    Mix.shell().info("")
    Mix.shell().info(String.duplicate("=", 64))
    Mix.shell().info(title)
    Mix.shell().info(String.duplicate("=", 64))
  end
end
