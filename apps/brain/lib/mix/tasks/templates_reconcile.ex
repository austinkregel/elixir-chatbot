defmodule Mix.Tasks.Templates.Reconcile do
  @shortdoc "Align templates.json keys with the intent registry using declared transformations"

  @moduledoc """
  Brings `priv/response/templates.json` into line with `priv/analysis/intent_registry.json`.

      mix templates.reconcile          # report only
      mix templates.reconcile --save   # apply the resolved renames

  ## What it applies

  The same two transformations `mix rebuild_gold_standard` applied to the corpus,
  and nothing else:

    * a ` - context: …` suffix is stripped, since it names a context of an intent
      rather than a separate intent
    * `smarthome.lights.X` becomes `smarthome.device.X`, lights being a kind of
      device

  When the transformed key already has templates the two lists are merged,
  de-duplicated on `{text, condition}`. Otherwise the entry moves.

  ## What it refuses to do

  A key that neither transformation resolves is **left alone and reported**. It is
  not deleted, and no rename is invented for it. A lookup table of one-off
  renames is how `cleanup_gold_standard` accumulated 123 rules and reproduced 96%
  of the collapse task 079 blamed on a classifier; a template key is not worth
  starting a second one.

  Deciding those requires knowing whether a label is still reachable, and the
  registry alone does not answer that — the pipeline can emit labels that are not
  registry keys. `"question.factual"` is a literal default at
  `Brain.Analysis.Pipeline`'s speech-act mapping, and `"unknown"` is the value
  most of `speech_act_intent_map.json` routes to. Both have templates, both would
  look dead by a registry check, and deleting either would remove the response for
  unrecognised input.

  So the report grades each unresolved key by where it is referenced, and
  distinguishes a reference in a *rename map* from a reference on a live path. A
  key cited only by `cleanup_gold_standard.ex` is cited by a table of labels
  someone already renamed away from, which is evidence it is dead rather than
  evidence it is live.
  """

  use Mix.Task

  @requirements ["app.config"]

  @switches [save: :boolean]

  @registry_path "analysis/intent_registry.json"
  @templates_path "response/templates.json"
  @speech_act_map_path "analysis/speech_act_intent_map.json"

  @lights_prefix "smarthome.lights."
  @device_prefix "smarthome.device."

  # Referenced only from rename tables, so a hit here is evidence a label was
  # renamed away from, not evidence anything still emits it.
  @rename_map_files ~w(cleanup_gold_standard.ex normalize_gold_standard.ex rebuild_gold_standard.ex)

  # This task quotes example labels in its own moduledoc, which would otherwise
  # grade them as live.
  @self "templates_reconcile.ex"

  # A label short enough to occur as a domain or a bare word cannot be graded by
  # grep: `"weather"` appears wherever the weather *domain* is named. Graded
  # UNGRADEABLE rather than reported as reachable on a match that proves nothing.
  @ungradeable_by_grep ~w(weather unknown message smarthome)

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("templates.reconcile: unknown options #{inspect(invalid)}")
    if positional != [], do: Mix.raise("templates.reconcile: unexpected arguments #{inspect(positional)}")

    registry = read_json!(@registry_path)
    templates = read_json!(@templates_path)
    sa_targets = read_json!(@speech_act_map_path) |> Map.values() |> MapSet.new()

    orphans = templates |> Map.keys() |> Enum.reject(&Map.has_key?(registry, &1)) |> Enum.sort()

    {resolved, unresolved} =
      Enum.split_with(orphans, fn key -> Map.has_key?(registry, transform(key)) end)

    report(templates, registry, resolved, unresolved, sa_targets, load_sources())

    if opts[:save] do
      save!(templates, resolved)
    else
      Mix.shell().info("\n  Report only. Pass --save to apply the #{length(resolved)} resolved renames.\n")
    end
  end

  # -- transformations --------------------------------------------------------

  defp transform(key), do: key |> strip_context() |> fold_lights()

  defp strip_context(key), do: key |> String.split(" - ") |> List.first() |> String.trim()

  defp fold_lights(@lights_prefix <> rest), do: @device_prefix <> rest
  defp fold_lights(key), do: key

  # -- applying ---------------------------------------------------------------

  defp save!(templates, resolved) do
    {merged, moved, combined} =
      Enum.reduce(resolved, {templates, 0, 0}, fn key, {acc, moved, combined} ->
        target = transform(key)
        incoming = get_in(acc, [key, "templates"]) || []

        case Map.get(acc, target) do
          nil ->
            {acc |> Map.put(target, %{"templates" => incoming}) |> Map.delete(key), moved + 1, combined}

          %{"templates" => existing} ->
            union = dedupe(existing ++ incoming)
            {acc |> Map.put(target, %{"templates" => union}) |> Map.delete(key), moved, combined + 1}
        end
      end)

    path = priv(@templates_path)
    File.write!(path, encode_sorted!(merged) <> "\n")

    Mix.shell().info("")
    Mix.shell().info("  moved   #{moved}  (target had no templates)")
    Mix.shell().info("  merged  #{combined}  (target already had templates)")
    Mix.shell().info("  keys    #{map_size(templates)} -> #{map_size(merged)}")
    Mix.shell().info("")
  end

  defp dedupe(templates) do
    Enum.uniq_by(templates, fn t -> {Map.get(t, "text"), Map.get(t, "condition")} end)
  end

  # `Jason.encode!` emits a map in Erlang's iteration order, which is stable for a
  # given map but not for a given *content*: adding or removing one key reshuffles
  # the whole file. Writing 242 keys that way turns a 24-key rename into a 3,100
  # line diff, so the change cannot be reviewed and two runs producing the same
  # templates do not produce the same file. Sorting every object makes the output a
  # function of the content alone.
  defp encode_sorted!(data), do: data |> sort_deep() |> Jason.encode!(pretty: true)

  defp sort_deep(%{} = map) when not is_struct(map) do
    %Jason.OrderedObject{
      values: map |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(fn {k, v} -> {k, sort_deep(v)} end)
    }
  end

  defp sort_deep(list) when is_list(list), do: Enum.map(list, &sort_deep/1)
  defp sort_deep(other), do: other

  # -- reporting --------------------------------------------------------------

  defp report(templates, registry, resolved, unresolved, sa_targets, sources) do
    corpus_missing = corpus_intents_without_templates(templates, registry, resolved)

    Mix.shell().info("")
    Mix.shell().info(String.duplicate("=", 72))
    Mix.shell().info("TEMPLATES vs REGISTRY")
    Mix.shell().info(String.duplicate("=", 72))
    Mix.shell().info("  template keys          #{map_size(templates)}")
    Mix.shell().info("  registry entries       #{map_size(registry)}")
    Mix.shell().info("  keys not in registry   #{length(resolved) + length(unresolved)}")
    Mix.shell().info("    resolved by transform  #{length(resolved)}")
    Mix.shell().info("    unresolved             #{length(unresolved)}")
    Mix.shell().info("  registry intents with no template, after applying: #{corpus_missing}")
    Mix.shell().info("")

    if resolved != [] do
      Mix.shell().info("  RESOLVED — these move or merge onto a registry key:")

      Enum.each(resolved, fn key ->
        target = transform(key)
        n = length(get_in(templates, [key, "templates"]) || [])
        existing = length(get_in(templates, [target, "templates"]) || [])
        verb = if existing > 0, do: "merge into #{existing}", else: "move"
        Mix.shell().info("    #{String.pad_trailing(key, 46)} #{n} tmpl  -> #{target}  (#{verb})")
      end)

      Mix.shell().info("")
    end

    if unresolved != [] do
      Mix.shell().info("  UNRESOLVED — left untouched, each needs a decision:")
      Mix.shell().info("")

      unresolved
      |> Enum.map(fn key -> {key, grade(key, sa_targets, sources)} end)
      |> Enum.group_by(fn {_k, {g, _w}} -> g end)
      |> Enum.sort_by(fn {g, _} -> order(g) end)
      |> Enum.each(fn {grade, entries} ->
        Mix.shell().info("    #{grade} (#{length(entries)}):")

        Enum.each(entries, fn {key, {_g, where}} ->
          n = length(get_in(templates, [key, "templates"]) || [])
          Mix.shell().info("      #{String.pad_trailing(key, 44)} #{n} tmpl   #{where}")
        end)

        Mix.shell().info("")
      end)
    end
  end

  # How a label earns "still reachable": named on a live path, or routed to by the
  # speech-act map. A hit only in a rename table is the opposite of evidence.
  defp grade(key, sa_targets, sources) do
    cond do
      MapSet.member?(sa_targets, key) ->
        {"REACHABLE — speech_act_intent_map routes to it", "keep"}

      key in @ungradeable_by_grep ->
        {"UNGRADEABLE — too generic to grep for", "decide by hand"}

      true ->
        case live_references(key, sources) do
          [] ->
            case rename_map_references(key, sources) do
              [] -> {"NO REFERENCE ANYWHERE", "-"}
              files -> {"RENAME-TABLE ONLY — renamed away from", Enum.join(files, ", ")}
            end

          files ->
            {"REACHABLE — named on a live path", Enum.join(files, ", ")}
        end
    end
  end

  defp order("REACHABLE — speech_act_intent_map routes to it"), do: 0
  defp order("REACHABLE — named on a live path"), do: 1
  defp order("RENAME-TABLE ONLY — renamed away from"), do: 2
  defp order(_), do: 3

  defp live_references(key, sources) do
    key |> referencing_files(sources) |> Enum.reject(&(&1 in @rename_map_files))
  end

  defp rename_map_references(key, sources) do
    key |> referencing_files(sources) |> Enum.filter(&(&1 in @rename_map_files))
  end

  # Matches the label as a complete quoted string, so `"music.stop"` does not hit
  # on `"music.stop_playback"`. It cannot tell an intent label from a same-named
  # domain, which is what @ungradeable_by_grep exists for.
  defp referencing_files(key, sources) do
    needle = ~s("#{key}")

    for {basename, contents} <- sources,
        basename != @self,
        String.contains?(contents, needle),
        do: basename
  end

  # Read every source once and keep it, rather than shelling out per label: 42
  # labels against three app trees is 42 subprocesses otherwise.
  defp load_sources do
    ~w(apps/brain/lib apps/chat_web/lib apps/world/lib)
    |> Enum.flat_map(fn dir -> Path.wildcard(Path.join(dir, "**/*.ex")) end)
    |> Enum.map(fn path -> {Path.basename(path), File.read!(path)} end)
  end

  defp corpus_intents_without_templates(templates, registry, resolved) do
    after_apply =
      Enum.reduce(resolved, MapSet.new(Map.keys(templates)), fn key, acc ->
        acc |> MapSet.delete(key) |> MapSet.put(transform(key))
      end)

    registry |> Map.keys() |> Enum.count(&(not MapSet.member?(after_apply, &1)))
  end

  # -- io ---------------------------------------------------------------------

  defp priv(rel), do: Brain.priv_path(rel)

  defp read_json!(rel) do
    path = priv(rel)

    unless File.regular?(path), do: Mix.raise("templates.reconcile: no file at #{path}")

    path |> File.read!() |> Jason.decode!()
  end
end
