defmodule Mix.Tasks.Registry.Derive do
  @shortdoc "Derive intent_registry.json entries for corpus intents that have none"

  @moduledoc """
  Builds registry entries for every intent in the gold standard that
  `intent_registry.json` does not list, taking each field from the Dialogflow
  export rather than inventing it.

      mix registry.derive                        # write the candidate file
      mix registry.derive --judgment PATH        # merge in category/speech_act/description
      mix registry.derive --judgment PATH --save # merge into intent_registry.json

  ## Why this exists

  `Brain.Analysis.Pipeline.determine_intent/6` gates on
  `registered_intent?/1`. When the classifier emits a label the registry does not
  list, the pipeline discards it and infers from the speech act instead. Measured
  on 1,000 held-out rows: 475 carry a label the registry lacks, and the pipeline
  turns 256 correct predictions into wrong ones, which is the whole gap between
  the classifier's 38.9% and the pipeline's 15.5%.

  ## What is derived, and from where

  Everything except three fields comes out of `data/intents/<intent>.json`:

  | registry field | Dialogflow source |
  |---|---|
  | `domain` | first dotted part of the name |
  | `required` | `parameters` with `required: true` |
  | `optional` | `parameters` with `required: false` |
  | `entity_mappings` | each parameter's `dataType`, as the runtime entity type it stands for |
  | `clarification_templates` | each parameter's first `prompts[].value` |
  | `defaults` | each parameter's non-empty `defaultValue` |
  | `input_contexts` | `contexts`, plus each `followupContexts[].context` |
  | `output_contexts` | `responses[0].affectedContexts` |

  ## Slot types

  A slot's types are runtime entity types that `analysis/entity_types.json`
  declares in its `type_hierarchy`, which is what an extracted entity is
  compared with. A `dataType` of the export's own (`@room`) is that type, and
  raises when the hierarchy does not declare it. A Dialogflow system entity
  (`@sys.number`) is the runtime type `config.export_system_types` maps it to,
  and raises when it maps it to none (`@sys.any`, any text, has no type) or
  when the slot's name is a declared type the mapped type is not related to in
  the hierarchy: a `start_year` slot typed `@sys.number` is a year, not any
  number, and that is not the export's to say.

  Slot names, entity types and context names are written with underscores in
  the export and the registry alike (`final_value`, `sys.date_time`,
  `device_volume`). A hyphenated name raises rather than being copied, since a
  slot name becomes an entity type at runtime and `analysis/entity_types.json`
  does not accept hyphens.

  `category`, `speech_act` and `description` are **not** derivable. `category` is
  one of three values and `speech_act` one of fourteen, and neither follows from
  the domain -- smalltalk alone uses ten distinct pairs in the existing registry.
  Those come from a judgment file, keyed by intent.

  An intent folded by `mix rebuild_gold_standard` keeps its definition under the
  pre-fold name, so `smarthome.device.hue.check` reads
  `smarthome.lights.hue.check.json`.
  """

  use Mix.Task

  alias Brain.ML.EvaluationStore

  @requirements ["app.config"]

  @switches [judgment: :string, save: :boolean, out: :string]

  @registry_path "analysis/intent_registry.json"
  @entity_types_path "analysis/entity_types.json"
  @device_prefix "smarthome.device."
  @lights_prefix "smarthome.lights."

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("registry.derive: unknown options #{inspect(invalid)}")
    if positional != [], do: Mix.raise("registry.derive: unexpected arguments #{inspect(positional)}")

    registry = load_registry!()
    corpus = corpus_intents!()
    missing = corpus |> MapSet.difference(MapSet.new(Map.keys(registry))) |> Enum.sort()

    Mix.shell().info("  corpus intents        #{MapSet.size(corpus)}")
    Mix.shell().info("  registry entries      #{map_size(registry)}")
    Mix.shell().info("  missing from registry #{length(missing)}")

    judgment = load_judgment(opts[:judgment])
    {entries, undecided} = derive_all(missing, judgment, load_entity_types!())

    report(entries, undecided, judgment)

    cond do
      opts[:save] && undecided != [] ->
        Mix.raise("""
        registry.derive: #{length(undecided)} intent(s) have no category/speech_act judgment.

        Refusing to write a registry entry with a guessed category. Supply them via
        --judgment and run again.
        """)

      opts[:save] ->
        save!(registry, entries)

      true ->
        write_candidates!(entries, opts[:out])
    end
  end

  # -- inputs -----------------------------------------------------------------

  defp load_registry! do
    Brain.priv_path(@registry_path) |> File.read!() |> Jason.decode!()
  end

  defp corpus_intents! do
    "intent"
    |> EvaluationStore.load_gold_standard(:all)
    |> MapSet.new(& &1["intent"])
  end

  # The declared runtime types, each with the types above it, and the runtime
  # type each export system entity stands for.
  @doc false
  def load_entity_types! do
    path = Brain.priv_path(@entity_types_path)
    data = path |> File.read!() |> Jason.decode!()

    hierarchy =
      case data do
        %{"type_hierarchy" => %{} = hierarchy} -> hierarchy
        _ -> Mix.raise("registry.derive: #{path} has no type_hierarchy object")
      end

    system =
      case get_in(data, ["config", "export_system_types"]) do
        %{} = mappings -> Map.delete(mappings, "description")
        other -> Mix.raise("registry.derive: #{path} has no config.export_system_types object, got #{inspect(other)}")
      end

    parents =
      Enum.reduce(hierarchy, %{}, fn {parent, children}, acc ->
        acc = Map.put_new(acc, parent, [])
        Enum.reduce(children, acc, fn child, acc -> Map.update(acc, child, [parent], &[parent | &1]) end)
      end)

    entity_types = %{parents: parents, system: system, path: path}

    Enum.each(system, fn {export, type} ->
      unless is_binary(type) and Map.has_key?(parents, type) do
        Mix.raise("registry.derive: #{path} maps #{export} to #{inspect(type)}, which its type_hierarchy does not declare")
      end
    end)

    entity_types
  end

  defp load_judgment(nil), do: %{}

  defp load_judgment(path) do
    unless File.regular?(path), do: Mix.raise("registry.derive: no judgment file at #{path}")

    path |> File.read!() |> Jason.decode!()
  end

  # -- derivation -------------------------------------------------------------

  defp derive_all(missing, judgment, entity_types) do
    Enum.reduce(missing, {%{}, []}, fn intent, {entries, undecided} ->
      case definition_for(intent) do
        nil ->
          Mix.raise("""
          registry.derive: no Dialogflow definition for #{intent}.

          Looked for #{definition_path(intent)} and its pre-fold name. Every corpus
          intent comes from the export, so a missing definition means the export and
          the gold standard disagree -- rebuild with `mix rebuild_gold_standard`.
          """)

        definition ->
          decided = Map.get(judgment, intent)
          entry = build_entry(intent, definition, decided, entity_types)

          if decided, do: {Map.put(entries, intent, entry), undecided},
            else: {Map.put(entries, intent, entry), [intent | undecided]}
      end
    end)
    |> then(fn {entries, undecided} -> {entries, Enum.reverse(undecided)} end)
  end

  defp definition_path(intent), do: Brain.data_path("intents/#{intent}.json")

  # Two places a definition can live: under the intent's own name, or under its
  # pre-fold name for anything `mix rebuild_gold_standard` folded from
  # `smarthome.lights.*`. Follow-up variants are collapsed into their intent's
  # definition (`followupContexts`), so there is no variant file to look for.
  defp definition_for(intent) do
    paths =
      case intent do
        @device_prefix <> rest -> [definition_path(intent), definition_path(@lights_prefix <> rest)]
        _ -> [definition_path(intent)]
      end

    case Enum.find(paths, &File.regular?/1) do
      nil -> nil
      path -> path |> File.read!() |> Jason.decode!()
    end
  end

  defp build_entry(intent, definition, decided, entity_types) do
    params = definition |> Map.get("responses", []) |> List.first(%{}) |> Map.get("parameters", [])

    {required, optional} = Enum.split_with(params, &Map.get(&1, "required", false))

    base = %{
      "domain" => intent |> String.split(".") |> List.first(),
      "required" => Enum.map(required, &canonical(&1["name"])),
      "optional" => Enum.map(optional, &canonical(&1["name"])),
      "entity_mappings" => entity_mappings(params, entity_types),
      "clarification_templates" => clarification_templates(params),
      "input_contexts" => input_contexts(definition),
      "output_contexts" => output_contexts(definition)
    }

    base
    |> maybe_put("defaults", defaults(params))
    |> Map.merge(judgment_fields(decided))
  end

  @doc false
  # Each parameter's slot and the runtime type its `dataType` stands for (see
  # the moduledoc's "Slot types").
  def entity_mappings(params, entity_types) do
    params
    |> Enum.flat_map(fn p ->
      case p["dataType"] do
        "@" <> type when type != "" ->
          slot = canonical(p["name"])
          [{slot, [runtime_type!(canonical(type), slot, entity_types)]}]

        _ ->
          []
      end
    end)
    |> Map.new()
  end

  defp runtime_type!("sys." <> _ = system, slot, %{system: system_types, path: path} = entity_types) do
    case Map.fetch(system_types, system) do
      {:ok, type} ->
        if Map.has_key?(entity_types.parents, slot) and not related?(type, slot, entity_types.parents) do
          Mix.raise(
            "registry.derive: the slot #{slot} is typed @#{system}, which stands for #{type}; #{slot} is a type " <>
              "#{path} declares, unrelated to #{type} in its type_hierarchy, so the slot's type is not the export's " <>
              "to say. Type the slot by hand."
          )
        end

        type

      :error ->
        Mix.raise(
          "registry.derive: the slot #{slot} is typed @#{system}, a system entity #{path}'s " <>
            "config.export_system_types maps to no runtime type. Map it there, or type the slot by hand."
        )
    end
  end

  defp runtime_type!(type, slot, %{parents: parents, path: path}) do
    unless Map.has_key?(parents, type) do
      Mix.raise("registry.derive: the slot #{slot} is typed @#{type}, which #{path}'s type_hierarchy does not declare")
    end

    type
  end

  # Whether one type is the other or lies above or below it in the hierarchy.
  defp related?(a, b, parents), do: b in ancestors(a, parents) or a in ancestors(b, parents)

  defp ancestors(type, parents) do
    parents
    |> Map.get(type, [])
    |> Enum.reduce([type], fn parent, acc -> acc ++ ancestors(parent, parents) end)
  end

  defp clarification_templates(params) do
    params
    |> Enum.flat_map(fn p ->
      case p |> Map.get("prompts", []) |> List.first() do
        %{"value" => v} when is_binary(v) and v != "" -> [{canonical(p["name"]), v}]
        _ -> []
      end
    end)
    |> Map.new()
  end

  defp defaults(params) do
    params
    |> Enum.flat_map(fn p ->
      case p["defaultValue"] do
        v when is_binary(v) and v != "" -> [{canonical(p["name"]), v}]
        _ -> []
      end
    end)
    |> Map.new()
  end

  # Names in the export are written with underscores; a hyphenated one is data
  # that was not migrated, and copying it would bring a second spelling back.
  defp canonical(name) when is_binary(name) do
    if String.contains?(name, "-") do
      Mix.raise("registry.derive: #{inspect(name)} contains a hyphen; names in data/intents use underscores")
    end

    name
  end

  # The contexts the intent can continue from: its own required `contexts` and
  # the contexts of the follow-ups collapsed into it (`followupContexts`).
  # Dialogflow input contexts carry no lifespan; the registry's existing entries
  # use 2 for theirs.
  defp input_contexts(definition) do
    followups = definition |> Map.get("followupContexts", []) |> Enum.map(&Map.fetch!(&1, "context"))

    (Map.get(definition, "contexts", []) ++ followups)
    |> Enum.uniq()
    |> Enum.map(fn name -> %{"name" => canonical(name), "lifespan" => 2} end)
  end

  defp output_contexts(definition) do
    definition
    |> Map.get("responses", [])
    |> List.first(%{})
    |> Map.get("affectedContexts", [])
    |> Enum.map(fn ctx -> %{"name" => ctx["name"], "lifespan" => ctx["lifespan"]} end)
  end

  defp judgment_fields(nil), do: %{}

  defp judgment_fields(decided) do
    %{}
    |> maybe_put("category", decided["category"])
    |> maybe_put("speech_act", decided["speech_act"])
    |> maybe_put("description", decided["description"])
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, _key, empty) when empty == %{}, do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  # -- output -----------------------------------------------------------------

  defp write_candidates!(entries, out) do
    path = out || Path.join(System.tmp_dir!(), "registry_candidates.json")
    File.write!(path, Jason.encode!(entries, pretty: true) <> "\n")
    Mix.shell().info("\n  Wrote #{map_size(entries)} candidate entries to #{path}")
    Mix.shell().info("  Nothing merged. Pass --judgment PATH --save to write the registry.\n")
  end

  defp save!(registry, entries) do
    path = Brain.priv_path(@registry_path)
    stamp = DateTime.utc_now() |> DateTime.to_iso8601() |> String.replace(":", "-")
    File.cp!(path, "#{path}.#{stamp}.bak")

    merged = Map.merge(registry, entries)
    File.write!(path, Jason.encode!(merged, pretty: true) <> "\n")

    Mix.shell().info("\n  Backed up to #{path}.#{stamp}.bak")
    Mix.shell().info("  #{map_size(registry)} + #{map_size(entries)} = #{map_size(merged)} entries\n")
  end

  defp report(entries, undecided, judgment) do
    slotted = Enum.count(entries, fn {_, e} -> e["required"] != [] or e["optional"] != [] end)
    prompts = Enum.count(entries, fn {_, e} -> e["clarification_templates"] != %{} end)

    Mix.shell().info("")
    Mix.shell().info("  derived               #{map_size(entries)}")
    Mix.shell().info("    with slots          #{slotted}")
    Mix.shell().info("    with prompts        #{prompts}")
    Mix.shell().info("  judgment supplied     #{map_size(judgment)}")
    Mix.shell().info("  still undecided       #{length(undecided)}")

    if undecided != [] do
      Mix.shell().info("")
      Mix.shell().info("  no category/speech_act for:")
      Enum.each(Enum.take(undecided, 12), fn i -> Mix.shell().info("    #{i}") end)
      if length(undecided) > 12, do: Mix.shell().info("    ... and #{length(undecided) - 12} more")
    end
  end
end
