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
  | `entity_mappings` | each parameter's `dataType`, minus the `@` |
  | `clarification_templates` | each parameter's first `prompts[].value` |
  | `defaults` | each parameter's non-empty `defaultValue` |
  | `input_contexts` | `contexts` |
  | `output_contexts` | `responses[0].affectedContexts` |

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
    {entries, undecided} = derive_all(missing, judgment)

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

  defp load_judgment(nil), do: %{}

  defp load_judgment(path) do
    unless File.regular?(path), do: Mix.raise("registry.derive: no judgment file at #{path}")

    path |> File.read!() |> Jason.decode!()
  end

  # -- derivation -------------------------------------------------------------

  defp derive_all(missing, judgment) do
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
          entry = build_entry(intent, definition, decided)

          if decided, do: {Map.put(entries, intent, entry), undecided},
            else: {Map.put(entries, intent, entry), [intent | undecided]}
      end
    end)
    |> then(fn {entries, undecided} -> {entries, Enum.reverse(undecided)} end)
  end

  defp definition_path(intent), do: Brain.data_path("intents/#{intent}.json")

  # Three places a definition can live:
  #
  #   * under the intent's own name
  #   * under its pre-fold name, for anything `mix rebuild_gold_standard` folded
  #     from `smarthome.lights.*`
  #   * under a ` - context_...` variant, for an intent that exists in the export
  #     only as a follow-up and has no base definition of its own
  #     (`smarthome.device.switch` is one)
  defp definition_for(intent) do
    exact =
      case intent do
        @device_prefix <> rest -> [definition_path(intent), definition_path(@lights_prefix <> rest)]
        _ -> [definition_path(intent)]
      end

    case Enum.find(exact, &File.regular?/1) || context_variant(intent) do
      nil -> nil
      path -> path |> File.read!() |> Jason.decode!()
    end
  end

  defp context_variant(intent) do
    names =
      case intent do
        @device_prefix <> rest -> [intent, @lights_prefix <> rest]
        _ -> [intent]
      end

    names
    |> Enum.flat_map(fn name -> Path.wildcard(Brain.data_path("intents/#{name} - *.json")) end)
    |> Enum.reject(&String.ends_with?(&1, "_usersays_en.json"))
    |> Enum.sort()
    |> List.first()
  end

  defp build_entry(intent, definition, decided) do
    params = definition |> Map.get("responses", []) |> List.first(%{}) |> Map.get("parameters", [])

    {required, optional} = Enum.split_with(params, &Map.get(&1, "required", false))

    base = %{
      "domain" => intent |> String.split(".") |> List.first(),
      "required" => Enum.map(required, & &1["name"]),
      "optional" => Enum.map(optional, & &1["name"]),
      "entity_mappings" => entity_mappings(params),
      "clarification_templates" => clarification_templates(params),
      "input_contexts" => input_contexts(definition),
      "output_contexts" => output_contexts(definition)
    }

    base
    |> maybe_put("defaults", defaults(params))
    |> Map.merge(judgment_fields(decided))
  end

  defp entity_mappings(params) do
    params
    |> Enum.flat_map(fn p ->
      case p["dataType"] do
        "@" <> type when type != "" -> [{p["name"], [type]}]
        _ -> []
      end
    end)
    |> Map.new()
  end

  defp clarification_templates(params) do
    params
    |> Enum.flat_map(fn p ->
      case p |> Map.get("prompts", []) |> List.first() do
        %{"value" => v} when is_binary(v) and v != "" -> [{p["name"], v}]
        _ -> []
      end
    end)
    |> Map.new()
  end

  defp defaults(params) do
    params
    |> Enum.flat_map(fn p ->
      case p["defaultValue"] do
        v when is_binary(v) and v != "" -> [{p["name"], v}]
        _ -> []
      end
    end)
    |> Map.new()
  end

  # Dialogflow input contexts carry no lifespan; the registry's existing entries
  # use 2 for theirs.
  defp input_contexts(definition) do
    definition
    |> Map.get("contexts", [])
    |> Enum.map(fn name -> %{"name" => name, "lifespan" => 2} end)
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
