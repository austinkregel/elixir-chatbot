defmodule Mix.Tasks.CleanupGoldStandard do
  @shortdoc "Clean up gold standard data: resolve label conflicts, normalize formats, deduplicate"
  @moduledoc """
  Applies transformation rules to the intent and sentiment gold standard
  data to resolve label contradictions, normalize naming conventions, and
  remove duplicates.

  ## Usage

      mix cleanup_gold_standard              # Clean intent gold standard + registry
      mix cleanup_gold_standard --sentiment  # Also deduplicate sentiment gold standard
      mix cleanup_gold_standard --dry-run    # Preview changes without writing

  ## Rules applied (in order)

  1. Merge smarthome.device.brightness/switch into smarthome.lights.*
  2. Merge news/web.search navigation into navigation.*
  3. Normalize music_player_control.* to music.player.*
  4. Promote flat "continuation" to "dialog.continuation"
  5. Fix known mislabels
  6. Deduplicate (same text + same label)

  Remaining conflicts (same text, different labels after cleanup) are
  printed for manual review but not removed.
  """

  use Mix.Task
  require Logger

  @device_to_lights_map %{
    "smarthome.device.brightness.down" => "smarthome.lights.brightness.down",
    "smarthome.device.brightness.up" => "smarthome.lights.brightness.up",
    "smarthome.device.brightness.set" => "smarthome.lights.brightness.set",
    "smarthome.device.brightness.check" => "smarthome.lights.brightness.check",
    "smarthome.device.brightness.down.implicit" => "smarthome.lights.brightness.down.implicit",
    "smarthome.device.brightness.up.implicit" => "smarthome.lights.brightness.up.implicit",
    "smarthome.device.brightness.check.implicit" => "smarthome.lights.brightness.check.implicit",
    "smarthome.device.brightness.schedule.up" => "smarthome.lights.brightness.schedule.up",
    "smarthome.device.brightness.schedule.down" => "smarthome.lights.brightness.schedule.down",
    "smarthome.device.switch.on" => "smarthome.lights.switch.on",
    "smarthome.device.switch.off" => "smarthome.lights.switch.off",
    "smarthome.device.switch.check.on" => "smarthome.lights.switch.check.on",
    "smarthome.device.switch.check.off" => "smarthome.lights.switch.check.off",
    "smarthome.device.switch.schedule.on" => "smarthome.lights.switch.schedule.on",
    "smarthome.device.switch.schedule.off" => "smarthome.lights.switch.schedule.off"
  }

  @nav_map %{
    "news.search - next" => "navigation.next",
    "news.search - previous" => "navigation.previous",
    "news.search - repeat" => "navigation.repeat",
    "web.search - next" => "navigation.next",
    "web.search - previous" => "navigation.previous",
    "web.search - repeat" => "navigation.repeat"
  }

  @music_map %{
    "music_player_control.play" => "music.player.play",
    "music_player_control.pause" => "music.player.pause",
    "music_player_control.stop" => "music.player.stop",
    "music_player_control.resume" => "music.player.resume",
    "music_player_control.repeat" => "music.player.repeat",
    "music_player_control.shuffle" => "music.player.shuffle",
    "music_player_control.skip_forward" => "music.player.skip_forward",
    "music_player_control.skip_backward" => "music.player.skip_backward",
    "music_player_control.add_favorite" => "music.player.add_favorite",
    "music_player_control.add_playlist" => "music.player.add_playlist"
  }

  @all_renames Map.merge(@device_to_lights_map, @nav_map) |> Map.merge(@music_map)

  @mislabels [
    {"lights brightness up to 15 percents in the bedroom", "smarthome.lights.brightness.down",
     "smarthome.lights.brightness.up"}
  ]

  @impl Mix.Task
  def run(args) do
    dry_run? = "--dry-run" in args
    do_sentiment? = "--sentiment" in args

    intent_path = intent_gold_standard_path()
    registry_path = intent_registry_path()

    Mix.shell().info("[cleanup] Loading intent gold standard from #{intent_path}")

    entries = load_json!(intent_path)
    original_count = length(entries)

    Mix.shell().info("[cleanup] Loaded #{original_count} entries")

    {entries, stats} = apply_all_rules(entries)

    print_stats(stats, original_count, length(entries))
    print_remaining_conflicts(entries)

    unless dry_run? do
      write_json!(intent_path, entries)
      Mix.shell().info("[cleanup] Wrote #{length(entries)} entries to #{intent_path}")

      cleanup_registry(registry_path)

      if do_sentiment? do
        cleanup_sentiment()
      end
    else
      Mix.shell().info("[cleanup] Dry run -- no files written")
    end
  end

  defp apply_all_rules(entries) do
    stats = %{rule1: 0, rule2: 0, rule3: 0, rule4: 0, rule5: 0, dupes_removed: 0}

    {entries, stats} = apply_renames(entries, stats)
    {entries, stats} = apply_mislabel_fixes(entries, stats)
    {entries, stats} = deduplicate(entries, stats)

    {entries, stats}
  end

  defp apply_renames(entries, stats) do
    Enum.reduce(entries, {[], stats}, fn entry, {acc, s} ->
      intent = Map.get(entry, "intent", "")

      cond do
        Map.has_key?(@device_to_lights_map, intent) ->
          new_intent = Map.fetch!(@device_to_lights_map, intent)
          {[Map.put(entry, "intent", new_intent) | acc], %{s | rule1: s.rule1 + 1}}

        Map.has_key?(@nav_map, intent) ->
          new_intent = Map.fetch!(@nav_map, intent)
          {[Map.put(entry, "intent", new_intent) | acc], %{s | rule2: s.rule2 + 1}}

        Map.has_key?(@music_map, intent) ->
          new_intent = Map.fetch!(@music_map, intent)
          {[Map.put(entry, "intent", new_intent) | acc], %{s | rule3: s.rule3 + 1}}

        intent == "continuation" ->
          {[Map.put(entry, "intent", "dialog.continuation") | acc], %{s | rule4: s.rule4 + 1}}

        true ->
          {[entry | acc], s}
      end
    end)
    |> then(fn {entries, stats} -> {Enum.reverse(entries), stats} end)
  end

  defp apply_mislabel_fixes(entries, stats) do
    Enum.reduce(entries, {[], stats}, fn entry, {acc, s} ->
      text = Map.get(entry, "text", "")
      intent = Map.get(entry, "intent", "")

      case Enum.find(@mislabels, fn {t, old, _new} -> t == text and old == intent end) do
        {_t, _old, new_intent} ->
          {[Map.put(entry, "intent", new_intent) | acc], %{s | rule5: s.rule5 + 1}}

        nil ->
          {[entry | acc], s}
      end
    end)
    |> then(fn {entries, stats} -> {Enum.reverse(entries), stats} end)
  end

  defp deduplicate(entries, stats) do
    {unique, _seen} =
      Enum.reduce(entries, {[], MapSet.new()}, fn entry, {acc, seen} ->
        key = {Map.get(entry, "text", ""), Map.get(entry, "intent", "")}

        if MapSet.member?(seen, key) do
          {acc, seen}
        else
          {[entry | acc], MapSet.put(seen, key)}
        end
      end)

    unique = Enum.reverse(unique)
    dupes_removed = length(entries) - length(unique)
    {unique, %{stats | dupes_removed: dupes_removed}}
  end

  defp print_stats(stats, original_count, final_count) do
    Mix.shell().info("[cleanup] Rule 1: Merged #{stats.rule1} entries from smarthome.device.* to smarthome.lights.*")
    Mix.shell().info("[cleanup] Rule 2: Merged #{stats.rule2} entries into navigation.*")
    Mix.shell().info("[cleanup] Rule 3: Renamed #{stats.rule3} entries from music_player_control.* to music.player.*")
    Mix.shell().info("[cleanup] Rule 4: Promoted #{stats.rule4} continuation entries to dialog.continuation")
    Mix.shell().info("[cleanup] Rule 5: Fixed #{stats.rule5} mislabel(s)")
    Mix.shell().info("[cleanup] Rule 6: Removed #{stats.dupes_removed} exact duplicates")
    Mix.shell().info("[cleanup] Total: #{original_count} -> #{final_count} entries")
  end

  defp print_remaining_conflicts(entries) do
    conflicts =
      entries
      |> Enum.group_by(&Map.get(&1, "text"))
      |> Enum.filter(fn {_text, group} ->
        group |> Enum.map(&Map.get(&1, "intent")) |> Enum.uniq() |> length() > 1
      end)
      |> Enum.sort_by(fn {_text, group} -> -length(group) end)

    if conflicts == [] do
      Mix.shell().info("[cleanup] No remaining conflicts!")
    else
      Mix.shell().info("[cleanup] Remaining conflicts (#{length(conflicts)} texts with multiple labels):")

      Enum.each(conflicts, fn {text, group} ->
        labels = group |> Enum.map(&Map.get(&1, "intent")) |> Enum.uniq() |> Enum.join(", ")
        Mix.shell().info("  - #{inspect(text)} -> #{labels}")
      end)
    end
  end

  defp cleanup_registry(registry_path) do
    Mix.shell().info("[cleanup] Updating intent registry at #{registry_path}")

    registry = load_json!(registry_path)

    renamed =
      Enum.reduce(@all_renames, registry, fn {old_key, new_key}, acc ->
        case Map.pop(acc, old_key) do
          {nil, acc} -> acc
          {value, acc} ->
            new_domain = new_key |> String.split(".") |> List.first()
            updated_value = Map.put(value, "domain", new_domain)
            Map.put(acc, new_key, updated_value)
        end
      end)

    renamed =
      case Map.pop(renamed, "continuation") do
        {nil, r} -> r
        {value, r} -> Map.put(r, "dialog.continuation", Map.put(value, "domain", "dialog"))
      end

    nav_template = %{
      "category" => "directive",
      "clarification_templates" => %{},
      "domain" => "navigation",
      "entity_mappings" => %{},
      "input_contexts" => [],
      "optional" => [],
      "output_contexts" => [],
      "required" => []
    }

    renamed =
      Enum.reduce(["navigation.next", "navigation.previous", "navigation.repeat"], renamed, fn key, acc ->
        Map.put_new(acc, key, nav_template)
      end)

    write_json!(registry_path, renamed)
    Mix.shell().info("[cleanup] Registry updated (#{map_size(renamed)} entries)")
  end

  defp cleanup_sentiment do
    path = sentiment_gold_standard_path()
    Mix.shell().info("[cleanup] Deduplicating sentiment gold standard at #{path}")

    entries = load_json!(path)
    original_count = length(entries)

    unique =
      entries
      |> Enum.reduce({[], MapSet.new()}, fn entry, {acc, seen} ->
        key = {Map.get(entry, "text", ""), Map.get(entry, "sentiment", "")}

        if MapSet.member?(seen, key) do
          {acc, seen}
        else
          {[entry | acc], MapSet.put(seen, key)}
        end
      end)
      |> elem(0)
      |> Enum.reverse()

    removed = original_count - length(unique)
    write_json!(path, unique)
    Mix.shell().info("[cleanup] Sentiment: removed #{removed} duplicates (#{original_count} -> #{length(unique)})")
  end

  defp load_json!(path) do
    path
    |> File.read!()
    |> Jason.decode!()
  end

  defp write_json!(path, data) do
    json = Jason.encode!(data, pretty: true)
    File.write!(path, json <> "\n")
  end

  defp intent_gold_standard_path do
    Path.join([brain_source_root(), "priv", "evaluation", "intent", "gold_standard.json"])
  end

  defp sentiment_gold_standard_path do
    Path.join([brain_source_root(), "priv", "evaluation", "sentiment", "gold_standard.json"])
  end

  defp intent_registry_path do
    Path.join([brain_source_root(), "priv", "analysis", "intent_registry.json"])
  end

  defp brain_source_root do
    umbrella_root = File.cwd!()
    source = Path.join(umbrella_root, "apps/brain")
    if File.dir?(source), do: source, else: umbrella_root
  end
end
