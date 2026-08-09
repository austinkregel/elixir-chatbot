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

  0. Strip context-variant suffixes (`*_-_context:*` -> base intent)
  1. Merge smarthome.device.brightness/switch into smarthome.lights.*
  2. Merge news/web.search navigation variants into base search intents
  3. Normalize music_player_control.* to music.player.*
  4. Promote flat "continuation" to "dialog.continuation"
  5. Fix known mislabels
  6. Rename extra labels + consolidate fine-grained smalltalk into broad
     social-formula labels (greet, farewell, thanks, apology, compliment,
     acknowledgment) + remap phantom labels + roll up smalltalk.user.* mood/affect
     into smalltalk.user.emotion
  7. Consolidate 62 smarthome.* intents into 6 action intents (device_up/down/set/check,
     switch, schedule_create)
  8. Deduplicate (same text + same label)

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
    "news.search - next" => "news.search",
    "news.search - previous" => "news.search",
    "news.search - repeat" => "news.search",
    "news.search_-_next" => "news.search",
    "news.search_-_previous" => "news.search",
    "news.search_-_repeat" => "news.search",
    "web.search - next" => "web.search",
    "web.search - previous" => "web.search",
    "web.search - repeat" => "web.search"
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

  @consolidation_renames %{
    "smalltalk.greetings.hello" => "smalltalk.greet",
    "smalltalk.greetings.how_are_you" => "smalltalk.greet",
    "smalltalk.greetings.whatsup" => "smalltalk.greet",
    "smalltalk.greetings.nice_to_meet_you" => "smalltalk.greet",
    "smalltalk.greetings.nice_to_see_you" => "smalltalk.greet",
    "smalltalk.greetings.goodnight" => "smalltalk.greet",
    "smalltalk.greetings.goodmorning" => "smalltalk.greet",
    "smalltalk.greetings.goodevening" => "smalltalk.greet",
    "smalltalk.greetings.nice_to_talk_to_you" => "smalltalk.greet",
    "smalltalk.greetings.bye" => "smalltalk.farewell",
    "smalltalk.appraisal.thank_you" => "smalltalk.thanks",
    "smalltalk.dialog.sorry" => "smalltalk.apology",
    "smalltalk.user.likes_agent" => "smalltalk.compliment",
    "smalltalk.agent.beautiful" => "smalltalk.compliment",
    "smalltalk.agent.clever" => "smalltalk.compliment",
    "smalltalk.agent.funny" => "smalltalk.compliment",
    "smalltalk.appraisal.no_problem" => "smalltalk.acknowledgment"
  }

  # Mood / affect / self-reported state: fine-grained labels are redundant with the
  # pipeline's sentiment and tonal dimensions.
  @smalltalk_user_emotion_renames %{
    "smalltalk.user.angry" => "smalltalk.user.emotion",
    "smalltalk.user.bored" => "smalltalk.user.emotion",
    "smalltalk.user.busy" => "smalltalk.user.emotion",
    "smalltalk.user.can_not_sleep" => "smalltalk.user.emotion",
    "smalltalk.user.does_not_want_to_talk" => "smalltalk.user.emotion",
    "smalltalk.user.excited" => "smalltalk.user.emotion",
    "smalltalk.user.good" => "smalltalk.user.emotion",
    "smalltalk.user.happy" => "smalltalk.user.emotion",
    "smalltalk.user.joking" => "smalltalk.user.emotion",
    "smalltalk.user.lonely" => "smalltalk.user.emotion",
    "smalltalk.user.loves_agent" => "smalltalk.user.emotion",
    "smalltalk.user.misses_agent" => "smalltalk.user.emotion",
    "smalltalk.user.sad" => "smalltalk.user.emotion",
    "smalltalk.user.sleepy" => "smalltalk.user.emotion",
    "smalltalk.user.tired" => "smalltalk.user.emotion"
  }

  @phantom_renames %{
    "music.stop" => "music.player.stop",
    "music.skip" => "music.player.skip_forward",
    "news.query" => "news.search",
    "search.web" => "web.search",
    "outfit.recommend" => "weather.outfit",
    "account.query" => "account.balance.check",
    "smalltalk.user.name" => "smalltalk.user.introduction"
  }

  @smarthome_consolidation %{
    "smarthome.device.volume.check" => "smarthome.device_check",
    "smarthome.device.volume.check.implicit" => "smarthome.device_check",
    "smarthome.device.volume.down" => "smarthome.device_down",
    "smarthome.device.volume.down.implicit" => "smarthome.device_down",
    "smarthome.device.volume.mute" => "smarthome.switch",
    "smarthome.device.volume.mute.implicit" => "smarthome.switch",
    "smarthome.device.volume.set" => "smarthome.device_set",
    "smarthome.device.volume.unmute" => "smarthome.switch",
    "smarthome.device.volume.unmute.implicit" => "smarthome.switch",
    "smarthome.device.volume.up" => "smarthome.device_up",
    "smarthome.device.volume.up.implicit" => "smarthome.device_up",
    "smarthome.heating.check" => "smarthome.device_check",
    "smarthome.heating.check.implicit" => "smarthome.device_check",
    "smarthome.heating.down" => "smarthome.device_down",
    "smarthome.heating.down.implicit" => "smarthome.device_down",
    "smarthome.heating.schedule.down" => "smarthome.schedule_create",
    "smarthome.heating.schedule.set" => "smarthome.schedule_create",
    "smarthome.heating.schedule.up" => "smarthome.schedule_create",
    "smarthome.heating.set" => "smarthome.device_set",
    "smarthome.heating.switch.off" => "smarthome.switch",
    "smarthome.heating.switch.on" => "smarthome.switch",
    "smarthome.heating.switch.schedule.off" => "smarthome.schedule_create",
    "smarthome.heating.switch.schedule.on" => "smarthome.schedule_create",
    "smarthome.heating.up" => "smarthome.device_up",
    "smarthome.heating.up.implicit" => "smarthome.device_up",
    "smarthome.lights.brightness.check" => "smarthome.device_check",
    "smarthome.lights.brightness.check.implicit" => "smarthome.device_check",
    "smarthome.lights.brightness.down" => "smarthome.device_down",
    "smarthome.lights.brightness.down.implicit" => "smarthome.device_down",
    "smarthome.lights.brightness.schedule.down" => "smarthome.schedule_create",
    "smarthome.lights.brightness.schedule.up" => "smarthome.schedule_create",
    "smarthome.lights.brightness.set" => "smarthome.device_set",
    "smarthome.lights.brightness.up" => "smarthome.device_up",
    "smarthome.lights.brightness.up.implicit" => "smarthome.device_up",
    "smarthome.lights.hue.check" => "smarthome.device_check",
    "smarthome.lights.hue.down" => "smarthome.device_down",
    "smarthome.lights.hue.set" => "smarthome.device_set",
    "smarthome.lights.hue.up" => "smarthome.device_up",
    "smarthome.lights.saturation.check" => "smarthome.device_check",
    "smarthome.lights.saturation.down" => "smarthome.device_down",
    "smarthome.lights.saturation.set" => "smarthome.device_set",
    "smarthome.lights.saturation.up" => "smarthome.device_up",
    "smarthome.lights.switch.check" => "smarthome.switch",
    "smarthome.lights.switch.check.off" => "smarthome.switch",
    "smarthome.lights.switch.check.on" => "smarthome.switch",
    "smarthome.lights.switch.off" => "smarthome.switch",
    "smarthome.lights.switch.on" => "smarthome.switch",
    "smarthome.lights.switch.schedule.off" => "smarthome.schedule_create",
    "smarthome.lights.switch.schedule.on" => "smarthome.schedule_create",
    "smarthome.locks.check" => "smarthome.switch",
    "smarthome.locks.check.close" => "smarthome.switch",
    "smarthome.locks.check.lock" => "smarthome.switch",
    "smarthome.locks.check.open" => "smarthome.switch",
    "smarthome.locks.check.unlock" => "smarthome.switch",
    "smarthome.locks.close" => "smarthome.switch",
    "smarthome.locks.lock" => "smarthome.switch",
    "smarthome.locks.open" => "smarthome.switch",
    "smarthome.locks.schedule.close" => "smarthome.schedule_create",
    "smarthome.locks.schedule.lock" => "smarthome.schedule_create",
    "smarthome.locks.schedule.open" => "smarthome.schedule_create",
    "smarthome.locks.schedule.unlock" => "smarthome.schedule_create",
    "smarthome.locks.unlock" => "smarthome.switch"
  }

  @smarthome_canonical_intents [
    "smarthome.device_down",
    "smarthome.device_up",
    "smarthome.device_set",
    "smarthome.device_check",
    "smarthome.switch",
    "smarthome.schedule_create"
  ]

  @registry_deletions [
    "action.request",
    "information.request",
    "question.factual",
    "question.opinion",
    "dialog.continuation",
    "unknown",
    "navigation.directions",
    "navigation.next",
    "navigation.previous",
    "navigation.repeat",
    "smalltalk.user.age",
    "smalltalk.user.location",
    "smalltalk.user.origin",
    "meta.trust_check",
    "smalltalk.backchannel"
  ]

  @extra_renames %{
    "weather" => "weather.query",
    "default_welcome_intent" => "smalltalk.greet",
    # The corpus carries Dialogflow's own display name, spaces and capitals
    # intact. The snake_case rule above was written against a normalised form
    # this data never had, so it could never fire — and nothing normalises
    # label case (`mix normalize_gold_standard` handles context variants, empty
    # text and duplicates, not casing). Matching the literal label is what
    # actually retires the last 47 unregistered examples.
    "Default Welcome Intent" => "smalltalk.greet",
    "message" => "communication.text"
  }
  |> Map.merge(@consolidation_renames)
  |> Map.merge(@smalltalk_user_emotion_renames)
  |> Map.merge(@phantom_renames)

  @all_renames @device_to_lights_map
              |> Map.merge(@nav_map)
              |> Map.merge(@music_map)
              |> Map.merge(@extra_renames)

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

      validate_against_registry(entries, registry_path)

      if do_sentiment? do
        cleanup_sentiment()
      end
    else
      Mix.shell().info("[cleanup] Dry run -- no files written (registry validation skipped)")
    end
  end

  defp apply_all_rules(entries) do
    stats = %{
      rule0: 0,
      rule1: 0,
      rule2: 0,
      rule3: 0,
      rule4: 0,
      rule5: 0,
      rule6: 0,
      rule7: 0,
      dupes_removed: 0
    }

    {entries, stats} = strip_context_variants(entries, stats)
    {entries, stats} = apply_renames(entries, stats)
    {entries, stats} = apply_mislabel_fixes(entries, stats)
    {entries, stats} = apply_smarthome_consolidation(entries, stats)
    {entries, stats} = deduplicate(entries, stats)

    {entries, stats}
  end

  defp strip_context_variants(entries, stats) do
    Enum.reduce(entries, {[], stats}, fn entry, {acc, s} ->
      intent = Map.get(entry, "intent", "")

      case String.split(intent, "_-_context:", parts: 2) do
        [base, _rest] ->
          {[Map.put(entry, "intent", base) | acc], %{s | rule0: s.rule0 + 1}}

        _ ->
          {[entry | acc], s}
      end
    end)
    |> then(fn {entries, stats} -> {Enum.reverse(entries), stats} end)
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

        Map.has_key?(@extra_renames, intent) ->
          new_intent = Map.fetch!(@extra_renames, intent)
          {[Map.put(entry, "intent", new_intent) | acc], %{s | rule6: s.rule6 + 1}}

        true ->
          {[entry | acc], s}
      end
    end)
    |> then(fn {entries, stats} -> {Enum.reverse(entries), stats} end)
  end

  defp apply_smarthome_consolidation(entries, stats) do
    Enum.reduce(entries, {[], stats}, fn entry, {acc, s} ->
      intent = Map.get(entry, "intent", "")

      if Map.has_key?(@smarthome_consolidation, intent) do
        new_intent = Map.fetch!(@smarthome_consolidation, intent)
        {[Map.put(entry, "intent", new_intent) | acc], %{s | rule7: s.rule7 + 1}}
      else
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
    Mix.shell().info("[cleanup] Rule 0: Stripped #{stats.rule0} context-variant suffixes")
    Mix.shell().info("[cleanup] Rule 1: Merged #{stats.rule1} entries from smarthome.device.* to smarthome.lights.*")
    Mix.shell().info("[cleanup] Rule 2: Merged #{stats.rule2} navigation-variant entries into base search intents")
    Mix.shell().info("[cleanup] Rule 3: Renamed #{stats.rule3} entries from music_player_control.* to music.player.*")
    Mix.shell().info("[cleanup] Rule 4: Promoted #{stats.rule4} continuation entries to dialog.continuation")
    Mix.shell().info("[cleanup] Rule 5: Fixed #{stats.rule5} mislabel(s)")
    Mix.shell().info("[cleanup] Rule 6: Renamed #{stats.rule6} label(s) (smalltalk + phantom + user emotion rollup + extras)")
    Mix.shell().info("[cleanup] Rule 7: Consolidated #{stats.rule7} smarthome label(s) into 6 action intents")
    Mix.shell().info("[cleanup] Dedup: Removed #{stats.dupes_removed} exact duplicates")
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

  defp validate_against_registry(entries, registry_path) do
    registry = load_json!(registry_path)
    registry_intents = MapSet.new(Map.keys(registry))

    unregistered =
      entries
      |> Enum.map(&Map.get(&1, "intent", ""))
      |> Enum.reject(&MapSet.member?(registry_intents, &1))
      |> Enum.frequencies()
      |> Enum.sort_by(fn {_, count} -> -count end)

    if unregistered == [] do
      Mix.shell().info("[cleanup] Validation: all gold standard labels exist in intent_registry.json")
    else
      total = Enum.reduce(unregistered, 0, fn {_, c}, acc -> acc + c end)

      Mix.shell().error(
        "[cleanup] WARNING: #{length(unregistered)} gold standard labels (#{total} examples) " <>
          "are NOT in intent_registry.json — these intents can never be predicted correctly:"
      )

      Enum.each(unregistered, fn {intent, count} ->
        Mix.shell().error("  #{count} examples: #{intent}")
      end)

      Mix.shell().error(
        "[cleanup] Add rename rules for these labels or register them in intent_registry.json"
      )
    end
  end

  defp cleanup_registry(registry_path) do
    Mix.shell().info("[cleanup] Updating intent registry at #{registry_path}")

    registry = load_json!(registry_path)
    before_count = map_size(registry)

    renamed =
      Enum.reduce(@all_renames, registry, fn {old_key, new_key}, acc ->
        case Map.pop(acc, old_key) do
          {nil, acc} ->
            acc

          {value, acc} ->
            new_domain = new_key |> String.split(".") |> List.first()
            updated_value = Map.put(value, "domain", new_domain)
            Map.put_new(acc, new_key, updated_value)
        end
      end)

    renamed =
      case Map.pop(renamed, "continuation") do
        {nil, r} -> r
        {value, r} -> Map.put_new(r, "dialog.continuation", Map.put(value, "domain", "dialog"))
      end

    renamed = apply_smarthome_registry_consolidation(renamed)
    renamed = merge_smarthome_canonical_metadata(renamed)

    pruned = Map.drop(renamed, @registry_deletions)
    deleted_count = map_size(renamed) - map_size(pruned)

    if deleted_count > 0 do
      Mix.shell().info("[cleanup] Removed #{deleted_count} phantom registry entries")
    end

    write_json!(registry_path, pruned)
    Mix.shell().info("[cleanup] Registry updated (#{before_count} -> #{map_size(pruned)} entries)")
  end

  defp apply_smarthome_registry_consolidation(registry) do
    Enum.reduce(@smarthome_consolidation, registry, fn {old_key, new_key}, acc ->
      case Map.pop(acc, old_key) do
        {nil, acc} ->
          acc

        {value, acc} ->
          new_domain = new_key |> String.split(".") |> List.first()
          updated_value = Map.put(value, "domain", new_domain)
          Map.put_new(acc, new_key, updated_value)
      end
    end)
  end

  defp merge_smarthome_canonical_metadata(registry) do
    base = %{
      "category" => "directive",
      "domain" => "smarthome",
      "clarification_templates" => %{},
      "entity_mappings" => %{},
      "optional" => [],
      "required" => [],
      "speech_act" => "request"
    }

    Enum.reduce(@smarthome_canonical_intents, registry, fn key, acc ->
      case Map.fetch(acc, key) do
        {:ok, meta} ->
          Map.put(acc, key, Map.merge(meta, base))

        :error ->
          acc
      end
    end)
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
