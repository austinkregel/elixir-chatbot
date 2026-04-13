# Populates input_contexts in intent_registry.json based on output_contexts patterns.
# Run with: mix run scripts/populate_input_contexts.exs

context_consumer_patterns = %{
  "heating" => fn intent -> String.starts_with?(intent, "smarthome.heating.") end,
  "room" => fn intent -> String.starts_with?(intent, "smarthome.") end,
  "music-player-control" => fn intent -> String.starts_with?(intent, "music.player.") end,
  "play-music" => fn intent -> String.starts_with?(intent, "music.") end,
  "search-music" => fn intent -> String.starts_with?(intent, "music.") end,
  "schedule" => fn intent -> String.contains?(intent, "schedule") end,
  "weather" => fn intent -> String.starts_with?(intent, "weather.") end,
  "device-brightness" => fn intent -> String.starts_with?(intent, "smarthome.lights.brightness.") end,
  "device-switch" => fn intent ->
    String.starts_with?(intent, "smarthome.lights.switch.") or
    String.starts_with?(intent, "smarthome.heating.switch.")
  end,
  "device-volume" => fn intent -> String.starts_with?(intent, "smarthome.device.volume.") end,
  "text-context" => fn intent -> String.starts_with?(intent, "communication.") end,
  "calendar-context" => fn intent -> String.starts_with?(intent, "calendar.") end,
  "reminder-context" => fn intent -> String.starts_with?(intent, "reminder.") end,
  "alarm-context" => fn intent -> String.starts_with?(intent, "alarm.") end,
  "todo-context" => fn intent -> String.starts_with?(intent, "todo.") end,
  "websearch-followup" => fn intent ->
    String.starts_with?(intent, "web.") or String.starts_with?(intent, "navigation.")
  end,
  "web-search" => fn intent -> String.starts_with?(intent, "web.") end,
  "control_lists_web_search" => fn intent -> String.starts_with?(intent, "web.") end,
  "news-search" => fn intent -> String.starts_with?(intent, "news.") end,
  "newssearch-followup" => fn intent -> String.starts_with?(intent, "news.") end,
  "balance" => fn intent -> String.starts_with?(intent, "account.") or String.starts_with?(intent, "payment.") end,
  "spending" => fn intent -> String.starts_with?(intent, "account.") end,
  "earning" => fn intent -> String.starts_with?(intent, "account.") end,
  "due_date" => fn intent -> String.starts_with?(intent, "account.") or String.starts_with?(intent, "payment.") end,
  "call-context" => fn intent -> String.starts_with?(intent, "communication.") end,
  "volume-check" => fn intent -> String.starts_with?(intent, "smarthome.device.volume.") end,
  "timer-context" => fn intent -> String.starts_with?(intent, "timer.") end,
  "lock" => fn intent -> String.starts_with?(intent, "smarthome.locks.") end,
  "unlock" => fn intent -> String.starts_with?(intent, "smarthome.locks.") end,
  "open" => fn intent -> String.starts_with?(intent, "smarthome.locks.") end,
  "close" => fn intent -> String.starts_with?(intent, "smarthome.locks.") end
}

path = "apps/brain/priv/analysis/intent_registry.json"
{:ok, content} = File.read(path)
{:ok, registry} = Jason.decode(content)

all_intents = Map.keys(registry)

active_contexts =
  registry
  |> Enum.flat_map(fn {_intent, meta} ->
    case meta do
      %{"output_contexts" => contexts} when is_list(contexts) ->
        Enum.flat_map(contexts, fn
          %{"name" => name} -> [name]
          _ -> []
        end)
      _ -> []
    end
  end)
  |> Enum.uniq()

context_to_consumers =
  Enum.reduce(active_contexts, %{}, fn ctx, acc ->
    matcher = Map.get(context_consumer_patterns, ctx)
    if matcher do
      consumers = Enum.filter(all_intents, matcher)
      Map.put(acc, ctx, consumers)
    else
      acc
    end
  end)

updated_registry =
  Enum.reduce(registry, registry, fn {intent_name, meta}, acc ->
    matching_contexts =
      context_to_consumers
      |> Enum.flat_map(fn {ctx_name, consumers} ->
        if intent_name in consumers do
          [%{"name" => ctx_name, "lifespan" => 2}]
        else
          []
        end
      end)
      |> Enum.uniq_by(fn %{"name" => n} -> n end)

    if matching_contexts != [] do
      Map.put(acc, intent_name, Map.put(meta, "input_contexts", matching_contexts))
    else
      acc
    end
  end)

json = Jason.encode!(updated_registry, pretty: true)
File.write!(path, json <> "\n")

populated_count = Enum.count(updated_registry, fn {_k, v} ->
  case v["input_contexts"] do
    list when is_list(list) and list != [] -> true
    _ -> false
  end
end)

IO.puts("Updated #{populated_count} intents with input_contexts")
IO.puts("Active contexts: #{length(active_contexts)}")
IO.puts("Written to #{path}")
