# scripts/materialize_orphan_intents.exs
#
# Materializes orphan gold-standard entries (from a prior LLM iteration) back
# into DialogFlow-style *_usersays_en.json source files under data/intents/.
#
# Two passes:
#   C.1  Fully-invented intents → new <intent>_usersays_en.json files
#   C.2  Augmentation entries   → appended to existing usersays files
#
# Idempotent: skips files / entries that already exist unless --force is passed.
#
# Usage:
#   mix run scripts/materialize_orphan_intents.exs
#   mix run scripts/materialize_orphan_intents.exs --force

force? = "--force" in System.argv()
gold_path = "apps/brain/priv/evaluation/intent/gold_standard.json"
intents_dir = "data/intents"

IO.puts("=== Materialize Orphan Intents ===")
IO.puts("Force mode: #{force?}\n")

# ── Step 1: Load gold standard ──────────────────────────────────────────────
gold =
  gold_path
  |> File.read!()
  |> Jason.decode!()

IO.puts("Loaded #{length(gold)} gold entries from #{gold_path}")

# ── Step 2: Build the set of all source-backed texts ────────────────────────
source_texts =
  Path.wildcard(Path.join(intents_dir, "*_usersays_en.json"))
  |> Enum.flat_map(fn path ->
    case Jason.decode(File.read!(path)) do
      {:ok, entries} when is_list(entries) ->
        Enum.map(entries, fn entry ->
          case entry do
            %{"data" => segments} when is_list(segments) ->
              segments |> Enum.map_join("", &Map.get(&1, "text", "")) |> String.trim()

            %{"text" => t} ->
              String.trim(t)

            _ ->
              ""
          end
        end)

      _ ->
        []
    end
  end)
  |> MapSet.new()

IO.puts("Found #{MapSet.size(source_texts)} unique source-backed texts across usersays files\n")

# ── Step 3: Identify orphan entries ─────────────────────────────────────────
orphans =
  Enum.filter(gold, fn entry ->
    not MapSet.member?(source_texts, entry["text"])
  end)

IO.puts("Identified #{length(orphans)} orphan entries (no source backing)\n")

orphans_by_intent = Enum.group_by(orphans, & &1["intent"])

# ── Step 4: Classify into C.1 (new intent) vs C.2 (augmentation) ───────────
existing_usersays =
  Path.wildcard(Path.join(intents_dir, "*_usersays_en.json"))
  |> Enum.map(fn path ->
    path |> Path.basename() |> String.replace_suffix("_usersays_en.json", "")
  end)
  |> MapSet.new()

{c1_intents, c2_intents} =
  Enum.split_with(orphans_by_intent, fn {intent, _entries} ->
    not MapSet.member?(existing_usersays, intent)
  end)

IO.puts("C.1 fully-invented intents: #{length(c1_intents)} (#{c1_intents |> Enum.map(fn {_, e} -> length(e) end) |> Enum.sum()} entries)")
IO.puts("C.2 augmentation intents:   #{length(c2_intents)} (#{c2_intents |> Enum.map(fn {_, e} -> length(e) end) |> Enum.sum()} entries)\n")

# ── Helper: build a DialogFlow usersays entry from a gold entry ─────────────
now_unix = System.os_time(:second)

build_usersays_entry = fn text, id_prefix ->
  hash = :crypto.hash(:md5, text) |> Base.encode16(case: :lower) |> binary_part(0, 8)

  %{
    "id" => "#{id_prefix}-#{hash}",
    "data" => [%{"text" => text, "userDefined" => false}],
    "isTemplate" => false,
    "count" => 0,
    "lang" => "en",
    "updated" => now_unix
  }
end

# ── Step 5: C.1 — Write new usersays files ──────────────────────────────────
IO.puts("── C.1: Writing new usersays files ──")

c1_written = 0

c1_written =
  Enum.reduce(c1_intents, 0, fn {intent, entries}, count ->
    usersays_path = Path.join(intents_dir, "#{intent}_usersays_en.json")
    definition_path = Path.join(intents_dir, "#{intent}.json")

    if File.exists?(usersays_path) and not force? do
      IO.puts("  SKIP #{intent} (#{usersays_path} already exists)")
      count
    else
      usersays_data =
        entries
        |> Enum.uniq_by(& &1["text"])
        |> Enum.map(fn entry -> build_usersays_entry.(entry["text"], "orphan") end)

      File.write!(usersays_path, Jason.encode!(usersays_data, pretty: true))
      IO.puts("  WROTE #{usersays_path} (#{length(usersays_data)} utterances)")

      unless File.exists?(definition_path) do
        uuid =
          :crypto.strong_rand_bytes(16)
          |> Base.encode16(case: :lower)
          |> then(fn hex ->
            <<a::binary-size(8), b::binary-size(4), c::binary-size(4), d::binary-size(4), e::binary-size(12)>> = hex
            "#{a}-#{b}-#{c}-#{d}-#{e}"
          end)

        definition = %{
          "id" => uuid,
          "name" => intent,
          "auto" => true,
          "contexts" => [],
          "responses" => [
            %{
              "resetContexts" => false,
              "action" => "",
              "affectedContexts" => [],
              "parameters" => [],
              "messages" => [
                %{"type" => "0", "title" => "", "textToSpeech" => "", "lang" => "en", "condition" => ""}
              ],
              "speech" => []
            }
          ],
          "priority" => 500_000,
          "webhookUsed" => false,
          "webhookForSlotFilling" => false,
          "fallbackIntent" => false,
          "events" => [],
          "conditionalResponses" => [],
          "condition" => "",
          "conditionalFollowupEvents" => []
        }

        File.write!(definition_path, Jason.encode!(definition, pretty: true))
        IO.puts("  WROTE #{definition_path} (definition stub)")
      end

      count + length(usersays_data)
    end
  end)

IO.puts("  Total C.1 entries written: #{c1_written}\n")

# ── Step 6: C.2 — Append to existing usersays files ────────────────────────
IO.puts("── C.2: Appending to existing usersays files ──")

c2_written =
  Enum.reduce(c2_intents, 0, fn {intent, entries}, count ->
    usersays_path = Path.join(intents_dir, "#{intent}_usersays_en.json")

    existing_data =
      case Jason.decode(File.read!(usersays_path)) do
        {:ok, data} when is_list(data) -> data
        _ -> []
      end

    existing_texts =
      existing_data
      |> Enum.map(fn entry ->
        case entry do
          %{"data" => segments} when is_list(segments) ->
            segments |> Enum.map_join("", &Map.get(&1, "text", "")) |> String.trim()

          %{"text" => t} ->
            String.trim(t)

          _ ->
            ""
        end
      end)
      |> MapSet.new()

    new_entries =
      entries
      |> Enum.uniq_by(& &1["text"])
      |> Enum.reject(fn entry -> MapSet.member?(existing_texts, entry["text"]) end)

    if new_entries == [] do
      IO.puts("  SKIP #{intent} (all #{length(entries)} orphans already present)")
      count
    else
      new_usersays =
        Enum.map(new_entries, fn entry ->
          build_usersays_entry.(entry["text"], "augmented")
        end)

      merged = existing_data ++ new_usersays
      File.write!(usersays_path, Jason.encode!(merged, pretty: true))

      IO.puts(
        "  APPENDED #{length(new_usersays)} to #{usersays_path} (was #{length(existing_data)}, now #{length(merged)})"
      )

      count + length(new_usersays)
    end
  end)

IO.puts("  Total C.2 entries appended: #{c2_written}\n")

IO.puts("=== Done ===")
IO.puts("C.1: #{c1_written} entries in new files")
IO.puts("C.2: #{c2_written} entries appended to existing files")
IO.puts("Total: #{c1_written + c2_written} orphan entries materialized into data/intents/")
