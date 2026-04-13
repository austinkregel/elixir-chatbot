defmodule Mix.Tasks.FactDatabase do
  @moduledoc "Mix task for interacting with the fact database.\n\n## Usage\n\n    mix fact_database              # Show stats\n    mix fact_database stats        # Show detailed stats\n    mix fact_database list         # List all facts\n    mix fact_database categories   # List categories\n    mix fact_database search TERM  # Search facts\n    mix fact_database entity NAME  # Get facts about an entity\n    mix fact_database sync         # Sync facts to epistemic system\n    mix fact_database add ENTITY FACT  # Add a new fact\n"

  alias Brain.FactDatabase.Integration
  alias Brain.FactDatabase
  use Mix.Task

  @shortdoc "Interact with the fact database"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    case args do
      [] -> show_stats()
      ["stats"] -> show_stats()
      ["list"] -> list_facts()
      ["categories"] -> list_categories()
      ["search" | terms] -> search_facts(Enum.join(terms, " "))
      ["entity" | names] -> get_entity_facts(Enum.join(names, " "))
      ["sync"] -> sync_to_beliefs()
      ["add", entity | fact_parts] -> add_fact(entity, Enum.join(fact_parts, " "))
      _ -> show_help()
    end
  end

  defp show_stats do
    stats = FactDatabase.stats()
    categories = FactDatabase.list_categories()

    IO.puts("\n=== Fact Database Stats ===\n")
    IO.puts("  Total facts:  #{stats.total_facts}")
    IO.puts("  Categories:   #{stats.categories}")
    IO.puts("  Entities:     #{stats.entities}")
    IO.puts("  Loaded at:    #{format_timestamp(stats.loaded_at)}")
    IO.puts("
  Categories: #{Enum.join(categories, ", ")}")
    IO.puts("")
  end

  defp list_facts do
    facts = FactDatabase.query(limit: 100)

    IO.puts("
=== All Facts (#{length(facts)}) ===
")

    facts
    |> Enum.group_by(& &1.category)
    |> Enum.sort_by(fn {cat, _} -> cat end)
    |> Enum.each(fn {category, cat_facts} ->
      IO.puts("## #{String.upcase(category)} (#{length(cat_facts)})")

      Enum.each(cat_facts, fn fact ->
        IO.puts("  [#{fact.id}] #{fact.entity}: #{fact.fact}")
      end)

      IO.puts("")
    end)
  end

  defp list_categories do
    categories = FactDatabase.list_categories()

    IO.puts("\n=== Categories ===\n")

    Enum.each(categories, fn cat ->
      facts = FactDatabase.get_category_facts(cat)
      IO.puts("  #{cat}: #{length(facts)} facts")
    end)

    IO.puts("")
  end

  defp search_facts(term) do
    facts = FactDatabase.query(search: term, limit: 20)

    IO.puts("
=== Search: \"#{term}\" (#{length(facts)} results) ===
")

    if facts == [] do
      IO.puts("  No facts found matching \"#{term}\"")
    else
      Enum.each(facts, fn fact ->
        IO.puts("  [#{fact.category}] #{fact.entity}: #{fact.fact}")
      end)
    end

    IO.puts("")
  end

  defp get_entity_facts(entity) do
    facts = FactDatabase.get_entity_facts(entity)

    IO.puts("
=== Facts about \"#{entity}\" (#{length(facts)}) ===
")

    if facts == [] do
      IO.puts("  No facts found for entity \"#{entity}\"")
    else
      Enum.each(facts, fn fact ->
        IO.puts("  [#{fact.category}] #{fact.fact}")
        IO.puts("    Source: #{fact.verification_source}")
        IO.puts("    Confidence: #{fact.confidence}")
        IO.puts("")
      end)
    end
  end

  defp sync_to_beliefs do
    IO.puts("\n=== Syncing Facts to Epistemic System ===\n")

    {:ok, count} = Integration.sync_facts_to_beliefs()
    IO.puts("  Successfully synced #{count} facts to beliefs")

    IO.puts("")
  end

  defp add_fact(entity, fact_text) do
    IO.puts("\n=== Adding New Fact ===\n")
    IO.puts("  Entity: #{entity}")
    IO.puts("  Fact: #{fact_text}")

    {:ok, fact_id, _fact} =
      Integration.add_fact(entity, fact_text,
        category: "learned",
        confidence: 0.8,
        verification_source: "manual_entry"
      )

    IO.puts("
  Success! Fact ID: #{fact_id}")
    IO.puts("")
  end

  defp show_help do
    IO.puts(@moduledoc)
  end

  defp format_timestamp(unix_seconds) do
    case DateTime.from_unix(unix_seconds) do
      {:ok, dt} -> Calendar.strftime(dt, "%Y-%m-%d %H:%M:%S UTC")
      _ -> "unknown"
    end
  end
end