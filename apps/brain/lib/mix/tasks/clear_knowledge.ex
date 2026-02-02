defmodule Mix.Tasks.ClearKnowledge do
  @moduledoc """
  Clears learned knowledge without affecting training data.

  Usage:
    mix clear_knowledge             # Clear all knowledge
    mix clear_knowledge --memory    # Clear only cognitive memory
    mix clear_knowledge --learned   # Clear only learned facts (people, pets, etc.)
    mix clear_knowledge --brain     # Clear brain conversation memory
    mix clear_knowledge --entities  # Clear admin-added gazetteer entities
    mix clear_knowledge --reload    # Clear & reload gazetteer from data files
  """
  use Mix.Task

  @shortdoc "Clear learned knowledge (not training data)"

  @impl Mix.Task
  def run(args) do
    # Start the application
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          memory: :boolean,
          learned: :boolean,
          brain: :boolean,
          entities: :boolean,
          reload: :boolean,
          all: :boolean
        ]
      )

    # Default to all if no specific option (except reload which is explicit)
    clear_all = opts[:all] || opts == []

    IO.puts("\n=== Clearing Knowledge ===\n")

    # Clear cognitive memory
    if clear_all || opts[:memory] do
      clear_cognitive_memory()
    end

    # Clear learned knowledge (KnowledgeStore)
    if clear_all || opts[:learned] do
      clear_learned_knowledge()
    end

    # Clear brain memory
    if clear_all || opts[:brain] do
      clear_brain_memory()
    end

    # Clear admin-added entities
    if clear_all || opts[:entities] do
      clear_admin_entities()
    end

    # Reload gazetteer from files
    if opts[:reload] do
      reload_gazetteer()
    end

    IO.puts("\n=== Knowledge Cleared ===\n")
  end

  defp clear_cognitive_memory do
    IO.puts("Clearing cognitive memory...")

    case Brain.Memory.Store.stats() do
      stats when is_map(stats) ->
        before_episodes = stats.episode_count
        before_semantics = stats.semantic_count

        Brain.Memory.Store.clear()

        IO.puts("  ✓ Cleared #{before_episodes} episodes")
        IO.puts("  ✓ Cleared #{before_semantics} semantic facts")

      _ ->
        IO.puts("  ✓ Cognitive memory already empty")
    end
  end

  defp clear_learned_knowledge do
    IO.puts("Clearing learned knowledge...")

    # Get all agent names (default to Echo)
    agents = ["Echo"]

    Enum.each(agents, fn agent ->
      knowledge = Brain.KnowledgeStore.get_knowledge(agent)

      counts =
        knowledge
        |> Enum.map(fn {key, value} ->
          count =
            cond do
              is_map(value) -> map_size(value)
              is_list(value) -> length(value)
              true -> 0
            end

          {key, count}
        end)
        |> Enum.filter(fn {_, c} -> c > 0 end)

      if length(counts) > 0 do
        Brain.KnowledgeStore.clear(agent)
        IO.puts("  ✓ Cleared knowledge for #{agent}:")

        Enum.each(counts, fn {key, count} ->
          IO.puts("    - #{key}: #{count} entries")
        end)
      else
        IO.puts("  ✓ No learned knowledge for #{agent}")
      end
    end)

    # Also delete persisted files
    knowledge_dir = Brain.priv_path("knowledge")

    if File.exists?(knowledge_dir) do
      File.rm_rf!(knowledge_dir)
      File.mkdir_p!(knowledge_dir)
      IO.puts("  ✓ Cleared persisted knowledge files")
    end
  end

  defp clear_brain_memory do
    IO.puts("Clearing brain conversation memory...")

    status = Brain.get_status()
    before_size = status.global_memory_size

    # End all active conversations
    conversations = GenServer.call(Brain, :get_conversations, 60_000)

    Enum.each(conversations, fn conv ->
      Brain.end_conversation(conv.id)
    end)

    # Clear memory directory
    memory_dir = Brain.priv_path("memory")

    if File.exists?(memory_dir) do
      File.rm_rf!(memory_dir)
      File.mkdir_p!(memory_dir)
    end

    IO.puts("  ✓ Ended #{length(conversations)} conversations")
    IO.puts("  ✓ Cleared #{before_size} memory entries")
    IO.puts("  ✓ Cleared persisted memory files")
  end

  defp clear_admin_entities do
    IO.puts("Clearing admin-added entities...")

    case Brain.ML.Gazetteer.clear_admin_entries() do
      {:ok, count} ->
        IO.puts("  ✓ Cleared #{count} admin-added entities")

      {:error, reason} ->
        IO.puts("  ✗ Failed to clear: #{inspect(reason)}")
    end
  end

  defp reload_gazetteer do
    IO.puts("Reloading gazetteer from data files...")

    case Brain.ML.Gazetteer.clear_all() do
      {:ok, count} ->
        IO.puts("  ✓ Cleared #{count} entities")

      _ ->
        :ok
    end

    case Brain.ML.Gazetteer.load_all() do
      {:ok, stats} ->
        IO.puts("  ✓ Loaded #{stats[:entities] || 0} entities from data files")

      {:error, reason} ->
        IO.puts("  ✗ Failed to reload: #{inspect(reason)}")
    end
  end
end
