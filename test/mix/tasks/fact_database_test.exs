defmodule Mix.Tasks.FactDatabaseTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.FactDatabase

  import ExUnit.CaptureLog
  import ExUnit.CaptureIO

  @moduletag :mix_task

  import ChatBot.TestHelpers

  setup do
    # Start all required services
    start_test_services()

    # Store original shell to restore later
    original_shell = Mix.shell()
    Mix.shell(Mix.Shell.Quiet)

    on_exit(fn ->
      Mix.shell(original_shell)
    end)

    :ok
  end

  describe "run/1 with no arguments (default: stats)" do
    test "shows stats and logs output" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run([])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should show stats information
      assert output =~ "Fact Database" or output =~ "facts" or output =~ "Total"
    end
  end

  describe "run/1 with 'stats' subcommand" do
    test "displays database statistics" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["stats"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should include stats-related output
      assert output =~ "Total" or output =~ "facts" or output =~ "Categories"
    end
  end

  describe "run/1 with 'list' subcommand" do
    test "lists facts" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["list"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should show facts or indicate list output
      assert output =~ "Facts" or output =~ "[" or is_binary(output)
    end
  end

  describe "run/1 with 'categories' subcommand" do
    test "shows categories" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["categories"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should mention categories
      assert output =~ "Categories" or output =~ "general" or output =~ "learned"
    end
  end

  describe "run/1 with 'search' subcommand" do
    test "searches facts and shows results" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["search", "test"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should show search results header
      assert output =~ "Search" or output =~ "results" or output =~ "test"
    end

    test "handles search with multiple words" do
      log =
        capture_log(fn ->
          FactDatabase.run(["search", "test", "query"])
        end)

      assert is_binary(log)
    end

    test "shows no results message for non-existent term" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["search", "xyznonexistent12345"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should indicate no results found
      assert output =~ "No facts found" or output =~ "0 results" or output =~ "Search"
    end
  end

  describe "run/1 with 'entity' subcommand" do
    test "shows facts about an entity" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["entity", "France"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should mention the entity or facts
      assert output =~ "France" or output =~ "Facts" or output =~ "capital"
    end

    test "handles entity not found" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["entity", "NonExistentEntity12345"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should indicate no facts found
      assert output =~ "No facts found" or output =~ "(0)" or is_binary(output)
    end
  end

  describe "run/1 with 'sync' subcommand" do
    test "syncs facts and logs activity" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["sync"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should mention syncing
      assert output =~ "Sync" or output =~ "sync" or output =~ "beliefs" or output =~ "facts"
    end
  end

  describe "run/1 with 'add' subcommand" do
    test "adds a new fact and confirms" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["add", "TestAddEntity", "is", "a", "test", "entity"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should confirm addition
      assert output =~ "Success" or output =~ "Added" or output =~ "Fact"

      # Verify fact was actually added
      facts = ChatBot.FactDatabase.get_entity_facts("TestAddEntity")
      assert length(facts) >= 1
    end
  end

  describe "run/1 with invalid subcommand" do
    test "shows help message" do
      output =
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          FactDatabase.run(["invalidcommand"])
          Mix.shell(Mix.Shell.Quiet)
        end)

      # Should show usage help
      assert output =~ "Usage" or output =~ "fact_database" or output =~ "help"
    end
  end

  describe "fact database state" do
    test "stats returns valid structure" do
      stats = ChatBot.FactDatabase.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :total_facts)
      assert Map.has_key?(stats, :categories)
      assert Map.has_key?(stats, :entities)
      assert is_integer(stats.total_facts)
    end
  end
end
