defmodule Mix.Tasks.MigrateFactsTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.MigrateFacts

  import ExUnit.CaptureIO

  @moduletag :mix_task

  setup do
    # Create a unique temp directory for each test
    test_id = "#{System.system_time(:nanosecond)}_#{:rand.uniform(100_000)}"
    tmp_dir = Path.join(System.tmp_dir!(), "migrate_facts_test_#{test_id}")
    facts_dir = Path.join(tmp_dir, "data/facts")
    File.mkdir_p!(facts_dir)

    # Store original cwd to restore later
    original_cwd = File.cwd!()
    original_shell = Mix.shell()

    on_exit(fn ->
      # Restore original cwd before cleanup
      File.cd!(original_cwd)
      Mix.shell(original_shell)
      File.rm_rf!(tmp_dir)
    end)

    %{tmp_dir: tmp_dir, facts_dir: facts_dir, original_cwd: original_cwd}
  end

  describe "run/1 with --dry-run" do
    test "shows what would be changed without modifying files", %{
      facts_dir: facts_dir,
      original_cwd: original_cwd
    } do
      # Create a test fact file with legacy format (entity_type but no entity)
      legacy_fact = %{
        "facts" => [
          %{
            "entity_type" => "John",
            "category" => "people",
            "fact" => "lives in New York",
            "confidence" => 0.9,
            "verification_source" => "user"
          }
        ]
      }

      test_file = Path.join(facts_dir, "test.json")
      File.write!(test_file, Jason.encode!(legacy_fact))

      # Change to the temp directory so the task finds data/facts
      File.cd!(Path.dirname(facts_dir) |> Path.dirname())

      try do
        output =
          capture_io(fn ->
            Mix.shell(Mix.Shell.IO)
            MigrateFacts.run(["--dry-run"])
          end)

        # Should mention dry run mode
        assert output =~ "Dry run" or output =~ "dry-run" or output =~ "DRY RUN"

        # Verify file was NOT modified
        {:ok, content} = File.read(test_file)
        {:ok, data} = Jason.decode(content)
        fact = hd(data["facts"])

        # Should still have legacy format
        assert Map.has_key?(fact, "entity_type")
        refute Map.has_key?(fact, "entity")
      after
        File.cd!(original_cwd)
      end
    end

    test "reports migration counts", %{facts_dir: facts_dir, original_cwd: original_cwd} do
      # Create a test fact file with legacy format
      legacy_facts = %{
        "facts" => [
          %{
            "entity_type" => "Alice",
            "category" => "people",
            "fact" => "is a developer",
            "confidence" => 0.9,
            "verification_source" => "user"
          },
          %{
            "entity_type" => "Bob",
            "category" => "people",
            "fact" => "works at a company",
            "confidence" => 0.8,
            "verification_source" => "user"
          }
        ]
      }

      test_file = Path.join(facts_dir, "people.json")
      File.write!(test_file, Jason.encode!(legacy_facts))

      File.cd!(Path.dirname(facts_dir) |> Path.dirname())

      try do
        output =
          capture_io(fn ->
            Mix.shell(Mix.Shell.IO)
            MigrateFacts.run(["--dry-run", "--verbose"])
          end)

        # Should show migration info
        assert output =~ "Migrating" or output =~ "Migration" or output =~ "facts"
      after
        File.cd!(original_cwd)
      end
    end
  end

  describe "run/1 without --dry-run" do
    test "migrates legacy entity_type to entity field", %{
      facts_dir: facts_dir,
      original_cwd: original_cwd
    } do
      # Create a test fact file with legacy format
      legacy_fact = %{
        "facts" => [
          %{
            "entity_type" => "Sarah",
            "category" => "people",
            "fact" => "has a pet cat",
            "confidence" => 0.95,
            "verification_source" => "user"
          }
        ]
      }

      test_file = Path.join(facts_dir, "migrate_test.json")
      File.write!(test_file, Jason.encode!(legacy_fact))

      File.cd!(Path.dirname(facts_dir) |> Path.dirname())

      try do
        capture_io(fn ->
          Mix.shell(Mix.Shell.IO)
          MigrateFacts.run([])
        end)

        # Verify file was modified
        {:ok, content} = File.read(test_file)
        {:ok, data} = Jason.decode(content)
        fact = hd(data["facts"])

        # Should have both entity and entity_type now
        assert Map.has_key?(fact, "entity")
        assert Map.has_key?(fact, "entity_type")
        assert fact["entity"] == "Sarah"
      after
        File.cd!(original_cwd)
      end
    end

    test "deduplicates entries in learned.json", %{
      facts_dir: facts_dir,
      original_cwd: original_cwd
    } do
      # Create learned.json with duplicate facts
      duplicate_facts = %{
        "facts" => [
          %{
            "entity" => "Mike",
            "entity_type" => "person",
            "category" => "people",
            "fact" => "likes pizza",
            "confidence" => 0.9,
            "verification_source" => "user"
          },
          %{
            "entity" => "Mike",
            "entity_type" => "person",
            "category" => "people",
            "fact" => "likes pizza",
            "confidence" => 0.85,
            "verification_source" => "user"
          },
          %{
            "entity" => "Mike",
            "entity_type" => "person",
            "category" => "people",
            "fact" => "has a dog",
            "confidence" => 0.9,
            "verification_source" => "user"
          }
        ]
      }

      test_file = Path.join(facts_dir, "learned.json")
      File.write!(test_file, Jason.encode!(duplicate_facts))

      File.cd!(Path.dirname(facts_dir) |> Path.dirname())

      try do
        output =
          capture_io(fn ->
            Mix.shell(Mix.Shell.IO)
            MigrateFacts.run([])
          end)

        # Verify duplicates were removed
        {:ok, content} = File.read(test_file)
        {:ok, data} = Jason.decode(content)

        # Should have only 2 unique facts (deduplicated by entity + fact)
        assert length(data["facts"]) == 2

        # Output should mention something about the migration
        assert output =~ "duplicate" or output =~ "Migration" or output =~ "complete"
      after
        File.cd!(original_cwd)
      end
    end
  end

  describe "module loading" do
    test "module is loaded correctly" do
      assert Code.ensure_loaded?(Mix.Tasks.MigrateFacts)
    end
  end
end
