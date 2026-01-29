defmodule Mix.Tasks.MigrateTrainingDataTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.MigrateTrainingData

  @moduletag :mix_task

  setup do
    # Create a unique temp directory for each test
    test_id = "#{System.system_time(:nanosecond)}_#{:rand.uniform(100_000)}"
    tmp_dir = Path.join(System.tmp_dir!(), "migrate_training_data_test_#{test_id}")
    data_dir = Path.join(tmp_dir, "data")
    intents_dir = Path.join(data_dir, "intents")
    entities_dir = Path.join(data_dir, "entities")

    File.mkdir_p!(intents_dir)
    File.mkdir_p!(entities_dir)

    # Create minimal test data
    intent_data = [
      %{
        "text" => "hello there",
        "intent" => "greeting"
      }
    ]
    File.write!(Path.join(intents_dir, "greeting.json"), Jason.encode!(intent_data))

    entity_data = [
      %{"value" => "New York", "synonyms" => ["NYC", "NY"]}
    ]
    File.write!(Path.join(entities_dir, "location.json"), Jason.encode!(entity_data))

    # Capture Mix.shell output
    Mix.shell(Mix.Shell.Process)

    # Store original cwd to restore later
    original_cwd = File.cwd!()

    on_exit(fn ->
      # Restore original cwd before cleanup
      File.cd!(original_cwd)
      Mix.shell(Mix.Shell.IO)
      File.rm_rf!(tmp_dir)
    end)

    %{tmp_dir: tmp_dir, data_dir: data_dir, intents_dir: intents_dir, entities_dir: entities_dir}
  end

  describe "run/1 with --dry-run" do
    test "shows what directories would be created", %{tmp_dir: tmp_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        MigrateTrainingData.run(["--dry-run", "--skip-python"])

        output = collect_shell_output()

        assert output =~ "DRY RUN"
        assert output =~ "Step 1"
        assert output =~ "Would create"
      after
        File.cd!(original_cwd)
      end
    end

    test "does not create directories in dry-run mode", %{tmp_dir: tmp_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        MigrateTrainingData.run(["--dry-run", "--skip-python"])

        # Legacy and training directories should NOT exist
        refute File.exists?(Path.join(tmp_dir, "data/legacy"))
        refute File.exists?(Path.join(tmp_dir, "data/training"))
      after
        File.cd!(original_cwd)
      end
    end
  end

  describe "run/1 with --skip-python" do
    test "creates directory structure", %{tmp_dir: tmp_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        MigrateTrainingData.run(["--skip-python", "--force"])

        # Verify directories were created
        assert File.exists?(Path.join(tmp_dir, "data/legacy"))
        assert File.exists?(Path.join(tmp_dir, "data/legacy/intents"))
        assert File.exists?(Path.join(tmp_dir, "data/legacy/entities"))
        assert File.exists?(Path.join(tmp_dir, "data/training"))
        assert File.exists?(Path.join(tmp_dir, "data/training/intents"))
        assert File.exists?(Path.join(tmp_dir, "data/training/entities"))
        assert File.exists?(Path.join(tmp_dir, "data/training/pos"))
        assert File.exists?(Path.join(tmp_dir, "data/training/disambiguation"))
      after
        File.cd!(original_cwd)
      end
    end

    test "copies intent files to legacy directory", %{tmp_dir: tmp_dir, intents_dir: intents_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        MigrateTrainingData.run(["--skip-python", "--force"])

        # Verify intent file was copied to legacy
        legacy_intent = Path.join(tmp_dir, "data/legacy/intents/greeting.json")
        assert File.exists?(legacy_intent)

        # Original should still exist
        assert File.exists?(Path.join(intents_dir, "greeting.json"))
      after
        File.cd!(original_cwd)
      end
    end

    test "copies entity files to training directory", %{tmp_dir: tmp_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        MigrateTrainingData.run(["--skip-python", "--force"])

        # Verify entity file was copied to training
        training_entity = Path.join(tmp_dir, "data/training/entities/location.json")
        assert File.exists?(training_entity)
      after
        File.cd!(original_cwd)
      end
    end

    test "shows next steps in summary", %{tmp_dir: tmp_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        MigrateTrainingData.run(["--skip-python", "--force"])

        output = collect_shell_output()

        assert output =~ "Migration Complete" or output =~ "MIGRATION SUMMARY"
        assert output =~ "Next steps"
      after
        File.cd!(original_cwd)
      end
    end
  end

  describe "run/1 with --keep-legacy" do
    test "copies files instead of moving", %{tmp_dir: tmp_dir, intents_dir: intents_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        MigrateTrainingData.run(["--skip-python", "--force", "--keep-legacy"])

        # Both original and legacy should exist
        assert File.exists?(Path.join(intents_dir, "greeting.json"))
        assert File.exists?(Path.join(tmp_dir, "data/legacy/intents/greeting.json"))
      after
        File.cd!(original_cwd)
      end
    end
  end

  describe "option parsing" do
    test "parses all supported options", %{tmp_dir: tmp_dir} do
      original_cwd = File.cwd!()
      File.cd!(tmp_dir)

      try do
        # This should not raise
        MigrateTrainingData.run([
          "--dry-run",
          "--skip-python",
          "--keep-legacy",
          "--force",
          "--validate"
        ])

        output = collect_shell_output()
        assert output =~ "DRY RUN"
      after
        File.cd!(original_cwd)
      end
    end
  end

  # Helper to collect all shell output
  defp collect_shell_output do
    collect_shell_output([])
  end

  defp collect_shell_output(acc) do
    receive do
      {:mix_shell, :info, [msg]} ->
        collect_shell_output([msg | acc])

      {:mix_shell, :error, [msg]} ->
        collect_shell_output([msg | acc])
    after
      100 ->
        acc |> Enum.reverse() |> Enum.join("\n")
    end
  end
end
