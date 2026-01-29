defmodule Mix.Tasks.TrainingWorldTest do
  use ExUnit.Case, async: false

  @moduletag :mix_task
  @moduletag timeout: 30_000

  import ChatBot.TestHelpers

  setup do
    # Start required services including WorldManager
    start_world_test_services()

    # Setup world sandbox for automatic cleanup
    setup_world_sandbox()

    # Capture Mix.shell output to prevent test pollution
    Mix.shell(Mix.Shell.Process)

    on_exit(fn ->
      Mix.shell(Mix.Shell.IO)
    end)

    :ok
  end

  describe "module loading" do
    test "base module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld)
    end

    test "Create module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Create)
    end

    test "List module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.List)
    end

    test "Metrics module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Metrics)
    end

    test "Entities module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Entities)
    end

    test "Events module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Events)
    end

    test "Ambiguous module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Ambiguous)
    end

    test "Compare module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Compare)
    end

    test "Export module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Export)
    end

    test "Import module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Import)
    end

    test "Merge module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Merge)
    end

    test "Destroy module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Destroy)
    end

    test "Checkpoint module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Checkpoint)
    end

    test "Load module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Load)
    end

    test "Ingest module is loaded" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainingWorld.Ingest)
    end
  end

  describe "Mix.Tasks.TrainingWorld (base task)" do
    test "runs without error" do
      # Should not raise
      Mix.Tasks.TrainingWorld.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Create" do
    test "creates a new ephemeral world" do
      world_name = "test_world_#{:rand.uniform(100_000)}"

      Mix.Tasks.TrainingWorld.Create.run([world_name])

      worlds = ChatBot.Learning.WorldManager.list_worlds()
      assert Enum.any?(worlds, fn w -> w.name == world_name end)
    end

    test "handles missing name without crashing" do
      # Should not raise
      Mix.Tasks.TrainingWorld.Create.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.List" do
    test "runs without error" do
      # Should not raise
      Mix.Tasks.TrainingWorld.List.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Metrics" do
    test "handles missing world_id without crashing" do
      Mix.Tasks.TrainingWorld.Metrics.run([])
    end

    test "handles non-existent world" do
      Mix.Tasks.TrainingWorld.Metrics.run(["nonexistent_12345"])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Entities" do
    test "handles missing world_id without crashing" do
      Mix.Tasks.TrainingWorld.Entities.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Events" do
    test "handles missing world_id without crashing" do
      Mix.Tasks.TrainingWorld.Events.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Destroy" do
    test "handles non-existent world" do
      Mix.Tasks.TrainingWorld.Destroy.run(["nonexistent_world_12345"])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Export" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Export.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Import" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Import.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Merge" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Merge.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Ingest" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Ingest.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Load" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Load.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Checkpoint" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Checkpoint.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Compare" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Compare.run([])
    end
  end

  describe "Mix.Tasks.TrainingWorld.Ambiguous" do
    test "handles missing arguments without crashing" do
      Mix.Tasks.TrainingWorld.Ambiguous.run([])
    end
  end
end
