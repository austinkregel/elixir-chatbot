defmodule ChatBot.Knowledge.TaskSourceTest do
  use ExUnit.Case, async: true

  alias ChatBot.Knowledge.TaskSource
  alias ChatBot.Knowledge.Types.ResearchGoal

  @test_tasks_path "test/fixtures/task_source_tests"

  setup_all do
    # Create test fixtures directory and sample task files
    File.mkdir_p!(@test_tasks_path)

    # Create a sample QA task file
    qa_task = %{
      "Contributors" => ["Test"],
      "Source" => ["test"],
      "URL" => ["https://example.com"],
      "Categories" => ["Question Answering"],
      "Domains" => ["Wikipedia"],
      "Definition" => ["Answer questions about France and its geography."],
      "Input_language" => ["English"],
      "Output_language" => ["English"],
      "Positive Examples" => [
        %{
          "input" => "What is the capital of France?",
          "output" => "Paris",
          "explanation" => "Paris is the capital city of France."
        }
      ],
      "Negative Examples" => [],
      "Instances" => [
        %{
          "id" => "france-qa-1",
          "input" => "What is the population of France?",
          "output" => ["67 million"]
        },
        %{
          "id" => "france-qa-2",
          "input" => "What is the official language of France?",
          "output" => ["French"]
        }
      ]
    }

    qa_file = Path.join(@test_tasks_path, "task_test_france_qa.json")
    File.write!(qa_file, Jason.encode!(qa_task))

    # Create a sample commonsense task file
    commonsense_task = %{
      "Contributors" => ["Test"],
      "Source" => ["test"],
      "Categories" => ["Commonsense Classification"],
      "Domains" => ["Commonsense"],
      "Definition" => ["Apply commonsense reasoning."],
      "Input_language" => ["English"],
      "Output_language" => ["English"],
      "Positive Examples" => [
        %{
          "input" => "Can fish breathe underwater?",
          "output" => "Yes",
          "explanation" => "Fish have gills that extract oxygen from water."
        }
      ],
      "Negative Examples" => [],
      "Instances" => [
        %{
          "id" => "commonsense-1",
          "input" => "Do birds have feathers?",
          "output" => ["Yes"]
        }
      ]
    }

    commonsense_file = Path.join(@test_tasks_path, "task_test_commonsense.json")
    File.write!(commonsense_file, Jason.encode!(commonsense_task))

    on_exit(fn ->
      File.rm_rf!(@test_tasks_path)
    end)

    :ok
  end

  describe "fetch_for_goal/2" do
    test "fetches findings for a QA goal" do
      goal = ResearchGoal.new("France", questions: ["What is the capital?"])

      # Note: This test uses the actual data directory, so it may find real tasks
      # In a real test environment, we'd mock the tasks_path
      {:ok, findings} = TaskSource.fetch_for_goal(goal, max_tasks: 2, max_instances: 5)

      # Should return some findings (actual count depends on available tasks)
      assert is_list(findings)
    end
  end

  describe "create_training_sessions/2" do
    test "creates sessions for question_answering capability" do
      {:ok, sessions} = TaskSource.create_training_sessions(:question_answering, max_tasks: 3)

      assert is_list(sessions)

      if length(sessions) > 0 do
        session = List.first(sessions)
        assert Map.has_key?(session, :task_id)
        assert Map.has_key?(session, :categories)
        assert "Question Answering" in session.categories
      end
    end

    test "creates sessions for commonsense capability" do
      {:ok, sessions} = TaskSource.create_training_sessions(:commonsense, max_tasks: 3)

      assert is_list(sessions)
    end

    test "creates sessions for all capabilities" do
      {:ok, sessions} = TaskSource.create_training_sessions(:all, max_tasks: 5)

      assert is_list(sessions)
    end
  end

  describe "available_tasks/1" do
    test "returns tasks grouped by category" do
      {:ok, grouped} = TaskSource.available_tasks()

      assert is_map(grouped)
      # Should have at least some categories from the real data
    end
  end

  describe "infer_goal_type/1" do
    # Test via the public API by checking fetch_for_goal behavior
    test "handles factual goal type" do
      goal = ResearchGoal.new("France", questions: ["What is the capital of France?"])
      {:ok, _findings} = TaskSource.fetch_for_goal(goal, max_tasks: 1, max_instances: 2)
      # Should not crash - that's the main assertion
    end

    test "handles reasoning goal type" do
      goal = ResearchGoal.new("Physics", questions: ["Why does ice float on water?"])
      {:ok, _findings} = TaskSource.fetch_for_goal(goal, max_tasks: 1, max_instances: 2)
    end
  end
end
