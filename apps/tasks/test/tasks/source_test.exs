defmodule Tasks.SourceTest do
  @moduledoc "Tests for Tasks.Source module.\n\nThese tests use the real shipped dataset in priv/domain_tasks/ to verify\nthe task source functionality works end-to-end with actual NLP benchmark data.\n"
  use ExUnit.Case, async: false

  alias Tasks.{Analyzer, Source}
  alias Brain.Knowledge.Types.ResearchGoal

  describe "fetch_for_goal/2" do
    test "fetches findings for a QA goal" do
      goal = ResearchGoal.new("France", questions: ["What is the capital?"])

      {:ok, findings} = Source.fetch_for_goal(goal, max_tasks: 2, max_instances: 5)
      assert is_list(findings)
    end
  end

  describe "create_training_sessions/2" do
    test "creates sessions for question_answering capability" do
      {:ok, sessions} = Source.create_training_sessions(:question_answering, max_tasks: 3)

      assert is_list(sessions)
      assert sessions != [], "Expected to find QA tasks in shipped dataset"

      session = List.first(sessions)
      assert Map.has_key?(session, :task_id)
      assert Map.has_key?(session, :categories)
      assert "Question Answering" in session.categories
    end

    test "creates sessions for commonsense capability" do
      {:ok, sessions} = Source.create_training_sessions(:commonsense, max_tasks: 3)

      assert is_list(sessions)
    end

    test "creates sessions for all capabilities" do
      {:ok, sessions} = Source.create_training_sessions(:all, max_tasks: 5)

      assert is_list(sessions)
      assert sessions != [], "Expected to find useful tasks in shipped dataset"
    end
  end

  describe "available_tasks/1" do
    test "returns tasks grouped by category" do
      {:ok, grouped} = Source.available_tasks()

      assert is_map(grouped)
      assert map_size(grouped) > 0, "Expected to find task categories in shipped dataset"
    end
  end

  describe "infer_goal_type/1 via fetch_for_goal" do
    test "handles factual goal type" do
      goal = ResearchGoal.new("France", questions: ["What is the capital of France?"])
      {:ok, findings} = Source.fetch_for_goal(goal, max_tasks: 1, max_instances: 2)
      assert is_list(findings)
    end

    test "handles reasoning goal type" do
      goal = ResearchGoal.new("Physics", questions: ["Why does ice float on water?"])
      {:ok, findings} = Source.fetch_for_goal(goal, max_tasks: 1, max_instances: 2)
      assert is_list(findings)
    end
  end

  describe "Analyzer.default_tasks_path/0" do
    test "returns a path that exists and contains task files" do
      path = Analyzer.default_tasks_path()

      assert File.dir?(path), "Expected #{path} to be a directory"

      files = Path.wildcard(Path.join(path, "*.json"))
      assert files != [], "Expected to find JSON files in #{path}"
    end
  end
end