defmodule Tasks.AnalyzerTest do
  use ExUnit.Case, async: false

  alias Tasks.Analyzer

  @test_tasks_path "test/fixtures/domain_tasks"

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
      "Definition" => ["Answer the question."],
      "Input_language" => ["English"],
      "Output_language" => ["English"],
      "Positive Examples" => [
        %{
          "input" => "What is the capital of France?",
          "output" => "Paris",
          "explanation" => "Paris is the capital of France."
        }
      ],
      "Negative Examples" => [],
      "Instances" => [
        %{
          "id" => "test-qa-1",
          "input" => "Who wrote Romeo and Juliet?",
          "output" => ["William Shakespeare"]
        },
        %{
          "id" => "test-qa-2",
          "input" => "What is the largest planet?",
          "output" => ["Jupiter"]
        }
      ]
    }

    qa_file = Path.join(@test_tasks_path, "task_test_qa.json")
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
          "id" => "test-cs-1",
          "input" => "Is ice colder than steam?",
          "output" => ["Yes"]
        }
      ]
    }

    commonsense_file = Path.join(@test_tasks_path, "task_test_commonsense.json")
    File.write!(commonsense_file, Jason.encode!(commonsense_task))

    # Create a translation task (should be skipped)
    translation_task = %{
      "Contributors" => ["Test"],
      "Source" => ["test"],
      "Categories" => ["Translation"],
      "Domains" => ["Wikipedia"],
      "Definition" => ["Translate to Spanish."],
      "Input_language" => ["English"],
      "Output_language" => ["Spanish"],
      "Positive Examples" => [],
      "Negative Examples" => [],
      "Instances" => [
        %{
          "id" => "test-trans-1",
          "input" => "Hello world",
          "output" => ["Hola mundo"]
        }
      ]
    }

    translation_file = Path.join(@test_tasks_path, "task_test_translation.json")
    File.write!(translation_file, Jason.encode!(translation_task))

    on_exit(fn ->
      File.rm_rf!(@test_tasks_path)
    end)

    :ok
  end

  describe "analyze_all/1" do
    test "analyzes all task files in directory" do
      {:ok, result} = Analyzer.analyze_all(tasks_path: @test_tasks_path)

      assert result.total_tasks == 3
      assert length(result.useful_tasks) == 2
      assert length(result.skipped_tasks) == 1
    end

    test "filters by english only" do
      {:ok, result} = Analyzer.analyze_all(tasks_path: @test_tasks_path, english_only: true)

      # Translation task outputs Spanish, so only 2 are english-only
      assert result.english_only == 2
    end

    test "counts tasks by category" do
      {:ok, result} = Analyzer.analyze_all(tasks_path: @test_tasks_path)

      assert Map.get(result.by_category, "Question Answering") == 1
      assert Map.get(result.by_category, "Commonsense Classification") == 1
      assert Map.get(result.by_category, "Translation") == 1
    end
  end

  describe "parse_task_file/1" do
    test "parses task file and extracts metadata" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")
      metadata = Analyzer.parse_task_file(file_path)

      assert metadata.task_id == "task_test_qa"
      assert metadata.categories == ["Question Answering"]
      assert metadata.domains == ["Wikipedia"]
      assert metadata.input_language == "English"
      assert metadata.output_language == "English"
      assert metadata.instance_count == 2
      assert metadata.positive_example_count == 1
    end

    test "returns nil for invalid file" do
      metadata = Analyzer.parse_task_file("nonexistent.json")
      assert is_nil(metadata)
    end
  end

  describe "useful_task?/1" do
    test "returns true for useful categories" do
      task = %{
        categories: ["Question Answering"],
        domains: ["Wikipedia"],
        input_language: "English",
        output_language: "English"
      }

      assert Analyzer.useful_task?(task)
    end

    test "returns false for skip categories" do
      task = %{
        categories: ["Translation"],
        domains: ["Wikipedia"],
        input_language: "English",
        output_language: "Spanish"
      }

      refute Analyzer.useful_task?(task)
    end
  end

  describe "english_task?/1" do
    test "returns true for english-only task" do
      task = %{
        input_language: "English",
        output_language: "English"
      }

      assert Analyzer.english_task?(task)
    end

    test "returns false for non-english task" do
      task = %{
        input_language: "English",
        output_language: "German"
      }

      refute Analyzer.english_task?(task)
    end
  end

  describe "load_instances/2" do
    test "loads instances from task file" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")
      {:ok, instances} = Analyzer.load_instances(file_path)

      # Should include examples + instances
      assert length(instances) == 3
    end

    test "respects max_instances option" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")
      {:ok, instances} = Analyzer.load_instances(file_path, max_instances: 2)

      assert length(instances) == 2
    end

    test "can exclude examples" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")
      {:ok, instances} = Analyzer.load_instances(file_path, include_examples: false)

      # Only instances, no examples
      assert length(instances) == 2
    end
  end

  describe "get_definition/1" do
    test "extracts task definition" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")
      {:ok, definition} = Analyzer.get_definition(file_path)

      assert definition == "Answer the question."
    end
  end

  describe "group_by_category/1" do
    test "groups tasks by category" do
      tasks = [
        %{categories: ["Question Answering"], task_id: "t1"},
        %{categories: ["Question Answering"], task_id: "t2"},
        %{categories: ["Commonsense Classification"], task_id: "t3"}
      ]

      grouped = Analyzer.group_by_category(tasks)

      assert length(grouped["Question Answering"]) == 2
      assert length(grouped["Commonsense Classification"]) == 1
    end
  end
end
