defmodule ChatBot.Learning.TaskTransformerTest do
  use ExUnit.Case, async: true

  alias ChatBot.Learning.TaskTransformer

  @test_tasks_path "test/fixtures/domain_tasks_transformer"

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
          "input" => "What is the largest planet in our solar system?",
          "output" => ["Jupiter"]
        }
      ]
    }

    qa_file = Path.join(@test_tasks_path, "task_test_qa.json")
    File.write!(qa_file, Jason.encode!(qa_task))

    # Create a sample sentiment task file
    sentiment_task = %{
      "Contributors" => ["Test"],
      "Source" => ["test"],
      "Categories" => ["Sentiment Analysis"],
      "Domains" => ["Reviews"],
      "Definition" => ["Classify sentiment."],
      "Input_language" => ["English"],
      "Output_language" => ["English"],
      "Positive Examples" => [],
      "Negative Examples" => [],
      "Instances" => [
        %{
          "id" => "test-sent-1",
          "input" => "This movie was absolutely fantastic!",
          "output" => ["positive"]
        },
        %{
          "id" => "test-sent-2",
          "input" => "Terrible experience, would not recommend.",
          "output" => ["negative"]
        }
      ]
    }

    sentiment_file = Path.join(@test_tasks_path, "task_test_sentiment.json")
    File.write!(sentiment_file, Jason.encode!(sentiment_task))

    # Create a sample paraphrasing task file
    paraphrase_task = %{
      "Contributors" => ["Test"],
      "Source" => ["test"],
      "Categories" => ["Paraphrasing"],
      "Domains" => ["General"],
      "Definition" => ["Paraphrase the sentence."],
      "Input_language" => ["English"],
      "Output_language" => ["English"],
      "Positive Examples" => [],
      "Negative Examples" => [],
      "Instances" => [
        %{
          "id" => "test-para-1",
          "input" => "The quick brown fox jumps over the lazy dog.",
          "output" => ["A fast brown fox leaps over an idle dog."]
        }
      ]
    }

    paraphrase_file = Path.join(@test_tasks_path, "task_test_paraphrase.json")
    File.write!(paraphrase_file, Jason.encode!(paraphrase_task))

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

    on_exit(fn ->
      File.rm_rf!(@test_tasks_path)
    end)

    :ok
  end

  describe "transform_task/2" do
    test "transforms QA task into training samples and knowledge facts" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")

      {:ok, result} = TaskTransformer.transform_task(file_path, extract_entities: false)

      # Should have training samples (examples + instances)
      assert length(result.training_samples) == 3

      # Check sample structure
      sample = Enum.find(result.training_samples, &(&1.id == "test-qa-1"))
      assert sample.text == "Who wrote Romeo and Juliet?"
      assert sample.intent == "factual_question"
      assert is_list(sample.tokens)
      assert is_list(sample.pos_tags)

      # Should have knowledge facts
      assert length(result.knowledge_facts) >= 2

      fact = Enum.find(result.knowledge_facts, &(&1.subject =~ "Romeo"))
      assert fact.predicate == "has_answer"
      assert fact.object =~ "Shakespeare"
    end

    test "transforms sentiment task into training samples" do
      file_path = Path.join(@test_tasks_path, "task_test_sentiment.json")

      {:ok, result} = TaskTransformer.transform_task(file_path, extract_entities: false)

      assert length(result.training_samples) == 2

      # Check intent naming
      positive_sample = Enum.find(result.training_samples, &(&1.text =~ "fantastic"))
      assert positive_sample.intent == "sentiment.positive"

      negative_sample = Enum.find(result.training_samples, &(&1.text =~ "Terrible"))
      assert negative_sample.intent == "sentiment.negative"

      # Sentiment tasks don't generate knowledge facts
      assert result.knowledge_facts == []
    end

    test "transforms paraphrasing task into multiple samples" do
      file_path = Path.join(@test_tasks_path, "task_test_paraphrase.json")

      {:ok, result} = TaskTransformer.transform_task(file_path, extract_entities: false)

      # Should have original + paraphrase(s)
      assert length(result.training_samples) >= 2

      # Check we have both original and variant
      intents = Enum.map(result.training_samples, & &1.intent)
      assert "paraphrase.original" in intents
      assert "paraphrase.variant" in intents
    end

    test "transforms commonsense task with explanations" do
      file_path = Path.join(@test_tasks_path, "task_test_commonsense.json")

      {:ok, result} = TaskTransformer.transform_task(file_path, extract_entities: false)

      assert length(result.training_samples) == 2

      # Check sample intent
      sample = List.first(result.training_samples)
      assert sample.intent == "commonsense_query"

      # Should have knowledge facts from examples with explanations
      assert length(result.knowledge_facts) >= 1
    end

    test "respects max_instances option" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")

      {:ok, result} = TaskTransformer.transform_task(file_path,
        max_instances: 1,
        extract_entities: false
      )

      # Should only have 1 sample
      assert length(result.training_samples) == 1
    end

    test "returns error for nonexistent file" do
      {:error, reason} = TaskTransformer.transform_task("nonexistent.json")
      assert reason == :parse_failed
    end
  end

  describe "text_to_training_sample/4" do
    test "creates training sample with tokenization and POS tagging" do
      sample = TaskTransformer.text_to_training_sample(
        "What is the weather today?",
        "weather.query",
        "test-id-1",
        extract_entities: false
      )

      assert sample.text == "What is the weather today?"
      assert sample.intent == "weather.query"
      assert sample.id == "test-id-1"
      assert is_list(sample.tokens)
      assert length(sample.tokens) > 0
      assert is_list(sample.pos_tags)
      assert length(sample.pos_tags) == length(sample.tokens)
    end

    test "includes metadata when provided" do
      sample = TaskTransformer.text_to_training_sample(
        "Hello world",
        "greeting",
        "test-id-2",
        metadata: %{source: "test"}
      )

      assert sample.metadata == %{source: "test"}
    end
  end

  describe "transform_tasks/2" do
    test "transforms multiple task files" do
      files = [
        Path.join(@test_tasks_path, "task_test_qa.json"),
        Path.join(@test_tasks_path, "task_test_sentiment.json")
      ]

      {:ok, result} = TaskTransformer.transform_tasks(files, extract_entities: false)

      # Should combine samples from both files
      assert length(result.training_samples) >= 5
    end

    test "reports progress via callback" do
      files = [
        Path.join(@test_tasks_path, "task_test_qa.json"),
        Path.join(@test_tasks_path, "task_test_sentiment.json")
      ]

      progress_reports = :ets.new(:progress, [:set, :public])

      {:ok, _result} = TaskTransformer.transform_tasks(files,
        extract_entities: false,
        progress_callback: fn progress ->
          :ets.insert(progress_reports, {progress.current, progress})
        end
      )

      # Should have received progress updates
      assert :ets.info(progress_reports, :size) == 2

      :ets.delete(progress_reports)
    end
  end

  describe "convert_qa/4" do
    test "extracts question-answer pairs as knowledge facts" do
      instances = [
        %{
          "id" => "qa-1",
          "input" => "What year did World War 2 end?",
          "output" => ["1945"]
        }
      ]

      metadata = %{task_id: "test_qa", categories: ["Question Answering"]}

      result = TaskTransformer.convert_qa(instances, metadata, "", extract_entities: false)

      assert length(result.training_samples) == 1
      assert length(result.knowledge_facts) == 1

      fact = List.first(result.knowledge_facts)
      assert fact.subject =~ "World War"
      assert fact.object == "1945"
      assert fact.predicate == "has_answer"
    end
  end

  describe "convert_sentiment/4" do
    test "creates intent based on sentiment label" do
      instances = [
        %{"id" => "s1", "input" => "Great product!", "output" => ["positive"]},
        %{"id" => "s2", "input" => "Awful service.", "output" => ["negative"]},
        %{"id" => "s3", "input" => "It was okay.", "output" => ["neutral"]}
      ]

      metadata = %{task_id: "test_sent", categories: ["Sentiment Analysis"]}

      result = TaskTransformer.convert_sentiment(instances, metadata, "", extract_entities: false)

      assert length(result.training_samples) == 3

      intents = Enum.map(result.training_samples, & &1.intent) |> Enum.sort()
      assert intents == ["sentiment.negative", "sentiment.neutral", "sentiment.positive"]
    end
  end

  describe "convert_commonsense/4" do
    test "extracts reasoning facts from explanations" do
      instances = [
        %{
          "id" => "cs-1",
          "input" => "Can birds fly?",
          "output" => ["Most can"],
          "explanation" => "Most birds have wings adapted for flight."
        }
      ]

      metadata = %{task_id: "test_cs", categories: ["Commonsense Classification"]}

      result = TaskTransformer.convert_commonsense(instances, metadata, "", extract_entities: false)

      assert length(result.training_samples) == 1
      assert length(result.knowledge_facts) == 1

      fact = List.first(result.knowledge_facts)
      assert fact.predicate == "implies"
      assert fact.object =~ "wings"
    end
  end
end
