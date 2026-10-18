defmodule Tasks.TransformerTest do
  use ExUnit.Case, async: false

  alias Tasks.Transformer

  @test_tasks_path "test/fixtures/domain_tasks_transformer"

  # The world the test suite runs in. Extraction reads that world's entity
  # model, so every call that extracts names it.
  @world_id "default"

  setup_all do
    File.mkdir_p!(@test_tasks_path)

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

      {:ok, result} =
        Transformer.transform_task(file_path, extract_entities: false, world_id: @world_id)

      assert length(result.training_samples) == 3
      sample = Enum.find(result.training_samples, &(&1.id == "test-qa-1"))
      assert sample.text == "Who wrote Romeo and Juliet?"
      assert sample.intent == "factual_question"
      assert sample.tokens != []
      assert length(sample.pos_tags) == length(sample.tokens)
      assert sample.entities == []

      # One fact per answer: the positive example's and the two instances'.
      assert length(result.knowledge_facts) == 3

      fact = Enum.find(result.knowledge_facts, &(&1.subject =~ "Romeo"))
      assert fact.predicate == "has_answer"
      assert fact.object =~ "Shakespeare"
    end

    test "a Question Answering task without a world raises, since its answers are always extracted" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")

      assert_raise ArgumentError, ~r/needs :world_id/, fn ->
        Transformer.transform_task(file_path, extract_entities: false)
      end
    end

    test "extracting entities without a world raises" do
      file_path = Path.join(@test_tasks_path, "task_test_sentiment.json")

      assert_raise ArgumentError, ~r/needs :world_id/, fn ->
        Transformer.transform_task(file_path)
      end
    end

    test "transforms sentiment task into training samples" do
      file_path = Path.join(@test_tasks_path, "task_test_sentiment.json")

      {:ok, result} = Transformer.transform_task(file_path, extract_entities: false)

      assert length(result.training_samples) == 2
      positive_sample = Enum.find(result.training_samples, &(&1.text =~ "fantastic"))
      assert positive_sample.intent == "sentiment.positive"

      negative_sample = Enum.find(result.training_samples, &(&1.text =~ "Terrible"))
      assert negative_sample.intent == "sentiment.negative"
      assert result.knowledge_facts == []
    end

    test "transforms paraphrasing task into multiple samples" do
      file_path = Path.join(@test_tasks_path, "task_test_paraphrase.json")

      {:ok, result} = Transformer.transform_task(file_path, extract_entities: false)

      # The one instance's input and its one paraphrase.
      intents = Enum.map(result.training_samples, & &1.intent)
      assert intents == ["paraphrase.original", "paraphrase.variant"]
    end

    test "transforms commonsense task with explanations" do
      file_path = Path.join(@test_tasks_path, "task_test_commonsense.json")

      {:ok, result} = Transformer.transform_task(file_path, extract_entities: false)

      assert length(result.training_samples) == 2
      assert Enum.all?(result.training_samples, &(&1.intent == "commonsense_query"))

      # The positive example carries an explanation and the instance does not.
      predicates = result.knowledge_facts |> Enum.map(& &1.predicate) |> Enum.sort()
      assert predicates == ["commonsense_answer", "implies"]
    end

    test "respects max_instances option" do
      file_path = Path.join(@test_tasks_path, "task_test_qa.json")

      {:ok, result} =
        Transformer.transform_task(file_path,
          max_instances: 1,
          extract_entities: false,
          world_id: @world_id
        )

      assert length(result.training_samples) == 1
    end

    test "returns error for nonexistent file" do
      {:error, reason} = Transformer.transform_task("nonexistent.json")
      assert reason == :parse_failed
    end
  end

  describe "text_to_training_sample/4" do
    test "creates training sample with tokenization and POS tagging" do
      sample =
        Transformer.text_to_training_sample(
          "What is the weather today?",
          "weather.query",
          "test-id-1",
          extract_entities: false
        )

      assert sample.text == "What is the weather today?"
      assert sample.intent == "weather.query"
      assert sample.id == "test-id-1"
      assert is_list(sample.tokens)
      assert sample.tokens != []
      assert is_list(sample.pos_tags)
      assert length(sample.pos_tags) == length(sample.tokens)
    end

    test "includes metadata when provided" do
      sample =
        Transformer.text_to_training_sample(
          "Hello world",
          "greeting",
          "test-id-2",
          metadata: %{source: "test"},
          world_id: @world_id
        )

      assert sample.metadata == %{source: "test"}
    end

    test "extracting entities without a world raises, naming the option" do
      assert_raise ArgumentError, ~r/needs :world_id/, fn ->
        Transformer.text_to_training_sample("Hello world", "greeting", "test-id-3")
      end
    end

    test "extracting entities with a world id that is not a string raises" do
      for world_id <- [nil, ""] do
        assert_raise ArgumentError, ~r/:world_id must be a world id/, fn ->
          Transformer.text_to_training_sample("Hello world", "greeting", "test-id-4", world_id: world_id)
        end
      end
    end

    test "extracts in the world it is given, which must hold an entity model" do
      # No world with this id exists, so it has no entity model: extraction
      # reads the named world's model and nothing else.
      world_id = "test_no_entity_model_#{System.unique_integer([:positive])}"

      assert_raise RuntimeError, ~r/world "#{world_id}" has no entity model/, fn ->
        Transformer.text_to_training_sample("Hello world", "greeting", "test-id-5", world_id: world_id)
      end
    end

    test "needs no world when entities are not extracted" do
      sample =
        Transformer.text_to_training_sample("Hello world", "greeting", "test-id-6", extract_entities: false)

      assert sample.entities == []
    end
  end

  describe "text_to_training_sample/4 with extracted entities" do
    test "annotates each extracted entity with its type and the tokens it spans" do
      text = "What is the weather in Paris tomorrow?"
      positioned = Brain.ML.Tokenizer.tokenize(text)

      extracted =
        Brain.ML.EntityExtractor.extract_entities(text, world_id: @world_id, skip_disambiguation: true)

      assert extracted != [], "extraction decided no entity of #{inspect(text)} in world #{@world_id}"

      sample = Transformer.text_to_training_sample(text, "weather.query", "test-id-7", world_id: @world_id)

      assert sample.tokens == Enum.map(positioned, & &1.text)
      assert length(sample.entities) == length(extracted)

      for {annotation, entity} <- Enum.zip(sample.entities, extracted) do
        assert annotation["text"] == entity.value
        assert annotation["type"] == entity.entity_type
        assert Enum.at(positioned, annotation["start"]).start_pos == entity.start_pos
        assert Enum.at(positioned, annotation["end"]).end_pos == entity.end_pos
      end
    end
  end

  # Decided entities as Brain.ML.EntityExtractor.extract_entities/2 returns
  # them, built here so the tests are about the transformer.
  @decided %{
    entity_type: "city",
    value: "New York",
    match: "New York",
    start_pos: 11,
    end_pos: 18,
    confidence: 0.83,
    source: :gazetteer,
    resolved: true
  }

  describe "entity_annotation!/2" do
    test "gives the entity's type and the indexes of the first and last token it spans" do
      tokens = Brain.ML.Tokenizer.tokenize("Weather in New York today")

      assert Transformer.entity_annotation!(@decided, tokens) == %{
               "text" => "New York",
               "type" => "city",
               "start" => 2,
               "end" => 3
             }
    end

    test "an entity missing a key a decided entity carries raises, naming it" do
      tokens = Brain.ML.Tokenizer.tokenize("Weather in New York today")

      for key <- [:value, :entity_type, :start_pos, :end_pos] do
        assert_raise ArgumentError, ~r/has no #{inspect(key)}/, fn ->
          Transformer.entity_annotation!(Map.delete(@decided, key), tokens)
        end
      end
    end

    test "a type under :entity or positions under :start and :end are not read" do
      tokens = Brain.ML.Tokenizer.tokenize("Weather in New York today")
      legacy_type = @decided |> Map.delete(:entity_type) |> Map.put(:entity, "city")
      legacy_positions = @decided |> Map.drop([:start_pos, :end_pos]) |> Map.merge(%{start: 2, end: 3})

      assert_raise ArgumentError, ~r/has no :entity_type/, fn ->
        Transformer.entity_annotation!(legacy_type, tokens)
      end

      assert_raise ArgumentError, ~r/has no :start_pos/, fn ->
        Transformer.entity_annotation!(legacy_positions, tokens)
      end
    end

    test "a span that does not begin and end on token boundaries raises" do
      tokens = Brain.ML.Tokenizer.tokenize("Weather in New York today")

      assert_raise ArgumentError, ~r/token boundaries/, fn ->
        Transformer.entity_annotation!(%{@decided | start_pos: 12}, tokens)
      end
    end
  end

  describe "entity_candidate!/2" do
    test "gives the entity's value, type and confidence, and the task it came from" do
      assert Transformer.entity_candidate!(@decided, "task_1") == %{
               value: "New York",
               inferred_type: "city",
               confidence: 0.83,
               source: "task_1"
             }
    end

    test "an entity missing a key a decided entity carries raises, naming it" do
      for key <- [:value, :entity_type, :confidence] do
        assert_raise ArgumentError, ~r/has no #{inspect(key)}/, fn ->
          Transformer.entity_candidate!(Map.delete(@decided, key), "task_1")
        end
      end
    end
  end

  describe "transform_tasks/2" do
    test "transforms multiple task files" do
      files = [
        Path.join(@test_tasks_path, "task_test_qa.json"),
        Path.join(@test_tasks_path, "task_test_sentiment.json")
      ]

      {:ok, result} =
        Transformer.transform_tasks(files, extract_entities: false, world_id: @world_id)

      # Three from the Question Answering task, two from the sentiment task.
      assert length(result.training_samples) == 5
    end

    test "reports progress via callback" do
      files = [
        Path.join(@test_tasks_path, "task_test_qa.json"),
        Path.join(@test_tasks_path, "task_test_sentiment.json")
      ]

      progress_reports = :ets.new(:progress, [:set, :public])

      {:ok, _result} =
        Transformer.transform_tasks(files,
          extract_entities: false,
          world_id: @world_id,
          progress_callback: fn progress ->
            :ets.insert(progress_reports, {progress.current, progress})
          end
        )

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

      result =
        Transformer.convert_qa(instances, metadata, "", extract_entities: false, world_id: @world_id)

      assert length(result.training_samples) == 1
      assert length(result.knowledge_facts) == 1

      fact = List.first(result.knowledge_facts)
      assert fact.subject =~ "World War"
      assert fact.object == "1945"
      assert fact.predicate == "has_answer"
    end

    test "raises without a world, since answers are always extracted as entity candidates" do
      instances = [%{"id" => "qa-1", "input" => "What year did World War 2 end?", "output" => ["1945"]}]
      metadata = %{task_id: "test_qa", categories: ["Question Answering"]}

      assert_raise ArgumentError, ~r/:world_id/, fn ->
        Transformer.convert_qa(instances, metadata, "", extract_entities: false)
      end
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

      result =
        Transformer.convert_sentiment(instances, metadata, "", extract_entities: false)

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

      result =
        Transformer.convert_commonsense(instances, metadata, "", extract_entities: false)

      assert length(result.training_samples) == 1
      assert length(result.knowledge_facts) == 1

      fact = List.first(result.knowledge_facts)
      assert fact.predicate == "implies"
      assert fact.object =~ "wings"
    end
  end
end