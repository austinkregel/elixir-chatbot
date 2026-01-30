defmodule ChatBot.Testing.CapabilityTest do
  @moduledoc """
  Tests NLP capabilities using domain-specific benchmark tasks.

  This module implements the scientific method for capability testing:
  1. **Hypothesis**: Our NLP system can handle task X
  2. **Prediction**: Given input, system will produce expected output
  3. **Investigation**: Run tests against multiple task instances
  4. **Evaluation**: Measure accuracy and track failures
  5. **Conclusion**: Capability supported, falsified, or needs improvement

  ## Capabilities Tested

  - **Question Answering**: Extract answers from passages
  - **Entity Recognition**: Identify named entities
  - **Temporal Reasoning**: Understand time relationships
  - **Coreference Resolution**: Resolve pronouns to entities
  - **Sentiment Analysis**: Detect emotional tone

  ## Example

      # Test question answering capability
      {:ok, results} = CapabilityTest.test_capability(:question_answering, limit: 50)
      
      # Run full benchmark
      {:ok, report} = CapabilityTest.run_benchmark()
  """

  require Logger

  alias ChatBot.Learning.TaskAnalyzer
  alias ChatBot.Testing.TaskSchema
  alias ChatBot.Analysis.Pipeline
  alias ChatBot.ML.{Tokenizer, POSTagger, Gazetteer, IntentClassifierSimple}
  alias ChatBot.Knowledge.Types.{Investigation, Hypothesis}

  @tasks_dir "data/domain_specific_tasks"

  defp tasks_path do
    Path.join(File.cwd!(), @tasks_dir)
  end

  # Capability mappings to task categories
  @capability_categories %{
    question_answering: ["Question Answering", "Reading Comprehension"],
    entity_recognition: ["Named Entity Recognition", "Entity Detection"],
    temporal_reasoning: ["Temporal Reasoning"],
    coreference: ["Coreference Resolution"],
    sentiment: ["Sentiment Analysis", "Emotion Detection"],
    commonsense: ["Commonsense Reasoning", "Reasoning"],
    classification: ["Text Classification", "Classification"]
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Tests a specific capability using relevant benchmark tasks.

  ## Options
    - :limit - Maximum instances per task (default: 20)
    - :max_tasks - Maximum task files to use (default: 5)
    - :verbose - Log detailed results (default: false)

  ## Returns

  Investigation struct with hypothesis results.
  """
  @spec test_capability(atom(), keyword()) :: {:ok, Investigation.t()} | {:error, term()}
  def test_capability(capability, opts \\ []) when is_atom(capability) do
    limit = Keyword.get(opts, :limit, 20)
    max_tasks = Keyword.get(opts, :max_tasks, 5)
    verbose = Keyword.get(opts, :verbose, false)

    categories = Map.get(@capability_categories, capability, [])

    if categories == [] do
      {:error, {:unknown_capability, capability}}
    else
      Logger.info("Testing capability: #{capability}",
        categories: categories,
        limit: limit
      )

      # Find relevant tasks
      tasks = find_tasks_for_categories(categories, max_tasks)

      if tasks == [] do
        {:error, {:no_tasks_found, capability}}
      else
        # Create investigation
        investigation = Investigation.new("Capability: #{capability}")

        # For each task: create hypothesis, run tests, add evidence directly
        investigation =
          tasks
          |> Enum.reduce(investigation, fn task_file, inv ->
            test_task_with_hypothesis(inv, task_file, capability, limit, verbose)
          end)

        # Conclude the investigation
        investigation = conclude_investigation(investigation)

        log_investigation_summary(investigation)

        {:ok, investigation}
      end
    end
  end

  @doc """
  Runs a full benchmark across all capabilities.

  ## Options
    - :capabilities - List of capabilities to test (default: all)
    - :limit - Instances per task (default: 10)
    - :max_tasks - Tasks per capability (default: 3)

  ## Returns

  Map of capability -> Investigation results.
  """
  @spec run_benchmark(keyword()) :: {:ok, map()}
  def run_benchmark(opts \\ []) do
    capabilities = Keyword.get(opts, :capabilities, Map.keys(@capability_categories))
    limit = Keyword.get(opts, :limit, 10)
    max_tasks = Keyword.get(opts, :max_tasks, 3)

    Logger.info("Starting capability benchmark",
      capabilities: capabilities
    )

    results =
      capabilities
      |> Enum.reduce(%{}, fn capability, acc ->
        case test_capability(capability, limit: limit, max_tasks: max_tasks) do
          {:ok, investigation} ->
            Map.put(acc, capability, investigation)
          {:error, _} ->
            acc
        end
      end)

    report = generate_benchmark_report(results)
    Logger.info("Benchmark complete", report: report)

    {:ok, results}
  end

  @doc """
  Lists available capabilities that can be tested.
  """
  @spec list_capabilities() :: [atom()]
  def list_capabilities do
    Map.keys(@capability_categories)
  end

  @doc """
  Gets task statistics by category.
  """
  @spec task_stats() :: map()
  def task_stats do
    case list_task_files() do
      {:ok, files} ->
        files
        |> Enum.reduce(%{}, fn file, acc ->
          case get_task_metadata(file) do
            %{categories: categories} when is_list(categories) ->
              Enum.reduce(categories, acc, fn cat, a ->
                Map.update(a, cat, 1, &(&1 + 1))
              end)
            _ ->
              acc
          end
        end)
      _ ->
        %{}
    end
  end

  defp list_task_files do
    path = tasks_path()
    if File.dir?(path) do
      files =
        path
        |> File.ls!()
        |> Enum.filter(&String.ends_with?(&1, ".json"))
        |> Enum.map(&Path.join(path, &1))
      {:ok, files}
    else
      {:error, :not_found}
    end
  end

  # Load full task file with instances
  defp load_full_task(file_path) do
    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} -> {:ok, data}
          {:error, _} -> {:error, :parse_error}
        end
      {:error, _} -> {:error, :read_error}
    end
  end

  # Get task metadata for filtering
  defp get_task_metadata(file_path) do
    TaskAnalyzer.parse_task_file(file_path)
  end

  # ============================================================================
  # Private Functions - Task Discovery
  # ============================================================================

  defp find_tasks_for_categories(categories, max_tasks) do
    case list_task_files() do
      {:ok, files} ->
        files
        |> Enum.filter(fn file ->
          case get_task_metadata(file) do
            %{categories: task_categories} when is_list(task_categories) ->
              Enum.any?(categories, &(&1 in task_categories))
            _ ->
              false
          end
        end)
        |> Enum.take(max_tasks)

      _ ->
        []
    end
  end

  defp test_task_with_hypothesis(investigation, task_file, capability, limit, verbose) do
    task_name = Path.basename(task_file, ".json")

    # Load and parse the task using the schema
    case load_full_task(task_file) do
      {:ok, raw_task} ->
        {:ok, task} = TaskSchema.parse(raw_task)

        # Extract task metadata for better hypothesis
        definition = TaskSchema.get_definition(task) |> String.slice(0, 100)
        reasoning_types = TaskSchema.get_reasoning_types(task)
        patterns = TaskSchema.extract_patterns(task)

        # Create hypothesis with task-specific details
        hypothesis = Hypothesis.new(
          "System can perform #{capability} on #{task_name}",
          entity: to_string(capability),
          derived_from: definition,
          prediction: build_prediction(capability, reasoning_types, patterns)
        )

        # Get positive examples for calibration (helps understand expected output format)
        positive_examples = TaskSchema.get_positive_examples(task, 3)

        # Run tests for this task with examples context
        results = test_task_instances(task, capability, limit, positive_examples, verbose)

        # Calculate pass/fail metrics
        passed = Enum.count(results, & &1.passed)
        failed = length(results) - passed
        total = length(results)

        # Create evidence findings based on results
        alias ChatBot.Knowledge.Types.{Finding, SourceInfo}

        source = SourceInfo.new("test://#{task_name}",
          title: task_name,
          reliability_score: 1.0
        )

        # Add supporting/contradicting evidence with actual test details
        hypothesis =
          if total > 0 do
            # Add passed tests as supporting evidence
            hypothesis =
              results
              |> Enum.filter(& &1.passed)
              |> Enum.reduce(hypothesis, fn result, hyp ->
                finding = Finding.new(
                  "Passed: expected #{inspect(Enum.take(result.expected, 2))}, got match",
                  task_name,
                  source,
                  confidence: 1.0,
                  raw_context: String.slice(result.input, 0, 100)
                )
                Hypothesis.add_supporting_evidence(hyp, finding)
              end)

            # Add failed tests as contradicting evidence
            hypothesis =
              results
              |> Enum.reject(& &1.passed)
              |> Enum.reduce(hypothesis, fn result, hyp ->
                finding = Finding.new(
                  "Failed: expected #{inspect(Enum.take(result.expected, 2))}, got #{inspect(Enum.take(result.actual, 2))}",
                  task_name,
                  source,
                  confidence: 1.0,
                  raw_context: String.slice(result.input, 0, 100)
                )
                Hypothesis.add_contradicting_evidence(hyp, finding)
              end)

            hypothesis
          else
            hypothesis
          end

        # Evaluate the hypothesis
        hypothesis = Hypothesis.evaluate(hypothesis)

        if verbose do
          Logger.debug("Task hypothesis evaluated",
            task: task_name,
            definition: definition,
            reasoning: reasoning_types,
            positive_examples: patterns.positive_count,
            status: hypothesis.status,
            confidence: hypothesis.confidence,
            passed: passed,
            failed: failed
          )
        end

        # Add hypothesis to investigation
        Investigation.add_hypothesis(investigation, hypothesis)

      {:error, _} ->
        # Skip tasks that can't be loaded
        investigation
    end
  end

  defp build_prediction(capability, reasoning_types, patterns) do
    reasoning_str = if reasoning_types != [], do: " using #{Enum.join(reasoning_types, ", ")}", else: ""
    examples_str = if patterns.positive_count > 0, do: " (#{patterns.positive_count} examples available)", else: ""

    "If the system has #{capability} capability#{reasoning_str}, it will correctly process task instances#{examples_str}."
  end

  defp test_task_instances(task, capability, limit, positive_examples, verbose) do
    instances = TaskSchema.get_instances(task, limit)

    instances
    |> Enum.map(fn instance ->
      result = test_instance_with_examples(instance, capability, positive_examples)

      if verbose do
        Logger.debug("Test result",
          passed: result.passed,
          expected: Enum.take(result.expected, 2),
          actual: Enum.take(result.actual, 2)
        )
      end

      result
    end)
  end

  defp test_instance_with_examples(instance, capability, _positive_examples) do
    # For now, use the standard test logic
    # Future: Use positive_examples to calibrate expected output format
    input = instance.input
    expected = instance.output |> Enum.map(&String.downcase/1) |> Enum.map(&String.trim/1)

    actual = run_capability_test(capability, input)

    passed = evaluate_result(expected, actual, capability)

    %{
      input: input,
      expected: expected,
      actual: actual,
      passed: passed,
      capability: capability
    }
  end

  defp conclude_investigation(investigation) do
    # Determine overall conclusion based on hypothesis statuses
    supported = Enum.count(investigation.hypotheses, &(&1.status == :supported))
    falsified = Enum.count(investigation.hypotheses, &(&1.status == :falsified))
    total = length(investigation.hypotheses)

    conclusion =
      cond do
        total == 0 -> :inconclusive
        supported == total -> :hypotheses_supported
        falsified == total -> :hypotheses_falsified
        supported > falsified -> :mixed
        true -> :inconclusive
      end

    %{investigation |
      status: :concluded,
      conclusion: conclusion,
      concluded_at: DateTime.utc_now()
    }
  end

  # ============================================================================
  # Private Functions - Capability-Specific Tests
  # ============================================================================

  defp run_capability_test(:question_answering, input) do
    # Extract question and passage
    {passage, question} = parse_qa_input(input)

    # Use our pipeline to analyze
    case Pipeline.process(question, analyze_discourse: false) do
      {:ok, analysis} ->
        # Try to find answer using entity extraction and analysis
        extract_answer_candidates(passage, question, analysis)
      _ ->
        []
    end
  end

  defp run_capability_test(:entity_recognition, input) do
    # Use Gazetteer and entity extraction via lookup_spans
    tokens = Tokenizer.tokenize_words(input)

    case Gazetteer.lookup_spans(tokens) do
      {:ok, spans} ->
        spans
        |> Enum.map(fn %{text: text} -> String.downcase(text) end)
      _ ->
        # Fallback: use POS tagger to find proper nouns
        extract_names_from_sentence(input)
        |> Enum.map(&String.downcase/1)
    end
  end

  defp run_capability_test(:sentiment, input) do
    # Basic sentiment analysis using keyword heuristics
    analyze_sentiment(input)
  end

  defp run_capability_test(:classification, input) do
    # Use intent classifier
    case IntentClassifierSimple.classify(input) do
      {:ok, intent, _confidence} -> [String.downcase(intent)]
      _ -> []
    end
  end

  defp run_capability_test(:coreference, input) do
    # Extract pronouns and potential referents
    resolve_coreferences(input)
  end

  defp run_capability_test(:temporal_reasoning, input) do
    # Extract temporal expressions
    extract_temporal_expressions(input)
  end

  defp run_capability_test(:commonsense, input) do
    # Use pipeline for commonsense reasoning
    case Pipeline.process(input, analyze_discourse: false) do
      {:ok, analysis} ->
        extract_commonsense_answer(analysis)
      _ ->
        []
    end
  end

  defp run_capability_test(_, _input) do
    []
  end

  # ============================================================================
  # Private Functions - QA Helpers
  # ============================================================================

  defp parse_qa_input(input) do
    # Many tasks use "Passage: ... Question: ..." format
    case String.split(input, ~r/\nQuestion:\s*/i, parts: 2) do
      [passage_part, question] ->
        passage = String.replace(passage_part, ~r/^Passage:\s*/i, "")
        {String.trim(passage), String.trim(question)}
      _ ->
        {input, input}
    end
  end

  defp extract_answer_candidates(passage, question, _analysis) do
    # Tokenize question and passage
    question_tokens = Tokenizer.tokenize_words(question) |> MapSet.new()
    passage_sentences = Tokenizer.split_sentences(passage)

    # Find sentences with most overlap to question
    passage_sentences
    |> Enum.map(fn sentence ->
      sentence_tokens = Tokenizer.tokenize_words(sentence) |> MapSet.new()
      overlap = MapSet.intersection(question_tokens, sentence_tokens) |> MapSet.size()
      {sentence, overlap}
    end)
    |> Enum.sort_by(fn {_, overlap} -> -overlap end)
    |> Enum.take(3)
    |> Enum.flat_map(fn {sentence, _} ->
      # Extract entities/names from best matching sentences
      extract_names_from_sentence(sentence)
    end)
    |> Enum.map(&String.downcase/1)
    |> Enum.uniq()
  end

  defp extract_names_from_sentence(sentence) do
    tokens = Tokenizer.tokenize_words(sentence)

    case POSTagger.load_model() do
      {:ok, model} ->
        tags = POSTagger.predict_tags(tokens, model)

        Enum.zip(tokens, tags)
        |> Enum.filter(fn {_token, tag} -> tag in ["PROPN", "NOUN"] end)
        |> Enum.map(fn {token, _} -> token end)

      _ ->
        # Fallback: capitalize words are likely names
        tokens
        |> Enum.filter(fn token ->
          first_char = String.first(token) || ""
          String.upcase(first_char) == first_char and first_char != ""
        end)
    end
  end

  # ============================================================================
  # Private Functions - Sentiment Analysis
  # ============================================================================

  defp analyze_sentiment(text) do
    lower = String.downcase(text)

    positive_words = ~w(good great excellent happy positive love like best wonderful amazing)
    negative_words = ~w(bad terrible awful sad negative hate dislike worst horrible)

    positive_count = Enum.count(positive_words, &String.contains?(lower, &1))
    negative_count = Enum.count(negative_words, &String.contains?(lower, &1))

    cond do
      positive_count > negative_count -> ["positive"]
      negative_count > positive_count -> ["negative"]
      true -> ["neutral"]
    end
  end

  # ============================================================================
  # Private Functions - Coreference Resolution
  # ============================================================================

  defp resolve_coreferences(text) do
    tokens = Tokenizer.tokenize_words(text)

    case POSTagger.load_model() do
      {:ok, model} ->
        tags = POSTagger.predict_tags(tokens, model)

        # Find pronouns (for future more sophisticated resolution)
        _pronouns = Enum.zip(tokens, tags)
          |> Enum.filter(fn {_token, tag} -> tag == "PRON" end)
          |> Enum.map(fn {token, _} -> token end)

        # Find proper nouns (potential referents)
        proper_nouns = Enum.zip(tokens, tags)
          |> Enum.filter(fn {_token, tag} -> tag == "PROPN" end)
          |> Enum.map(fn {token, _} -> token end)

        # Simple heuristic: return proper nouns as resolved references
        proper_nouns
        |> Enum.map(&String.downcase/1)
        |> Enum.uniq()

      _ ->
        []
    end
  end

  # ============================================================================
  # Private Functions - Temporal Reasoning
  # ============================================================================

  defp extract_temporal_expressions(text) do
    tokens = Tokenizer.tokenize_words(text)

    # Common temporal patterns
    temporal_words = ~w(before after during while when until since now yesterday tomorrow today)

    tokens
    |> Enum.filter(fn token ->
      String.downcase(token) in temporal_words
    end)
    |> Enum.map(&String.downcase/1)
  end

  # ============================================================================
  # Private Functions - Commonsense
  # ============================================================================

  defp extract_commonsense_answer(analysis) do
    # Use entities from analysis as potential answers
    chunks = Map.get(analysis, :chunks, [])

    chunks
    |> Enum.flat_map(fn chunk ->
      entities = Map.get(chunk, :entities, [])
      Enum.map(entities, fn entity ->
        String.downcase(Map.get(entity, :text, ""))
      end)
    end)
    |> Enum.reject(&(&1 == ""))
    |> Enum.uniq()
  end

  # ============================================================================
  # Private Functions - Evaluation
  # ============================================================================

  defp evaluate_result(expected, actual, _capability) do
    # Check if any actual answer matches any expected answer
    expected_set = MapSet.new(expected)
    actual_set = MapSet.new(actual)

    # Exact match
    if not MapSet.disjoint?(expected_set, actual_set) do
      true
    else
      # Partial match: check if any expected is contained in any actual
      Enum.any?(expected, fn exp ->
        Enum.any?(actual, fn act ->
          String.contains?(act, exp) or String.contains?(exp, act)
        end)
      end)
    end
  end

  defp log_investigation_summary(investigation) do
    summary = Investigation.summary(investigation)

    Logger.info("Capability test complete",
      topic: summary.topic,
      hypotheses: summary.total_hypotheses,
      supported: summary.supported,
      falsified: summary.falsified,
      conclusion: summary.conclusion
    )
  end

  defp generate_benchmark_report(results) do
    results
    |> Enum.map(fn {capability, investigation} ->
      summary = Investigation.summary(investigation)
      {capability, %{
        hypotheses: summary.total_hypotheses,
        supported: summary.supported,
        falsified: summary.falsified,
        conclusion: summary.conclusion
      }}
    end)
    |> Map.new()
  end
end
