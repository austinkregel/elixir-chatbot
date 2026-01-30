defmodule Mix.Tasks.ValidateNlp do
  @moduledoc """
  Validates NLP capabilities against benchmark task files.

  This task uses domain-specific benchmark files for their intended purpose:
  testing and validating our NLP system's capabilities.

  ## Usage

      # Run all validation tasks
      mix validate_nlp

      # Run a specific task file
      mix validate_nlp --task task019_mctaco_temporal_reasoning_category

      # Run with more instances
      mix validate_nlp --limit 100

      # Run with verbose output
      mix validate_nlp --verbose

      # List available tasks by category
      mix validate_nlp --list

  ## Output

  Reports accuracy metrics for each capability tested:
  - Passed/Failed counts
  - Accuracy percentage
  - Sample failures for debugging
  """

  use Mix.Task

  alias ChatBot.Testing.{TaskSchema, TaskSolver}
  alias ChatBot.Analysis.Pipeline
  alias ChatBot.ML.{Tokenizer, POSTagger}

  @shortdoc "Validates NLP capabilities against benchmark tasks"

  @tasks_dir "data/domain_specific_tasks"

  # Map task categories to our validation functions
  @validators %{
    "Question Understanding" => :validate_question_understanding,
    "Temporal Reasoning" => :validate_temporal_reasoning,
    "Named Entity Recognition" => :validate_entity_recognition,
    "Sentiment Analysis" => :validate_sentiment,
    "Text Classification" => :validate_classification,
    "Pos Tagging" => :validate_pos_tagging,
    "Commonsense Reasoning" => :validate_commonsense,
    # Textual entailment / NLI
    "Textual Entailment" => :validate_entailment,
    # Summarization using our extractive approach
    "Summarization" => :validate_summarization,
    # Generative tasks that truly need LLM
    "Question Generation" => :validate_generative_unsupported,
    "Text Completion" => :validate_generative_unsupported,
    "Dialogue Generation" => :validate_generative_unsupported
  }

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      switches: [
        task: :string,
        limit: :integer,
        verbose: :boolean,
        list: :boolean,
        category: :string
      ],
      aliases: [t: :task, l: :limit, v: :verbose, c: :category]
    )

    # Start the application
    Mix.Task.run("app.start")

    cond do
      opts[:list] ->
        list_available_tasks()

      opts[:task] ->
        run_single_task(opts[:task], opts)

      opts[:category] ->
        run_category(opts[:category], opts)

      true ->
        run_quick_validation(opts)
    end
  end

  # ============================================================================
  # Task Discovery
  # ============================================================================

  defp list_available_tasks do
    tasks = load_task_index()

    Mix.shell().info("\n=== Available Validation Tasks ===\n")

    tasks
    |> Enum.group_by(fn {_file, meta} -> List.first(meta.categories) || "Unknown" end)
    |> Enum.sort_by(fn {cat, _} -> cat end)
    |> Enum.each(fn {category, task_list} ->
      supported = if Map.has_key?(@validators, category), do: " [SUPPORTED]", else: ""
      Mix.shell().info("#{category}#{supported}")

      task_list
      |> Enum.take(5)
      |> Enum.each(fn {file, meta} ->
        task_id = Path.basename(file, ".json")
        Mix.shell().info("  - #{task_id} (#{meta.instance_count} instances)")
      end)

      if length(task_list) > 5 do
        Mix.shell().info("  ... and #{length(task_list) - 5} more")
      end

      Mix.shell().info("")
    end)

    supported_count = Enum.count(tasks, fn {_, meta} ->
      Enum.any?(meta.categories, &Map.has_key?(@validators, &1))
    end)

    Mix.shell().info("Total: #{length(tasks)} tasks, #{supported_count} with validators")
  end

  defp load_task_index do
    tasks_path()
    |> File.ls!()
    |> Enum.filter(&String.ends_with?(&1, ".json"))
    |> Enum.map(fn file ->
      path = Path.join(tasks_path(), file)
      case load_task_metadata(path) do
        {:ok, meta} -> {file, meta}
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp load_task_metadata(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            {:ok, %{
              categories: Map.get(data, "Categories", []),
              reasoning: Map.get(data, "Reasoning", []),
              instance_count: length(Map.get(data, "Instances", []))
            }}
          _ -> {:error, :parse_error}
        end
      _ -> {:error, :read_error}
    end
  end

  # ============================================================================
  # Running Validation
  # ============================================================================

  defp run_single_task(task_name, opts) do
    task_file = find_task_file(task_name)

    case task_file do
      nil ->
        Mix.shell().error("Task not found: #{task_name}")
        Mix.shell().info("Use --list to see available tasks")

      path ->
        Mix.shell().info("\n=== Validating: #{task_name} ===\n")
        run_validation(path, opts)
    end
  end

  defp run_category(category, opts) do
    tasks = load_task_index()
    |> Enum.filter(fn {_, meta} -> category in meta.categories end)

    if Enum.empty?(tasks) do
      Mix.shell().error("No tasks found for category: #{category}")
    else
      Mix.shell().info("\n=== Validating Category: #{category} ===")
      Mix.shell().info("Found #{length(tasks)} tasks\n")

      results = Enum.map(tasks, fn {file, _} ->
        path = Path.join(tasks_path(), file)
        {file, run_validation(path, Keyword.put(opts, :quiet, true))}
      end)

      # Summary
      print_category_summary(category, results)
    end
  end

  defp run_quick_validation(opts) do
    Mix.shell().info("\n=== Quick NLP Validation ===\n")

    # Run one task from each supported category
    tasks = load_task_index()

    @validators
    |> Map.keys()
    |> Enum.each(fn category ->
      case Enum.find(tasks, fn {_, meta} -> category in meta.categories end) do
        {file, _meta} ->
          Mix.shell().info("Testing #{category}...")
          path = Path.join(tasks_path(), file)
          run_validation(path, Keyword.merge(opts, limit: opts[:limit] || 20, quiet: true))

        nil ->
          Mix.shell().info("No tasks available for: #{category}")
      end
    end)
  end

  defp run_validation(path, opts) do
    limit = Keyword.get(opts, :limit, 50)
    verbose = Keyword.get(opts, :verbose, false)
    quiet = Keyword.get(opts, :quiet, false)

    case load_task(path) do
      {:ok, task} ->
        category = TaskSchema.get_primary_category(task)
        validator = Map.get(@validators, category, :validate_generic)

        # Show task info
        unless quiet do
          Mix.shell().info("Category: #{category}")
          Mix.shell().info("Definition: #{TaskSchema.get_definition(task) |> String.slice(0, 100)}...")
          Mix.shell().info("Reasoning: #{Enum.join(TaskSchema.get_reasoning_types(task), ", ")}")
          Mix.shell().info("")
        end

        # Run validation
        instances = TaskSchema.get_instances(task, limit)
        results = run_instances(instances, validator, task, verbose)

        # Report results
        report_results(path, results, quiet)

        results

      {:error, reason} ->
        Mix.shell().error("Failed to load task: #{reason}")
        %{passed: 0, failed: 0, errors: []}
    end
  end

  defp run_instances(instances, validator, task, verbose) do
    results = Enum.map(instances, fn instance ->
      result = apply(__MODULE__, validator, [instance, task])

      if verbose do
        status = if result.passed, do: "✓", else: "✗"
        Mix.shell().info("#{status} #{String.slice(instance.input, 0, 60)}...")
        unless result.passed do
          Mix.shell().info("  Expected: #{inspect(instance.output)}")
          Mix.shell().info("  Got: #{inspect(result.actual)}")
        end
      end

      result
    end)

    passed = Enum.count(results, & &1.passed)
    failed = length(results) - passed
    errors = Enum.reject(results, & &1.passed) |> Enum.take(3)

    %{passed: passed, failed: failed, total: length(results), errors: errors}
  end

  defp report_results(path, results, quiet) do
    task_name = Path.basename(path, ".json")
    accuracy = if results.total > 0, do: Float.round(results.passed / results.total * 100, 1), else: 0.0

    color = cond do
      accuracy >= 80 -> :green
      accuracy >= 50 -> :yellow
      true -> :red
    end

    Mix.shell().info([
      color,
      "#{task_name}: #{results.passed}/#{results.total} (#{accuracy}%)",
      :reset
    ])

    unless quiet or Enum.empty?(results.errors) do
      Mix.shell().info("\nSample failures:")
      Enum.each(results.errors, fn error ->
        Mix.shell().info("  Input: #{String.slice(error.input, 0, 80)}...")
        Mix.shell().info("  Expected: #{inspect(Enum.take(error.expected, 2))}")
        Mix.shell().info("  Got: #{inspect(error.actual)}")
        Mix.shell().info("")
      end)
    end
  end

  defp print_category_summary(category, results) do
    total_passed = Enum.sum(Enum.map(results, fn {_, r} -> r.passed end))
    total_failed = Enum.sum(Enum.map(results, fn {_, r} -> r.failed end))
    total = total_passed + total_failed

    accuracy = if total > 0, do: Float.round(total_passed / total * 100, 1), else: 0.0

    Mix.shell().info("\n=== #{category} Summary ===")
    Mix.shell().info("Tasks tested: #{length(results)}")
    Mix.shell().info("Total instances: #{total}")
    Mix.shell().info("Accuracy: #{accuracy}%")
  end

  # ============================================================================
  # Validators - These test specific NLP capabilities
  # ============================================================================

  @doc false
  def validate_question_understanding(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # Parse the input to extract sentence, question, and category
    parts = parse_mctaco_input(input)

    # Use our pipeline to analyze the question
    result = analyze_temporal_question(parts)

    passed = result in expected or String.downcase(result) in Enum.map(expected, &String.downcase/1)

    %{
      passed: passed,
      input: input,
      expected: expected,
      actual: result
    }
  end

  @doc false
  def validate_temporal_reasoning(instance, task) do
    # Same as question understanding for MCTACO-style tasks
    validate_question_understanding(instance, task)
  end

  @doc false
  def validate_entity_recognition(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # Extract entities using our pipeline
    case Pipeline.process(input, analyze_discourse: false) do
      {:ok, analysis} ->
        entities = extract_entities_from_analysis(analysis)
        passed = check_entity_overlap(entities, expected)

        %{passed: passed, input: input, expected: expected, actual: entities}

      _ ->
        %{passed: false, input: input, expected: expected, actual: []}
    end
  end

  @doc false
  def validate_sentiment(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # Use our sentiment analysis
    sentiment = analyze_sentiment(input)
    passed = sentiment in expected or matches_sentiment?(sentiment, expected)

    %{passed: passed, input: input, expected: expected, actual: sentiment}
  end

  @doc false
  def validate_classification(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # Use intent classifier
    case ChatBot.ML.IntentClassifierSimple.classify(input) do
      {:ok, intent, _confidence} ->
        # Check if our classification matches expected
        passed = classification_matches?(intent, expected)
        %{passed: passed, input: input, expected: expected, actual: intent}

      _ ->
        %{passed: false, input: input, expected: expected, actual: "unknown"}
    end
  end

  @doc false
  def validate_pos_tagging(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # Use our POS tagger - tokenize to plain strings
    tokens = Tokenizer.tokenize(input)
    token_strings = Enum.map(tokens, fn
      %{text: text} -> text
      t when is_binary(t) -> t
      _ -> ""
    end)

    # Get POS tags using the stored model
    case POSTagger.load_model() do
      {:ok, model} ->
        tagged = POSTagger.predict(token_strings, model)
        # predict returns list of {word, tag} tuples
        result = Enum.map(tagged, fn
          {word, tag} -> "#{word}/#{tag}"
          tag when is_binary(tag) -> tag
          _ -> "UNK"
        end) |> Enum.join(" ")
        passed = pos_matches?(result, expected)
        %{passed: passed, input: input, expected: expected, actual: result}

      _ ->
        %{passed: false, input: input, expected: expected, actual: "model_not_loaded"}
    end
  end

  @doc false
  def validate_commonsense(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # For commonsense, we check if our pipeline can at least parse correctly
    case Pipeline.process(input, analyze_discourse: false) do
      {:ok, _analysis} ->
        # Basic pass if we can process - commonsense requires more sophisticated evaluation
        %{passed: false, input: input, expected: expected, actual: "requires_llm"}

      _ ->
        %{passed: false, input: input, expected: expected, actual: "parse_error"}
    end
  end

  @doc false
  def validate_entailment(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    try do
      # Parse premise and hypothesis
      {premise, hypothesis} = TaskSolver.parse_nli_input(input)

      # Solve entailment
      result = TaskSolver.solve_entailment(premise, hypothesis)

      # Check if our answer matches expected
      passed = result in expected or
               String.downcase(result) in Enum.map(expected, &String.downcase/1)

      %{passed: passed, input: input, expected: expected, actual: result}
    rescue
      _ ->
        %{passed: false, input: input, expected: expected, actual: "parse_error"}
    end
  end

  @doc false
  def validate_summarization(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    try do
      # Generate summary using our extractive approach
      summary = TaskSolver.summarize_dialogue(input)

      # For summarization, we can't expect exact match
      # Instead, check for key entity/fact overlap
      passed = summarization_quality(summary, expected) > 0.3

      %{passed: passed, input: input, expected: expected, actual: summary}
    rescue
      _ ->
        %{passed: false, input: input, expected: expected, actual: "parse_error"}
    end
  end

  defp summarization_quality(generated, expected_list) do
    # Calculate overlap between generated summary and expected
    gen_tokens = generated
                 |> String.downcase()
                 |> String.split(~r/\W+/)
                 |> Enum.reject(&(&1 in ~w(a an the is are was were be)))

    expected_tokens = expected_list
                      |> Enum.join(" ")
                      |> String.downcase()
                      |> String.split(~r/\W+/)
                      |> Enum.reject(&(&1 in ~w(a an the is are was were be)))

    # Calculate Jaccard similarity
    gen_set = MapSet.new(gen_tokens)
    exp_set = MapSet.new(expected_tokens)

    intersection = MapSet.intersection(gen_set, exp_set) |> MapSet.size()
    union = MapSet.union(gen_set, exp_set) |> MapSet.size()

    if union == 0, do: 0.0, else: intersection / union
  end

  @doc false
  def validate_generic(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # Generic validation - just check if we can process
    try do
      case Pipeline.process(input, analyze_discourse: false) do
        {:ok, _} ->
          %{passed: false, input: input, expected: expected, actual: "no_validator"}

        _ ->
          %{passed: false, input: input, expected: expected, actual: "pipeline_error"}
      end
    rescue
      _ ->
        %{passed: false, input: input, expected: expected, actual: "exception"}
    end
  end

  @doc false
  def validate_generative_unsupported(instance, _task) do
    input = instance.input
    expected = normalize_outputs(instance.output)

    # These tasks require text generation (LLM) - mark as unsupported
    # We can still verify we can parse the input
    try do
      # For dialogue inputs, just check we can tokenize
      tokens = Tokenizer.tokenize(input)
      token_count = length(tokens)

      %{
        passed: false,
        input: input,
        expected: expected,
        actual: "requires_llm (#{token_count} tokens parsed)"
      }
    rescue
      _ ->
        %{passed: false, input: input, expected: expected, actual: "requires_llm"}
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp tasks_path do
    Path.join(File.cwd!(), @tasks_dir)
  end

  defp find_task_file(task_name) do
    # Try exact match first
    path = Path.join(tasks_path(), "#{task_name}.json")
    if File.exists?(path), do: path, else: find_partial_match(task_name)
  end

  defp find_partial_match(task_name) do
    tasks_path()
    |> File.ls!()
    |> Enum.find(fn file ->
      String.contains?(file, task_name) and String.ends_with?(file, ".json")
    end)
    |> case do
      nil -> nil
      file -> Path.join(tasks_path(), file)
    end
  end

  defp load_task(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} -> TaskSchema.parse(data)
          error -> error
        end
      error -> error
    end
  end

  defp normalize_outputs(outputs) when is_list(outputs) do
    Enum.map(outputs, &String.trim/1)
  end
  defp normalize_outputs(output) when is_binary(output), do: [String.trim(output)]
  defp normalize_outputs(_), do: []

  defp parse_mctaco_input(input) do
    # Parse format: "Sentence: X \nQuestion: Y \nCategory: Z."
    parts = %{sentence: "", question: "", category: ""}

    input
    |> String.split("\n")
    |> Enum.reduce(parts, fn line, acc ->
      cond do
        String.starts_with?(line, "Sentence:") ->
          %{acc | sentence: String.trim_leading(line, "Sentence:") |> String.trim()}
        String.starts_with?(line, "Question:") ->
          %{acc | question: String.trim_leading(line, "Question:") |> String.trim()}
        String.starts_with?(line, "Category:") ->
          %{acc | category: String.trim_leading(line, "Category:") |> String.trim() |> String.trim_trailing(".")}
        true ->
          acc
      end
    end)
  end

  defp analyze_temporal_question(parts) do
    question = parts.question
    category = parts.category

    # Analyze the question to determine if it matches the category
    question_lower = String.downcase(question)

    matches = cond do
      # Event Duration keywords
      category == "Event Duration" ->
        String.contains?(question_lower, ["how long", "duration", "how much time", "how many hours", "how many minutes", "how many days"])

      # Frequency keywords
      category == "Frequency" ->
        String.contains?(question_lower, ["how often", "how frequently", "how many times", "frequency"])

      # Event Ordering keywords
      category == "Event Ordering" ->
        String.contains?(question_lower, ["before", "after", "first", "then", "next", "prior", "following", "sequence"])

      # Absolute Timepoint keywords
      category == "Absolute Timepoint" ->
        String.contains?(question_lower, ["when", "what time", "what day", "what year", "what month", "at what"])

      # Transient v. Stationary
      category == "Transient v. Stationary" ->
        String.contains?(question_lower, ["still", "anymore", "permanent", "temporary", "always", "forever"])

      true ->
        false
    end

    if matches, do: "Yes.", else: "No."
  end

  defp extract_entities_from_analysis(analysis) do
    # Extract entities from pipeline analysis
    analysis
    |> Map.get(:chunks, [])
    |> Enum.flat_map(fn chunk ->
      Map.get(chunk, :entities, [])
    end)
    |> Enum.map(& &1.text)
  end

  defp check_entity_overlap(found, expected) do
    found_lower = Enum.map(found, &String.downcase/1)
    expected_lower = Enum.map(expected, &String.downcase/1)

    Enum.any?(expected_lower, fn exp ->
      Enum.any?(found_lower, fn found_e ->
        String.contains?(found_e, exp) or String.contains?(exp, found_e)
      end)
    end)
  end

  defp analyze_sentiment(text) do
    # Simple keyword-based sentiment
    text_lower = String.downcase(text)

    positive_words = ~w(good great excellent happy love like wonderful amazing beautiful)
    negative_words = ~w(bad terrible awful hate dislike horrible ugly sad angry)

    pos_count = Enum.count(positive_words, &String.contains?(text_lower, &1))
    neg_count = Enum.count(negative_words, &String.contains?(text_lower, &1))

    cond do
      pos_count > neg_count -> "positive"
      neg_count > pos_count -> "negative"
      true -> "neutral"
    end
  end

  defp matches_sentiment?(actual, expected) do
    expected_lower = Enum.map(expected, &String.downcase/1)
    actual_lower = String.downcase(actual)

    Enum.any?(expected_lower, fn exp ->
      String.contains?(exp, actual_lower) or String.contains?(actual_lower, exp)
    end)
  end

  defp classification_matches?(intent, expected) do
    intent_lower = String.downcase(intent)
    expected_lower = Enum.map(expected, &String.downcase/1)

    Enum.any?(expected_lower, fn exp ->
      String.jaro_distance(intent_lower, exp) > 0.7
    end)
  end

  defp pos_matches?(actual, expected) do
    # POS matching is complex - check for overlap
    actual_tags = String.split(actual, " ")
    expected_str = Enum.join(expected, " ")
    expected_tags = String.split(expected_str, " ")

    # Count matching tags (order-independent for now)
    matches = Enum.count(actual_tags, fn tag ->
      Enum.any?(expected_tags, &(&1 == tag))
    end)

    matches / max(length(actual_tags), 1) > 0.5
  end
end
