defmodule Tasks.Source do
  @moduledoc "Source adapter that provides domain-specific NLP tasks to Research Agents.\n\nThis allows the Learning Center to train its child agents using curated\nbenchmark tasks instead of (or in addition to) web sources.\n\n## Benefits\n\n- High-quality, human-verified Q&A pairs\n- Commonsense reasoning examples with explanations\n- Diverse domains (Wikipedia, News, Science, etc.)\n- No rate limiting or network latency\n- Reproducible training data\n\n## Usage\n\nConfigure agents to use the :task source:\n\n    LearningCenter.start_session(\"commonsense\", sources: [:task])\n\nOr combine with web sources:\n\n    LearningCenter.start_session(\"France\", sources: [:task, :web])\n\n## Task Selection\n\nTasks are matched to research goals by:\n- Category (e.g., \"Question Answering\" for factual queries)\n- Domain (e.g., \"Wikipedia\" for general knowledge)\n- Keywords in the topic/questions\n"

  alias Tasks.Analyzer
  alias Brain.Knowledge.Types
  alias Brain.ML.Tokenizer
  require Logger

  alias Types.{Finding, SourceInfo}

  @category_mappings %{
    factual: ["Question Answering"],
    reasoning: ["Commonsense Classification", "Coherence Classification"],
    explanatory: ["Explanation", "Question Decomposition"],
    sentiment: ["Sentiment Analysis"],
    general: ["Question Answering", "Commonsense Classification", "Text Categorization"]
  }

  @doc "Fetches findings from domain-specific tasks matching a research goal.\n\n## Options\n  - `:max_tasks` - Maximum task files to use (default: 5)\n  - `:max_instances` - Maximum instances per task (default: 20)\n  - `:goal_type` - Type of goal for task selection (:factual, :reasoning, :general)\n"
  @spec fetch_for_goal(map(), keyword()) :: {:ok, [Finding.t()]} | {:error, term()}
  def fetch_for_goal(goal, opts \\ []) do
    max_tasks = Keyword.get(opts, :max_tasks, 5)
    max_instances = Keyword.get(opts, :max_instances, 20)
    goal_type = Keyword.get(opts, :goal_type, infer_goal_type(goal))

    Logger.debug("TaskSource fetching for goal",
      topic: goal.topic,
      goal_type: goal_type,
      max_tasks: max_tasks
    )

    case select_tasks(goal, goal_type, max_tasks) do
      {:ok, tasks} ->
        findings = extract_findings_from_tasks(tasks, goal, max_instances)

        Logger.info("TaskSource completed",
          topic: goal.topic,
          tasks_used: length(tasks),
          findings: length(findings)
        )

        {:ok, findings}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Gets available tasks by category for browsing/selection.\n\nUses ETS-based caching to avoid re-scanning files on every call.\nCache expires after 5 minutes.\n"
  @spec available_tasks(keyword()) :: {:ok, map()} | {:error, term()}
  def available_tasks(_opts \\ []) do
    cache_key = :task_source_available_tasks
    cache_ttl_ms = 300_000

    case Process.get(cache_key) do
      {cached_at, result} when is_integer(cached_at) ->
        if System.monotonic_time(:millisecond) - cached_at < cache_ttl_ms do
          {:ok, result}
        else
          fetch_and_cache_tasks(cache_key)
        end

      _ ->
        fetch_and_cache_tasks(cache_key)
    end
  end

  defp fetch_and_cache_tasks(cache_key) do
    case Analyzer.analyze_all(english_only: true) do
      {:ok, analysis} ->
        grouped = Analyzer.group_by_category(analysis.useful_tasks)
        Process.put(cache_key, {System.monotonic_time(:millisecond), grouped})
        {:ok, grouped}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Creates training sessions from domain tasks for specific capabilities.\n\nThis is useful for systematic training of agent capabilities.\n\n## Capabilities\n  - :question_answering - Train on factual Q&A\n  - :commonsense - Train on reasoning and common knowledge\n  - :sentiment - Train on emotion detection\n  - :all - Train on all useful categories\n"
  @spec create_training_sessions(atom(), keyword()) :: {:ok, [map()]} | {:error, term()}
  def create_training_sessions(capability, opts \\ []) do
    max_tasks = Keyword.get(opts, :max_tasks, 10)

    categories = capability_to_categories(capability)

    case Analyzer.analyze_all(categories: categories) do
      {:ok, analysis} ->
        tasks = Enum.take(analysis.useful_tasks, max_tasks)

        sessions =
          Enum.map(tasks, fn task ->
            %{
              task_id: task.task_id,
              categories: task.categories,
              domains: task.domains,
              instance_count: task.instance_count,
              file: task.file
            }
          end)

        {:ok, sessions}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp infer_goal_type(goal) do
    topic = String.downcase(goal.topic || "")
    questions = Enum.map(goal.questions || [], &String.downcase/1)
    all_text = topic <> " " <> Enum.join(questions, " ")

    case Brain.ML.MicroClassifiers.classify(:goal_type, all_text) do
      {:ok, label, _score} -> String.to_existing_atom(label)
      _ -> :general
    end
  end

  defp select_tasks(goal, goal_type, max_tasks) do
    categories = Map.get(@category_mappings, goal_type, @category_mappings.general)

    case Analyzer.analyze_all(
           categories: categories,
           english_only: true
         ) do
      {:ok, analysis} ->
        scored_tasks =
          analysis.useful_tasks
          |> Enum.map(fn task ->
            score = score_task_relevance(task, goal)
            {task, score}
          end)
          |> Enum.sort_by(fn {_, score} -> -score end)
          |> Enum.take(max_tasks)
          |> Enum.map(fn {task, _} -> task end)

        {:ok, scored_tasks}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp score_task_relevance(task, goal) do
    topic_words = goal.topic |> Tokenizer.tokenize_normalized(min_length: 2)

    question_words =
      (goal.questions || [])
      |> Enum.flat_map(&Tokenizer.tokenize_normalized(&1, min_length: 2))

    all_words = MapSet.new(topic_words ++ question_words)

    domain_score =
      task.domains
      |> Enum.count(fn domain ->
        domain_words = domain |> Tokenizer.tokenize_normalized(min_length: 2) |> MapSet.new()
        not MapSet.disjoint?(domain_words, all_words)
      end)

    definition_words =
      task.definition
      |> Tokenizer.tokenize_normalized(min_length: 2)
      |> MapSet.new()

    keyword_overlap = MapSet.intersection(all_words, definition_words) |> MapSet.size()
    domain_score * 2 + keyword_overlap + task.instance_count / 1000
  end

  defp extract_findings_from_tasks(tasks, goal, max_instances) do
    tasks
    |> Enum.flat_map(fn task ->
      extract_findings_from_task(task, goal, max_instances)
    end)
  end

  defp extract_findings_from_task(task, goal, max_instances) do
    case Analyzer.load_instances(task.file, max_instances: max_instances) do
      {:ok, instances} ->
        source_info =
          SourceInfo.new(
            "task://#{task.task_id}",
            title: task.task_id,
            reliability_score: 0.95,
            trust_tier: :verified
          )

        Enum.flat_map(instances, fn instance ->
          instance_to_findings(instance, task, source_info, goal)
        end)

      {:error, _} ->
        []
    end
  end

  defp instance_to_findings(instance, task, source_info, _goal) do
    input = Map.get(instance, "input", "")
    outputs = get_outputs(instance)
    explanation = Map.get(instance, "explanation", "")
    _instance_id = Map.get(instance, "id", "unknown")

    category = List.first(task.categories) || "Unknown"

    case category do
      "Question Answering" ->
        Enum.map(outputs, fn answer ->
          Finding.new(
            answer,
            extract_entity(input),
            source_info,
            entity_type: "question_answer",
            raw_context: input,
            confidence: 0.9
          )
        end)

      "Commonsense Classification" ->
        base_claim =
          if explanation != "" do
            explanation
          else
            "#{input} → #{Enum.join(outputs, ", ")}"
          end

        [
          Finding.new(
            base_claim,
            extract_entity(input),
            source_info,
            entity_type: "commonsense",
            raw_context: input,
            confidence: 0.85
          )
        ]

      "Explanation" ->
        Enum.map(outputs, fn explanation_text ->
          Finding.new(
            explanation_text,
            extract_entity(input),
            source_info,
            entity_type: "explanation",
            raw_context: input,
            confidence: 0.88
          )
        end)

      _ ->
        Enum.map(outputs, fn output ->
          Finding.new(
            output,
            extract_entity(input),
            source_info,
            entity_type: category,
            raw_context: input,
            confidence: 0.8
          )
        end)
    end
  end

  # Extract the primary entity/subject from input text
  defp extract_entity(input) when is_binary(input) do
    input
    |> Tokenizer.tokenize_words()
    |> Enum.map(&String.downcase/1)
    |> Enum.reject(&(&1 in ~w(what who where when why how is are was were the a an ? ! .)))
    |> Enum.take(3)
    |> Enum.join(" ")
    |> case do
      "" -> "unknown"
      entity -> entity
    end
  end

  defp extract_entity(_), do: "unknown"

  defp get_outputs(instance) do
    case Map.get(instance, "output") do
      nil -> []
      list when is_list(list) -> list
      single -> [single]
    end
  end

  @doc "Returns the task categories for a given capability atom.

  ## Examples

      iex> Tasks.Source.capability_categories(:question_answering)
      [\"Question Answering\"]

      iex> Tasks.Source.capability_categories(:all)
      # All useful categories from Analyzer
  "
  def capability_categories(capability), do: capability_to_categories(capability)

  defp capability_to_categories(capability) do
    case capability do
      :question_answering ->
        ["Question Answering"]

      :commonsense ->
        ["Commonsense Classification", "Coherence Classification", "Word Semantics"]

      :sentiment ->
        ["Sentiment Analysis"]

      :reasoning ->
        ["Explanation", "Question Decomposition", "Coreference Resolution"]

      :all ->
        Analyzer.useful_categories()

      _ ->
        Analyzer.useful_categories()
    end
  end
end
