defmodule ChatBot.Learning.TaskAnalyzer do
  @moduledoc """
  Analyzes domain-specific task files from NLP benchmarks.

  Parses task metadata, categorizes tasks by their NLP category and domain,
  and filters tasks suitable for chatbot training.

  ## Task File Format

  Each task file contains:
  - Contributors, Source, URL: Metadata
  - Categories: NLP task category (e.g., "Question Answering")
  - Domains: Subject domain (e.g., "Wikipedia", "Commonsense")
  - Definition: Task instruction
  - Positive/Negative Examples: Demonstrations
  - Instances: Actual task instances with input/output pairs
  """

  require Logger

  @type task_metadata :: %{
          file: String.t(),
          task_id: String.t(),
          categories: [String.t()],
          domains: [String.t()],
          input_language: String.t(),
          output_language: String.t(),
          instance_count: non_neg_integer(),
          positive_example_count: non_neg_integer(),
          definition: String.t()
        }

  @type analysis_result :: %{
          total_tasks: non_neg_integer(),
          english_only: non_neg_integer(),
          by_category: %{String.t() => non_neg_integer()},
          by_domain: %{String.t() => non_neg_integer()},
          useful_tasks: [task_metadata()],
          skipped_tasks: [task_metadata()]
        }

  # Categories that are useful for chatbot training
  @useful_categories [
    "Question Answering",
    "Commonsense Classification",
    "Sentiment Analysis",
    "Question Generation",
    "Text Categorization",
    "Data to Text",
    "Paraphrasing",
    "Coreference Resolution",
    "Explanation",
    "Sentence Composition",
    "Word Semantics",
    "Coherence Classification",
    "Story Composition",
    "Text Completion"
  ]

  # Categories to skip (translation, code generation, etc.)
  @skip_categories [
    "Translation",
    "Text to Code",
    "Code to Text",
    "Program Execution"
  ]

  @default_tasks_path "data/domain_specific_tasks"

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Analyzes all task files in the domain_specific_tasks directory.

  ## Options
    - `:tasks_path` - Path to the tasks directory (default: data/domain_specific_tasks)
    - `:english_only` - Only include English tasks (default: true)
    - `:categories` - List of categories to include (default: all useful categories)

  Returns an analysis result with task counts, category breakdown, and filtered task list.
  """
  @spec analyze_all(keyword()) :: {:ok, analysis_result()} | {:error, term()}
  def analyze_all(opts \\ []) do
    tasks_path = Keyword.get(opts, :tasks_path, @default_tasks_path)
    english_only = Keyword.get(opts, :english_only, true)
    categories = Keyword.get(opts, :categories, @useful_categories)

    case list_task_files(tasks_path) do
      {:ok, files} ->
        Logger.info("Analyzing domain tasks", %{file_count: length(files)})

        # Parse all task files in parallel
        tasks =
          files
          |> Task.async_stream(&parse_task_file/1, max_concurrency: 10, timeout: 30_000)
          |> Enum.map(fn
            {:ok, result} -> result
            {:exit, _reason} -> nil
          end)
          |> Enum.reject(&is_nil/1)

        # Filter and categorize
        {useful, skipped} = filter_tasks(tasks, english_only, categories)

        result = %{
          total_tasks: length(tasks),
          english_only: Enum.count(tasks, &english_task?/1),
          by_category: count_by_category(tasks),
          by_domain: count_by_domain(tasks),
          useful_tasks: useful,
          skipped_tasks: skipped
        }

        {:ok, result}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Parses a single task file and extracts metadata.
  """
  @spec parse_task_file(String.t()) :: task_metadata() | nil
  def parse_task_file(file_path) do
    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            extract_metadata(file_path, data)

          {:error, reason} ->
            # Only log at debug level to avoid flooding logs when scanning many files
            Logger.debug("Failed to parse JSON: #{Path.basename(file_path)} - #{inspect(reason)}")
            nil
        end

      {:error, reason} ->
        Logger.debug("Failed to read file: #{Path.basename(file_path)} - #{inspect(reason)}")
        nil
    end
  end

  @doc """
  Returns the list of useful categories for chatbot training.
  """
  @spec useful_categories() :: [String.t()]
  def useful_categories, do: @useful_categories

  @doc """
  Returns categories that should be skipped.
  """
  @spec skip_categories() :: [String.t()]
  def skip_categories, do: @skip_categories

  @doc """
  Checks if a task is useful for chatbot training based on its categories.
  """
  @spec useful_task?(task_metadata()) :: boolean()
  def useful_task?(task) do
    task_categories = MapSet.new(task.categories)
    useful_set = MapSet.new(@useful_categories)
    skip_set = MapSet.new(@skip_categories)

    has_useful = not MapSet.disjoint?(task_categories, useful_set)
    has_skip = not MapSet.disjoint?(task_categories, skip_set)

    has_useful and not has_skip
  end

  @doc """
  Checks if a task is English-only.
  """
  @spec english_task?(task_metadata()) :: boolean()
  def english_task?(task) do
    task.input_language == "English" and task.output_language == "English"
  end

  @doc """
  Groups tasks by their primary category.
  """
  @spec group_by_category([task_metadata()]) :: %{String.t() => [task_metadata()]}
  def group_by_category(tasks) do
    Enum.group_by(tasks, fn task ->
      List.first(task.categories) || "Unknown"
    end)
  end

  @doc """
  Groups tasks by their primary domain.
  """
  @spec group_by_domain([task_metadata()]) :: %{String.t() => [task_metadata()]}
  def group_by_domain(tasks) do
    Enum.group_by(tasks, fn task ->
      # Extract top-level domain (before "->")
      task.domains
      |> List.first()
      |> case do
        nil -> "Unknown"
        domain -> domain |> String.split("->") |> List.first() |> String.trim()
      end
    end)
  end

  @doc """
  Loads instances from a task file.

  ## Options
    - `:max_instances` - Maximum instances to load (default: all)
    - `:include_examples` - Include positive/negative examples (default: true)
  """
  @spec load_instances(String.t(), keyword()) :: {:ok, [map()]} | {:error, term()}
  def load_instances(file_path, opts \\ []) do
    max_instances = Keyword.get(opts, :max_instances, :all)
    include_examples = Keyword.get(opts, :include_examples, true)

    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            instances = Map.get(data, "Instances", [])

            # Optionally include examples
            examples =
              if include_examples do
                positive = Map.get(data, "Positive Examples", [])
                negative = Map.get(data, "Negative Examples", [])
                positive ++ negative
              else
                []
              end

            all_instances = examples ++ instances

            # Apply limit
            limited =
              case max_instances do
                :all -> all_instances
                n when is_integer(n) -> Enum.take(all_instances, n)
              end

            {:ok, limited}

          {:error, reason} ->
            {:error, {:json_parse_error, reason}}
        end

      {:error, reason} ->
        {:error, {:file_read_error, reason}}
    end
  end

  @doc """
  Gets the task definition/instruction from a task file.
  """
  @spec get_definition(String.t()) :: {:ok, String.t()} | {:error, term()}
  def get_definition(file_path) do
    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            definition =
              data
              |> Map.get("Definition", [])
              |> List.first()
              |> Kernel.||("")

            {:ok, definition}

          {:error, reason} ->
            {:error, {:json_parse_error, reason}}
        end

      {:error, reason} ->
        {:error, {:file_read_error, reason}}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp list_task_files(tasks_path) do
    pattern = Path.join(tasks_path, "*.json")
    files = Path.wildcard(pattern)

    # Filter out README.md or other non-task files
    task_files =
      files
      |> Enum.filter(&String.ends_with?(&1, ".json"))
      |> Enum.reject(&String.contains?(&1, "README"))

    if length(task_files) == 0 do
      {:error, :no_task_files_found}
    else
      {:ok, task_files}
    end
  end

  defp extract_metadata(file_path, data) do
    task_id = extract_task_id(file_path)

    %{
      file: file_path,
      task_id: task_id,
      categories: Map.get(data, "Categories", []),
      domains: Map.get(data, "Domains", []),
      input_language: data |> Map.get("Input_language", ["Unknown"]) |> List.first(),
      output_language: data |> Map.get("Output_language", ["Unknown"]) |> List.first(),
      instance_count: data |> Map.get("Instances", []) |> length(),
      positive_example_count: data |> Map.get("Positive Examples", []) |> length(),
      definition: data |> Map.get("Definition", []) |> List.first() |> Kernel.||("")
    }
  end

  defp extract_task_id(file_path) do
    file_path
    |> Path.basename()
    |> String.replace_suffix(".json", "")
  end

  defp filter_tasks(tasks, english_only, categories) do
    category_set = MapSet.new(categories)

    Enum.split_with(tasks, fn task ->
      lang_ok = not english_only or english_task?(task)
      task_categories = MapSet.new(task.categories)
      cat_ok = not MapSet.disjoint?(task_categories, category_set)

      # Also check it's not in skip categories
      skip_set = MapSet.new(@skip_categories)
      not_skipped = MapSet.disjoint?(task_categories, skip_set)

      lang_ok and cat_ok and not_skipped
    end)
  end

  defp count_by_category(tasks) do
    tasks
    |> Enum.flat_map(fn task -> task.categories end)
    |> Enum.frequencies()
  end

  defp count_by_domain(tasks) do
    tasks
    |> Enum.flat_map(fn task ->
      Enum.map(task.domains, fn domain ->
        domain |> String.split("->") |> List.first() |> String.trim()
      end)
    end)
    |> Enum.frequencies()
  end
end
