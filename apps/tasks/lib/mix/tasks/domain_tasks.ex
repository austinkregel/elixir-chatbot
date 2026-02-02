defmodule Mix.Tasks.DomainTasks.Analyze do
  @shortdoc "Analyzes domain-specific task files for chatbot training potential"

  @moduledoc """
  Analyzes the domain_specific_tasks directory and reports on available tasks.

  This task scans all NLP benchmark task files and categorizes them by:
  - NLP category (Question Answering, Commonsense, etc.)
  - Domain (Wikipedia, Movies, Commonsense, etc.)
  - Language (English-only by default)
  - Suitability for chatbot training

  ## Usage

      mix domain_tasks.analyze [options]

  ## Options

    * `--tasks-path` - Path to tasks directory (default: data/domain_specific_tasks)
    * `--all-languages` - Include non-English tasks
    * `--show-skipped` - Show skipped task details
    * `--output` - Output file for JSON report (optional)
  """

  use Mix.Task

  alias Tasks.Analyzer, as: TaskAnalyzer

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          tasks_path: :string,
          all_languages: :boolean,
          show_skipped: :boolean,
          output: :string
        ]
      )

    # Start application for Logger
    Mix.Task.run("app.start")

    tasks_path = Keyword.get(opts, :tasks_path, "data/domain_specific_tasks")
    english_only = not Keyword.get(opts, :all_languages, false)
    show_skipped = Keyword.get(opts, :show_skipped, false)
    output_file = Keyword.get(opts, :output)

    Mix.shell().info("Analyzing domain tasks in: #{tasks_path}")
    Mix.shell().info("")

    case TaskAnalyzer.analyze_all(tasks_path: tasks_path, english_only: english_only) do
      {:ok, result} ->
        print_summary(result, show_skipped)

        if output_file do
          write_report(output_file, result)
        end

      {:error, reason} ->
        Mix.shell().error("Analysis failed: #{inspect(reason)}")
    end
  end

  defp print_summary(result, show_skipped) do
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("DOMAIN TASKS ANALYSIS REPORT")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")

    Mix.shell().info("Overview:")
    Mix.shell().info("  Total tasks: #{result.total_tasks}")
    Mix.shell().info("  English-only tasks: #{result.english_only}")
    Mix.shell().info("  Useful for training: #{length(result.useful_tasks)}")
    Mix.shell().info("  Skipped: #{length(result.skipped_tasks)}")
    Mix.shell().info("")

    # Calculate total instances
    useful_instances = result.useful_tasks |> Enum.map(& &1.instance_count) |> Enum.sum()
    Mix.shell().info("  Total useful instances: #{useful_instances}")
    Mix.shell().info("")

    Mix.shell().info("-" |> String.duplicate(60))
    Mix.shell().info("Tasks by Category:")
    Mix.shell().info("-" |> String.duplicate(60))

    result.by_category
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.each(fn {category, count} ->
      useful_marker =
        if category in TaskAnalyzer.useful_categories(), do: " [USEFUL]", else: ""

      skip_marker = if category in TaskAnalyzer.skip_categories(), do: " [SKIP]", else: ""
      Mix.shell().info("  #{category}: #{count}#{useful_marker}#{skip_marker}")
    end)

    Mix.shell().info("")
    Mix.shell().info("-" |> String.duplicate(60))
    Mix.shell().info("Tasks by Domain:")
    Mix.shell().info("-" |> String.duplicate(60))

    result.by_domain
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.take(15)
    |> Enum.each(fn {domain, count} ->
      Mix.shell().info("  #{domain}: #{count}")
    end)

    if map_size(result.by_domain) > 15 do
      Mix.shell().info("  ... and #{map_size(result.by_domain) - 15} more domains")
    end

    Mix.shell().info("")

    if show_skipped and length(result.skipped_tasks) > 0 do
      Mix.shell().info("-" |> String.duplicate(60))
      Mix.shell().info("Skipped Tasks (first 20):")
      Mix.shell().info("-" |> String.duplicate(60))

      result.skipped_tasks
      |> Enum.take(20)
      |> Enum.each(fn task ->
        categories = Enum.join(task.categories, ", ")
        Mix.shell().info("  #{task.task_id}: #{categories}")
      end)
    end

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
  end

  defp write_report(output_file, result) do
    report = %{
      summary: %{
        total_tasks: result.total_tasks,
        english_only: result.english_only,
        useful_count: length(result.useful_tasks),
        skipped_count: length(result.skipped_tasks)
      },
      by_category: result.by_category,
      by_domain: result.by_domain,
      useful_tasks:
        Enum.map(result.useful_tasks, fn t ->
          %{
            task_id: t.task_id,
            categories: t.categories,
            domains: t.domains,
            instance_count: t.instance_count
          }
        end)
    }

    case Jason.encode(report, pretty: true) do
      {:ok, json} ->
        File.write!(output_file, json)
        Mix.shell().info("Report written to: #{output_file}")

      {:error, reason} ->
        Mix.shell().error("Failed to write report: #{inspect(reason)}")
    end
  end
end

defmodule Mix.Tasks.DomainTasks.Transform do
  @shortdoc "Transforms domain tasks into chatbot training data"

  @moduledoc """
  Transforms domain-specific NLP task files into training data for the chatbot.

  This task reads NLP benchmark task files and converts them into:
  - Intent training samples (for classifier training)
  - Knowledge facts (for semantic memory)
  - Entity candidates (for gazetteer)

  ## Usage

      mix domain_tasks.transform [options]

  ## Options

    * `--world-id` - Target training world (default: "default")
    * `--tasks-path` - Path to tasks directory (default: data/domain_specific_tasks)
    * `--categories` - Comma-separated list of categories to include
    * `--max-instances` - Maximum instances per task (default: 1000)
    * `--max-tasks` - Maximum number of tasks to process (default: all)
    * `--dry-run` - Preview without writing to disk
    * `--output-dir` - Output directory for transformed data (default: data/training/domain_tasks)
    * `--skip-entity-extraction` - Skip entity extraction for faster processing

  ## Examples

      # Process all useful tasks
      mix domain_tasks.transform

      # Process only Question Answering tasks with limit
      mix domain_tasks.transform --categories "Question Answering" --max-instances 500

      # Dry run to preview
      mix domain_tasks.transform --dry-run --max-tasks 10
  """

  use Mix.Task

  require Logger

  alias Tasks.Analyzer, as: TaskAnalyzer
  alias Tasks.Transformer, as: TaskTransformer

  @default_output_dir "data/training/domain_tasks"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          world_id: :string,
          tasks_path: :string,
          categories: :string,
          max_instances: :integer,
          max_tasks: :integer,
          dry_run: :boolean,
          output_dir: :string,
          skip_entity_extraction: :boolean
        ]
      )

    # Start application
    Mix.Task.run("app.start")

    world_id = Keyword.get(opts, :world_id, "default")
    tasks_path = Keyword.get(opts, :tasks_path, "data/domain_specific_tasks")
    max_instances = Keyword.get(opts, :max_instances, 1000)
    max_tasks = Keyword.get(opts, :max_tasks, nil)
    dry_run = Keyword.get(opts, :dry_run, false)
    output_dir = Keyword.get(opts, :output_dir, @default_output_dir)
    extract_entities = not Keyword.get(opts, :skip_entity_extraction, false)

    categories =
      case Keyword.get(opts, :categories) do
        nil -> TaskAnalyzer.useful_categories()
        cats -> String.split(cats, ",") |> Enum.map(&String.trim/1)
      end

    Mix.shell().info("Domain Tasks Transformation")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("  World ID: #{world_id}")
    Mix.shell().info("  Tasks path: #{tasks_path}")
    Mix.shell().info("  Categories: #{length(categories)} selected")
    Mix.shell().info("  Max instances per task: #{max_instances}")
    Mix.shell().info("  Entity extraction: #{if extract_entities, do: "enabled", else: "disabled"}")
    Mix.shell().info("  Dry run: #{dry_run}")
    Mix.shell().info("")

    # Analyze tasks first
    case TaskAnalyzer.analyze_all(tasks_path: tasks_path, categories: categories) do
      {:ok, analysis} ->
        tasks_to_process =
          if max_tasks do
            Enum.take(analysis.useful_tasks, max_tasks)
          else
            analysis.useful_tasks
          end

        Mix.shell().info("Found #{length(tasks_to_process)} tasks to process")
        Mix.shell().info("")

        if length(tasks_to_process) == 0 do
          Mix.shell().info("No tasks to process. Check your category filter.")
        else
          process_tasks(tasks_to_process, %{
            world_id: world_id,
            max_instances: max_instances,
            dry_run: dry_run,
            output_dir: output_dir,
            extract_entities: extract_entities
          })
        end

      {:error, reason} ->
        Mix.shell().error("Analysis failed: #{inspect(reason)}")
    end
  end

  defp process_tasks(tasks, config) do
    total = length(tasks)
    start_time = System.monotonic_time(:second)

    # Group tasks by category for better organization
    by_category = TaskAnalyzer.group_by_category(tasks)

    Mix.shell().info("Processing #{total} tasks across #{map_size(by_category)} categories...")
    Mix.shell().info("")

    # Process each task
    {all_samples, all_facts, all_entities, processed} =
      tasks
      |> Enum.with_index(1)
      |> Enum.reduce({[], [], [], 0}, fn {task, idx}, {samples, facts, entities, count} ->
        progress = Float.round(idx / total * 100, 1)
        IO.write("\r  Processing: #{idx}/#{total} (#{progress}%) - #{task.task_id}")

        transform_opts = [
          max_instances: config.max_instances,
          extract_entities: config.extract_entities
        ]

        case TaskTransformer.transform_task(task.file, transform_opts) do
          {:ok, result} ->
            {
              result.training_samples ++ samples,
              result.knowledge_facts ++ facts,
              result.entity_candidates ++ entities,
              count + 1
            }

          {:error, _reason} ->
            {samples, facts, entities, count}
        end
      end)

    IO.write("\n")
    Mix.shell().info("")

    elapsed = System.monotonic_time(:second) - start_time

    Mix.shell().info("-" |> String.duplicate(60))
    Mix.shell().info("Transformation Results:")
    Mix.shell().info("-" |> String.duplicate(60))
    Mix.shell().info("  Tasks processed: #{processed}/#{total}")
    Mix.shell().info("  Training samples: #{length(all_samples)}")
    Mix.shell().info("  Knowledge facts: #{length(all_facts)}")
    Mix.shell().info("  Entity candidates: #{length(all_entities)}")
    Mix.shell().info("  Time elapsed: #{elapsed}s")
    Mix.shell().info("")

    if config.dry_run do
      Mix.shell().info("[DRY RUN] No data written to disk.")
      Mix.shell().info("")

      # Show sample of data
      if length(all_samples) > 0 do
        Mix.shell().info("Sample training data:")
        sample = Enum.take(all_samples, 3)

        Enum.each(sample, fn s ->
          Mix.shell().info("  - [#{s.intent}] #{String.slice(s.text, 0, 60)}...")
        end)
      end
    else
      # Write data to disk
      write_training_data(all_samples, all_facts, all_entities, config)

      # Integrate with world if not dry run
      if config.world_id != "none" do
        integrate_with_world(all_samples, all_facts, all_entities, config)
      end
    end

    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Done!")
  end

  defp write_training_data(samples, facts, entities, config) do
    output_dir = config.output_dir

    # Create output directory
    File.mkdir_p!(output_dir)

    # Group samples by intent
    by_intent = Enum.group_by(samples, & &1.intent)

    Mix.shell().info("Writing training data to: #{output_dir}")

    # Write intent files
    intents_dir = Path.join(output_dir, "intents")
    File.mkdir_p!(intents_dir)

    Enum.each(by_intent, fn {intent, intent_samples} ->
      file_name = "#{normalize_filename(intent)}.json"
      file_path = Path.join(intents_dir, file_name)

      # Convert to expected format
      data =
        Enum.map(intent_samples, fn s ->
          %{
            "text" => s.text,
            "tokens" => s.tokens,
            "pos_tags" => s.pos_tags,
            "entities" => s.entities,
            "id" => s.id,
            "intent" => s.intent
          }
        end)

      case Jason.encode(data, pretty: true) do
        {:ok, json} -> File.write!(file_path, json)
        {:error, _} -> :ok
      end
    end)

    Mix.shell().info("  Written #{map_size(by_intent)} intent files")

    # Write knowledge facts
    if length(facts) > 0 do
      facts_file = Path.join(output_dir, "knowledge_facts.json")

      facts_data =
        Enum.map(facts, fn f ->
          %{
            "subject" => f.subject,
            "predicate" => f.predicate,
            "object" => f.object,
            "source" => f.source,
            "confidence" => f.confidence
          }
        end)

      case Jason.encode(facts_data, pretty: true) do
        {:ok, json} -> File.write!(facts_file, json)
        {:error, _} -> :ok
      end

      Mix.shell().info("  Written #{length(facts)} knowledge facts")
    end

    # Write entity candidates
    if length(entities) > 0 do
      entities_file = Path.join(output_dir, "entity_candidates.json")

      entities_data =
        entities
        |> Enum.uniq_by(fn e -> {e.value, e.inferred_type} end)
        |> Enum.map(fn e ->
          %{
            "value" => e.value,
            "entity_type" => e.inferred_type,
            "confidence" => e.confidence,
            "source" => e.source
          }
        end)

      case Jason.encode(entities_data, pretty: true) do
        {:ok, json} -> File.write!(entities_file, json)
        {:error, _} -> :ok
      end

      Mix.shell().info("  Written #{length(entities_data)} unique entity candidates")
    end

    Mix.shell().info("")
  end

  defp integrate_with_world(_samples, facts, entities, config) do
    world_id = config.world_id

    Mix.shell().info("Integrating with world: #{world_id}")

    # Check if world exists
    case World.Manager.get(world_id) do
      {:ok, _world} ->
        # Add entity candidates
        if length(entities) > 0 do
          unique_entities = Enum.uniq_by(entities, fn e -> {e.value, e.inferred_type} end)

          Enum.each(unique_entities, fn entity ->
            candidate = %{
              value: entity.value,
              inferred_type: entity.inferred_type,
              confidence: entity.confidence,
              source: :domain_tasks,
              metadata: %{original_source: entity.source},
              discovered_at: DateTime.utc_now()
            }

            World.Manager.add_candidate(world_id, candidate)
          end)

          Mix.shell().info("  Added #{length(unique_entities)} entity candidates to world")
        end

        # Save knowledge facts to world
        if length(facts) > 0 do
          world_path = World.Persistence.world_path(world_id)
          knowledge_file = Path.join(world_path, "domain_knowledge.json")

          case Jason.encode(facts, pretty: true) do
            {:ok, json} ->
              File.write!(knowledge_file, json)
              Mix.shell().info("  Saved #{length(facts)} knowledge facts to world")

            {:error, _} ->
              Mix.shell().error("  Failed to save knowledge facts")
          end
        end

        # Checkpoint the world
        World.Manager.checkpoint(world_id)
        Mix.shell().info("  World checkpoint saved")

      {:error, :not_found} ->
        Mix.shell().warning("  World '#{world_id}' not found. Skipping world integration.")
        Mix.shell().info("  Create the world first with: mix training_world.create #{world_id}")
    end

    Mix.shell().info("")
  end

  defp normalize_filename(name) do
    name
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/, "_")
    |> String.trim("_")
  end
end

defmodule Mix.Tasks.DomainTasks.List do
  @shortdoc "Lists available domain task files"

  @moduledoc """
  Lists domain task files with optional filtering.

  ## Usage

      mix domain_tasks.list [options]

  ## Options

    * `--category` - Filter by category
    * `--domain` - Filter by domain
    * `--limit` - Maximum tasks to show (default: 50)
  """

  use Mix.Task

  alias Tasks.Analyzer, as: TaskAnalyzer

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          category: :string,
          domain: :string,
          limit: :integer
        ]
      )

    Mix.Task.run("app.start")

    category_filter = Keyword.get(opts, :category)
    domain_filter = Keyword.get(opts, :domain)
    limit = Keyword.get(opts, :limit, 50)

    case TaskAnalyzer.analyze_all() do
      {:ok, result} ->
        tasks =
          result.useful_tasks
          |> maybe_filter_category(category_filter)
          |> maybe_filter_domain(domain_filter)
          |> Enum.take(limit)

        Mix.shell().info("Available tasks (showing #{length(tasks)}):")
        Mix.shell().info("")

        Enum.each(tasks, fn task ->
          categories = Enum.join(task.categories, ", ")
          Mix.shell().info("  #{task.task_id}")
          Mix.shell().info("    Categories: #{categories}")
          Mix.shell().info("    Instances: #{task.instance_count}")
          Mix.shell().info("")
        end)

      {:error, reason} ->
        Mix.shell().error("Failed: #{inspect(reason)}")
    end
  end

  defp maybe_filter_category(tasks, nil), do: tasks

  defp maybe_filter_category(tasks, category) do
    Enum.filter(tasks, fn task ->
      Enum.any?(task.categories, &String.contains?(String.downcase(&1), String.downcase(category)))
    end)
  end

  defp maybe_filter_domain(tasks, nil), do: tasks

  defp maybe_filter_domain(tasks, domain) do
    Enum.filter(tasks, fn task ->
      Enum.any?(task.domains, &String.contains?(String.downcase(&1), String.downcase(domain)))
    end)
  end
end
