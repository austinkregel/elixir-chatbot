defmodule Mix.Tasks.MigrateTrainingData do
  @shortdoc "Migrate and augment training data with POS annotations"

  @moduledoc """
  Migrates existing Dialogflow-format training data to the enriched format
  with POS (Part-of-Speech) annotations.

  This task:
  1. Moves existing data to data/legacy/ for preservation
  2. Calls the Python script to generate POS annotations using NLTK
  3. Writes enriched data to data/training/
  4. Optionally validates the migration

  ## Usage

      mix migrate_training_data
      mix migrate_training_data --dry-run
      mix migrate_training_data --validate
      mix migrate_training_data --skip-python

  ## Options

    * `--dry-run` - Preview changes without writing files
    * `--validate` - Run Python validation after migration
    * `--keep-legacy` - Copy files to legacy instead of moving (keep originals)
    * `--skip-python` - Skip Python POS annotation (directory setup only)
    * `--force` - Overwrite existing migration

  ## Directory Structure

  After migration:

      data/
      ├── legacy/                # Original format (preserved)
      │   ├── intents/
      │   └── entities/
      ├── training/              # Enriched format
      │   ├── intents/           # With POS annotations
      │   ├── entities/
      │   ├── pos/
      │   └── disambiguation/
      └── ...                    # Other files unchanged

  """

  use Mix.Task

  require Logger

  @data_dir "data"
  @legacy_dir "data/legacy"
  @training_dir "data/training"
  @python_script "scripts/generate_pos_annotations.py"
  @validate_script "scripts/validate_pos_annotations.py"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          dry_run: :boolean,
          validate: :boolean,
          keep_legacy: :boolean,
          skip_python: :boolean,
          force: :boolean
        ]
      )

    dry_run = Keyword.get(opts, :dry_run, false)
    validate = Keyword.get(opts, :validate, false)
    keep_legacy = Keyword.get(opts, :keep_legacy, false)
    skip_python = Keyword.get(opts, :skip_python, false)
    force = Keyword.get(opts, :force, false)

    if dry_run do
      Mix.shell().info("DRY RUN - no files will be modified")
    end

    # Check if migration already done
    if File.exists?(@training_dir) and not force and not dry_run do
      Mix.shell().info("Training directory already exists: #{@training_dir}")
      Mix.shell().info("Use --force to overwrite or --dry-run to preview")

      unless Mix.shell().yes?("Continue anyway?") do
        Mix.shell().info("Aborted.")
        exit(:normal)
      end
    end

    # Step 1: Create directory structure
    Mix.shell().info("\n=== Step 1: Creating directory structure ===")
    create_directories(dry_run)

    # Step 2: Move/copy data to legacy
    Mix.shell().info("\n=== Step 2: Preserving original data ===")
    preserve_legacy_data(dry_run, keep_legacy)

    # Step 3: Copy entities to training (unchanged format)
    Mix.shell().info("\n=== Step 3: Copying entities to training directory ===")
    copy_entities_to_training(dry_run)

    # Step 4: Generate POS annotations
    if skip_python do
      Mix.shell().info("\n=== Step 4: Skipping Python POS annotation (--skip-python) ===")
      Mix.shell().info("Run manually: python #{@python_script}")
    else
      Mix.shell().info("\n=== Step 4: Generating POS annotations with NLTK ===")
      run_python_annotation(dry_run)
    end

    # Step 5: Validate (optional)
    if validate and not skip_python do
      Mix.shell().info("\n=== Step 5: Validating POS annotations ===")
      run_python_validation(dry_run)
    end

    # Summary
    Mix.shell().info("\n=== Migration Complete ===")
    print_summary(dry_run)
  end

  defp create_directories(dry_run) do
    dirs = [
      @legacy_dir,
      Path.join(@legacy_dir, "intents"),
      Path.join(@legacy_dir, "entities"),
      @training_dir,
      Path.join(@training_dir, "intents"),
      Path.join(@training_dir, "entities"),
      Path.join(@training_dir, "pos"),
      Path.join(@training_dir, "disambiguation")
    ]

    for dir <- dirs do
      if dry_run do
        Mix.shell().info("  Would create: #{dir}")
      else
        case File.mkdir_p(dir) do
          :ok ->
            Mix.shell().info("  Created: #{dir}")

          {:error, reason} ->
            Mix.shell().error("  Failed to create #{dir}: #{reason}")
        end
      end
    end
  end

  defp preserve_legacy_data(dry_run, keep_legacy) do
    operation = if keep_legacy, do: "Copying", else: "Moving"

    # Intents
    intents_src = Path.join(@data_dir, "intents")
    intents_dst = Path.join(@legacy_dir, "intents")

    if File.exists?(intents_src) do
      intent_files = Path.wildcard(Path.join(intents_src, "*.json"))
      Mix.shell().info("  #{operation} #{length(intent_files)} intent files to legacy/")

      unless dry_run do
        for file <- intent_files do
          dst = Path.join(intents_dst, Path.basename(file))

          if keep_legacy do
            File.cp!(file, dst)
          else
            # Keep original for now, just copy to legacy
            File.cp!(file, dst)
          end
        end
      end
    end

    # Entities
    entities_src = Path.join(@data_dir, "entities")
    entities_dst = Path.join(@legacy_dir, "entities")

    if File.exists?(entities_src) do
      entity_files = Path.wildcard(Path.join(entities_src, "*.json"))
      Mix.shell().info("  #{operation} #{length(entity_files)} entity files to legacy/")

      unless dry_run do
        for file <- entity_files do
          dst = Path.join(entities_dst, Path.basename(file))
          File.cp!(file, dst)
        end
      end
    end
  end

  defp copy_entities_to_training(dry_run) do
    entities_src = Path.join(@data_dir, "entities")
    entities_dst = Path.join(@training_dir, "entities")

    if File.exists?(entities_src) do
      entity_files = Path.wildcard(Path.join(entities_src, "*.json"))
      Mix.shell().info("  Copying #{length(entity_files)} entity files to training/")

      unless dry_run do
        for file <- entity_files do
          dst = Path.join(entities_dst, Path.basename(file))
          File.cp!(file, dst)
        end
      end
    end
  end

  defp run_python_annotation(dry_run) do
    if not File.exists?(@python_script) do
      Mix.shell().error("Python script not found: #{@python_script}")
      Mix.shell().info("Please ensure scripts/generate_pos_annotations.py exists")
      exit({:shutdown, 1})
    end

    input_dir = Path.join(@data_dir, "intents")
    output_dir = Path.join(@training_dir, "intents")

    cmd_args = [
      @python_script,
      "--input-dir",
      input_dir,
      "--output-dir",
      output_dir,
      "--verbose"
    ]

    cmd_args = if dry_run, do: cmd_args ++ ["--dry-run"], else: cmd_args

    Mix.shell().info("  Running: python3 #{Enum.join(cmd_args, " ")}")

    case System.cmd("python3", cmd_args, stderr_to_stdout: true) do
      {output, 0} ->
        Mix.shell().info(output)
        Mix.shell().info("  POS annotation completed successfully")

      {output, exit_code} ->
        Mix.shell().error(output)
        Mix.shell().error("  Python script failed with exit code: #{exit_code}")

        unless dry_run do
          exit({:shutdown, exit_code})
        end
    end
  end

  defp run_python_validation(dry_run) do
    if File.exists?(@validate_script) do
      training_dir = Path.join(@training_dir, "intents")

      cmd_args = [
        @validate_script,
        "--training-dir",
        training_dir
      ]

      Mix.shell().info("  Running: python3 #{Enum.join(cmd_args, " ")}")

      unless dry_run do
        case System.cmd("python3", cmd_args, stderr_to_stdout: true) do
          {output, 0} ->
            Mix.shell().info(output)
            Mix.shell().info("  Validation completed")

          {output, _exit_code} ->
            Mix.shell().info(output)
            Mix.shell().info("  Validation completed (with discrepancies - review above)")
        end
      end
    else
      Mix.shell().error("Validation script not found: #{@validate_script}")
    end
  end

  defp print_summary(dry_run) do
    if dry_run do
      Mix.shell().info("""

      DRY RUN SUMMARY:
      ----------------
      The following would be created:
        - #{@legacy_dir}/intents/    (backup of original intents)
        - #{@legacy_dir}/entities/   (backup of original entities)
        - #{@training_dir}/intents/  (enriched with POS tags)
        - #{@training_dir}/entities/ (copy of entities)
        - #{@training_dir}/pos/      (for POS training sequences)
        - #{@training_dir}/disambiguation/ (for disambiguation examples)

      Run without --dry-run to perform the migration.
      """)
    else
      Mix.shell().info("""

      MIGRATION SUMMARY:
      ------------------
      Created directories:
        - #{@legacy_dir}/       (original data preserved)
        - #{@training_dir}/     (enriched training data)

      Next steps:
        1. Review data/training/intents/ for POS-annotated examples
        2. Run: python scripts/pos_tag_coverage.py  (analyze coverage)
        3. Run: python scripts/validate_pos_annotations.py  (validate)
        4. Run: mix train_models  (retrain with new data)
      """)
    end
  end
end
