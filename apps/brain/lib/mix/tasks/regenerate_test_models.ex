defmodule Mix.Tasks.RegenerateTestModels do
  @moduledoc """
  Regenerate LSTM model .term files for test compatibility.

  When LSTM model files become incompatible with current library versions
  (Nx, EXLA, Axon, OTP), this task regenerates them.

  ## Usage

      mix regenerate_test_models [options]

  ## Options

    --all            Regenerate all model types
    --unified        Regenerate unified model
    --multi-task     Regenerate multi-task model
    --response       Regenerate response scorer model
    --minimal        Generate minimal test models (fast, small vocab)
    --check          Check compatibility without regenerating

  ## Examples

      # Check if models need regeneration
      mix regenerate_test_models --check

      # Regenerate all models
      mix regenerate_test_models --all

      # Generate minimal models for fast testing
      mix regenerate_test_models --minimal

  ## Version Compatibility

  Models are serialized using `:erlang.term_to_binary/1`. The internal format
  of Nx tensors depends on:

  - Nx version
  - EXLA version (for EXLA-backed tensors)
  - OTP version (for term_to_binary format)

  When you update these libraries, models need to be regenerated.

  ## Workflow

  1. Run `mix regenerate_test_models --check` to see if models are incompatible
  2. Run `mix regenerate_test_models --all` to regenerate
  3. Run `mix test` to verify tests pass
  4. Commit the new .term files
  """

  use Mix.Task
  require Logger

  @shortdoc "Regenerate LSTM model .term files for test compatibility"

  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [
        all: :boolean,
        unified: :boolean,
        multi_task: :boolean,
        response: :boolean,
        minimal: :boolean,
        check: :boolean
      ]
    )

    # Start the application
    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("LSTM Model Compatibility Tool")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")

    # Show current versions
    show_versions()

    cond do
      Keyword.get(opts, :check, false) ->
        check_compatibility()

      Keyword.get(opts, :minimal, false) ->
        generate_minimal_models()

      Keyword.get(opts, :all, false) or opts == [] ->
        regenerate_all_models()

      true ->
        regenerate_selected_models(opts)
    end
  end

  defp show_versions do
    Mix.shell().info("Current Library Versions:")
    Mix.shell().info("  Nx:     #{get_version(:nx)}")
    Mix.shell().info("  EXLA:   #{get_version(:exla)}")
    Mix.shell().info("  Axon:   #{get_version(:axon)}")
    Mix.shell().info("  OTP:    #{System.otp_release()}")
    Mix.shell().info("  Elixir: #{System.version()}")
    Mix.shell().info("")
  end

  defp get_version(app) do
    case Application.spec(app, :vsn) do
      nil -> "not loaded"
      vsn -> to_string(vsn)
    end
  end

  defp check_compatibility do
    Mix.shell().info("Checking model compatibility...")
    Mix.shell().info("")

    models = [
      {:unified, "Unified Model"},
      {:multi_task, "Multi-Task Model"},
      {:response_scorer, "Response Scorer"}
    ]

    all_ok = Enum.reduce(models, true, fn {type, name}, acc ->
      result = Brain.LSTMTestHelpers.check_model_compatibility(type)

      case result do
        :ok ->
          Mix.shell().info("  ✓ #{name}: Compatible")
          acc

        {:error, :model_not_found, _} ->
          Mix.shell().info("  - #{name}: Not found (optional)")
          acc

        {:error, :decode_failed, _} ->
          Mix.shell().error("  ✗ #{name}: INCOMPATIBLE - regeneration needed")
          false

        {:error, reason, _} ->
          Mix.shell().info("  ? #{name}: #{reason}")
          acc
      end
    end)

    Mix.shell().info("")

    if all_ok do
      Mix.shell().info("All models are compatible!")
    else
      Mix.shell().error("Some models need regeneration.")
      Mix.shell().info("")
      Mix.shell().info("Run: mix regenerate_test_models --all")
      Mix.shell().info("Or:  mix train_models")
    end
  end

  defp generate_minimal_models do
    Mix.shell().info("Generating minimal test models...")
    Mix.shell().info("")
    Mix.shell().info("Note: These models are NOT suitable for inference.")
    Mix.shell().info("They are only for testing model loading/saving logic.")
    Mix.shell().info("")

    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    lstm_path = Path.join(models_path, "lstm")
    File.mkdir_p!(lstm_path)

    # Generate minimal unified model
    {:ok, path} = Brain.LSTMTestHelpers.generate_test_model(:unified,
      vocab_size: 100,
      embedding_size: 32,
      hidden_size: 32,
      output_dir: lstm_path
    )

    # Rename to standard filename
    dest = Path.join(lstm_path, "unified_model.term")
    File.rename!(path, dest)
    Mix.shell().info("  ✓ Generated: #{dest}")

    Mix.shell().info("")
    Mix.shell().info("Minimal models generated. For full models, run:")
    Mix.shell().info("  mix train_models")
  end

  defp regenerate_all_models do
    Mix.shell().info("Regenerating all LSTM models...")
    Mix.shell().info("")
    Mix.shell().info("This will retrain all models from training data.")
    Mix.shell().info("This may take several minutes.")
    Mix.shell().info("")

    # Delegate to the main training tasks
    Mix.shell().info("Running: mix train_unified")
    Mix.Task.run("train_unified", [])

    Mix.shell().info("")
    Mix.shell().info("Running: mix train_response")
    Mix.Task.run("train_response", [])

    Mix.shell().info("")
    Mix.shell().info("All models regenerated!")
    Mix.shell().info("")
    Mix.shell().info("Don't forget to commit the new .term files.")
  end

  defp regenerate_selected_models(opts) do
    if Keyword.get(opts, :unified, false) do
      Mix.shell().info("Regenerating unified model...")
      Mix.Task.run("train_unified", [])
    end

    if Keyword.get(opts, :multi_task, false) do
      Mix.shell().info("Regenerating multi-task model...")
      # Multi-task training would be handled by train_lstm
      Mix.Task.run("train_lstm", ["--type", "multitask"])
    end

    if Keyword.get(opts, :response, false) do
      Mix.shell().info("Regenerating response scorer model...")
      Mix.Task.run("train_response", [])
    end

    Mix.shell().info("")
    Mix.shell().info("Selected models regenerated!")
  end
end
