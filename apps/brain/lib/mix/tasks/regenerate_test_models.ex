defmodule Mix.Tasks.RegenerateTestModels do
  @moduledoc "Regenerate LSTM model .term files for test compatibility.\n\nWhen LSTM model files become incompatible with current library versions\n(Nx, EXLA, Axon, OTP), this task regenerates them.\n\n## Usage\n\n    mix regenerate_test_models [options]\n\n## Options\n\n  --all            Regenerate all model types\n  --unified        Regenerate unified model\n  --multi-task     Regenerate multi-task model\n  --response       Regenerate response scorer model\n  --minimal        Generate minimal test models (fast, small vocab)\n  --check          Check compatibility without regenerating\n\n## Examples\n\n    # Check if models need regeneration\n    mix regenerate_test_models --check\n\n    # Regenerate all models\n    mix regenerate_test_models --all\n\n    # Generate minimal models for fast testing\n    mix regenerate_test_models --minimal\n\n## Version Compatibility\n\nModels are serialized using `:erlang.term_to_binary/1`. The internal format\nof Nx tensors depends on:\n\n- Nx version\n- EXLA version (for EXLA-backed tensors)\n- OTP version (for term_to_binary format)\n\nWhen you update these libraries, models need to be regenerated.\n\n## Workflow\n\n1. Run `mix regenerate_test_models --check` to see if models are incompatible\n2. Run `mix regenerate_test_models --all` to regenerate\n3. Run `mix test` to verify tests pass\n4. Commit the new .term files\n"

  # Brain.LSTMTestHelpers is only available in test environment.
  # This mix task is typically run in test context.
  @compile {:no_warn_undefined, Brain.LSTMTestHelpers}

  alias Brain.LSTMTestHelpers
  use Mix.Task
  require Logger

  @shortdoc "Regenerate LSTM model .term files for test compatibility"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          all: :boolean,
          unified: :boolean,
          multi_task: :boolean,
          response: :boolean,
          minimal: :boolean,
          check: :boolean
        ]
      )

    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("LSTM Model Compatibility Tool")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
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
      unified: "Unified Model",
      multi_task: "Multi-Task Model",
      response_scorer: "Response Scorer"
    ]

    all_ok =
      Enum.reduce(models, true, fn {type, name}, acc ->
        result = LSTMTestHelpers.check_model_compatibility(type)

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

    {:ok, path} =
      LSTMTestHelpers.generate_test_model(:unified,
        vocab_size: 100,
        embedding_size: 32,
        hidden_size: 32,
        output_dir: lstm_path
      )

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