defmodule Mix.Tasks.RegenerateTestModels do
  @moduledoc """
  Regenerate ML model .term files for test compatibility.

  When model files become incompatible with current library versions
  (Nx, EXLA, OTP), this task regenerates them.

  ## Usage

      mix regenerate_test_models [options]

  ## Options

    --check          Check compatibility without regenerating

  ## Examples

      # Check if models need regeneration
      mix regenerate_test_models --check

      # Regenerate all models
      mix regenerate_test_models
  """

  use Mix.Task
  require Logger

  @shortdoc "Regenerate ML model .term files for test compatibility"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          check: :boolean
        ]
      )

    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Model Compatibility Tool")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    show_versions()

    if Keyword.get(opts, :check, false) do
      check_compatibility()
    else
      regenerate_models()
    end
  end

  defp show_versions do
    Mix.shell().info("Current Library Versions:")
    Mix.shell().info("  Nx:     #{get_version(:nx)}")
    Mix.shell().info("  EXLA:   #{get_version(:exla)}")
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

    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")

    model_files = [
      {"classifier.term", "Intent Classifier"},
      {"entity_model.term", "Entity Model"},
      {"pos_model.term", "POS Tagger"},
      {"gazetteer.term", "Gazetteer"},
      {"embedder.term", "Embedder Vocabulary"}
    ]

    Enum.each(model_files, fn {filename, name} ->
      path = Path.join(models_path, filename)

      if File.exists?(path) do
        case File.read(path) do
          {:ok, binary} ->
            try do
              :erlang.binary_to_term(binary)
              Mix.shell().info("  ✓ #{name}: Compatible")
            rescue
              _ ->
                Mix.shell().error("  ✗ #{name}: INCOMPATIBLE - regeneration needed")
            end

          {:error, reason} ->
            Mix.shell().info("  ? #{name}: #{reason}")
        end
      else
        Mix.shell().info("  - #{name}: Not found (optional)")
      end
    end)

    Mix.shell().info("")
  end

  defp regenerate_models do
    Mix.shell().info("Regenerating models via mix train_models...")
    Mix.shell().info("")
    Mix.Task.run("train_models", [])
    Mix.shell().info("")
    Mix.shell().info("Models regenerated!")
  end
end
