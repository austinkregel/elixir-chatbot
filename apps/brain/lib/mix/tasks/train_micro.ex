defmodule Mix.Tasks.TrainMicro do
  @moduledoc """
  Train all micro-classifiers from data/classifiers/*.json files.

  Each JSON file should contain an array of objects with "text" and "label" keys.
  Trained models are saved to priv/ml_models/micro/{name}.term.

  ## Usage

      mix train_micro [options]

  ## Options

      --only NAME    Train only the named classifier (e.g., --only personal_question)
      --list         List available classifiers and their data file status
      --verbose      Show per-classifier training details
  """

  use Mix.Task
  require Logger

  alias Brain.ML.SimpleClassifier
  alias Brain.ML.ModelStore

  @shortdoc "Train micro-classifiers from data/classifiers/*.json"

  @classifier_names ~w(
    personal_question
    clarification_response
    modal_directive
    fallback_response
    goal_type
    entity_type
    user_fact_type
    directed_at_bot
    event_argument_role
  )

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        switches: [only: :string, list: :boolean, verbose: :boolean, publish: :boolean],
        aliases: [o: :only, l: :list, v: :verbose]
      )

    if opts[:list] do
      list_classifiers()
    else
      names =
        case opts[:only] do
          nil -> @classifier_names
          name -> [name]
        end

      output_dir = Path.join(get_models_path(), "micro")
      File.mkdir_p!(output_dir)

      publish? = opts[:publish] || false

      results =
        Enum.map(names, fn name ->
          train_classifier(name, output_dir, opts[:verbose] || false, publish?)
        end)

      successes = Enum.count(results, fn {status, _, _} -> status == :ok end)
      failures = Enum.count(results, fn {status, _, _} -> status == :error end)

      Mix.shell().info("\n--- Micro-Classifier Training Summary ---")
      Mix.shell().info("  Trained: #{successes}")
      Mix.shell().info("  Failed:  #{failures}")

      Enum.each(results, fn
        {:ok, name, count} ->
          Mix.shell().info("  [OK]    #{name} (#{count} examples)")

        {:error, name, reason} ->
          Mix.shell().error("  [FAIL]  #{name}: #{inspect(reason)}")
      end)
    end
  end

  defp train_classifier(name, output_dir, verbose, publish?) do
    data_path = data_file_path(name)

    case File.read(data_path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} ->
            training_data =
              Enum.map(entries, fn entry ->
                {Map.get(entry, "text", ""), Map.get(entry, "label", "unknown")}
              end)

            if verbose do
              labels = training_data |> Enum.map(&elem(&1, 1)) |> Enum.frequencies()
              Mix.shell().info("Training #{name}: #{length(training_data)} examples, labels: #{inspect(labels)}")
            end

            model = SimpleClassifier.train(training_data)
            model_path = Path.join(output_dir, "#{name}.term")
            binary = :erlang.term_to_binary(model)
            File.write!(model_path, binary)

            if publish? do
              remote_key = ModelStore.version_prefix() <> "micro/#{name}.term"
              ModelStore.publish(model_path, remote_key)
            end

            if verbose do
              Mix.shell().info("  Saved to #{model_path}")
            end

            {:ok, name, length(training_data)}

          {:error, reason} ->
            {:error, name, {:json_decode, reason}}
        end

      {:error, reason} ->
        {:error, name, {:file_read, reason, data_path}}
    end
  end

  defp list_classifiers do
    Mix.shell().info("Available micro-classifiers:\n")

    Enum.each(@classifier_names, fn name ->
      data_path = data_file_path(name)
      model_path = Path.join([get_models_path(), "micro", "#{name}.term"])

      data_status =
        case File.read(data_path) do
          {:ok, json} ->
            case Jason.decode(json) do
              {:ok, entries} -> "#{length(entries)} examples"
              _ -> "invalid JSON"
            end

          _ ->
            "missing"
        end

      model_status = if File.exists?(model_path), do: "trained", else: "not trained"

      Mix.shell().info("  #{name}")
      Mix.shell().info("    Data:  #{data_status} (#{data_path})")
      Mix.shell().info("    Model: #{model_status}")
    end)
  end

  defp get_models_path do
    Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
  end

  defp data_file_path(name) do
    priv_dir = :code.priv_dir(:brain) |> to_string()

    umbrella_root =
      case File.read_link(priv_dir) do
        {:ok, link_target} ->
          parent = Path.dirname(priv_dir)
          real_priv = Path.join(parent, link_target) |> Path.expand()
          Path.join(real_priv, "../../..") |> Path.expand()

        {:error, _} ->
          Path.join(priv_dir, "../../../../..") |> Path.expand()
      end

    Path.join(umbrella_root, "data/classifiers/#{name}.json")
  end
end
