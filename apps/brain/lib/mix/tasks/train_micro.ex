defmodule Mix.Tasks.TrainMicro do
  @moduledoc """
  Train **all** TF-IDF micro-classifiers from `data/classifiers/<name>.json`.

  Each JSON file is an array of `{"text": "...", "label": "..."}` objects. Models are
  written to `priv/ml_models/micro/<name>.term` (resolved via `:brain` `:ml` `:models_path`
  when set).

  ## Classifiers (15)

  **Pragmatic / routing:** `personal_question`, `clarification_response`,
  `modal_directive`, `fallback_response`, `goal_type`, `entity_type`,
  `user_fact_type`, `directed_at_bot`, `event_argument_role`.

  **ChunkProfile axes** (labels derived from `gold_standard.json` via `mix gen_micro_data`):
  `intent_full`, `intent_domain`, `tense_class`, `aspect_class`, `urgency`,
  `certainty_level`, `coarse_semantic_class`.

  ## Workflow

  1. After changing `priv/evaluation/intent/gold_standard.json` or axis heuristics,
     regenerate axis JSON: `mix gen_micro_data`.
  2. Train everything: `mix train_micro`, or one model: `mix train_micro --only intent_domain`.
  3. Full pipeline also runs this task as stage 9 of `mix train` unless `--skip-micro`.

  ## Usage

      mix train_micro [options]

  ## Options

      --only NAME    Train only one classifier (e.g. `--only intent_domain`)
      --list         List classifiers and data/model file status
      --verbose      Per-classifier label counts and output paths
      --publish      Publish trained `.term` files via ModelStore when configured
  """

  use Mix.Task
  require Logger

  alias Brain.ML.FeatureVectorClassifier
  alias Brain.ML.ModelStore
  alias Brain.ML.SimpleClassifier
  alias Brain.ML.WeightOptimizer

  @shortdoc "Train micro-classifiers from data/classifiers/*.json"

  # Axis classifiers that operate on dense feature vectors rather than text.
  # Training records for these MUST have shape %{"feature_vector" => [..], "label" => ..}.
  @feature_vector_classifiers ~w(
    intent_full
    intent_domain
    tense_class
    aspect_class
    urgency
    certainty_level
  )

  @text_classifiers ~w(
    personal_question
    clarification_response
    modal_directive
    fallback_response
    goal_type
    entity_type
    user_fact_type
    directed_at_bot
    event_argument_role
    coarse_semantic_class
  )

  @classifier_names @feature_vector_classifiers ++ @text_classifiers

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          only: :string,
          list: :boolean,
          verbose: :boolean,
          publish: :boolean,
          skip_weight_optimization: :boolean,
          no_balance: :boolean
        ],
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

      skip_optimization? = opts[:skip_weight_optimization] || false
      balance? = !(opts[:no_balance] || false)

      results =
        Enum.map(names, fn name ->
          train_classifier(name, output_dir, opts[:verbose] || false, publish?, skip_optimization?, balance?)
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

      validate_intent_map_sync(output_dir)
    end
  end

  defp validate_intent_map_sync(output_dir) do
    intent_model_path = Path.join(output_dir, "intent_full.term")

    intent_map_path =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv/analysis/speech_act_intent_map.json"
        priv -> Path.join(priv, "analysis/speech_act_intent_map.json")
      end

    with {:ok, model_bin} <- File.read(intent_model_path),
         model <- :erlang.binary_to_term(model_bin),
         {:ok, map_json} <- File.read(intent_map_path),
         {:ok, intent_map} <- Jason.decode(map_json) do
      model_labels =
        case model do
          %{label_centroids: lc} when is_map(lc) ->
            lc |> Map.keys() |> Enum.map(&to_string/1) |> MapSet.new()

          _ ->
            MapSet.new()
        end

      if MapSet.size(model_labels) > 0 do
        map_values = intent_map |> Map.values() |> Enum.reject(&(&1 == "unknown")) |> MapSet.new()
        missing = MapSet.difference(map_values, model_labels)

        if MapSet.size(missing) > 0 do
          Mix.shell().error(
            "\n[WARN] speech_act_intent_map.json references intents not in intent_full model: #{inspect(MapSet.to_list(missing))}"
          )
        else
          Mix.shell().info("\n[OK] speech_act_intent_map.json labels validated against intent_full model")
        end
      end
    else
      _ -> :ok
    end
  end

  defp train_classifier(name, output_dir, verbose, publish?, skip_optimization?, balance?) do
    data_path = data_file_path(name)

    with {:ok, json} <- read_file(data_path),
         {:ok, entries} <- decode_json(json),
         {:ok, training_data, kind} <- to_training_data(name, entries) do
      if verbose do
        labels = training_data |> Enum.map(&elem(&1, 1)) |> Enum.frequencies()

        Mix.shell().info(
          "Training #{name} (#{kind}): #{length(training_data)} examples, " <>
            "labels: #{inspect(labels)}"
        )
      end

      model = train_model(kind, training_data, skip_optimization?, name, balance?)
      model_path = Path.join(output_dir, "#{name}.term")
      File.write!(model_path, :erlang.term_to_binary(model))

      if publish? do
        remote_key = ModelStore.version_prefix() <> "micro/#{name}.term"
        ModelStore.publish(model_path, remote_key)
      end

      if verbose do
        Mix.shell().info("  Saved to #{model_path}")
      end

      {:ok, name, length(training_data)}
    else
      {:error, stage, reason} -> {:error, name, {stage, reason, data_path}}
      {:error, reason} -> {:error, name, reason}
    end
  end

  defp read_file(path) do
    case File.read(path) do
      {:ok, content} -> {:ok, content}
      {:error, reason} -> {:error, :file_read, reason}
    end
  end

  defp decode_json(json) do
    case Jason.decode(json) do
      {:ok, entries} -> {:ok, entries}
      {:error, reason} -> {:error, :json_decode, reason}
    end
  end

  # Convert JSON-decoded records into `{input, label}` training tuples.
  # Axis classifiers expect `"feature_vector"` records; text classifiers
  # expect `"text"` records. If the shape doesn't match, we return a
  # structured error rather than silently training on mismatched input.
  defp to_training_data(name, entries) do
    cond do
      name in @feature_vector_classifiers ->
        extract_feature_vector_pairs(name, entries)

      name in @text_classifiers ->
        extract_text_pairs(name, entries)

      true ->
        {:error, :shape_mismatch, {:unknown_classifier, name}}
    end
  end

  defp extract_feature_vector_pairs(name, entries) do
    pairs =
      Enum.flat_map(entries, fn
        %{"feature_vector" => vec, "label" => label}
        when is_list(vec) and is_binary(label) and length(vec) > 0 ->
          [{vec, label}]

        _ ->
          []
      end)

    case pairs do
      [] ->
        {:error, :shape_mismatch,
         {name, "no records with non-empty :feature_vector found. " <>
                  "Did you run `mix gen_micro_data` after the axis classifier migration?"}}

      _ ->
        {:ok, pairs, :feature_vector}
    end
  end

  defp extract_text_pairs(_name, entries) do
    pairs =
      Enum.map(entries, fn entry ->
        {Map.get(entry, "text", ""), Map.get(entry, "label", "unknown")}
      end)

    {:ok, pairs, :text}
  end

  defp train_model(:feature_vector, training_data, skip_optimization?, name, balance?) do
    n_classes = training_data |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> length()

    balance_label = if balance?, do: " (balanced centroids)", else: ""

    cond do
      n_classes < 2 ->
        Mix.shell().info("  [#{name}] Only #{n_classes} class — skipping GA (nothing to optimize)")
        FeatureVectorClassifier.train(training_data, balance: balance?)

      skip_optimization? ->
        Mix.shell().info("  [#{name}] Skipping weight optimization (--skip-weight-optimization)#{balance_label}")
        FeatureVectorClassifier.train(training_data, balance: balance?)

      true ->
        Mix.shell().info("  [#{name}] Running GA weight optimization (#{n_classes} classes, balanced fitness)...")

        result = WeightOptimizer.optimize(training_data, verbose: true, classifier: name)

        Mix.shell().info(
          "  [#{name}] GA complete: #{Float.round(result.fitness * 100, 1)}% " <>
            "balanced accuracy at generation #{result.generation}"
        )

        alive = Enum.count(result.weights, &(&1 > 0.01))
        Mix.shell().info("  [#{name}] #{alive}/#{length(result.weights)} dimensions active (weight > 0.01)")

        FeatureVectorClassifier.train(training_data, weights: result.weights, balance: balance?)
    end
  end

  defp train_model(:text, training_data, _skip_optimization?, _name, _balance?) do
    SimpleClassifier.train(training_data)
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
