defmodule Mix.Tasks.Axes.Experiment do
  @shortdoc "Compare one flat intent classifier against one classifier per dotted position"

  @moduledoc """
  Measures whether predicting an intent as several independent parts beats
  predicting it as one opaque label.

      mix axes.experiment
      mix axes.experiment --verbose

  Both sides see the same feature vectors, the same train/held-out split and the
  same training seed, so the only difference is how the label is carved.

  ## How the axes are derived

  By dotted position, with no vocabulary of any kind. `smarthome.device.switch.on`
  contributes `p1=smarthome`, `p2=device`, `p3=switch`, `p4=on`, `p5=(none)`. A
  label shorter than the deepest one pads with `(none)`, which is a real class: it
  means "this intent has no part here".

  Position is the whole rule because the taxonomy already encodes its axes
  positionally. Naming the axes instead — deciding that `on` is an *operation* and
  `brightness` an *object* — would need a list of which words are which, and a
  list like that is the thing it would exist to avoid.

  ## What is reported

    * flat accuracy: one classifier over every distinct intent
    * per-position accuracy: one classifier per position, scored independently
    * joint accuracy: the positions predicted separately and rejoined with `.`,
      scored against the true label

  Joint accuracy is the comparable number. Per-position accuracy is always higher
  and does not mean the scheme works: every position has to be right for the
  rejoined label to be right.

  ## Inputs

  Train vectors come from `data/classifiers/intent_full.json`, which
  `mix gen_micro_data` writes, so this does not recompute them. Held-out vectors
  are computed here, because that file holds the train split only.
  """

  use Mix.Task

  alias Brain.Analysis.{FeatureExtractor, Pipeline}
  alias Brain.ML.{EvaluationStore, FeatureVectorClassifier}

  @requirements ["app.start"]

  @switches [verbose: :boolean]

  @none "(none)"

  @impl Mix.Task
  def run(args) do
    {opts, positional, invalid} = OptionParser.parse(args, strict: @switches)

    if invalid != [], do: Mix.raise("axes.experiment: unknown options #{inspect(invalid)}")
    if positional != [], do: Mix.raise("axes.experiment: unexpected arguments #{inspect(positional)}")

    train = load_train_vectors!()
    held_out = compute_held_out_vectors!()

    depth = max_depth(Enum.map(train ++ held_out, & &1.label))
    Mix.shell().info("  deepest label: #{depth} parts\n")

    flat = evaluate_flat(train, held_out)
    deployed = evaluate_deployed(held_out)
    positions = Enum.map(1..depth, fn pos -> evaluate_position(train, held_out, pos) end)
    joint = evaluate_joint(train, held_out, depth)

    report(flat, deployed, positions, joint, depth, opts[:verbose] || false)
  end

  # The model `mix train_micro` actually wrote, scored on the same vectors as the
  # flat baseline. It differs from that baseline in one respect: train_micro runs
  # a GA weight search and passes `weights:`, while the baseline trains unweighted.
  defp evaluate_deployed(held_out) do
    path = Brain.data_path("../apps/brain/priv/ml_models/micro/intent_full.term")
    path = if File.regular?(path), do: path, else: Brain.priv_path("ml_models/micro/intent_full.term")

    if File.regular?(path) do
      model = path |> File.read!() |> :erlang.binary_to_term()
      correct = Enum.count(held_out, fn row -> predict(model, row.vector) == row.label end)
      weights = Map.get(model, :weights)

      %{
        present: true,
        weighted: is_list(weights),
        weight_count: if(is_list(weights), do: length(weights), else: 0),
        classes: map_size(model.label_centroids),
        correct: correct,
        total: length(held_out),
        accuracy: correct / length(held_out)
      }
    else
      %{present: false}
    end
  end

  # -- inputs -----------------------------------------------------------------

  defp load_train_vectors!() do
    path = Brain.data_path("classifiers/intent_full.json")

    unless File.regular?(path) do
      Mix.raise("axes.experiment: no train vectors at #{path}. Run `mix gen_micro_data` first.")
    end

    rows =
      path
      |> File.read!()
      |> Jason.decode!()
      |> Enum.flat_map(fn row ->
        case {row["feature_vector"], row["label"]} do
          {vec, label} when is_list(vec) and is_binary(label) -> [%{vector: vec, label: label}]
          _ -> []
        end
      end)

    if rows == [] do
      Mix.raise("axes.experiment: #{path} holds no usable {feature_vector, label} rows")
    end

    Mix.shell().info("  train rows:    #{length(rows)} (from #{Path.relative_to_cwd(path)})")
    rows
  end

  defp compute_held_out_vectors!() do
    held_out = EvaluationStore.load_gold_standard("intent", :held_out)

    if held_out == [] do
      Mix.raise("axes.experiment: held_out.json is empty. Run `mix split.held_out intent`.")
    end

    Mix.shell().info("  held-out rows: #{length(held_out)}, computing vectors ...")

    rows =
      held_out
      |> Task.async_stream(
        fn entry ->
          analysis = Pipeline.analyze_chunk(entry["text"], side_effects: false)
          {vector, _word_feats} = FeatureExtractor.extract(analysis)
          %{vector: vector, label: entry["intent"]}
        end,
        # Deliberately below System.schedulers_online(). analyze_chunk/2 reaches
        # Brain.Memory.Store.query_semantic through ContextAccumulator, and that
        # is one GenServer with a 5s call timeout: enough parallel pipeline runs
        # queue behind it and the call times out rather than the work being slow.
        max_concurrency: 4,
        timeout: 120_000,
        ordered: false
      )
      |> Enum.map(fn {:ok, row} -> row end)

    # No partial evaluation: a held-out row that failed would silently make the
    # measurement easier for both sides.
    if length(rows) != length(held_out) do
      Mix.raise("axes.experiment: #{length(held_out) - length(rows)} held-out rows failed the pipeline")
    end

    rows
  end

  # -- label carving ----------------------------------------------------------

  defp parts(label), do: String.split(label, ".")

  defp max_depth(labels), do: labels |> Enum.map(&length(parts(&1))) |> Enum.max()

  defp part_at(label, pos) do
    case Enum.at(parts(label), pos - 1) do
      nil -> @none
      part -> part
    end
  end

  # -- evaluation -------------------------------------------------------------

  defp evaluate_flat(train, held_out) do
    model = FeatureVectorClassifier.train(Enum.map(train, &{&1.vector, &1.label}))
    correct = Enum.count(held_out, fn row -> predict(model, row.vector) == row.label end)

    %{
      classes: model.label_centroids |> map_size(),
      correct: correct,
      total: length(held_out),
      accuracy: correct / length(held_out)
    }
  end

  defp evaluate_position(train, held_out, pos) do
    model =
      FeatureVectorClassifier.train(Enum.map(train, &{&1.vector, part_at(&1.label, pos)}))

    correct =
      Enum.count(held_out, fn row -> predict(model, row.vector) == part_at(row.label, pos) end)

    %{
      position: pos,
      classes: map_size(model.label_centroids),
      correct: correct,
      total: length(held_out),
      accuracy: correct / length(held_out)
    }
  end

  defp evaluate_joint(train, held_out, depth) do
    models =
      Map.new(1..depth, fn pos ->
        {pos, FeatureVectorClassifier.train(Enum.map(train, &{&1.vector, part_at(&1.label, pos)}))}
      end)

    correct =
      Enum.count(held_out, fn row ->
        rejoined =
          1..depth
          |> Enum.map(fn pos -> predict(models[pos], row.vector) end)
          |> Enum.reject(&(&1 in [@none, nil]))
          |> Enum.join(".")

        rejoined == row.label
      end)

    %{correct: correct, total: length(held_out), accuracy: correct / length(held_out)}
  end

  defp predict(model, vector) do
    case FeatureVectorClassifier.classify(vector, model) do
      {:ok, label, _confidence, _details} -> label
      :error -> nil
    end
  end

  # -- reporting --------------------------------------------------------------

  defp report(flat, deployed, positions, joint, depth, verbose?) do
    Mix.shell().info("")
    Mix.shell().info(String.duplicate("=", 66))
    Mix.shell().info("FLAT vs POSITIONAL, #{flat.total} held-out examples")
    Mix.shell().info(String.duplicate("=", 66))
    Mix.shell().info("")
    Mix.shell().info("  flat, unweighted (one classifier over every intent)")
    Mix.shell().info("    classes  #{flat.classes}")
    Mix.shell().info("    accuracy #{pct(flat.accuracy)}  (#{flat.correct}/#{flat.total})")
    Mix.shell().info("")

    if deployed.present do
      Mix.shell().info("  deployed intent_full.term (same vectors)")
      Mix.shell().info("    classes  #{deployed.classes}")

      Mix.shell().info(
        "    weights  #{if deployed.weighted, do: "#{deployed.weight_count} GA weights", else: "none"}"
      )

      Mix.shell().info("    accuracy #{pct(deployed.accuracy)}  (#{deployed.correct}/#{deployed.total})")
      Mix.shell().info("")
    end

    Mix.shell().info("  per position, scored independently")

    Enum.each(positions, fn p ->
      Mix.shell().info(
        "    p#{p.position}  classes #{String.pad_leading(to_string(p.classes), 4)}   " <>
          "accuracy #{pct(p.accuracy)}"
      )
    end)

    Mix.shell().info("")
    Mix.shell().info("  joint (#{depth} positions predicted separately, rejoined)")
    Mix.shell().info("    accuracy #{pct(joint.accuracy)}  (#{joint.correct}/#{joint.total})")
    Mix.shell().info("")

    delta = joint.accuracy - flat.accuracy

    verdict =
      cond do
        delta > 0.02 -> "positional wins by #{pct(delta)}"
        delta < -0.02 -> "flat wins by #{pct(-delta)}"
        true -> "no meaningful difference (#{pct(abs(delta))})"
      end

    Mix.shell().info("  #{verdict}")
    Mix.shell().info("")

    if verbose? do
      Mix.shell().info("  Per-position accuracy is not the comparable number. Every position")
      Mix.shell().info("  must be right for the rejoined label to be right, so the product of")
      Mix.shell().info("  the per-position rates bounds the joint rate from above:")

      bound = positions |> Enum.map(& &1.accuracy) |> Enum.reduce(1.0, &(&1 * &2))
      Mix.shell().info("    product of per-position accuracies: #{pct(bound)}")
      Mix.shell().info("    measured joint:                     #{pct(joint.accuracy)}")
      Mix.shell().info("")
    end
  end

  defp pct(x), do: "#{Float.round(x * 100, 1)}%"
end
