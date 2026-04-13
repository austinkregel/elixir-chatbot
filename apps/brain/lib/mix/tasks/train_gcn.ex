defmodule Mix.Tasks.TrainGcn do
  @shortdoc "Train the GCN text classification model"
  @moduledoc """
  Trains a Graph Convolutional Network for text classification.

  Builds a text graph from intent training data, trains a two-layer GCN,
  and saves the model to `priv/ml_models/{world_id}/gcn/model.term`.

  ## Usage

      mix train_gcn [options]

  ## Options

    * `--world` - World ID to scope training (default: "default")
    * `--epochs` - Number of training epochs (default: 200)
    * `--hidden-dim` - Hidden layer dimension (default: 200)
    * `--learning-rate` - Learning rate (default: 0.01)
    * `--vocab-size` - Maximum vocabulary size (default: 2000)
    * `--verbose` - Print detailed progress
  """

  use Mix.Task

  alias Brain.ML.GCN.{TextGraph, Model}
  alias Brain.ML.ModelStore

  @default_epochs 200
  @default_hidden_dim 200
  @default_learning_rate 0.01
  @default_vocab_size 2000

  @impl true
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [
        world: :string,
        epochs: :integer,
        hidden_dim: :integer,
        learning_rate: :float,
        vocab_size: :integer,
        verbose: :boolean,
        publish: :boolean
      ],
      aliases: [w: :world, e: :epochs, v: :verbose]
    )

    Mix.Task.run("app.start")

    world_id = Keyword.get(opts, :world, "default")
    epochs = Keyword.get(opts, :epochs, @default_epochs)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    learning_rate = Keyword.get(opts, :learning_rate, @default_learning_rate)
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    verbose = Keyword.get(opts, :verbose, false)

    Mix.shell().info("Training GCN model for world: #{world_id}")

    documents = load_training_data()

    if Enum.empty?(documents) do
      Mix.shell().error("No training data found in data/intents/*.json")
      exit({:shutdown, 1})
    end

    Mix.shell().info("Loaded #{length(documents)} training examples")

    if verbose do
      labels = Enum.map(documents, fn {_text, label} -> label end) |> Enum.frequencies()
      Mix.shell().info("Label distribution: #{inspect(labels)}")
    end

    Mix.shell().info("Building text graph (vocab_size: #{vocab_size})...")
    text_graph = TextGraph.build(documents, vocab_size: vocab_size)

    Mix.shell().info(
      "Graph: #{text_graph.num_docs} docs + #{text_graph.num_words} words = " <>
      "#{text_graph.num_docs + text_graph.num_words} nodes, " <>
      "#{text_graph.num_classes} classes"
    )

    Mix.shell().info("Training GCN (epochs: #{epochs}, hidden: #{hidden_dim}, lr: #{learning_rate})...")

    {:ok, _model, params, adjacency_norm, config} =
      Model.train(text_graph,
        epochs: epochs,
        learning_rate: learning_rate,
        hidden_dim: hidden_dim,
        verbose: true
      )

    output_path = model_path(world_id)
    Model.save_model(params, text_graph, config, output_path)
    Mix.shell().info("Model saved to #{output_path}")

    if opts[:publish] do
      remote_key = ModelStore.version_prefix() <> "#{world_id}/gcn/model.term"
      ModelStore.publish(output_path, remote_key)
    end

    if verbose do
      evaluate_model(params, text_graph, adjacency_norm)
    end
  end

  defp load_training_data do
    data_dir = resolve_data_dir()

    docs = data_dir
    |> Path.join("*.json")
    |> Path.wildcard()
    |> Enum.flat_map(&load_intent_file/1)

    if Enum.empty?(docs) do
      load_gold_standard_fallback()
    else
      docs
    end
  end

  defp load_intent_file(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, entries} when is_list(entries) ->
            Enum.flat_map(entries, fn
              %{"text" => text, "intent" => intent} -> [{text, intent}]
              %{"text" => text, "label" => label} -> [{text, label}]
              _ -> []
            end)

          _ -> []
        end

      _ -> []
    end
  end

  defp load_gold_standard_fallback do
    path = resolve_gold_standard_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, entries} when is_list(entries) ->
            Enum.flat_map(entries, fn
              %{"text" => text, "intent" => intent} -> [{text, intent}]
              _ -> []
            end)

          _ -> []
        end

      _ -> []
    end
  end

  defp resolve_data_dir do
    cond do
      File.dir?("data/intents") -> "data/intents"
      File.dir?("../../data/intents") -> "../../data/intents"
      true -> "data/intents"
    end
  end

  defp resolve_gold_standard_path do
    cond do
      File.exists?("priv/evaluation/intent/gold_standard.json") ->
        "priv/evaluation/intent/gold_standard.json"
      File.exists?("apps/brain/priv/evaluation/intent/gold_standard.json") ->
        "apps/brain/priv/evaluation/intent/gold_standard.json"
      true ->
        priv = :code.priv_dir(:brain) |> to_string()
        Path.join([priv, "evaluation", "intent", "gold_standard.json"])
    end
  end

  defp model_path(world_id) do
    priv = :code.priv_dir(:brain) |> to_string()
    Path.join([priv, "ml_models", world_id, "gcn", "model.term"])
  end

  defp evaluate_model(params, text_graph, adjacency_norm) do
    predict_fn = Nx.Defn.jit(&Model.gcn_forward/3, compiler: EXLA)
    output = predict_fn.(params, text_graph.features, adjacency_norm)

    predictions = Nx.argmax(output, axis: 1)
    labels = text_graph.labels
    mask = Nx.not_equal(labels, -1)

    correct = Nx.equal(predictions, Nx.max(labels, 0))
    |> Nx.multiply(mask)
    |> Nx.sum()
    |> Nx.to_number()

    total = Nx.sum(mask) |> Nx.to_number()

    accuracy = if total > 0, do: correct / total * 100, else: 0.0
    Mix.shell().info("Train accuracy: #{Float.round(accuracy, 1)}% (#{trunc(correct)}/#{trunc(total)})")
  end
end
