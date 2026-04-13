defmodule Mix.Tasks.TrainKgLstm do
  @shortdoc "Train the KG-LSTM triple scorer"
  @moduledoc """
  Trains a knowledge graph triple scorer using BiLSTM.

  ## Usage

      mix train_kg_lstm [options]

  ## Options

    * `--world` - World ID (default: "default")
    * `--epochs` - Number of training epochs (default: 50)
    * `--neg-ratio` - Negative samples per positive (default: 5)
    * `--verbose` - Print detailed progress
  """

  use Mix.Task

  alias Brain.ML.KnowledgeGraph.TripleScorer
  alias Brain.ML.ModelStore

  @impl true
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [
        world: :string,
        epochs: :integer,
        neg_ratio: :integer,
        verbose: :boolean,
        publish: :boolean
      ],
      aliases: [w: :world, e: :epochs, v: :verbose]
    )

    Mix.Task.run("app.start")

    world_id = Keyword.get(opts, :world, "default")
    epochs = Keyword.get(opts, :epochs, 50)
    neg_ratio = Keyword.get(opts, :neg_ratio, 5)
    verbose = Keyword.get(opts, :verbose, false)

    Mix.shell().info("Training KG-LSTM triple scorer for world: #{world_id}")

    triples = load_triples()

    if Enum.empty?(triples) do
      Mix.shell().error("No triples found")
      exit({:shutdown, 1})
    end

    Mix.shell().info("Loaded #{length(triples)} positive triples")

    {:ok, _model, params, vocab, config} = TripleScorer.train(triples,
      epochs: epochs,
      neg_ratio: neg_ratio,
      verbose: verbose
    )

    output_path = model_path(world_id)
    TripleScorer.save_model(params, vocab, config, output_path)
    Mix.shell().info("Model saved to #{output_path}")

    if opts[:publish] do
      remote_key = ModelStore.version_prefix() <> "#{world_id}/kg_lstm/triple_scorer.term"
      ModelStore.publish(output_path, remote_key)
    end
  end

  defp load_triples do
    hierarchy_triples = load_hierarchy_triples()
    knowledge_triples = load_knowledge_triples()

    (hierarchy_triples ++ knowledge_triples)
    |> Enum.uniq()
  end

  defp load_hierarchy_triples do
    path = resolve_entity_types_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"type_hierarchy" => hierarchy}} ->
            Enum.flat_map(hierarchy, fn {parent, children} when is_list(children) ->
              Enum.flat_map(children, fn child ->
                [
                  {child, "is_a", parent},
                  {parent, "has_subtype", child}
                ]
              end)
            end)

          _ -> []
        end

      _ -> []
    end
  end

  defp load_knowledge_triples do
    knowledge_dir = resolve_knowledge_dir()

    knowledge_dir
    |> Path.join("*.json")
    |> Path.wildcard()
    |> Enum.flat_map(&extract_triples_from_knowledge/1)
  end

  defp extract_triples_from_knowledge(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            device_triples = extract_device_triples(Map.get(data, "devices", %{}))
            fact_triples = extract_fact_triples(Map.get(data, "facts", []))
            device_triples ++ fact_triples

          _ -> []
        end

      _ -> []
    end
  end

  defp extract_device_triples(devices) when is_map(devices) do
    Enum.flat_map(devices, fn {name, attrs} when is_map(attrs) ->
      type_triple = case Map.get(attrs, "type") do
        nil -> []
        type -> [{name, "has_type", type}]
      end

      brand_triple = case Map.get(attrs, "brand") do
        nil -> []
        brand -> [{name, "made_by", brand}]
      end

      location_triple = case Map.get(attrs, "location") do
        nil -> []
        loc -> [{name, "located_in", loc}]
      end

      type_triple ++ brand_triple ++ location_triple
    end)
  end

  defp extract_device_triples(_), do: []

  defp extract_fact_triples(facts) when is_list(facts) do
    Enum.flat_map(facts, fn
      %{"entity" => entity} when is_binary(entity) and entity != "" ->
        [{entity, "mentioned_in", "knowledge_base"}]
      _ -> []
    end)
    |> Enum.uniq()
  end

  defp extract_fact_triples(_), do: []

  defp resolve_entity_types_path do
    cond do
      File.exists?("priv/analysis/entity_types.json") ->
        "priv/analysis/entity_types.json"
      File.exists?("apps/brain/priv/analysis/entity_types.json") ->
        "apps/brain/priv/analysis/entity_types.json"
      true ->
        priv = :code.priv_dir(:brain) |> to_string()
        Path.join([priv, "analysis", "entity_types.json"])
    end
  end

  defp resolve_knowledge_dir do
    cond do
      File.dir?("priv/knowledge") -> "priv/knowledge"
      File.dir?("apps/brain/priv/knowledge") -> "apps/brain/priv/knowledge"
      true ->
        priv = :code.priv_dir(:brain) |> to_string()
        Path.join(priv, "knowledge")
    end
  end

  defp model_path(world_id) do
    priv = :code.priv_dir(:brain) |> to_string()
    Path.join([priv, "ml_models", world_id, "kg_lstm", "triple_scorer.term"])
  end
end
