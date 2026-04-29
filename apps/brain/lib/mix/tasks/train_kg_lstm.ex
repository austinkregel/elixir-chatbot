defmodule Mix.Tasks.TrainKgLstm do
  @shortdoc "Train the KG-LSTM triple scorer"
  @moduledoc """
  Trains a knowledge graph triple scorer using BiLSTM with an expanded
  training set from hierarchy, knowledge, ConceptNet, SRL corpus, and
  BeliefStore triples.

  ## Usage

      mix train_kg_lstm [options]

  ## Options

    * `--world` - World ID (default: "default")
    * `--epochs` - Number of training epochs (default: 50)
    * `--neg-ratio` - Negative samples per positive (default: 5)
    * `--verbose` - Print detailed progress
    * `--publish` - Upload model to ModelStore after training
    * `--max-conceptnet` - Max ConceptNet triples to include (default: 50000)
    * `--include-srl` - Include SRL corpus triples (default: true)
    * `--include-conceptnet` - Include ConceptNet triples (default: true)
    * `--include-beliefs` - Include BeliefStore triples (default: true)

  ## Preconditions

  ConceptNet data must be ingested before training (unless `--no-include-conceptnet`).
  Run `mix download_wordnet && mix ingest_lexicon` if ConceptNet is missing.
  """

  use Mix.Task

  alias Brain.ML.KnowledgeGraph.{TripleScorer, PredicateNormalizer}
  alias Brain.ML.ModelStore
  alias Atlas.Graph.EdgeLabels

  @canonical_relations EdgeLabels.kg_relations() ++ ~w(
    MADE_BY AT_LOCATION OCCURRED_AT CAUSES HAS_PROPERTY
    PART_OF USED_FOR CAPABLE_OF HAS_A RECEIVES_ACTION DEFINED_AS SYMBOL_OF
    MANNER PURPOSE VISITED OWNS USES BELIEVES CLAIMS HOLDS SAYS KNOWS LIVES WORKS
  )

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          world: :string,
          epochs: :integer,
          neg_ratio: :integer,
          verbose: :boolean,
          publish: :boolean,
          max_conceptnet: :integer,
          include_srl: :boolean,
          include_conceptnet: :boolean,
          include_beliefs: :boolean
        ],
        aliases: [w: :world, e: :epochs, v: :verbose]
      )

    Mix.Task.run("app.start")

    world_id = Keyword.get(opts, :world, "default")
    epochs = Keyword.get(opts, :epochs, 50)
    neg_ratio = Keyword.get(opts, :neg_ratio, 5)
    verbose = Keyword.get(opts, :verbose, false)
    include_conceptnet = Keyword.get(opts, :include_conceptnet, true)
    include_srl = Keyword.get(opts, :include_srl, true)
    include_beliefs = Keyword.get(opts, :include_beliefs, true)
    max_conceptnet = Keyword.get(opts, :max_conceptnet, 50_000)

    if include_conceptnet do
      check_conceptnet_precondition!()
    end

    Mix.shell().info("Training KG-LSTM triple scorer for world: #{world_id}")

    triples = load_triples(opts, world_id, include_conceptnet, include_srl, include_beliefs, max_conceptnet)

    if Enum.empty?(triples) do
      Mix.shell().error("No triples found. Cannot train.")
      exit({:shutdown, 1})
    end

    relation_coverage = compute_relation_coverage(triples)
    Mix.shell().info("Loaded #{length(triples)} positive triples across #{map_size(relation_coverage)} relations")

    if verbose do
      Enum.each(Enum.sort_by(relation_coverage, fn {_, c} -> -c end), fn {rel, count} ->
        Mix.shell().info("  #{rel}: #{count}")
      end)
    end

    hard_negatives = load_hard_negatives()
    Mix.shell().info("Loaded #{length(hard_negatives)} hard negatives")

    model_version = build_model_version(include_conceptnet, include_srl)

    {:ok, _model, params, vocab, config} =
      TripleScorer.train(triples,
        epochs: epochs,
        neg_ratio: neg_ratio,
        verbose: verbose
      )

    output_path = model_path(world_id)

    TripleScorer.save_model(params, vocab, config, output_path,
      relation_coverage: relation_coverage,
      model_version: model_version
    )

    Mix.shell().info("Model saved to #{output_path}")
    Mix.shell().info("Model version: #{model_version}")
    Mix.shell().info("Relations trained: #{map_size(relation_coverage)}")

    if opts[:publish] do
      remote_key = ModelStore.version_prefix() <> "#{world_id}/kg_lstm/triple_scorer.term"
      ModelStore.publish(output_path, remote_key)
    end
  end

  defp check_conceptnet_precondition! do
    unless Brain.Lexicon.ConceptNet.has_data?("dog") do
      Mix.shell().error("""
      ConceptNet data is not loaded in the ETS table.

      The TripleScorer requires ConceptNet triples for meaningful training.
      Run the following commands first:

          mix download_wordnet && mix ingest_lexicon

      If you want to train without ConceptNet (not recommended), pass:

          mix train_kg_lstm --no-include-conceptnet
      """)

      exit({:shutdown, 1})
    end
  end

  defp load_triples(opts, world_id, include_conceptnet, include_srl, include_beliefs, max_conceptnet) do
    hierarchy_triples = load_hierarchy_triples()
    knowledge_triples = load_knowledge_triples()

    conceptnet_triples =
      if include_conceptnet do
        loaded = load_conceptnet_triples(max_conceptnet)
        Mix.shell().info("  ConceptNet: #{length(loaded)} triples")
        loaded
      else
        Mix.shell().info("  ConceptNet: skipped")
        []
      end

    srl_triples =
      if include_srl do
        loaded = load_srl_corpus_triples()
        Mix.shell().info("  SRL corpus: #{length(loaded)} triples")
        loaded
      else
        Mix.shell().info("  SRL corpus: skipped")
        []
      end

    belief_triples =
      if include_beliefs do
        loaded = load_belief_triples(world_id)
        Mix.shell().info("  Beliefs: #{length(loaded)} triples")
        loaded
      else
        Mix.shell().info("  Beliefs: skipped")
        []
      end

    _ = opts

    Mix.shell().info("  Hierarchy: #{length(hierarchy_triples)} triples")
    Mix.shell().info("  Knowledge: #{length(knowledge_triples)} triples")

    (hierarchy_triples ++ knowledge_triples ++ conceptnet_triples ++ srl_triples ++ belief_triples)
    |> normalize_all_predicates()
    |> Enum.uniq()
  end

  defp normalize_all_predicates(triples) do
    Enum.map(triples, fn {h, r, t} ->
      case PredicateNormalizer.normalize(r) do
        {:ok, canonical, _kind} -> {h, canonical, t}
        {:error, _} -> {h, r, t}
      end
    end)
  end

  defp compute_relation_coverage(triples) do
    Enum.reduce(triples, %{}, fn {_h, r, _t}, acc ->
      Map.update(acc, r, 1, &(&1 + 1))
    end)
  end

  defp build_model_version(include_conceptnet, include_srl) do
    sources =
      ["hierarchy", "knowledge"]
      |> then(fn s -> if include_conceptnet, do: s ++ ["conceptnet"], else: s end)
      |> then(fn s -> if include_srl, do: s ++ ["srl"], else: s end)
      |> Enum.join("+")

    ts = DateTime.utc_now() |> DateTime.to_unix()
    "v2-#{sources}-#{ts}"
  end

  # --- Triple sources ---

  defp load_hierarchy_triples do
    path = resolve_entity_types_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"type_hierarchy" => hierarchy}} ->
            Enum.flat_map(hierarchy, fn {parent, children} when is_list(children) ->
              Enum.flat_map(children, fn child ->
                [
                  {child, EdgeLabels.is_a(), parent},
                  {parent, EdgeLabels.has_subtype(), child}
                ]
              end)
            end)

          _ ->
            []
        end

      _ ->
        []
    end
  end

  defp load_knowledge_triples do
    knowledge_dir = resolve_knowledge_dir()

    knowledge_dir
    |> Path.join("*.json")
    |> Path.wildcard()
    |> Enum.flat_map(&extract_triples_from_knowledge/1)
  end

  defp load_conceptnet_triples(max_entries) do
    sample_concepts = ~w(
      dog cat car house person city water food music book
      computer phone tree animal bird fish school time money
      earth sun moon fire ice rain snow mountain river ocean
      doctor teacher lawyer student child parent family friend
      apple google microsoft facebook amazon
    )

    relations_map = Brain.Lexicon.ConceptNet.relations("dog")

    triples =
      if is_map(relations_map) do
        Enum.flat_map(sample_concepts, fn concept ->
          rels = Brain.Lexicon.ConceptNet.relations(concept)

          if is_map(rels) do
            Enum.flat_map(rels, fn {rel_type, related_list} ->
              related = if is_list(related_list), do: related_list, else: []

              normalized_rel = PredicateNormalizer.normalize(to_string(rel_type))
              if normalized_rel in @canonical_relations or to_string(rel_type) in @canonical_relations do
                Enum.map(related, fn target ->
                  target_str = if is_binary(target), do: target, else: to_string(target)
                  {concept, normalized_rel, target_str}
                end)
              else
                []
              end
            end)
          else
            []
          end
        end)
        |> Enum.uniq()
        |> Enum.take(max_entries)
      else
        []
      end

    triples
  end

  defp load_srl_corpus_triples do
    intent_dir = resolve_data_dir("intents")
    classifier_dir = resolve_data_dir("classifiers")

    texts =
      extract_texts_from_dir(intent_dir) ++
        extract_texts_from_dir(classifier_dir)

    texts
    |> Enum.uniq()
    |> Enum.flat_map(&srl_text_to_triples/1)
    |> filter_by_frequency(5)
  end

  defp load_belief_triples(_world_id) do
    if Brain.Epistemic.BeliefStore.ready?() do
      case Brain.Epistemic.BeliefStore.query_beliefs() do
        beliefs when is_list(beliefs) ->
          beliefs
          |> Enum.filter(fn b ->
            b.subject != :system and b.predicate != :consolidated_knowledge
          end)
          |> Enum.map(fn b ->
            {to_string(b.subject), to_string(b.predicate), to_string(b.object)}
          end)
          |> Enum.reject(fn {s, _p, o} -> s == "" or o == "" end)

        _ ->
          []
      end
    else
      []
    end
  rescue
    _ -> []
  end

  defp load_hard_negatives do
    path = resolve_hard_negatives_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, entries} when is_list(entries) ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "head", ""), Map.get(entry, "relation", ""), Map.get(entry, "tail", "")}
            end)

          _ ->
            []
        end

      {:error, _} ->
        []
    end
  end

  # --- SRL extraction from text ---

  defp srl_text_to_triples(text) do
    tokens = Brain.ML.Tokenizer.tokenize_words(text)

    if length(tokens) >= 2 do
      bio_tags = generate_minimal_bio_tags(tokens)
      frames = Brain.Analysis.SemanticRoleLabeler.label(tokens, bio_tags)
      Brain.Analysis.SemanticRoleLabeler.to_triples(frames)
    else
      []
    end
  rescue
    _ -> []
  end

  defp generate_minimal_bio_tags(tokens) do
    pos_tags =
    if Brain.ML.POSTagger.model_exists?() do
      case Brain.ML.POSTagger.load_model() do
        {:ok, model} ->
          tags = Brain.ML.POSTagger.predict(tokens, model)
          if is_list(tags) and length(tags) == length(tokens), do: tags, else: Enum.map(tokens, fn _ -> "NN" end)

        _ ->
          Enum.map(tokens, fn _ -> "NN" end)
      end
    else
      Enum.map(tokens, fn _ -> "NN" end)
    end

    pos_tags
    |> Enum.with_index()
    |> Enum.map(fn {tag, idx} ->
      cond do
        tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] -> if idx == 0, do: "B-V", else: "B-V"
        idx == 0 -> "B-ARG0"
        tag in ["NN", "NNS", "NNP", "NNPS", "PRP"] -> "B-ARG1"
        true -> "O"
      end
    end)
  end

  defp filter_by_frequency(triples, min_count) do
    freq =
      Enum.reduce(triples, %{}, fn {_h, r, _t}, acc ->
        Map.update(acc, r, 1, &(&1 + 1))
      end)

    Enum.filter(triples, fn {_h, r, _t} ->
      Map.get(freq, r, 0) >= min_count
    end)
  end

  # --- Text extraction from training data ---

  defp extract_texts_from_dir(dir) do
    if File.dir?(dir) do
      dir
      |> Path.join("*.json")
      |> Path.wildcard()
      |> Enum.flat_map(&extract_texts_from_file/1)
    else
      []
    end
  end

  defp extract_texts_from_file(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} -> extract_text_values(data)
          _ -> []
        end

      _ ->
        []
    end
  end

  defp extract_text_values(data) when is_map(data) do
    text_keys = ["text", "content", "utterance", "input", "query", "sentence"]

    direct =
      Enum.flat_map(text_keys, fn key ->
        case Map.get(data, key) do
          s when is_binary(s) and s != "" -> [s]
          _ -> []
        end
      end)

    nested = Enum.flat_map(data, fn {_k, v} -> extract_text_values(v) end)
    direct ++ nested
  end

  defp extract_text_values(data) when is_list(data) do
    Enum.flat_map(data, &extract_text_values/1)
  end

  defp extract_text_values(_), do: []

  # --- Knowledge triple extraction (existing) ---

  defp extract_triples_from_knowledge(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            device_triples = extract_device_triples(Map.get(data, "devices", %{}))
            fact_triples = extract_fact_triples(Map.get(data, "facts", []))
            device_triples ++ fact_triples

          _ ->
            []
        end

      _ ->
        []
    end
  end

  defp extract_device_triples(devices) when is_map(devices) do
    Enum.flat_map(devices, fn {name, attrs} when is_map(attrs) ->
      type_triple =
        case Map.get(attrs, "type") do
          nil -> []
          type -> [{name, "has_type", type}]
        end

      brand_triple =
        case Map.get(attrs, "brand") do
          nil -> []
          brand -> [{name, "made_by", brand}]
        end

      location_triple =
        case Map.get(attrs, "location") do
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

      _ ->
        []
    end)
    |> Enum.uniq()
  end

  defp extract_fact_triples(_), do: []

  # --- Path resolution ---

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

  defp resolve_data_dir(subdir) do
    cond do
      File.dir?("data/#{subdir}") -> "data/#{subdir}"
      File.dir?("apps/brain/data/#{subdir}") -> "apps/brain/data/#{subdir}"
      true -> "data/#{subdir}"
    end
  end

  defp resolve_hard_negatives_path do
    cond do
      File.exists?("data/kg/hard_negatives.json") ->
        "data/kg/hard_negatives.json"

      true ->
        "data/kg/hard_negatives.json"
    end
  end

  defp model_path(world_id) do
    priv = :code.priv_dir(:brain) |> to_string()
    Path.join([priv, "ml_models", world_id, "kg_lstm", "triple_scorer.term"])
  end
end
