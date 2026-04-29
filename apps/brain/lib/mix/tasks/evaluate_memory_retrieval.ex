defmodule Mix.Tasks.Evaluate.MemoryRetrieval do
  @shortdoc "Evaluate memory retrieval quality with ablation support"
  @moduledoc """
  Evaluates memory retrieval (Memory.Store.query_similar) against a gold
  standard dataset. Reports MRR@5, Recall@5, and NDCG@5.

  ## Usage

      mix evaluate.memory_retrieval              # Default (current config)
      mix evaluate.memory_retrieval --no-rerank  # TF-IDF only baseline
      mix evaluate.memory_retrieval --wider-only # TF-IDF k*4 pool, no KG blend
      mix evaluate.memory_retrieval --rerank     # Full KG re-rank
      mix evaluate.memory_retrieval --blend-weight 0.4  # Custom blend weight
      mix evaluate.memory_retrieval --save       # Save results to file
      mix evaluate.memory_retrieval --verbose    # Per-query detail

  ## Ablation modes

  - `--no-rerank` — TF-IDF top-k only (baseline, same as pre-phase-4 behaviour)
  - `--wider-only` — TF-IDF top-k*4 truncated to k, no KG blend
  - `--rerank` — full KG entity re-rank (the candidate change)
  - `--blend-weight FLOAT` — override the 0.7/0.3 TF-IDF/KG mix
  """

  use Mix.Task

  @k 5
  @gold_path "apps/brain/priv/evaluation/memory_retrieval_gold.json"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          no_rerank: :boolean,
          wider_only: :boolean,
          rerank: :boolean,
          blend_weight: :float,
          save: :boolean,
          verbose: :boolean
        ],
        aliases: [v: :verbose, s: :save]
      )

    Mix.Task.run("app.start")

    mode = determine_mode(opts)
    verbose = Keyword.get(opts, :verbose, false)
    blend_weight = Keyword.get(opts, :blend_weight, 0.3)

    Mix.shell().info("=== Memory Retrieval Evaluation ===")
    Mix.shell().info("Mode: #{mode}")
    Mix.shell().info("K: #{@k}")

    if mode == :rerank do
      Mix.shell().info("Blend weight (KG): #{blend_weight}")
    end

    gold_entries = load_gold_standard()
    Mix.shell().info("Gold standard entries: #{length(gold_entries)}")

    results =
      Enum.map(gold_entries, fn entry ->
        query = entry["query"]
        expected_tags = entry["expected_tags"] || []

        retrieved = retrieve(query, mode, blend_weight)

        relevance_scores =
          Enum.map(retrieved, fn {episode, _sim} ->
            episode_tags = Map.get(episode, :tags, [])
            overlap = length(expected_tags -- (expected_tags -- episode_tags))
            if overlap > 0, do: 1.0, else: 0.0
          end)

        mrr = compute_mrr(relevance_scores)
        recall = compute_recall(relevance_scores, length(expected_tags))
        ndcg = compute_ndcg(relevance_scores)

        if verbose do
          Mix.shell().info("  Q: #{query}")
          Mix.shell().info("    MRR: #{Float.round(mrr, 3)}, Recall: #{Float.round(recall, 3)}, NDCG: #{Float.round(ndcg, 3)}")
          Mix.shell().info("    Retrieved: #{length(retrieved)}, Expected tags: #{inspect(expected_tags)}")
        end

        %{query: query, mrr: mrr, recall: recall, ndcg: ndcg}
      end)

    avg_mrr = safe_avg(results, :mrr)
    avg_recall = safe_avg(results, :recall)
    avg_ndcg = safe_avg(results, :ndcg)

    Mix.shell().info("\n=== Results ===")
    Mix.shell().info("MRR@#{@k}:    #{Float.round(avg_mrr, 4)}")
    Mix.shell().info("Recall@#{@k}:  #{Float.round(avg_recall, 4)}")
    Mix.shell().info("NDCG@#{@k}:    #{Float.round(avg_ndcg, 4)}")
    Mix.shell().info("Queries evaluated: #{length(results)}")

    if Keyword.get(opts, :save, false) do
      save_results(mode, blend_weight, avg_mrr, avg_recall, avg_ndcg, length(results))
    end
  end

  defp determine_mode(opts) do
    cond do
      opts[:no_rerank] -> :no_rerank
      opts[:wider_only] -> :wider_only
      opts[:rerank] -> :rerank
      true -> :no_rerank
    end
  end

  defp retrieve(query, mode, blend_weight) do
    pool_multiplier =
      case mode do
        :no_rerank -> 2
        :wider_only -> 4
        :rerank -> 4
      end

    case Brain.Memory.Store.query_similar(query, @k * pool_multiplier) do
      {:ok, episodes} when is_list(episodes) ->
        results =
          case mode do
            :rerank ->
              rerank_with_kg(query, episodes, blend_weight)

            _ ->
              episodes
          end

        Enum.take(results, @k)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp rerank_with_kg(query, episodes, kg_weight) do
    tfidf_weight = 1.0 - kg_weight

    query_entities = extract_entity_names(query)

    if query_entities == [] do
      episodes
    else
      Enum.map(episodes, fn {episode, tfidf_score} ->
        episode_entities = extract_episode_entities(episode)

        kg_score =
          if episode_entities == [] do
            0.0
          else
            compute_max_entity_cosine(query_entities, episode_entities)
          end

        blended = tfidf_score * tfidf_weight + kg_score * kg_weight
        {episode, blended}
      end)
      |> Enum.sort_by(fn {_ep, score} -> -score end)
    end
  end

  defp extract_entity_names(text) do
    case Brain.ML.EntityExtractor.extract_entities(text) do
      entities when is_list(entities) ->
        Enum.map(entities, fn e ->
          Map.get(e, :value) || Map.get(e, :text, "")
        end)
        |> Enum.reject(&(&1 == ""))

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp extract_episode_entities(episode) do
    tags = Map.get(episode, :tags, [])
    state = Map.get(episode, :state, "")
    entity_names = extract_entity_names(state)
    Enum.uniq(tags ++ entity_names)
  end

  defp compute_max_entity_cosine(query_entities, episode_entities) do
    cache_mod = Brain.ML.KnowledgeGraph.EntityVectorCache

    if Code.ensure_loaded?(cache_mod) and function_exported?(cache_mod, :get_or_compute, 2) do
      query_vecs =
        Enum.flat_map(query_entities, fn name ->
          case cache_mod.get_or_compute("default", name) do
            {:ok, vec} -> [vec]
            _ -> []
          end
        end)

      episode_vecs =
        Enum.flat_map(episode_entities, fn name ->
          case cache_mod.get_or_compute("default", name) do
            {:ok, vec} -> [vec]
            _ -> []
          end
        end)

      if query_vecs == [] or episode_vecs == [] do
        0.0
      else
        max_cosine =
          for qv <- query_vecs, ev <- episode_vecs do
            FourthWall.Math.cosine_similarity(Nx.to_flat_list(qv), Nx.to_flat_list(ev))
          end
          |> Enum.max(fn -> 0.0 end)

        max(0.0, max_cosine)
      end
    else
      0.0
    end
  rescue
    _ -> 0.0
  end

  defp compute_mrr(relevance_scores) do
    case Enum.find_index(relevance_scores, &(&1 > 0)) do
      nil -> 0.0
      idx -> 1.0 / (idx + 1)
    end
  end

  defp compute_recall(relevance_scores, total_relevant) do
    if total_relevant == 0 do
      0.0
    else
      found = Enum.count(relevance_scores, &(&1 > 0))
      min(1.0, found / total_relevant)
    end
  end

  defp compute_ndcg(relevance_scores) do
    if relevance_scores == [] do
      0.0
    else
      dcg =
        relevance_scores
        |> Enum.with_index(1)
        |> Enum.reduce(0.0, fn {rel, pos}, acc ->
          acc + rel / :math.log2(pos + 1)
        end)

      ideal = Enum.sort(relevance_scores, :desc)

      idcg =
        ideal
        |> Enum.with_index(1)
        |> Enum.reduce(0.0, fn {rel, pos}, acc ->
          acc + rel / :math.log2(pos + 1)
        end)

      if idcg == 0.0, do: 0.0, else: dcg / idcg
    end
  end

  defp safe_avg(results, key) do
    if results == [] do
      0.0
    else
      total = Enum.reduce(results, 0.0, fn r, acc -> acc + Map.get(r, key, 0.0) end)
      total / length(results)
    end
  end

  defp load_gold_standard do
    path =
      cond do
        File.exists?(@gold_path) -> @gold_path
        true ->
          priv = :code.priv_dir(:brain) |> to_string()
          Path.join([priv, "evaluation", "memory_retrieval_gold.json"])
      end

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"entries" => entries}} when is_list(entries) -> entries
          _ -> raise "Invalid gold standard format at #{path}"
        end

      {:error, reason} ->
        raise "Cannot load gold standard at #{path}: #{inspect(reason)}"
    end
  end

  defp save_results(mode, blend_weight, mrr, recall, ndcg, query_count) do
    results_dir =
      cond do
        File.dir?("apps/brain/priv/evaluation/results") ->
          "apps/brain/priv/evaluation/results"

        true ->
          priv = :code.priv_dir(:brain) |> to_string()
          Path.join([priv, "evaluation", "results"])
      end

    File.mkdir_p!(results_dir)
    ts = DateTime.utc_now() |> DateTime.to_iso8601()

    result = %{
      timestamp: ts,
      mode: to_string(mode),
      blend_weight: blend_weight,
      k: @k,
      query_count: query_count,
      mrr_at_k: Float.round(mrr, 4),
      recall_at_k: Float.round(recall, 4),
      ndcg_at_k: Float.round(ndcg, 4)
    }

    filename = "memory_retrieval_#{mode}_#{System.system_time(:second)}.json"
    path = Path.join(results_dir, filename)
    File.write!(path, Jason.encode!(result, pretty: true))
    Mix.shell().info("Results saved to #{path}")
  end
end
