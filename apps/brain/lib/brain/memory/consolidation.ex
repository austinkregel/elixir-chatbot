defmodule Brain.Memory.Consolidation do
  @moduledoc "Consolidation logic for episodic memories.\n\nPorted from the Rust cognitive_memory_system consolidation module.\n\nClusters similar episodes and creates aggregated semantic facts.\nConsolidation allows the system to distill knowledge from raw\nexperiences into higher-level structures.\n"

  alias Brain.Memory.Types.SemanticFact
  alias Brain.Memory.{Store, VectorIndex}

  require Logger

  @doc "Consolidate similar episodes into semantic facts.\n\nEpisodes with embeddings whose cosine similarity exceeds the threshold\nwill be clustered together. For each cluster with at least min_cluster_size\nmembers, a new SemanticFact is created.\n\nOptions:\n- threshold: minimum cosine similarity for clustering (default: 0.8)\n- min_cluster_size: minimum episodes per cluster (default: 2)\n\nReturns the number of new semantic facts created.\n"
  def consolidate(opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.8)
    min_cluster_size = Keyword.get(opts, :min_cluster_size, 2)
    world_id = Keyword.get(opts, :world_id)
    store_opts = if world_id, do: [world_id: world_id], else: []

    Logger.info("Starting consolidation",
      threshold: threshold,
      min_cluster_size: min_cluster_size,
      world_id: world_id
    )

    {:ok, episodes} = Store.all_episodes(store_opts)

    if length(episodes) < min_cluster_size do
      Logger.info("Not enough episodes for consolidation", count: length(episodes))
      {:ok, 0}
    else
      clusters = find_clusters(episodes, threshold, min_cluster_size)

      Logger.info("Found clusters", count: length(clusters))

      new_semantics =
        Enum.reduce(clusters, 0, fn cluster, acc ->
          case create_semantic_from_cluster(cluster) do
            {:ok, _semantic_id} -> acc + 1
            {:error, _} -> acc
          end
        end)

      Logger.info("Consolidation complete", new_semantics: new_semantics)
      {:ok, new_semantics}
    end
  end

  @doc "Find clusters of similar episodes based on embedding similarity.\nUses greedy clustering - assigns each episode to the first cluster\nit's similar enough to, or creates a new cluster.\n"
  def find_clusters(episodes, threshold, min_cluster_size) do
    {clusters, _visited} =
      Enum.reduce(episodes, {[], MapSet.new()}, fn episode, {clusters_acc, visited} ->
        if MapSet.member?(visited, episode.id) do
          {clusters_acc, visited}
        else
          cluster =
            Enum.filter(episodes, fn other ->
              not MapSet.member?(visited, other.id) and
                (other.id == episode.id or
                   VectorIndex.cosine_similarity(episode.embedding, other.embedding) >= threshold)
            end)

          if length(cluster) >= min_cluster_size do
            new_visited =
              Enum.reduce(cluster, visited, fn ep, vis ->
                MapSet.put(vis, ep.id)
              end)

            {[cluster | clusters_acc], new_visited}
          else
            {clusters_acc, MapSet.put(visited, episode.id)}
          end
        end
      end)

    Enum.reverse(clusters)
  end

  @doc "Create a semantic fact from a cluster of episodes.\n"
  def create_semantic_from_cluster(cluster) when is_list(cluster) and cluster != [] do
    representation = aggregate_representation(cluster)
    embeddings = Enum.map(cluster, & &1.embedding)
    mean_embedding = VectorIndex.mean_vector(embeddings)
    evidence_ids = Enum.map(cluster, & &1.id)

    tags =
      cluster
      |> Enum.flat_map(& &1.tags)
      |> Enum.uniq()

    semantic = SemanticFact.new(representation, mean_embedding, evidence_ids, tags)

    case Store.add_semantic(semantic) do
      {:ok, semantic_id} ->
        Enum.each(evidence_ids, fn ep_id ->
          Store.link_episode_to_semantic(ep_id, semantic_id)
        end)

        # Bridge to BeliefStore for epistemic integration
        Brain.Epistemic.ConsolidationBridge.bridge_semantic_fact(semantic)

        Brain.Graph.Writer.write_semantic_cluster(semantic, evidence_ids)

        {:ok, semantic_id}

      error ->
        error
    end
  end

  def create_semantic_from_cluster([]) do
    {:error, :empty_cluster}
  end

  defp aggregate_representation(cluster) do
    actions = Enum.frequencies(Enum.map(cluster, & &1.action))
    {most_common_action, _count} = Enum.max_by(actions, fn {_action, count} -> count end)

    states =
      cluster
      |> Enum.map(& &1.state)
      |> Enum.take(3)
      |> Enum.join(" | ")

    "Action: #{most_common_action} | Examples: #{states}"
  end
end
