defmodule Brain.Memory.Consolidation do
  @moduledoc """
  Consolidation logic for episodic memories.

  Ported from the Rust cognitive_memory_system consolidation module.

  Clusters similar episodes and creates aggregated semantic facts.
  Consolidation allows the system to distill knowledge from raw
  experiences into higher-level structures.
  """

  alias Brain.Memory.Types.SemanticFact
  alias Brain.Memory.{Store, VectorIndex}

  require Logger

  @doc """
  Consolidate similar episodes into semantic facts.

  Episodes with embeddings whose cosine similarity exceeds the threshold
  will be clustered together. For each cluster with at least min_cluster_size
  members, a new SemanticFact is created.

  Options:
  - threshold: minimum cosine similarity for clustering (default: 0.8)
  - min_cluster_size: minimum episodes per cluster (default: 2)

  Returns the number of new semantic facts created.
  """
  def consolidate(opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.8)
    min_cluster_size = Keyword.get(opts, :min_cluster_size, 2)

    Logger.info("Starting consolidation",
      threshold: threshold,
      min_cluster_size: min_cluster_size
    )

    # Get all episodes
    {:ok, episodes} = Store.all_episodes()

    if length(episodes) < min_cluster_size do
      Logger.info("Not enough episodes for consolidation", count: length(episodes))
      {:ok, 0}
    else
      # Find clusters of similar episodes
      clusters = find_clusters(episodes, threshold, min_cluster_size)

      Logger.info("Found clusters", count: length(clusters))

      # Create semantic facts from clusters
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

  @doc """
  Find clusters of similar episodes based on embedding similarity.
  Uses greedy clustering - assigns each episode to the first cluster
  it's similar enough to, or creates a new cluster.
  """
  def find_clusters(episodes, threshold, min_cluster_size) do
    # Build clusters greedily
    {clusters, _visited} =
      Enum.reduce(episodes, {[], MapSet.new()}, fn episode, {clusters_acc, visited} ->
        if MapSet.member?(visited, episode.id) do
          {clusters_acc, visited}
        else
          # Find all episodes similar to this one
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
            # Mark as visited but don't create cluster
            {clusters_acc, MapSet.put(visited, episode.id)}
          end
        end
      end)

    Enum.reverse(clusters)
  end

  @doc """
  Create a semantic fact from a cluster of episodes.
  """
  def create_semantic_from_cluster(cluster) when is_list(cluster) and length(cluster) > 0 do
    # Aggregate representation
    representation = aggregate_representation(cluster)

    # Compute mean embedding
    embeddings = Enum.map(cluster, & &1.embedding)
    mean_embedding = VectorIndex.mean_vector(embeddings)

    # Collect evidence IDs
    evidence_ids = Enum.map(cluster, & &1.id)

    # Merge tags
    tags =
      cluster
      |> Enum.flat_map(& &1.tags)
      |> Enum.uniq()

    # Create the semantic fact
    semantic = SemanticFact.new(representation, mean_embedding, evidence_ids, tags)

    # Add to store
    case Store.add_semantic(semantic) do
      {:ok, semantic_id} ->
        # Link episodes to this semantic fact
        Enum.each(evidence_ids, fn ep_id ->
          Store.link_episode_to_semantic(ep_id, semantic_id)
        end)

        {:ok, semantic_id}

      error ->
        error
    end
  end

  def create_semantic_from_cluster([]), do: {:error, :empty_cluster}

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp aggregate_representation(cluster) do
    # Create a summary representation from the cluster
    # For now, we use the most common action and aggregate tags

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
