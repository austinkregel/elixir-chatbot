defmodule Brain.EmbeddingCaches do
  @moduledoc """
  The caches that hold vectors embedded with a world's vocabulary
  (`Brain.Worlds.Embedder`), built the first time the world uses them:

  - `Brain.Response.SemanticFactRetriever`: the world's fact index
  - `Brain.Response.ChunkSegmenter`: the world's chunk type centroids
  - `Brain.Response.TemplateBlender`: the world's chunk pool
  - `Brain.Response.TemplateStore`: the world's template embeddings

  A vector embedded with a vocabulary has that vocabulary's length, and means
  nothing beside one embedded with another, so a world's caches are forgotten
  (`forget_world/1`) when its vocabulary is rebuilt and when the world is
  destroyed, and built again with the world's vocabulary on their next use.

  `Brain.ML.KnowledgeGraph.EntityVectorCache` is not one of them: its vectors
  come from the world's triple scorer, not its vocabulary, and it forgets a
  world when the world's scorer is seeded or dropped.
  """

  alias Brain.Response.{ChunkSegmenter, SemanticFactRetriever, TemplateBlender, TemplateStore}

  @doc """
  Forgets every cache's vectors for the world. `Brain.Worlds.Embedder` calls it
  when a rebuild leaves the world's vocabulary ready, and `Brain.Worlds.stop!/1`
  when the world's tree stops.
  """
  @spec forget_world(String.t()) :: :ok
  def forget_world(world_id) when is_binary(world_id) and world_id != "" do
    :ok = SemanticFactRetriever.forget_world(world_id)
    :ok = ChunkSegmenter.forget_world(world_id)
    :ok = TemplateBlender.forget_world(world_id)
    :ok = TemplateStore.forget_world(world_id)
  end
end
