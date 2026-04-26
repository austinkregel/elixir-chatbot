defmodule Brain.Analysis.FeatureExtractor do
  @moduledoc """
  Orchestrates per-word and per-chunk feature extraction.

  Converts a ChunkAnalysis into a 326-dimension feature vector by:
  1. Running per-word feature extraction (POS + lexicon lookup + WSD)
  2. Aggregating word features into chunk-level groups
  3. Combining with existing analysis signals (speech act, discourse, etc.)

  The resulting vector is stored on ChunkAnalysis as `feature_vector`.
  """

  alias Brain.Analysis.FeatureExtractor.{WordFeatures, ChunkFeatures}

  @doc """
  Extracts the full feature vector for a ChunkAnalysis.

  Returns `{feature_vector, word_features}` where:
  - `feature_vector` is a flat list of floats (326 dims, see ChunkFeatures.vector_dimension/0)
  - `word_features` is the list of per-word feature maps
  """
  def extract(%{pos_tags: pos_tags} = analysis) do
    word_feats = WordFeatures.extract(pos_tags || [])
    chunk_vector = ChunkFeatures.extract(analysis, word_feats)
    {chunk_vector, word_feats}
  end

  def extract(%{} = analysis) do
    word_feats = WordFeatures.extract([])
    chunk_vector = ChunkFeatures.extract(analysis, word_feats)
    {chunk_vector, word_feats}
  end

  @doc """
  Extracts only the chunk-level feature vector (discards per-word details).
  """
  def extract_vector(analysis) do
    {vector, _word_feats} = extract(analysis)
    vector
  end

  @doc """
  Extracts features for multiple analyses in parallel.

  Returns a list of `{feature_vector, word_features}` tuples.
  """
  def extract_batch(analyses) when is_list(analyses) do
    analyses
    |> Task.async_stream(&extract/1, max_concurrency: System.schedulers_online(), timeout: 5_000)
    |> Enum.map(fn
      {:ok, result} -> result
      {:exit, _reason} -> {[], []}
    end)
  end

  @doc """
  Returns the total dimension count for chunk feature vectors.
  """
  def vector_dimension, do: ChunkFeatures.vector_dimension()

  @doc """
  Computes cosine similarity between two feature vectors.
  """
  def similarity(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    FourthWall.Math.cosine_similarity(vec1, vec2)
  rescue
    _ ->
      dot = Enum.zip(vec1, vec2) |> Enum.reduce(0.0, fn {a, b}, acc -> acc + a * b end)
      mag1 = :math.sqrt(Enum.reduce(vec1, 0.0, fn x, acc -> acc + x * x end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0.0, fn x, acc -> acc + x * x end))
      if mag1 == 0.0 or mag2 == 0.0, do: 0.0, else: dot / (mag1 * mag2)
  end
end
