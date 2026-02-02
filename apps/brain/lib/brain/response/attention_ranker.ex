defmodule Brain.Response.AttentionRanker do
  @moduledoc """
  Ranks response candidates using attention-based similarity.
  
  Given a query and multiple response candidates, computes attention
  scores to find which response best aligns with the query.
  
  Uses the seq2seq encoder to encode both query and candidates,
  then computes attention weights to determine relevance.
  """
  
  require Logger
  
  alias Brain.ML.Seq2Seq
  
  @doc """
  Rank response candidates by attention-based relevance to the query.
  
  ## Parameters
  - `query`: User's query text
  - `candidates`: List of candidate response maps with at least a `:text` field
  
  ## Options
  - `:world_id` - World ID for world-scoped models (default: "default")
  
  ## Returns
  List of candidates sorted by relevance (most relevant first).
  """
  def rank(query, candidates, opts \\ []) when is_binary(query) and is_list(candidates) do
    if not ready?(opts) do
      Logger.debug("AttentionRanker not ready, returning candidates unchanged")
      candidates
    else
      world_id = Keyword.get(opts, :world_id, "default")
      
      # Encode query
      query_encoding = encode_query(query, world_id)
      
      if query_encoding do
        # Score each candidate
        scored = 
          Enum.map(candidates, fn candidate ->
            candidate_text = extract_text(candidate)
            score = compute_relevance_score(query_encoding, candidate_text, world_id)
            {candidate, score}
          end)
        
        # Sort by score descending
        scored
        |> Enum.sort_by(fn {_, score} -> score end, :desc)
        |> Enum.map(fn {candidate, _} -> candidate end)
      else
        # Fallback: return unchanged
        candidates
      end
    end
  end
  
  @doc """
  Check if the attention ranker is ready.
  """
  def ready?(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, "default")
    Seq2Seq.ready?(world_id: world_id)
  end
  
  # ============================================================================
  # Private Functions
  # ============================================================================
  
  defp encode_query(query, world_id) do
    # Use seq2seq encoder to encode the query
    # For now, we'll use a simplified approach: encode via vocabulary
    # In a full implementation, would use the actual encoder model with world-scoped params
    
    alias Brain.ML.Seq2Seq.Vocabulary
    
    Logger.debug("Encoding query for world '#{world_id}': #{String.slice(query, 0, 50)}...")
    
    # TODO: Use world-scoped vocabulary when available
    # Currently uses global vocabulary; world_id is logged for traceability
    case Vocabulary.encode(query) do
      {:ok, indices} ->
        # Convert to tensor representation
        Nx.tensor([indices])
      
      {:error, reason} ->
        Logger.warning("Failed to encode query for world '#{world_id}': #{inspect(reason)}")
        nil
    end
  end
  
  defp extract_text(candidate) when is_binary(candidate), do: candidate
  defp extract_text(candidate) when is_map(candidate) do
    Map.get(candidate, :text) || 
    Map.get(candidate, "text") || 
    Map.get(candidate, :message) ||
    Map.get(candidate, "message") ||
    ""
  end
  defp extract_text(_), do: ""
  
  defp compute_relevance_score(query_encoding, candidate_text, world_id) do
    # Encode candidate using world-scoped vocabulary (when available)
    alias Brain.ML.Seq2Seq.Vocabulary
    
    # TODO: Use world-scoped vocabulary when available
    # Currently uses global vocabulary; world_id is used for logging
    case Vocabulary.encode(candidate_text) do
      {:ok, candidate_indices} ->
        candidate_encoding = Nx.tensor([candidate_indices])
        
        # Compute attention-based similarity
        # Uses cosine similarity between mean embeddings
        # In full implementation, would use actual attention mechanism with world-scoped model
        
        query_vec = Nx.mean(query_encoding, axes: [1])
        candidate_vec = Nx.mean(candidate_encoding, axes: [1])
        
        # Cosine similarity
        dot_product = Nx.sum(Nx.multiply(query_vec, candidate_vec))
        query_norm = Nx.sqrt(Nx.sum(Nx.multiply(query_vec, query_vec)))
        candidate_norm = Nx.sqrt(Nx.sum(Nx.multiply(candidate_vec, candidate_vec)))
        
        similarity = 
          if query_norm > 0 and candidate_norm > 0 do
            Nx.divide(dot_product, Nx.multiply(query_norm, candidate_norm))
            |> Nx.to_number()
          else
            0.0
          end
        
        Logger.debug("Relevance score for world '#{world_id}': #{Float.round(similarity, 4)}")
        similarity
      
      {:error, reason} ->
        Logger.debug("Failed to encode candidate for world '#{world_id}': #{inspect(reason)}")
        0.0
    end
  end
end
