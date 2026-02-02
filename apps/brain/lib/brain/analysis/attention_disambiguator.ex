defmodule Brain.Analysis.AttentionDisambiguator do
  @moduledoc """
  Uses attention to resolve ambiguous entities by focusing on
  contextual cues in the surrounding text.
  
  For example, in "I'm Austin", attention should focus on "I'm"
  which indicates introduction (person), not location query.
  
  Uses the seq2seq encoder to encode entity and context, then
  computes attention weights to determine which type is most likely.
  
  This module follows the project's data-driven approach:
  - Uses World.TypeInferrer for learned type patterns
  - Uses IntentRegistry for intent-driven scoring
  - Does NOT use hardcoded word lists
  """
  
  require Logger
  
  alias Brain.ML.Seq2Seq
  alias Brain.Analysis.IntentRegistry
  alias World.TypeInferrer
  
  @doc """
  Disambiguate an entity using attention over context.
  
  ## Parameters
  - `entity`: Entity map with at least `:value` field
  - `context_tokens`: List of tokens or POS-tagged tokens from surrounding context
  - `possible_types`: List of possible entity type maps
  
  ## Options
  - `:world_id` - World ID for world-scoped models (default: "default")
  - `:intent` - Current intent for domain-aware scoring (optional)
  
  ## Returns
  Selected entity type map (the most likely type based on attention).
  """
  def disambiguate(entity, context_tokens, possible_types, opts \\ []) when is_list(possible_types) do
    if length(possible_types) <= 1 do
      # No disambiguation needed
      List.first(possible_types) || entity
    else
      if not ready?(opts) do
        Logger.debug("AttentionDisambiguator not ready, returning first type")
        List.first(possible_types)
      else
        world_id = Keyword.get(opts, :world_id, "default")
        intent = Keyword.get(opts, :intent, "")
        entity_value = extract_entity_value(entity)
        context_text = build_context_text(context_tokens)
        
        # Get expected entity types from IntentRegistry for domain-aware scoring
        expected_types = IntentRegistry.expected_entity_types(intent)
        
        # Encode entity and context
        entity_encoding = encode_text(entity_value, world_id)
        context_encoding = encode_text(context_text, world_id)
        
        if entity_encoding and context_encoding do
          # Compute attention weights over context
          attention_weights = compute_attention_weights(entity_encoding, context_encoding)
          
          # Score each type based on attention, learned patterns, and intent expectations
          type_scores = 
            Enum.map(possible_types, fn type_info ->
              type_name = extract_type_name(type_info)
              base_score = score_type_from_attention(attention_weights, context_tokens, type_name, world_id)
              
              # Boost score if type is expected by the current intent
              intent_boost = if type_name in expected_types, do: 0.2, else: 0.0
              
              {type_info, base_score + intent_boost}
            end)
          
          # Select highest scoring type
          {best_type, _score} = Enum.max_by(type_scores, fn {_, score} -> score end)
          best_type
        else
          # Fallback: return first type
          List.first(possible_types)
        end
      end
    end
  end
  
  @doc """
  Check if the attention disambiguator is ready.
  """
  def ready?(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, "default")
    Seq2Seq.ready?(world_id: world_id)
  end
  
  # ============================================================================
  # Private Functions
  # ============================================================================
  
  defp extract_entity_value(entity) when is_map(entity) do
    Map.get(entity, :value) || Map.get(entity, "value") || ""
  end
  defp extract_entity_value(_), do: ""
  
  defp extract_type_name(type_info) when is_map(type_info) do
    Map.get(type_info, :entity_type) || 
    Map.get(type_info, "entity_type") ||
    Map.get(type_info, :type) ||
    Map.get(type_info, "type") ||
    "unknown"
  end
  defp extract_type_name(_), do: "unknown"
  
  defp build_context_text(context_tokens) do
    # Extract text from tokens (handle both plain tokens and POS-tagged)
    tokens = 
      Enum.map(context_tokens, fn
        {token, _tag} -> token
        token when is_binary(token) -> token
        %{text: text} -> text
        _ -> ""
      end)
      |> Enum.reject(&(&1 == ""))
    
    Enum.join(tokens, " ")
  end
  
  defp encode_text(text, _world_id) do
    alias Brain.ML.Seq2Seq.Vocabulary
    
    case Vocabulary.encode(text) do
      {:ok, indices} ->
        Nx.tensor([indices])
      
      {:error, _} ->
        nil
    end
  end
  
  defp compute_attention_weights(entity_encoding, context_encoding) do
    # Simplified attention computation
    # In full implementation, would use the actual attention mechanism
    
    # Compute similarity between entity and each context position
    entity_vec = Nx.mean(entity_encoding, axes: [1])
    
    # For each position in context
    context_len = Nx.axis_size(context_encoding, 1)
    
    weights = 
      for i <- 0..(context_len - 1) do
        context_pos = Nx.slice(context_encoding, [0, i], [1, 1])
        context_vec = Nx.squeeze(context_pos, axes: [0, 1])
        
        # Cosine similarity
        dot = Nx.sum(Nx.multiply(entity_vec, context_vec))
        entity_norm = Nx.sqrt(Nx.sum(Nx.multiply(entity_vec, entity_vec)))
        context_norm = Nx.sqrt(Nx.sum(Nx.multiply(context_vec, context_vec)))
        
        similarity = 
          if entity_norm > 0 and context_norm > 0 do
            Nx.divide(dot, Nx.multiply(entity_norm, context_norm))
            |> Nx.to_number()
          else
            0.0
          end
        
        similarity
      end
    
    # Normalize to attention weights (softmax)
    max_weight = Enum.max(weights)
    exp_weights = Enum.map(weights, fn w -> :math.exp(w - max_weight) end)
    sum_exp = Enum.sum(exp_weights)
    
    if sum_exp > 0 do
      Enum.map(exp_weights, fn w -> w / sum_exp end)
    else
      # Uniform distribution
      List.duplicate(1.0 / length(weights), length(weights))
    end
  end
  
  defp score_type_from_attention(attention_weights, context_tokens, type_name, world_id) do
    # Use data-driven approach: TypeInferrer for learned patterns, IntentRegistry for domains
    # NO hardcoded word lists - this follows the project's core design principle
    
    # Extract tokens and POS tags from context
    {context_words, context_tags} = extract_tokens_and_tags(context_tokens)
    
    # 1. Use TypeInferrer to get learned type inference from context patterns
    {inferred_type, type_confidence} = 
      TypeInferrer.infer_type("", context_words, context_tags, world_id)
    
    # 2. Score based on whether inferred type matches the candidate type
    type_match_score = 
      cond do
        # Direct match with inferred type
        inferred_type == type_name -> 
          type_confidence
        
        # Related types (location/city are related)
        inferred_type in ["location", "city"] and type_name in ["location", "city"] ->
          type_confidence * 0.9
        
        # Person/name are related
        inferred_type in ["person", "name"] and type_name in ["person", "name"] ->
          type_confidence * 0.9
        
        # Unknown type from inferrer - use neutral score
        inferred_type == "unknown" ->
          0.5
        
        # No match - low score
        true ->
          0.2
      end
    
    # 3. Weight by attention strength (how confident is the attention)
    avg_attention = 
      if length(attention_weights) > 0 do
        Enum.sum(attention_weights) / length(attention_weights)
      else
        0.5
      end
    
    # Combine type match score with attention weight
    type_match_score * (0.5 + avg_attention * 0.5)
  end
  
  # Extract tokens and POS tags from context_tokens list
  # Handles various formats: {token, tag}, plain strings, maps with :text
  defp extract_tokens_and_tags(context_tokens) do
    result = 
      Enum.map(context_tokens, fn
        {token, tag} when is_binary(token) and is_binary(tag) -> 
          {token, tag}
        
        {token, tag} when is_binary(token) -> 
          {token, to_string(tag)}
        
        token when is_binary(token) -> 
          {token, "X"}  # Unknown POS tag
        
        %{text: text, tag: tag} -> 
          {text, tag}
        
        %{text: text} -> 
          {text, "X"}
        
        _ -> 
          nil
      end)
      |> Enum.reject(&is_nil/1)
    
    tokens = Enum.map(result, fn {token, _tag} -> token end)
    tags = Enum.map(result, fn {_token, tag} -> tag end)
    
    {tokens, tags}
  end
end
