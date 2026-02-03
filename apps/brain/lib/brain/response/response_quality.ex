defmodule Brain.Response.ResponseQuality do
  @moduledoc """
  Response quality analysis and improvement using heuristics and LSTM scoring.
  
  This module catches common response quality issues:
  
  1. **Irrelevant responses** - Response doesn't match query topic
  2. **Incomplete responses** - Missing expected information
  3. **Awkward transitions** - Poor flow between parts
  4. **Tone mismatch** - Formal response to casual query or vice versa
  5. **Repetition** - Redundant phrases
  
  ## Usage
  
      # Check response quality
      ResponseQuality.analyze("What's the weather?", "The weather is nice.")
      # => %{
      #   score: 0.85,
      #   issues: [],
      #   suggestions: []
      # }
      
      # Improve a response
      ResponseQuality.improve("What's the weather?", "I don't understand that.")
      # => {:ok, "I can help with weather! What location are you interested in?"}
  """
  
  require Logger
  
  alias Brain.ML.Tokenizer
  alias Brain.Response.{LSTMResponse, Generator, TemplateStore}
  
  @quality_thresholds %{
    excellent: 0.85,
    good: 0.70,
    acceptable: 0.50,
    poor: 0.30
  }
  
  # Common issues and their detection patterns
  @issue_patterns [
    {:too_short, &__MODULE__.check_too_short/2},
    {:too_long, &__MODULE__.check_too_long/2},
    {:topic_mismatch, &__MODULE__.check_topic_mismatch/2},
    {:repetition, &__MODULE__.check_repetition/2},
    {:awkward_start, &__MODULE__.check_awkward_start/2},
    {:missing_entity_reference, &__MODULE__.check_entity_reference/2},
    {:generic_fallback, &__MODULE__.check_generic_fallback/2}
  ]
  
  # ============================================================================
  # Public API
  # ============================================================================
  
  @doc """
  Analyze response quality and return issues/score.
  """
  def analyze(query, response, opts \\ []) do
    entities = Keyword.get(opts, :entities, [])
    intent = Keyword.get(opts, :intent)
    
    context = %{
      query: query,
      response: response,
      entities: entities,
      intent: intent,
      query_tokens: Tokenizer.tokenize_words(query),
      response_tokens: Tokenizer.tokenize_words(response)
    }
    
    # Run all issue checks
    issues = @issue_patterns
      |> Enum.map(fn {issue_type, checker} ->
        case checker.(context, opts) do
          nil -> nil
          issue -> {issue_type, issue}
        end
      end)
      |> Enum.reject(&is_nil/1)
    
    # Calculate base score
    base_score = calculate_base_score(context)
    
    # Apply penalties for issues
    penalty = Enum.reduce(issues, 0.0, fn {type, issue}, acc ->
      acc + get_penalty(type, issue)
    end)
    
    final_score = max(0.0, base_score - penalty)
    
    # Get LSTM score if available
    lstm_score = get_lstm_score(query, response)
    
    # Blend scores
    blended_score = if lstm_score do
      final_score * 0.4 + lstm_score * 0.6
    else
      final_score
    end
    
    quality_level = score_to_level(blended_score)
    
    %{
      score: Float.round(blended_score, 3),
      level: quality_level,
      issues: Enum.map(issues, fn {type, issue} -> Map.put(issue, :type, type) end),
      suggestions: generate_suggestions(issues, context),
      lstm_score: lstm_score,
      heuristic_score: Float.round(final_score, 3)
    }
  end
  
  @doc """
  Attempt to improve a response if quality is below threshold.
  """
  def improve(query, response, opts \\ []) do
    intent = Keyword.get(opts, :intent)
    entities = Keyword.get(opts, :entities, [])
    threshold = Keyword.get(opts, :threshold, @quality_thresholds.acceptable)
    
    analysis = analyze(query, response, opts)
    
    if analysis.score >= threshold do
      {:ok, response, analysis}
    else
      # Try to get a better response
      improved = attempt_improvement(query, response, intent, entities, analysis)
      new_analysis = analyze(query, improved, opts)
      
      if new_analysis.score > analysis.score do
        {:improved, improved, new_analysis}
      else
        {:ok, response, analysis}
      end
    end
  end
  
  @doc """
  Get multiple candidate responses and return the best one.
  """
  def select_best_response(query, intent, entities, opts \\ []) do
    num_candidates = Keyword.get(opts, :candidates, 5)
    
    # Generate candidates
    candidates = generate_diverse_candidates(query, intent, entities, num_candidates)
    
    if Enum.empty?(candidates) do
      {:error, :no_candidates}
    else
      # Score each
      scored = candidates
        |> Enum.map(fn response ->
          analysis = analyze(query, response, intent: intent, entities: entities)
          {response, analysis.score, analysis}
        end)
        |> Enum.sort_by(fn {_, score, _} -> score end, :desc)
      
      {best, score, analysis} = hd(scored)
      
      {:ok, best, %{
        score: score,
        analysis: analysis,
        candidates_considered: length(candidates),
        all_scores: Enum.map(scored, fn {r, s, _} -> {String.slice(r, 0, 50), s} end)
      }}
    end
  end
  
  @doc """
  Quick quality check - returns :ok, :warning, or :poor.
  """
  def quick_check(query, response) do
    analysis = analyze(query, response)
    
    cond do
      analysis.score >= @quality_thresholds.good -> :ok
      analysis.score >= @quality_thresholds.acceptable -> :warning
      true -> :poor
    end
  end
  
  # ============================================================================
  # Issue Checkers
  # ============================================================================
  
  def check_too_short(%{response: response}, _opts) do
    if String.length(response) < 10 do
      %{severity: :medium, message: "Response is very short", length: String.length(response)}
    else
      nil
    end
  end
  
  def check_too_long(%{response: response}, _opts) do
    if String.length(response) > 500 do
      %{severity: :low, message: "Response may be too verbose", length: String.length(response)}
    else
      nil
    end
  end
  
  def check_topic_mismatch(%{query_tokens: qt, response_tokens: rt}, _opts) do
    # Check if query and response share meaningful words
    # Handle both string tokens and token maps
    query_words = qt |> Enum.map(&extract_token_text/1) |> Enum.map(&String.downcase/1) |> MapSet.new()
    response_words = rt |> Enum.map(&extract_token_text/1) |> Enum.map(&String.downcase/1) |> MapSet.new()
    
    # Remove common words
    stopwords = MapSet.new(~w(i you the a an is are was were be been being have has had do does did will would could should can may might must shall the to of and in that it for on with as at by from))
    
    query_content = MapSet.difference(query_words, stopwords)
    response_content = MapSet.difference(response_words, stopwords)
    
    overlap = MapSet.intersection(query_content, response_content) |> MapSet.size()
    query_size = MapSet.size(query_content)
    
    if query_size > 2 and overlap == 0 do
      %{severity: :high, message: "Response doesn't reference query topic"}
    else
      nil
    end
  end
  
  def check_repetition(%{response_tokens: tokens}, _opts) do
    # Check for repeated phrases (3+ words)
    # First extract text from token maps if needed
    token_texts = Enum.map(tokens, &extract_token_text/1)
    trigrams = token_texts
      |> Enum.chunk_every(3, 1, :discard)
      |> Enum.map(&Enum.join(&1, " "))
    
    unique = Enum.uniq(trigrams)
    
    if length(trigrams) > 3 and length(unique) < length(trigrams) * 0.8 do
      %{severity: :medium, message: "Response contains repetitive phrases"}
    else
      nil
    end
  end
  
  def check_awkward_start(%{response: response}, _opts) do
    lower = String.downcase(response)
    
    awkward_starts = [
      "well,",
      "so,",
      "actually,",
      "basically,",
      "i mean,",
      "like,"
    ]
    
    if Enum.any?(awkward_starts, &String.starts_with?(lower, &1)) do
      %{severity: :low, message: "Response starts with filler word"}
    else
      nil
    end
  end
  
  def check_entity_reference(%{entities: entities, response_tokens: tokens}, _opts) do
    if length(entities) > 0 do
      response_lower = tokens |> Enum.map(&extract_token_text/1) |> Enum.map(&String.downcase/1) |> MapSet.new()
      
      entity_values = entities
        |> Enum.map(fn e -> 
          value = Map.get(e, :value) || Map.get(e, "value") || ""
          String.downcase(value)
        end)
        |> Enum.filter(&(String.length(&1) > 0))
      
      referenced = Enum.any?(entity_values, fn val ->
        MapSet.member?(response_lower, val) or
        Enum.any?(response_lower, &String.contains?(&1, val))
      end)
      
      if not referenced and length(entity_values) > 0 do
        %{severity: :medium, message: "Response doesn't reference extracted entities"}
      else
        nil
      end
    else
      nil
    end
  end
  
  def check_generic_fallback(%{response: response}, _opts) do
    fallback_indicators = [
      "i don't understand",
      "i'm not sure what you mean",
      "could you rephrase",
      "i didn't get that",
      "sorry, i can't help with that"
    ]
    
    lower = String.downcase(response)
    
    if Enum.any?(fallback_indicators, &String.contains?(lower, &1)) do
      %{severity: :high, message: "Response appears to be a fallback/error message"}
    else
      nil
    end
  end
  
  # ============================================================================
  # Private: Scoring
  # ============================================================================
  
  defp calculate_base_score(context) do
    # Start with baseline
    score = 0.7
    
    # Bonus for appropriate length
    len = String.length(context.response)
    score = if len >= 20 and len <= 200, do: score + 0.1, else: score
    
    # Bonus for ending with proper punctuation
    score = if String.ends_with?(context.response, [". ", "!", "?", "."]), do: score + 0.05, else: score
    
    # Bonus for topic relevance (shared tokens)
    qt = context.query_tokens |> Enum.map(&extract_token_text/1) |> Enum.map(&String.downcase/1) |> MapSet.new()
    rt = context.response_tokens |> Enum.map(&extract_token_text/1) |> Enum.map(&String.downcase/1) |> MapSet.new()
    overlap = MapSet.intersection(qt, rt) |> MapSet.size()
    score = score + min(0.15, overlap * 0.03)
    
    min(1.0, score)
  end
  
  defp get_penalty(issue_type, issue) do
    severity = Map.get(issue, :severity, :medium)
    
    base_penalty = case issue_type do
      :too_short -> 0.1
      :too_long -> 0.05
      :topic_mismatch -> 0.25
      :repetition -> 0.15
      :awkward_start -> 0.05
      :missing_entity_reference -> 0.10
      :generic_fallback -> 0.30
      _ -> 0.05
    end
    
    severity_multiplier = case severity do
      :high -> 1.5
      :medium -> 1.0
      :low -> 0.5
    end
    
    base_penalty * severity_multiplier
  end
  
  defp get_lstm_score(query, response) do
    if LSTMResponse.ready?() do
      case LSTMResponse.score_response(query, response) do
        {:ok, score} -> score
        _ -> nil
      end
    else
      nil
    end
  end
  
  defp score_to_level(score) do
    cond do
      score >= @quality_thresholds.excellent -> :excellent
      score >= @quality_thresholds.good -> :good
      score >= @quality_thresholds.acceptable -> :acceptable
      score >= @quality_thresholds.poor -> :poor
      true -> :very_poor
    end
  end
  
  # ============================================================================
  # Private: Suggestions
  # ============================================================================
  
  defp generate_suggestions(issues, _context) do
    issues
    |> Enum.map(fn {type, _issue} ->
      case type do
        :too_short -> "Consider providing more detail or context"
        :too_long -> "Consider being more concise"
        :topic_mismatch -> "Ensure response addresses the query topic"
        :repetition -> "Remove repeated phrases"
        :awkward_start -> "Start with a more direct opening"
        :missing_entity_reference -> "Reference the specific entities mentioned"
        :generic_fallback -> "Try to provide a more helpful response"
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end
  
  # ============================================================================
  # Private: Improvement
  # ============================================================================
  
  defp attempt_improvement(query, _original_response, intent, entities, analysis) do
    # Strategy based on issues found
    issue_types = Enum.map(analysis.issues, & &1.type)
    
    cond do
      :generic_fallback in issue_types ->
        # Try to get a real response
        get_alternative_response(query, intent, entities)
        
      :topic_mismatch in issue_types ->
        # Try template-based response
        get_template_response(intent, entities)
        
      :too_short in issue_types ->
        # Try to get a more detailed response
        get_detailed_response(query, intent, entities)
        
      true ->
        # Try LSTM-ranked selection
        select_lstm_best(query, intent, entities)
    end
  end
  
  defp get_alternative_response(query, intent, entities) do
    case Generator.generate(intent, entities, query) do
      {:ok, response, _} -> response
      _ -> get_template_response(intent, entities)
    end
  end
  
  defp get_template_response(intent, _entities) do
    case TemplateStore.get_random_template(intent) do
      {:ok, response} when is_binary(response) -> response
      {:ok, %{text: text}} -> text
      _ -> "I'm here to help. Could you tell me more about what you're looking for?"
    end
  end
  
  defp get_detailed_response(query, intent, entities) do
    case Generator.generate(intent, entities, query) do
      {:ok, response, _} when byte_size(response) > 20 -> response
      _ -> get_template_response(intent, entities)
    end
  end
  
  defp select_lstm_best(query, intent, entities) do
    if LSTMResponse.ready?() do
      case LSTMResponse.generate(query, intent, entities) do
        {:ok, response, _score} -> response
        _ -> get_template_response(intent, entities)
      end
    else
      get_template_response(intent, entities)
    end
  end
  
  # ============================================================================
  # Private: Candidate Generation
  # ============================================================================
  
  defp generate_diverse_candidates(_query, intent, entities, num_candidates) do
    candidates = []
    
    # Template responses
    template_candidates = case TemplateStore.get_templates(intent) do
      templates when is_list(templates) -> 
        templates
        |> Enum.map(fn t ->
          cond do
            is_binary(t) -> t
            is_map(t) -> Map.get(t, :text) || Map.get(t, "text") || ""
            true -> ""
          end
        end)
        |> Enum.filter(&(String.length(&1) > 0))
        |> Enum.take(num_candidates)
      _ -> []
    end
    
    # Generator response
    generator_candidate = case Generator.generate(intent, entities) do
      {:ok, response, _} -> [response]
      _ -> []
    end
    
    (template_candidates ++ generator_candidate ++ candidates)
    |> Enum.uniq()
    |> Enum.filter(&is_binary/1)
    |> Enum.filter(&(String.length(&1) > 0))
    |> Enum.take(num_candidates)
  end

  # Helper to extract text from token (handles both string and map formats)
  defp extract_token_text(token) when is_binary(token), do: token
  defp extract_token_text(%{text: text}) when is_binary(text), do: text
  defp extract_token_text(%{"text" => text}) when is_binary(text), do: text
  defp extract_token_text(_), do: ""
end
