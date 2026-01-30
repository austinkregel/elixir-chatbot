defmodule ChatBot.Testing.TaskSolver do
  @moduledoc """
  Solves NLP benchmark tasks using our existing analysis capabilities.

  This module provides "third-person" analysis mode - instead of responding
  to a user, we analyze text to produce structured outputs like:
  - Textual entailment (premise/hypothesis relationship)
  - Summarization (key facts from dialogue)
  - Classification (categorize text)

  ## Approach

  Rather than using an LLM for generation, we leverage:
  - Entity extraction and comparison
  - Semantic similarity via token overlap
  - Pattern-based reasoning
  - Template-based response construction
  """

  alias ChatBot.ML.Tokenizer

  # ============================================================================
  # Textual Entailment / NLI
  # ============================================================================

  @doc """
  Determines the relationship between premise and hypothesis.

  Returns: "positive" | "negated" | "neutral"
  """
  @spec solve_entailment(String.t(), String.t()) :: String.t()
  def solve_entailment(premise, hypothesis) do
    # Tokenize and normalize
    premise_tokens = extract_content_tokens(premise)
    hypothesis_tokens = extract_content_tokens(hypothesis)

    # Extract numbers
    premise_numbers = extract_numbers(premise)
    hypothesis_numbers = extract_numbers(hypothesis)

    # Extract key nouns (the main subject nouns)
    premise_nouns = extract_subject_nouns(premise)
    hypothesis_nouns = extract_subject_nouns(hypothesis)

    # Extract proper names
    premise_names = extract_proper_names(premise)
    hypothesis_names = extract_proper_names(hypothesis)

    # Check for negation
    premise_negated = has_negation?(premise)
    _hypothesis_negated = has_negation?(hypothesis)

    # Calculate overlaps
    token_overlap = calculate_overlap(premise_tokens, hypothesis_tokens)
    noun_overlap = calculate_overlap(premise_nouns, hypothesis_nouns)
    name_overlap = calculate_overlap(premise_names, hypothesis_names)
    number_match = numbers_match?(premise_numbers, hypothesis_numbers)

    # Check for key noun substitution (projectors vs candles)
    has_different_subject = length(hypothesis_nouns) > 0 and
                            length(premise_nouns) > 0 and
                            noun_overlap < 0.5

    # Decision logic - order matters!
    cond do
      # Different subject nouns with same structure = neutral
      has_different_subject and number_match and name_overlap > 0.5 ->
        "neutral"

      # Same subject, same numbers, no negation = positive
      noun_overlap >= 0.5 and number_match and not premise_negated ->
        "positive"

      # Same structure but different numbers = negated
      noun_overlap >= 0.5 and not number_match ->
        "negated"

      # Numbers differ significantly = negated
      not number_match and length(premise_numbers) > 0 and length(hypothesis_numbers) > 0 ->
        "negated"

      # High token overlap, matching numbers = positive
      token_overlap > 0.5 and number_match ->
        "positive"

      # Low overall overlap = neutral
      token_overlap < 0.3 ->
        "neutral"

      # Default to neutral when uncertain
      true ->
        "neutral"
    end
  end

  defp extract_subject_nouns(text) do
    # Extract plural nouns that are likely subjects
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.filter(fn word ->
      # Look for plural nouns (ending in s, es) that aren't stopwords
      String.length(word) > 3 and
      (String.ends_with?(word, "s") or String.ends_with?(word, "es")) and
      word not in ~w(is was has does this these those always sometimes)
    end)
    |> Enum.take(3)
  end

  defp extract_proper_names(text) do
    # Extract capitalized words (proper nouns)
    Regex.scan(~r/\b([A-Z][a-z]+)\b/, text)
    |> Enum.map(fn [_, name] -> String.downcase(name) end)
    |> Enum.reject(&(&1 in ~w(all the there if had have has)))
  end

  @doc """
  Parses NLI input format into premise and hypothesis.
  """
  @spec parse_nli_input(String.t()) :: {String.t(), String.t()}
  def parse_nli_input(input) do
    # Format: "Premise : 'X','Hypothesis : Y'" or similar
    premise = extract_field(input, "Premise")
    hypothesis = extract_field(input, "Hypothesis")
    {premise, hypothesis}
  end

  # ============================================================================
  # Summarization
  # ============================================================================

  @doc """
  Generates a summary from dialogue/conversation text.

  Uses the full summarization pipeline for proper event extraction,
  pronoun resolution, and template-based generation.
  """
  @spec summarize_dialogue(String.t()) :: String.t()
  def summarize_dialogue(dialogue) do
    # Use the new pipeline for summarization
    alias ChatBot.Summarization.Pipeline

    Pipeline.summarize_text(dialogue, max_facts: 3)
  end

  # Legacy summarization functions removed - now using ChatBot.Summarization.Pipeline

  # ============================================================================
  # Classification
  # ============================================================================

  @doc """
  Classifies text into categories based on content analysis.
  """
  @spec classify_text(String.t(), [String.t()]) :: String.t()
  def classify_text(text, possible_labels) do
    text_lower = String.downcase(text)
    tokens = extract_content_tokens(text)

    # Score each label
    scores = Enum.map(possible_labels, fn label ->
      label_lower = String.downcase(label)
      label_tokens = Tokenizer.tokenize(label_lower)
                     |> Enum.map(fn t -> if is_map(t), do: t.text, else: t end)

      # Direct mention score
      mention_score = if String.contains?(text_lower, label_lower), do: 1.0, else: 0.0

      # Token overlap
      overlap = calculate_overlap(tokens, label_tokens)

      # Semantic hints
      semantic_score = semantic_label_match(text_lower, label)

      total = mention_score * 0.4 + overlap * 0.3 + semantic_score * 0.3
      {label, total}
    end)

    {best_label, _score} = Enum.max_by(scores, fn {_, s} -> s end)
    best_label
  end

  defp semantic_label_match(text, label) do
    # Keyword associations for common labels
    associations = %{
      "positive" => ~w(good great love happy wonderful amazing),
      "negative" => ~w(bad terrible hate sad awful horrible),
      "neutral" => ~w(okay fine normal average),
      "entailment" => ~w(therefore because so thus hence),
      "contradiction" => ~w(but however although despite yet),
      "yes" => ~w(agree correct true right),
      "no" => ~w(disagree wrong false incorrect)
    }

    keywords = Map.get(associations, String.downcase(label), [])
    matches = Enum.count(keywords, &String.contains?(text, &1))
    min(matches / max(length(keywords), 1), 1.0)
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp extract_content_tokens(text) do
    text
    |> String.downcase()
    |> Tokenizer.tokenize()
    |> Enum.map(fn t -> if is_map(t), do: t.text, else: t end)
    |> Enum.reject(&(&1 in stopwords()))
  end

  defp extract_numbers(text) do
    # Match spelled-out and numeric numbers
    spelled = %{
      "one" => 1, "two" => 2, "three" => 3, "four" => 4, "five" => 5,
      "six" => 6, "seven" => 7, "eight" => 8, "nine" => 9, "ten" => 10,
      "eleven" => 11, "twelve" => 12
    }

    # Find numeric
    numeric = Regex.scan(~r/\b(\d+)\b/, text)
              |> Enum.map(fn [_, n] -> String.to_integer(n) end)

    # Find spelled out
    text_lower = String.downcase(text)
    spelled_nums = spelled
                   |> Enum.filter(fn {word, _} -> String.contains?(text_lower, word) end)
                   |> Enum.map(fn {_, num} -> num end)

    Enum.uniq(numeric ++ spelled_nums)
  end

  defp has_negation?(text) do
    negation_words = ~w(not n't no never neither none nobody nothing nowhere)
    text_lower = String.downcase(text)
    Enum.any?(negation_words, &String.contains?(text_lower, &1))
  end

  defp calculate_overlap(list1, list2) when is_list(list1) and is_list(list2) do
    set1 = MapSet.new(list1)
    set2 = MapSet.new(list2)
    intersection = MapSet.intersection(set1, set2) |> MapSet.size()
    union = MapSet.union(set1, set2) |> MapSet.size()

    if union == 0, do: 0.0, else: intersection / union
  end

  defp numbers_match?(nums1, nums2) do
    case {nums1, nums2} do
      {[], _} -> true
      {_, []} -> true
      _ ->
        set1 = MapSet.new(nums1)
        set2 = MapSet.new(nums2)
        not MapSet.disjoint?(set1, set2)
    end
  end

  defp extract_field(input, field_name) do
    # The format is: Premise : 'text here.','Hypothesis : text here.'
    # Contractions like "don't" contain apostrophes, so we need smarter parsing

    # First, try to split on the field boundaries
    case Regex.run(~r/#{field_name}\s*:\s*'(.+?)(?:'(?:,|$|\s*,\s*'Hypothesis))/is, input) do
      [_, value] -> String.trim(value)
      _ ->
        # Fallback: try double quotes
        case Regex.run(~r/#{field_name}\s*:\s*"(.+?)"/is, input) do
          [_, value] -> String.trim(value)
          _ ->
            # Last resort: everything after the colon until comma or end
            case Regex.run(~r/#{field_name}\s*:\s*'?([^,]+)/i, input) do
              [_, value] -> String.trim(value) |> String.trim_trailing("'")
              _ -> ""
            end
        end
    end
  end

  defp stopwords do
    ~w(a an the is are was were be been being have has had do does did
       will would could should may might must shall can to of in for on
       with at by from as into through during before after above below
       between under again further then once here there when where why
       how all each few more most other some such no nor not only own
       same so than too very just don now that this these those)
  end
end
