defmodule Brain.Knowledge.ContradictionDetector do
  @moduledoc """
  Centralized contradiction detection using Tokenizer-based matching.

  Provides `has_negation_difference?/2` and `has_number_disagreement?/2`
  that were previously duplicated across learning_center, corroborator,
  academic_validator, and integration modules.

  Uses `Brain.ML.Tokenizer.tokenize_words/1` instead of `String.contains?`
  for NLP text analysis (per project rules).
  """

  alias Brain.ML.Tokenizer

  # Negation words that appear after contraction expansion and tokenization
  @negation_words ~w(not never no neither nor hardly barely scarcely cannot)

  @doc """
  Detects whether two text strings differ by a negation.

  Returns `true` if exactly one of the two texts contains negation words
  and the other does not, suggesting a contradictory relationship.

  ## Examples

      iex> ContradictionDetector.has_negation_difference?("Water is a liquid", "Water is not a liquid")
      true

      iex> ContradictionDetector.has_negation_difference?("I like cats", "I like cats")
      false
  """
  @spec has_negation_difference?(String.t(), String.t()) :: boolean()
  def has_negation_difference?(text1, text2)
      when is_binary(text1) and is_binary(text2) do
    tokens1 = text1 |> Tokenizer.expand_contractions() |> Tokenizer.tokenize_normalized(expand_contractions: false) |> MapSet.new()
    tokens2 = text2 |> Tokenizer.expand_contractions() |> Tokenizer.tokenize_normalized(expand_contractions: false) |> MapSet.new()

    negation_set = MapSet.new(@negation_words)

    c1_has_negation = not MapSet.disjoint?(tokens1, negation_set)
    c2_has_negation = not MapSet.disjoint?(tokens2, negation_set)

    c1_has_negation != c2_has_negation
  end

  def has_negation_difference?(_, _), do: false

  @doc """
  Detects whether two text strings contain numbers that disagree significantly.

  Extracts numeric tokens from both texts and checks if corresponding numbers
  differ by more than 20%. Uses Tokenizer for token extraction.

  ## Examples

      iex> ContradictionDetector.has_number_disagreement?("Population is 1000", "Population is 2000")
      true

      iex> ContradictionDetector.has_number_disagreement?("Score is 95", "Score is 98")
      false
  """
  @spec has_number_disagreement?(String.t(), String.t()) :: boolean()
  def has_number_disagreement?(text1, text2)
      when is_binary(text1) and is_binary(text2) do
    numbers1 = extract_numbers(text1)
    numbers2 = extract_numbers(text2)

    if numbers1 != [] and numbers2 != [] do
      Enum.any?(Enum.zip(numbers1, numbers2), fn {n1, n2} ->
        min_val = min(n1, n2)
        max_val = max(n1, n2)
        min_val > 0 and (max_val - min_val) / min_val > 0.2
      end)
    else
      false
    end
  end

  def has_number_disagreement?(_, _), do: false

  @doc """
  Checks whether two texts contradict each other.

  Combines negation difference and number disagreement detection.
  Also checks for "is" / "is not" opposition patterns using tokens.
  """
  @spec contradicts?(String.t(), String.t()) :: boolean()
  def contradicts?(text1, text2)
      when is_binary(text1) and is_binary(text2) do
    has_negation_difference?(text1, text2) or
      has_number_disagreement?(text1, text2) or
      has_opposite_meaning?(text1, text2)
  end

  def contradicts?(_, _), do: false

  # --- Private ---

  defp extract_numbers(text) do
    text
    |> Tokenizer.tokenize()
    |> Enum.filter(fn token -> token.type == :number end)
    |> Enum.map(fn token -> parse_number(token.text) end)
    |> Enum.reject(&is_nil/1)
  end

  defp parse_number(str) do
    cleaned = String.replace(str, ",", "")

    case Float.parse(cleaned) do
      {num, _} -> num
      :error -> nil
    end
  end

  defp has_opposite_meaning?(text1, text2) do
    tokens1 = text1 |> Tokenizer.expand_contractions() |> Tokenizer.tokenize_normalized(expand_contractions: false)
    tokens2 = text2 |> Tokenizer.expand_contractions() |> Tokenizer.tokenize_normalized(expand_contractions: false)

    has_is_not_pattern?(tokens1, tokens2) or has_is_not_pattern?(tokens2, tokens1)
  end

  defp has_is_not_pattern?(tokens1, tokens2) do
    # Check if tokens1 has "is" without "not" and tokens2 has "is" followed by "not"
    has_is_1 = "is" in tokens1
    has_not_1 = "not" in tokens1
    has_is_2 = "is" in tokens2
    has_not_2 = "not" in tokens2

    has_is_1 and not has_not_1 and has_is_2 and has_not_2
  end
end
