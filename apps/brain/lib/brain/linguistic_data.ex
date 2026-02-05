defmodule Brain.LinguisticData do
  @moduledoc """
  Provides linguistic data loaded from priv/knowledge/linguistic.json.

  This module centralizes access to commonly used linguistic patterns like
  negation words and sentiment lexicons, avoiding hardcoded lists scattered
  across the codebase.

  Data is loaded at compile time via @external_resource for efficiency.
  """

  @linguistic_path "priv/knowledge/linguistic.json"
  @external_resource @linguistic_path

  @data (case File.read(@linguistic_path) do
           {:ok, content} ->
             case Jason.decode(content) do
               {:ok, data} -> data
               {:error, _} -> %{}
             end

           {:error, _} ->
             %{}
         end)

  @doc """
  Returns the list of negation words.

  ## Examples

      iex> Brain.LinguisticData.negation_words()
      ["not", "no", "never", ...]
  """
  def negation_words do
    Map.get(@data, "negation_words", [
      "not", "no", "never", "none", "cannot", "can't", "won't", "don't"
    ])
  end

  @doc """
  Returns the list of positive sentiment words.

  ## Examples

      iex> Brain.LinguisticData.positive_words()
      ["good", "great", "love", ...]
  """
  def positive_words do
    get_in(@data, ["sentiment", "positive"]) || [
      "good", "great", "love", "happy", "thanks", "awesome", "nice", "excellent"
    ]
  end

  @doc """
  Returns the list of negative sentiment words.

  ## Examples

      iex> Brain.LinguisticData.negative_words()
      ["bad", "hate", "terrible", ...]
  """
  def negative_words do
    get_in(@data, ["sentiment", "negative"]) || [
      "bad", "hate", "terrible", "awful", "annoying", "frustrating", "worst"
    ]
  end

  @doc """
  Returns the list of intensifier words.

  ## Examples

      iex> Brain.LinguisticData.intensifiers()
      ["very", "really", "extremely", ...]
  """
  def intensifiers do
    Map.get(@data, "intensifiers", [
      "very", "really", "extremely", "quite", "absolutely"
    ])
  end

  @doc """
  Returns the list of hedge words.

  ## Examples

      iex> Brain.LinguisticData.hedges()
      ["maybe", "perhaps", "possibly", ...]
  """
  def hedges do
    Map.get(@data, "hedges", [
      "maybe", "perhaps", "possibly", "probably", "might"
    ])
  end

  @doc """
  Checks if a word is a negation word.

  ## Examples

      iex> Brain.LinguisticData.negation?("not")
      true

      iex> Brain.LinguisticData.negation?("happy")
      false
  """
  def negation?(word) when is_binary(word) do
    String.downcase(word) in negation_words()
  end

  @doc """
  Checks if a word has positive sentiment.
  """
  def positive?(word) when is_binary(word) do
    String.downcase(word) in positive_words()
  end

  @doc """
  Checks if a word has negative sentiment.
  """
  def negative?(word) when is_binary(word) do
    String.downcase(word) in negative_words()
  end
end
