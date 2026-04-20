defmodule Brain.LinguisticData do
  @moduledoc """
  Provides linguistic utility data loaded from priv/knowledge/linguistic.json.

  This module provides access to linguistic utility patterns such as negation
  words, intensifiers, and hedges. These are structural linguistic features,
  NOT classifiers.

  For sentiment classification, use `Brain.ML.SentimentClassifierSimple`.
  For speech act classification, use `Brain.Analysis.SpeechActClassifier`.

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
  Returns true if the given text contains at least one negation token.

  Token-level matching (case-insensitive) — avoids substring false positives
  like `"knot"` matching `"not"`.

  ## Examples

      iex> Brain.LinguisticData.has_negation?("I do not like rain")
      true

      iex> Brain.LinguisticData.has_negation?("I like rain")
      false
  """
  def has_negation?(text) when is_binary(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/, trim: true)
    |> Enum.any?(&(&1 in negation_words()))
  end

  def has_negation?(_), do: false
end
