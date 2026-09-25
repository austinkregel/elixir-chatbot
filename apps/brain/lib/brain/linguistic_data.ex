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

  alias Brain.Lexicon
  alias Brain.ML.Tokenizer

  @linguistic_path "priv/knowledge/linguistic.json"
  @external_resource @linguistic_path

  # This module is nothing but this file's contents; compiling it to %{} on a
  # read error would leave every accessor silently returning its default.
  @data @linguistic_path |> File.read!() |> Jason.decode!()

  @doc """
  Returns the list of negation words.

  ## Examples

      iex> Brain.LinguisticData.negation_words()
      ["not", "no", "never", ...]
  """
  def negation_words do
    Map.get(@data, "negation_words", [
      "not",
      "no",
      "never",
      "none",
      "cannot",
      "can't",
      "won't",
      "don't"
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
      "very",
      "really",
      "extremely",
      "quite",
      "absolutely"
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
      "maybe",
      "perhaps",
      "possibly",
      "probably",
      "might"
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

  Contractions are expanded before matching, because the vocabulary lists the
  negator itself (`"not"`), while a contraction carries it as a suffix. Splitting
  `"don't"` on non-word characters yields `["don", "t"]`, and neither is a
  negation word — so a naive split misses every contracted negation. Measured
  2026-09-12 before this was fixed: 55 of 206 negated sentences (26.7%) were
  reported as having no negation.

  Token-level matching (case-insensitive) still avoids substring false positives
  like `"knot"` matching `"not"`. This uses the same expansion and tokenisation
  path as `Brain.Knowledge.ContradictionDetector`, which was already correct.

  ## Examples

      iex> Brain.LinguisticData.has_negation?("I do not like rain")
      true

      iex> Brain.LinguisticData.has_negation?("I don't like rain")
      true

      iex> Brain.LinguisticData.has_negation?("I like rain")
      false

      iex> Brain.LinguisticData.has_negation?("She tied a knot in the rope")
      false
  """
  def has_negation?(text) when is_binary(text) do
    text
    |> Tokenizer.expand_contractions()
    |> Tokenizer.tokenize_normalized(expand_contractions: false)
    |> Enum.any?(&negator?/1)
  end

  def has_negation?(_), do: false

  @doc """
  Returns true if the word negates, from either of the two sources English uses.

  **Closed class** — `not`, `never`, `no`, `neither`, and the contractions.
  Finite by definition and listed in `priv/knowledge/linguistic.json`. WordNet
  contains no function words, so these cannot be derived from it.

  **Morphological** — `unable`, `impossible`, `hopeless`. Read from the brain's
  own lexicon, which `Brain.Lexicon.Seeder` fills from WordNet: a word negates
  when it is the antonym of another word **and** is that word plus a negative
  affix. The affix test is what separates negation from mere opposition —
  `hot`/`cold` are antonyms and neither negates, so the pair is correctly
  rejected.

  That half grows when WordNet grows, and can be corrected or retired once
  seeded, rather than requiring someone to notice a missing word and edit a
  list — the failure mode of a hand-maintained vocabulary. Measured against
  negation drawn from outside the 19-word closed-class list, the list alone
  recognised 1 of 12; seeding adds 1,374 words.

  ## Examples

      iex> Brain.LinguisticData.negator?("never")
      true

      iex> Brain.LinguisticData.negator?("unable")
      true

      iex> Brain.LinguisticData.negator?("cold")
      false
  """
  @spec negator?(String.t()) :: boolean()
  def negator?(word) when is_binary(word) do
    negation?(word) or morphological_negator?(word)
  end

  def negator?(_), do: false

  @doc """
  Returns true if the brain holds a morphological negation fact for the word.

  The facts are derived from WordNet by `Brain.Lexicon.Seeder` -- a word
  negates when it is the antonym of another word and is that word plus a
  negative affix -- and written into the brain's own lexicon. This reads them;
  it does not re-derive them, so a fact that is later corrected or retired
  changes the answer here.

  An unseeded lexicon holds no such facts, and every word reads as
  non-negating. `mix atlas.seed` fills it.
  """
  @spec morphological_negator?(String.t()) :: boolean()
  def morphological_negator?(word) when is_binary(word) do
    word
    |> String.downcase()
    |> Lexicon.owned_facts(kind: "property", key: "negation")
    |> Enum.any?(&(&1.value["kind"] == "morphological"))
  end

  def morphological_negator?(_), do: false
end
