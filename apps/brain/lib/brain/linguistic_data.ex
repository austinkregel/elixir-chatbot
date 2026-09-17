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

  # Negative affixes. This list is the rule, not the answer: it only decides
  # whether an antonym pair is related *by negation* rather than by plain
  # opposition. The negators themselves come from WordNet.
  @negative_prefixes ~w(un im in ir il non dis)
  @negative_suffix "less"

  @doc """
  Returns true if the word negates, from either of the two sources English uses.

  **Closed class** — `not`, `never`, `no`, `neither`, and the contractions.
  Finite by definition and listed in `priv/knowledge/linguistic.json`. WordNet
  contains no function words, so these cannot be derived from it.

  **Morphological** — `unable`, `impossible`, `hopeless`. Derived from WordNet:
  a word negates when it is the antonym of another word **and** is that word
  plus a negative affix. The affix test is what separates negation from mere
  opposition — `hot`/`cold` are antonyms and neither negates, so the pair is
  correctly rejected.

  The derived half grows when WordNet grows. It does not require anyone to
  notice a missing word and edit a list, which is the failure mode a
  hand-maintained vocabulary has: measured against negation drawn from outside
  the 19-word list, the list alone recognised 1 of 12.

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
  Returns true if the word is an antonym of some word it is derived from by a
  negative affix.

  Restricted to adjectives, adverbs and verbs. Nominalisations (`inability`,
  `impossibility`) are excluded: they name a negated concept rather than negate
  a clause, and admitting them makes every mention of a shortcoming a negation.

  Reads through `Brain.Lexicon`, whose WordNet base is supervised and
  required — it crashes on init if the WordNet data is missing, so there is no
  degraded mode here.
  """
  @spec morphological_negator?(String.t()) :: boolean()
  def morphological_negator?(word) when is_binary(word) do
    normalized = String.downcase(word)

    if clause_level_pos?(normalized) do
      normalized
      |> Lexicon.antonyms()
      |> Enum.any?(&derived_by_negative_affix?(normalized, &1))
    else
      false
    end
  end

  def morphological_negator?(_), do: false

  # POS atoms as `Brain.Lexicon.pos/2` returns them, per the WordNet mapping in
  # `Brain.ML.Lexicon.WordNetParser` (n->:noun, v->:verb, a->:adj,
  # s->:adj_satellite, r->:adv). Nouns are excluded deliberately; see
  # `morphological_negator?/1`.
  @clause_level_pos [:adj, :adj_satellite, :adv, :verb]

  defp clause_level_pos?(word) do
    Enum.any?(Lexicon.pos(word), &(&1 in @clause_level_pos))
  end

  # Two shapes of affixal negation, and they anchor differently.
  #
  #   prefix: the word is the antonym plus a prefix -- "unable" / "able".
  #   suffix: the word and its antonym share a stem, and the word is that stem
  #           plus "-less" -- "hopeless" / "hopeful" both sit on "hope". The
  #           antonym is not the root here, so the prefix test cannot be reused.
  defp derived_by_negative_affix?(word, antonym) do
    Enum.any?(@negative_prefixes, &(word == &1 <> antonym)) or
      privative_suffix?(word, antonym)
  end

  defp privative_suffix?(word, antonym) do
    case String.split(word, @negative_suffix) do
      [stem, ""] when byte_size(stem) >= 3 -> String.starts_with?(antonym, stem)
      _ -> false
    end
  end
end
