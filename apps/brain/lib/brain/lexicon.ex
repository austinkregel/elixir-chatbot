defmodule Brain.Lexicon do
  @moduledoc """
  Public API for the unified lexicon system, and the one place lexical lookups
  should come from.

  WordNet (`Brain.ML.Lexicon`) is the base reference. The brain's own facts
  (`Brain.Lexicon.UserDefined`, persisted in Atlas) extend it: list-valued
  lookups return WordNet's answers followed by owned relations, and every
  owned fact records its source so a seeded fact and a learned one that
  contradicts it can both be weighed from context. With no owned facts, every
  lookup here returns exactly what WordNet returns.

  Beyond the lookups, it provides:
  - Lexical domain lookup (from WordNet lexicographer file numbers)
  - ConceptNet relation queries
  - User-defined lexicon access (for OOV / learned words)
  - OOV detection and coarse semantic classification
  - Word sense disambiguation
  - Sense drift detection
  """

  alias Brain.ML.Lexicon, as: WordNet
  alias Brain.Lexicon.ConceptNet
  alias Brain.Lexicon.UserDefined

  @lexfile_to_domain %{
    0 => :adj_all,
    1 => :adj_pert,
    2 => :adv_all,
    3 => :noun_tops,
    4 => :noun_act,
    5 => :noun_animal,
    6 => :noun_artifact,
    7 => :noun_attribute,
    8 => :noun_body,
    9 => :noun_cognition,
    10 => :noun_communication,
    11 => :noun_event,
    12 => :noun_feeling,
    13 => :noun_food,
    14 => :noun_group,
    15 => :noun_location,
    16 => :noun_motive,
    17 => :noun_object,
    18 => :noun_person,
    19 => :noun_phenomenon,
    20 => :noun_plant,
    21 => :noun_possession,
    22 => :noun_process,
    23 => :noun_quantity,
    24 => :noun_relation,
    25 => :noun_shape,
    26 => :noun_state,
    27 => :noun_substance,
    28 => :noun_time,
    29 => :verb_body,
    30 => :verb_change,
    31 => :verb_cognition,
    32 => :verb_communication,
    33 => :verb_competition,
    34 => :verb_consumption,
    35 => :verb_contact,
    36 => :verb_creation,
    37 => :verb_emotion,
    38 => :verb_motion,
    39 => :verb_perception,
    40 => :verb_possession,
    41 => :verb_social,
    42 => :verb_stative,
    43 => :verb_weather,
    44 => :adj_ppl
  }

  @domain_atoms Map.values(@lexfile_to_domain) |> Enum.uniq()

  @doc "Returns the list of all lexical domain atoms."
  def domain_atoms, do: @domain_atoms

  @doc "Returns the mapping from lexicographer file number to domain atom."
  def lexfile_to_domain_map, do: @lexfile_to_domain

  @doc """
  Looks up a word in all lexicon tiers.

  Returns `{:wordnet, data}`, `{:user_defined, data}`, or `:oov`.
  """
  def lookup(word, pos \\ nil) when is_binary(word) do
    normalized = String.downcase(word)

    if WordNet.known_word?(normalized) do
      senses = WordNet.senses(normalized)

      filtered =
        if pos do
          wn_pos = to_wordnet_pos(pos)
          Enum.filter(senses, &(&1.pos == wn_pos))
        else
          senses
        end

      case filtered do
        [] -> check_user_defined(normalized, pos)
        senses -> {:wordnet, enrich_senses(senses)}
      end
    else
      check_user_defined(normalized, pos)
    end
  end

  @doc "Returns true if the word is known (in WordNet, Gazetteer, or user-defined lexicon)."
  def known?(word) when is_binary(word) do
    normalized = String.downcase(word)

    WordNet.known_word?(normalized) or
      UserDefined.has_entry?(normalized)
  end

  @doc "Returns true if the word is out-of-vocabulary (not in any lexicon tier)."
  def oov?(word) when is_binary(word) do
    not known?(word)
  end

  @doc """
  Returns the primary lexical domain for a word.

  Tries WordNet first, then user-defined lexicon.
  Returns the domain of the most frequent sense.
  """
  def primary_domain(word, pos \\ nil) when is_binary(word) do
    case lookup(word, pos) do
      {:wordnet, senses} ->
        senses
        |> Enum.max_by(& &1.tag_count, fn -> nil end)
        |> case do
          nil -> nil
          sense -> sense[:lexical_domain]
        end

      {:user_defined, entry} ->
        case entry.senses do
          [first | _] -> first[:coarse_class]
          _ -> nil
        end

      :oov ->
        nil
    end
  end

  @doc """
  Returns a lexical domain histogram for a list of words.

  Each word contributes its primary domain. Returns a map of
  `%{domain_atom => count}`.
  """
  def domain_histogram(words) when is_list(words) do
    words
    |> Enum.map(&primary_domain/1)
    |> Enum.reject(&is_nil/1)
    |> Enum.frequencies()
  end

  @doc """
  Returns the polysemy count (number of senses) for a word.
  """
  def polysemy_count(word) when is_binary(word) do
    normalized = String.downcase(word)

    case WordNet.senses(normalized) do
      [] -> 0
      senses -> length(senses)
    end
  end

  @doc """
  Returns the hypernym depth of the primary sense of a word.

  Depth is the number of steps from the word to the root entity.
  """
  def hypernym_depth(word, pos \\ nil) when is_binary(word) do
    chain = WordNet.hypernym_chain(word, to_wordnet_pos(pos))
    length(chain)
  end

  # ---------------------------------------------------------------------------
  # Lexical lookups — WordNet as the base, the brain's own facts on top
  # ---------------------------------------------------------------------------
  #
  # Every lexical lookup in the brain goes through these functions rather than
  # `Brain.ML.Lexicon` directly. WordNet supplies the base answer; the brain's
  # own facts (`Brain.Lexicon.UserDefined`) extend it.
  #
  # List-valued lookups return WordNet's answers followed by the owned
  # relation targets, de-duplicated. Owned relations are facts of kind
  # "relation" whose `key` is the relation ("synonym", "hypernym", "antonym")
  # and whose `ref` is the related word; a relation may carry `"pos"` in its
  # value, and when a POS filter is given only relations with that POS apply.
  #
  # Scalar lookups (`definition/2`, `lemma/1`) and structured ones (`senses/1`,
  # `hypernym_chain/3`) return WordNet's answer unchanged. The brain's own facts
  # about the same word are available from `owned_facts/2`; choosing between a
  # seeded answer and a learned one that contradicts it is a contextual
  # decision for the caller, not something this layer hard-codes.
  #
  # With no owned facts, every function here returns exactly what
  # `Brain.ML.Lexicon` returns. `test/brain/lexicon/lexicon_contract_test.exs`
  # holds that equivalence.
  #
  # The `:store` option exists for tests, which need an isolated store.

  @doc "Returns the brain's own facts about a word. See `Brain.Lexicon.UserDefined.facts/3`."
  @spec owned_facts(String.t(), keyword()) :: [Atlas.Schemas.LexiconFact.t()]
  def owned_facts(word, filters \\ []) when is_binary(word) do
    {store, filters} = Keyword.pop(filters, :store, UserDefined)
    UserDefined.facts(word, filters, store)
  end

  @doc "Returns synonyms for a word, optionally filtered by POS."
  @spec synonyms(String.t(), atom() | nil, keyword()) :: [String.t()]
  def synonyms(word, pos \\ nil, opts \\ []) when is_binary(word) do
    WordNet.synonyms(word, pos)
    |> with_owned_relations(word, "synonym", pos, opts)
  end

  @doc "Returns one-level hypernyms for a word, optionally filtered by POS."
  @spec hypernyms(String.t(), atom() | nil, keyword()) :: [String.t()]
  def hypernyms(word, pos \\ nil, opts \\ []) when is_binary(word) do
    WordNet.hypernyms(word, pos)
    |> with_owned_relations(word, "hypernym", pos, opts)
  end

  @doc "Returns antonym lemmas for a word."
  @spec antonyms(String.t(), keyword()) :: [String.t()]
  def antonyms(word, opts \\ []) when is_binary(word) do
    WordNet.antonyms(word)
    |> with_owned_relations(word, "antonym", nil, opts)
  end

  @doc """
  Returns the gloss of the first matching sense as `{:ok, definition}`, or
  `:not_found`.
  """
  @spec definition(String.t(), atom() | nil) :: {:ok, String.t()} | :not_found
  def definition(word, pos \\ nil) when is_binary(word), do: WordNet.definition(word, pos)

  @doc "Walks the hypernym chain from a word. See `Brain.ML.Lexicon.hypernym_chain/3`."
  @spec hypernym_chain(String.t(), atom() | nil, keyword()) :: [String.t()]
  def hypernym_chain(word, pos \\ nil, opts \\ []) when is_binary(word) do
    WordNet.hypernym_chain(word, pos, opts)
  end

  @doc "Returns every WordNet sense of a word, with synset id, POS and definition."
  @spec senses(String.t()) :: [map()]
  def senses(word) when is_binary(word), do: WordNet.senses(word)

  @doc """
  Returns the parts of speech a word takes: WordNet's, then those of the
  brain's own senses.
  """
  @spec pos(String.t(), keyword()) :: [atom()]
  def pos(word, opts \\ []) when is_binary(word) do
    owned =
      word
      |> owned_facts(kind: "sense", store: Keyword.get(opts, :store, UserDefined))
      |> Enum.map(&String.to_existing_atom(&1.key))

    Enum.uniq(WordNet.pos(word) ++ owned)
  end

  @doc "Returns true if the word is known. Same as `known?/1`."
  @spec known_word?(String.t()) :: boolean()
  def known_word?(word) when is_binary(word), do: known?(word)

  @doc """
  Expands a token list with synonyms for tokens missing from `vocabulary`.

  A token already in the vocabulary is kept alone. Otherwise its lemma is added
  if the lemma is in the vocabulary; failing that, the first synonym found in
  the vocabulary is added. `vocabulary` is a map keyed by word.
  """
  @spec expand_with_synonyms([String.t()], map(), keyword()) :: [String.t()]
  def expand_with_synonyms(tokens, vocabulary, opts \\ [])
      when is_list(tokens) and is_map(vocabulary) do
    Enum.flat_map(tokens, fn token ->
      if Map.has_key?(vocabulary, token) do
        [token]
      else
        lemmatized = lemma(token)

        if lemmatized != token and Map.has_key?(vocabulary, lemmatized) do
          [token, lemmatized]
        else
          case Enum.find(synonyms(token, nil, opts), &Map.has_key?(vocabulary, &1)) do
            nil -> [token]
            known -> [token, known]
          end
        end
      end
    end)
  end

  defp with_owned_relations(base, word, relation, pos, opts) do
    store = Keyword.get(opts, :store, UserDefined)
    normalized = String.downcase(word)

    owned =
      normalized
      |> owned_facts(kind: "relation", key: relation, store: store)
      |> Enum.filter(fn fact -> is_nil(pos) or fact.value["pos"] == to_string(pos) end)
      |> Enum.map(& &1.ref)
      |> Enum.uniq()
      |> Enum.reject(&(&1 == normalized or &1 in base))

    # WordNet's list is passed through untouched. Its functions do not all
    # clean their output the same way (hypernyms/3 keeps the word itself if a
    # parent synset contains it), so re-processing it here would change answers.
    base ++ owned
  end

  @doc """
  Returns ConceptNet relations for a concept.

  Returns `%{relation_type => [related_concepts]}` or empty map if not found.
  """
  def conceptnet_relations(word) when is_binary(word) do
    ConceptNet.relations(word)
  end

  @doc """
  Counts ConceptNet relations by type for a word.

  Returns `%{relation_type => count}`.
  """
  def conceptnet_relation_counts(word) when is_binary(word) do
    word
    |> conceptnet_relations()
    |> Enum.map(fn {rel_type, concepts} -> {rel_type, length(concepts)} end)
    |> Map.new()
  end

  @doc """
  Word sense disambiguation using lexical domain overlap with context.

  Given a word and its context words, returns the best-matching synset
  with its domain and confidence.
  """
  def disambiguate(word, pos \\ nil, context_words)

  def disambiguate(word, pos, context_words) when is_binary(word) and is_list(context_words) do
    normalized = String.downcase(word)
    senses = WordNet.senses(normalized)

    filtered =
      if pos do
        wn_pos = to_wordnet_pos(pos)
        Enum.filter(senses, &(&1.pos == wn_pos))
      else
        senses
      end

    case filtered do
      [] ->
        {:oov, nil, 0.0}

      [single] ->
        domain = lexical_domain_for_synset(single.synset_id)
        {:ok, %{synset_id: single.synset_id, domain: domain, confidence: 1.0}}

      multiple ->
        context_domains = domain_histogram(context_words)
        scored = Enum.map(multiple, &score_sense_against_context(&1, context_domains))

        best = Enum.max_by(scored, fn {_sense, score} -> score end)
        {best_sense, best_score} = best

        total = scored |> Enum.map(fn {_, s} -> s end) |> Enum.sum()
        confidence = if total > 0, do: best_score / total, else: 1.0 / length(multiple)

        domain = lexical_domain_for_synset(best_sense.synset_id)

        {:ok, %{synset_id: best_sense.synset_id, domain: domain, confidence: confidence}}
    end
  end

  @doc """
  Wu-Palmer similarity between two words.
  Delegates to the existing WordNet implementation.
  """
  def word_similarity(word1, word2) do
    WordNet.word_similarity(word1, word2)
  end

  @doc "Returns the base/lemma form of a word."
  def lemma(word) when is_binary(word), do: WordNet.lemma(word)

  @doc """
  Returns every lemma a word can be a form of, as `{lemma, pos}`: regular
  and irregular inflections and the word itself. See
  `Brain.ML.Lexicon.base_forms/2`.
  """
  @spec base_forms(String.t()) :: [{String.t(), atom()}]
  def base_forms(word) when is_binary(word), do: WordNet.base_forms(word)

  @doc """
  The grammatical number of a word read as a noun: `:plural`, `:singular`,
  or `nil` when it is no noun WordNet knows. See
  `Brain.ML.Lexicon.grammatical_number/2`.
  """
  @spec grammatical_number(String.t()) :: :plural | :singular | nil
  def grammatical_number(word) when is_binary(word), do: WordNet.grammatical_number(word)

  @doc "Returns all synset IDs for a word, optionally filtered by POS."
  def synset_ids(word, pos \\ nil) when is_binary(word) do
    senses = WordNet.senses(String.downcase(word))

    filtered =
      if pos do
        wn_pos = to_wordnet_pos(pos)
        Enum.filter(senses, &(&1.pos == wn_pos))
      else
        senses
      end

    Enum.map(filtered, & &1.synset_id)
  end

  @doc """
  Returns the lexical domain for a synset ID.

  Extracts the lexicographer file number from the sense key
  stored in the `:lexicon_sense_keys` ETS table (populated by Loader).
  Falls back to inferring from the synset ID range.
  """
  def lexical_domain_for_synset(synset_id) when is_integer(synset_id) do
    case :ets.lookup(:lexicon_sense_keys, synset_id) do
      [{^synset_id, lex_filenum}] ->
        Map.get(@lexfile_to_domain, lex_filenum, :unknown)

      _ ->
        infer_domain_from_synset_id(synset_id)
    end
  end

  # -- Private ----------------------------------------------------------------

  defp check_user_defined(word, _pos) do
    case UserDefined.get(word) do
      nil -> :oov
      entry -> {:user_defined, entry}
    end
  end

  defp enrich_senses(senses) do
    Enum.map(senses, fn sense ->
      domain = lexical_domain_for_synset(sense.synset_id)
      Map.put(sense, :lexical_domain, domain)
    end)
  end

  defp score_sense_against_context(sense, context_domains) when map_size(context_domains) == 0 do
    {sense, sense.tag_count + 1}
  end

  defp score_sense_against_context(sense, context_domains) do
    domain = lexical_domain_for_synset(sense.synset_id)
    domain_group = domain_to_group(domain)

    domain_score =
      context_domains
      |> Enum.reduce(0, fn {ctx_domain, count}, acc ->
        if domain_to_group(ctx_domain) == domain_group do
          acc + count
        else
          acc
        end
      end)

    tag_prior = :math.log(sense.tag_count + 1)
    {sense, domain_score + tag_prior}
  end

  defp domain_to_group(domain) when is_atom(domain) do
    domain
    |> Atom.to_string()
    |> String.split("_", parts: 2)
    |> case do
      [_pos, group] -> group
      [single] -> single
    end
  end

  defp infer_domain_from_synset_id(synset_id) do
    cond do
      synset_id >= 100_000_000 and synset_id < 200_000_000 -> :noun_tops
      synset_id >= 200_000_000 and synset_id < 300_000_000 -> :verb_stative
      synset_id >= 300_000_000 and synset_id < 400_000_000 -> :adj_all
      synset_id >= 400_000_000 -> :adv_all
      true -> :unknown
    end
  end

  defp to_wordnet_pos(nil), do: nil
  defp to_wordnet_pos(:noun), do: :noun
  defp to_wordnet_pos(:verb), do: :verb
  defp to_wordnet_pos(:adj), do: :adj
  defp to_wordnet_pos(:adv), do: :adv
  defp to_wordnet_pos(:NOUN), do: :noun
  defp to_wordnet_pos(:VERB), do: :verb
  defp to_wordnet_pos(:ADJ), do: :adj
  defp to_wordnet_pos(:ADV), do: :adv
  defp to_wordnet_pos(_), do: nil
end
