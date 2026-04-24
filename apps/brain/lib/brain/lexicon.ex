defmodule Brain.Lexicon do
  @moduledoc """
  Public API for the unified lexicon system.

  Wraps the existing WordNet-based `Brain.ML.Lexicon` and extends it with:
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

  @doc """
  Returns antonym lemmas for a word.
  """
  def antonyms(word) when is_binary(word) do
    WordNet.antonyms(word)
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
