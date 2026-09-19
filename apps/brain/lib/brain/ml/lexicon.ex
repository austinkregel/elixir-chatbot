defmodule Brain.ML.Lexicon do
  @moduledoc """
  ETS-backed lexical database powered by WordNet 3.1.

  Provides synonym lookup, hypernym chain walking, lemmatization,
  definitions, and word-sense information. Used throughout the analysis
  pipeline to bridge vocabulary gaps and enrich language understanding.

  WordNet is a required dependency. The GenServer crashes on init
  if the Prolog data files are missing or corrupt.
  """

  use GenServer
  require Logger

  alias Brain.ML.Lexicon.WordNetParser

  @words_table :lexicon_words
  @synsets_table :lexicon_synsets
  @hypernyms_table :lexicon_hypernyms
  @instances_table :lexicon_instances
  @antonyms_table :lexicon_antonyms
  @morph_table :lexicon_morph
  @endings_table :lexicon_endings
  @hyp_cache_table :lexicon_hyp_cache

  @wordnet_dir "wordnet"

  @required_files ["wn_s.pl", "wn_g.pl", "wn_hyp.pl", "wn_ins.pl", "wn_exc.pl", "wn_morphy.pl"]

  # A morphy rule for adjectives ("a") matches head adjectives and
  # satellites alike, as Princeton's morphy does.
  @rule_pos %{noun: [:noun], verb: [:verb], adj: [:adj, :adj_satellite]}

  # -- Client API -------------------------------------------------------------

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Returns synonyms for a word, optionally filtered by POS."
  def synonyms(word, pos \\ nil, name \\ __MODULE__)

  def synonyms(word, pos, name) when is_binary(word) do
    normalized = String.downcase(word)

    synset_ids = lookup_synset_ids(normalized, pos, name)

    synset_ids
    |> Enum.flat_map(fn synset_id ->
      case :ets.lookup(table_for(name, :synsets), synset_id) do
        [{^synset_id, words, _definition, _pos}] -> words
        _ -> []
      end
    end)
    |> Enum.uniq()
    |> Enum.reject(&(&1 == normalized))
  end

  @doc "Returns the definition (gloss) for a word, optionally filtered by POS."
  def definition(word, pos \\ nil, name \\ __MODULE__)

  def definition(word, pos, name) when is_binary(word) do
    normalized = String.downcase(word)
    synset_ids = lookup_synset_ids(normalized, pos, name)

    case synset_ids do
      [] ->
        :not_found

      [first | _] ->
        case :ets.lookup(table_for(name, :synsets), first) do
          [{^first, _words, definition, _pos}] -> {:ok, definition}
          _ -> :not_found
        end
    end
  end

  @doc "Returns hypernyms for a word (walks one level up the IS-A hierarchy)."
  def hypernyms(word, pos \\ nil, name \\ __MODULE__)

  def hypernyms(word, pos, name) when is_binary(word) do
    normalized = String.downcase(word)
    synset_ids = lookup_synset_ids(normalized, pos, name)

    synset_ids
    |> Enum.flat_map(fn sid ->
      case :ets.lookup(table_for(name, :hypernyms), sid) do
        [{^sid, parent_id}] -> synset_words(parent_id, name)
        _ -> []
      end
    end)
    |> Enum.uniq()
  end

  @doc """
  Walks the full hypernym chain from a word to the root.

  Returns a list of word lists, one per level. Stops at `max_depth`
  or when there are no more hypernyms.
  """
  def hypernym_chain(word, pos \\ nil, opts \\ [])

  def hypernym_chain(word, pos, opts) when is_binary(word) do
    name = Keyword.get(opts, :name, __MODULE__)
    max_depth = Keyword.get(opts, :max_depth, 15)
    normalized = String.downcase(word)

    synset_ids = lookup_synset_ids(normalized, pos, name)

    case synset_ids do
      [] ->
        []

      [first_sid | _] ->
        cache_table = table_for(name, :hyp_cache)

        case :ets.lookup(cache_table, {first_sid, max_depth}) do
          [{_, cached}] ->
            cached

          _ ->
            chain = walk_hypernyms(first_sid, max_depth, name, MapSet.new())
            :ets.insert(cache_table, {{first_sid, max_depth}, chain})
            chain
        end
    end
  end

  @doc """
  Returns every ancestor of one synset as `{synset_id, words, distance}`,
  nearest first. `distance` is the number of edges from the synset, so
  ancestors reached by different paths at the same distance can be told
  apart from nearer ones; within one distance they are ordered by synset id.

  Follows both hypernym edges and instance edges, so a named instance
  reaches its class: Paris reaches "national capital" and "city". Unlike
  `hypernym_chain/3`, which walks only a word's first sense, this works on a
  single sense, so each sense of a word can be classified on its own.
  """
  def synset_ancestors(synset_id, opts \\ []) when is_integer(synset_id) do
    name = Keyword.get(opts, :name, __MODULE__)
    max_depth = Keyword.get(opts, :max_depth, 15)

    collect_ancestors([synset_id], 1, max_depth, name, MapSet.new([synset_id]), [])
  end

  defp collect_ancestors(_frontier, distance, max_depth, _name, _seen, acc)
       when distance > max_depth,
       do: Enum.reverse(acc)

  defp collect_ancestors([], _distance, _max_depth, _name, _seen, acc), do: Enum.reverse(acc)

  defp collect_ancestors(frontier, distance, max_depth, name, seen, acc) do
    parents =
      frontier
      |> Enum.flat_map(fn sid ->
        hyps = :ets.lookup(table_for(name, :hypernyms), sid)
        insts = :ets.lookup(table_for(name, :instances), sid)
        Enum.map(hyps ++ insts, fn {_child, parent} -> parent end)
      end)
      |> Enum.uniq()
      |> Enum.reject(&MapSet.member?(seen, &1))
      |> Enum.sort()

    seen = Enum.reduce(parents, seen, &MapSet.put(&2, &1))

    acc =
      Enum.reduce(parents, acc, fn sid, acc ->
        [{sid, synset_words(sid, name), distance} | acc]
      end)

    collect_ancestors(parents, distance + 1, max_depth, name, seen, acc)
  end

  @doc """
  Returns one synset as `{:ok, %{words: words, definition: gloss, pos: pos}}`,
  or `:error` when WordNet holds no synset with that id.
  """
  def synset(synset_id, name \\ __MODULE__) when is_integer(synset_id) do
    case :ets.lookup(table_for(name, :synsets), synset_id) do
      [{^synset_id, words, definition, pos}] -> {:ok, %{words: words, definition: definition, pos: pos}}
      [] -> :error
    end
  end

  @doc """
  Returns every word WordNet holds, lowercased.

  Used to derive facts from the whole corpus at seed time.
  """
  def words(name \\ __MODULE__) do
    table_for(name, :words)
    |> :ets.tab2list()
    |> Enum.map(fn {word, _entries} -> word end)
  end

  @doc "Returns antonyms for a word."
  def antonyms(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)

    case :ets.lookup(table_for(name, :words), normalized) do
      [{^normalized, entries}] ->
        entries
        |> Enum.flat_map(fn {synset_id, w_num, _pos, _tag_count} ->
          table_for(name, :antonyms)
          |> :ets.lookup({synset_id, w_num})
          |> Enum.flat_map(fn {_, target_sid, _target_wnum} ->
            synset_words(target_sid, name)
          end)
        end)
        |> Enum.uniq()
        |> Enum.reject(&(&1 == normalized))

      _ ->
        []
    end
  end

  @doc """
  Returns every `{word, antonym}` pair WordNet holds.

  Used to derive facts from the whole corpus at seed time, rather than asking
  about one word at a time. Roughly 8,000 pairs.
  """
  def antonym_pairs(name \\ __MODULE__) do
    table_for(name, :words)
    |> :ets.tab2list()
    |> Enum.flat_map(fn {word, _entries} ->
      Enum.map(antonyms(word, name), &{word, &1})
    end)
  end

  @doc "Returns the base/lemma form of an inflected word using WordNet morphological exceptions."
  def lemma(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)

    case exception_bases(normalized, name) do
      [] -> normalized
      bases -> hd(bases)
    end
  end

  @doc "Returns all known base forms for an inflected word."
  def lemma_all(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)

    case exception_bases(normalized, name) do
      [] -> [normalized]
      bases -> bases
    end
  end

  # The base forms the exception list gives for a word, any part of speech.
  defp exception_bases(normalized, name) do
    case :ets.lookup(table_for(name, :morph), normalized) do
      [{^normalized, entries}] -> entries |> Enum.map(fn {base, _pos} -> base end) |> Enum.uniq()
      _ -> []
    end
  end

  @doc """
  Returns every WordNet lemma `word` can be a form of, as `{lemma, pos}`,
  the way WordNet's morphy (wn_morphy.pl) finds them.

  Candidates come from the exception list (`mice` -> `mouse`), from every
  suffix rule that applies (`lights` -> `light`, `cities` -> `city`), and
  from the word itself. A candidate is kept only when WordNet holds it as a
  lemma with that part of speech, so `bus` does not become `bu`. Sorted, so
  the result does not depend on table order.

  Unlike `lemma/1`, which reads only the exception list, this applies the
  regular suffix rules too.
  """
  @spec base_forms(String.t(), atom()) :: [{String.t(), atom()}]
  def base_forms(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)

    exceptions =
      case :ets.lookup(table_for(name, :morph), normalized) do
        [{^normalized, entries}] -> entries
        _ -> []
      end

    [{:endings, endings}] = :ets.lookup(table_for(name, :endings), :endings)

    by_rule =
      Enum.flat_map(endings, fn {rule_pos, inflected, base_ending} ->
        if String.ends_with?(normalized, inflected) and byte_size(normalized) > byte_size(inflected) do
          stem = binary_part(normalized, 0, byte_size(normalized) - byte_size(inflected))
          [{stem <> base_ending, rule_pos}]
        else
          []
        end
      end)

    itself = Enum.map([:noun, :verb, :adj, :adv], &{normalized, &1})

    (exceptions ++ by_rule ++ itself)
    |> Enum.map(fn {lemma, pos} -> {lemma, normalize_rule_pos(pos)} end)
    |> Enum.uniq()
    |> Enum.filter(fn {lemma, pos} -> lemma_with_pos?(lemma, pos, name) end)
    |> Enum.sort()
  end

  @doc """
  The grammatical number of `word` read as a noun: `:plural` when it is an
  inflected form of a different noun lemma (`lights`, `mice`), `:singular`
  when it is a noun lemma and no such form, and `nil` when WordNet knows no
  noun it could be. A word that is both (`glasses`, the eyewear, and the
  plural of `glass`) reads as `:plural`.
  """
  @spec grammatical_number(String.t(), atom()) :: :plural | :singular | nil
  def grammatical_number(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)
    nouns = for {lemma, :noun} <- base_forms(normalized, name), do: lemma

    cond do
      Enum.any?(nouns, &(&1 != normalized)) -> :plural
      normalized in nouns -> :singular
      true -> nil
    end
  end

  defp normalize_rule_pos(pos) when pos in [:adj, :adj_satellite], do: :adj
  defp normalize_rule_pos(pos), do: pos

  defp lemma_with_pos?(lemma, pos, name) do
    wanted = Map.get(@rule_pos, pos, [pos])

    case :ets.lookup(table_for(name, :words), lemma) do
      [{^lemma, entries}] -> Enum.any?(entries, fn {_sid, _w, p, _tc} -> p in wanted end)
      _ -> false
    end
  end

  @doc "Returns true if the word exists in WordNet."
  def known_word?(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)
    :ets.lookup(table_for(name, :words), normalized) != []
  end

  @doc "Returns the POS tags WordNet assigns to this word."
  def pos(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)

    case :ets.lookup(table_for(name, :words), normalized) do
      [{^normalized, entries}] ->
        entries
        |> Enum.map(fn {_synset_id, _w_num, pos, _tag_count} -> pos end)
        |> Enum.uniq()

      _ ->
        []
    end
  end

  @doc "Returns all senses for a word with synset IDs, POS, definitions, and tag counts."
  def senses(word, name \\ __MODULE__) when is_binary(word) do
    normalized = String.downcase(word)

    case :ets.lookup(table_for(name, :words), normalized) do
      [{^normalized, entries}] ->
        Enum.map(entries, fn {synset_id, _w_num, pos, tag_count} ->
          {_words, definition} =
            case :ets.lookup(table_for(name, :synsets), synset_id) do
              [{^synset_id, words, defn, _}] -> {words, defn}
              _ -> {[], ""}
            end

          %{synset_id: synset_id, pos: pos, definition: definition, tag_count: tag_count}
        end)

      _ ->
        []
    end
  end

  @doc """
  Expands a token list with synonyms for out-of-vocabulary tokens.

  For each token not in the given vocabulary map, looks up its
  synonyms and adds the first one that IS in the vocabulary.
  """
  def expand_with_synonyms(tokens, vocabulary, name \\ __MODULE__) when is_list(tokens) and is_map(vocabulary) do
    Enum.flat_map(tokens, fn token ->
      if Map.has_key?(vocabulary, token) do
        [token]
      else
        lemmatized = lemma(token, name)

        if lemmatized != token and Map.has_key?(vocabulary, lemmatized) do
          [token, lemmatized]
        else
          case synonyms(token, nil, name) do
            [] ->
              [token]

            syns ->
              known = Enum.find(syns, &Map.has_key?(vocabulary, &1))
              if known, do: [token, known], else: [token]
          end
        end
      end
    end)
  end

  @doc """
  Wu-Palmer similarity between two words via their hypernym paths.

  Returns a float between 0.0 and 1.0. Higher means more similar.
  """
  def word_similarity(word1, word2, name \\ __MODULE__) do
    w1 = String.downcase(word1)
    w2 = String.downcase(word2)

    chain1 = [w1 | hypernym_chain(w1, nil, name: name)]
    chain2 = [w2 | hypernym_chain(w2, nil, name: name)]
    set2 = MapSet.new(chain2)

    lcs_idx =
      chain1
      |> Enum.with_index()
      |> Enum.find_value(fn {word, idx} ->
        if MapSet.member?(set2, word), do: idx, else: nil
      end)

    case lcs_idx do
      nil ->
        0.0

      idx ->
        lcs_word = Enum.at(chain1, idx)
        lcs_pos_in_chain2 = Enum.find_index(chain2, &(&1 == lcs_word)) || 0
        depth1 = idx + 1
        depth2 = lcs_pos_in_chain2 + 1
        lcs_depth = length(chain1) - idx

        2.0 * lcs_depth / (2.0 * lcs_depth + depth1 - 1 + depth2 - 1)
    end
  end

  @doc "Returns stats about the loaded lexicon."
  def stats(name \\ __MODULE__) do
    GenServer.call(name, :stats)
  end

  # -- Server callbacks -------------------------------------------------------

  @impl true
  def init(opts) do
    wordnet_path = Keyword.get(opts, :wordnet_path, default_wordnet_path())

    validate_wordnet_files!(wordnet_path)

    tables = create_tables(opts)

    Logger.info("Lexicon: loading WordNet from #{wordnet_path}...")
    start_time = System.monotonic_time(:millisecond)

    load_senses(wordnet_path, tables)
    load_glosses(wordnet_path, tables)
    load_hypernyms(wordnet_path, tables)
    load_instances(wordnet_path, tables)
    load_antonyms(wordnet_path, tables)
    load_exceptions(wordnet_path, tables)
    load_endings(wordnet_path, tables)

    elapsed = System.monotonic_time(:millisecond) - start_time

    word_count = :ets.info(tables.words, :size)
    synset_count = :ets.info(tables.synsets, :size)

    Logger.info(
      "Lexicon: loaded #{word_count} words, #{synset_count} synsets in #{elapsed}ms"
    )

    {:ok,
     %{
       tables: tables,
       wordnet_path: wordnet_path,
       word_count: word_count,
       synset_count: synset_count,
       load_time_ms: elapsed
     }}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      word_count: state.word_count,
      synset_count: state.synset_count,
      load_time_ms: state.load_time_ms,
      hypernym_count: :ets.info(state.tables.hypernyms, :size),
      morph_count: :ets.info(state.tables.morph, :size),
      hyp_cache_size: :ets.info(state.tables.hyp_cache, :size)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call({:table_name, type}, _from, state) do
    {:reply, Map.fetch!(state.tables, type), state}
  end

  # -- Private ----------------------------------------------------------------

  defp default_wordnet_path do
    Brain.priv_path(@wordnet_dir)
  end

  defp validate_wordnet_files!(path) do
    unless File.dir?(path) do
      raise """
      WordNet data directory not found at #{path}.

      The Lexicon requires WordNet 3.1 Prolog files to function.
      Run: mix download_wordnet
      """
    end

    Enum.each(@required_files, fn file ->
      full_path = Path.join(path, file)

      unless File.exists?(full_path) do
        raise """
        Required WordNet file missing: #{file}

        Expected at: #{full_path}
        Run: mix download_wordnet
        """
      end
    end)
  end

  defp create_tables(opts) do
    prefix = Keyword.get(opts, :table_prefix)

    %{
      words: create_ets(prefix, @words_table, :set),
      synsets: create_ets(prefix, @synsets_table, :set),
      hypernyms: create_ets(prefix, @hypernyms_table, :bag),
      instances: create_ets(prefix, @instances_table, :bag),
      antonyms: create_ets(prefix, @antonyms_table, :bag),
      morph: create_ets(prefix, @morph_table, :set),
      endings: create_ets(prefix, @endings_table, :set),
      hyp_cache: create_ets(prefix, @hyp_cache_table, :set)
    }
  end

  defp create_ets(nil, name, type) do
    :ets.new(name, [type, :public, :named_table, read_concurrency: true])
    name
  end

  defp create_ets(prefix, name, type) do
    full_name = :"#{prefix}_#{name}"
    :ets.new(full_name, [type, :public, :named_table, read_concurrency: true])
    full_name
  end

  defp load_senses(path, tables) do
    senses = WordNetParser.parse_senses(Path.join(path, "wn_s.pl"))

    if senses == [] do
      raise "WordNet wn_s.pl parsed zero senses -- file may be corrupt"
    end

    word_groups =
      Enum.group_by(senses, fn s -> String.downcase(s.word) end)

    Enum.each(word_groups, fn {word, word_senses} ->
      entries =
        word_senses
        |> Enum.sort_by(fn s -> -s.tag_count end)
        |> Enum.map(fn s -> {s.synset_id, s.w_num, s.ss_type, s.tag_count} end)

      :ets.insert(tables.words, {word, entries})
    end)

    synset_groups =
      Enum.group_by(senses, fn s -> s.synset_id end)

    Enum.each(synset_groups, fn {synset_id, synset_senses} ->
      words = Enum.map(synset_senses, fn s -> String.downcase(s.word) end) |> Enum.uniq()
      pos = List.first(synset_senses).ss_type
      :ets.insert(tables.synsets, {synset_id, words, "", pos})
    end)
  end

  defp load_glosses(path, tables) do
    glosses = WordNetParser.parse_glosses(Path.join(path, "wn_g.pl"))

    Enum.each(glosses, fn {synset_id, definition} ->
      case :ets.lookup(tables.synsets, synset_id) do
        [{^synset_id, words, _old_def, pos}] ->
          :ets.insert(tables.synsets, {synset_id, words, definition, pos})

        _ ->
          :ets.insert(tables.synsets, {synset_id, [], definition, :unknown})
      end
    end)
  end

  defp load_hypernyms(path, tables) do
    hyps = WordNetParser.parse_hypernyms(Path.join(path, "wn_hyp.pl"))

    Enum.each(hyps, fn {child, parent} ->
      :ets.insert(tables.hypernyms, {child, parent})
    end)
  end

  # wn_ins.pl links a named instance to its class: Paris is an instance of
  # "national capital", a person to the kind of person they were. Proper nouns
  # have no ordinary hypernyms, so without these their ancestry is empty.
  defp load_instances(path, tables) do
    instances = WordNetParser.parse_instances(Path.join(path, "wn_ins.pl"))

    if instances == [] do
      raise "WordNet wn_ins.pl parsed zero instance relations -- file may be corrupt"
    end

    Enum.each(instances, fn {instance, class} ->
      :ets.insert(tables.instances, {instance, class})
    end)
  end

  defp load_antonyms(path, tables) do
    ant_path = Path.join(path, "wn_ant.pl")

    if File.exists?(ant_path) do
      ants = WordNetParser.parse_antonyms(ant_path)

      Enum.each(ants, fn {s1, w1, s2, _w2} ->
        :ets.insert(tables.antonyms, {{s1, w1}, s2, 0})
      end)
    end
  end

  # Irregular forms, keyed by inflected form, each with its base form and the
  # part of speech the exception is for.
  defp load_exceptions(path, tables) do
    exceptions = WordNetParser.parse_exceptions(Path.join(path, "wn_exc.pl"))

    if exceptions == [] do
      raise "WordNet wn_exc.pl parsed zero morphological exceptions -- file may be corrupt"
    end

    exceptions
    |> Enum.group_by(fn {inflected, _base, _pos} -> inflected end)
    |> Enum.each(fn {inflected, entries} ->
      bases = entries |> Enum.map(fn {_inflected, base, pos} -> {base, pos} end) |> Enum.uniq()
      :ets.insert(tables.morph, {inflected, bases})
    end)
  end

  # WordNet's regular suffix rules (wn_morphy.pl), in file order.
  defp load_endings(path, tables) do
    endings = WordNetParser.parse_endings(Path.join(path, "wn_morphy.pl"))

    if endings == [] do
      raise "WordNet wn_morphy.pl parsed zero suffix rules -- file may be corrupt"
    end

    :ets.insert(tables.endings, {:endings, endings})
  end

  defp lookup_synset_ids(normalized_word, nil, name) do
    case :ets.lookup(table_for(name, :words), normalized_word) do
      [{^normalized_word, entries}] ->
        Enum.map(entries, fn {synset_id, _w_num, _pos, _tag_count} -> synset_id end)

      _ ->
        []
    end
  end

  defp lookup_synset_ids(normalized_word, pos, name) when is_atom(pos) do
    case :ets.lookup(table_for(name, :words), normalized_word) do
      [{^normalized_word, entries}] ->
        entries
        |> Enum.filter(fn {_synset_id, _w_num, entry_pos, _tag_count} ->
          entry_pos == pos or
            (pos == :adj and entry_pos == :adj_satellite)
        end)
        |> Enum.map(fn {synset_id, _w_num, _pos, _tag_count} -> synset_id end)

      _ ->
        []
    end
  end

  defp synset_words(synset_id, name) do
    case :ets.lookup(table_for(name, :synsets), synset_id) do
      [{^synset_id, words, _def, _pos}] -> words
      _ -> []
    end
  end

  defp walk_hypernyms(_synset_id, 0, _name, _visited), do: []

  defp walk_hypernyms(synset_id, depth, name, visited) do
    if MapSet.member?(visited, synset_id) do
      []
    else
      visited = MapSet.put(visited, synset_id)

      parent_ids =
        table_for(name, :hypernyms)
        |> :ets.lookup(synset_id)
        |> Enum.map(fn {_child, parent} -> parent end)

      parent_words =
        parent_ids
        |> Enum.flat_map(&synset_words(&1, name))
        |> Enum.uniq()

      deeper =
        parent_ids
        |> Enum.flat_map(&walk_hypernyms(&1, depth - 1, name, visited))

      parent_words ++ deeper
    end
  end

  defp table_for(__MODULE__, :words), do: @words_table
  defp table_for(__MODULE__, :synsets), do: @synsets_table
  defp table_for(__MODULE__, :hypernyms), do: @hypernyms_table
  defp table_for(__MODULE__, :instances), do: @instances_table
  defp table_for(__MODULE__, :antonyms), do: @antonyms_table
  defp table_for(__MODULE__, :morph), do: @morph_table
  defp table_for(__MODULE__, :endings), do: @endings_table
  defp table_for(__MODULE__, :hyp_cache), do: @hyp_cache_table

  defp table_for(name, type) when is_atom(name) do
    GenServer.call(name, {:table_name, type})
  end
end
