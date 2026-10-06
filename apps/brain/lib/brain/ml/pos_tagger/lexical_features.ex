defmodule Brain.ML.POSTagger.LexicalFeatures do
  @moduledoc """
  What the lexicon knows about a token, as a fixed-width vector for the tagger.

  UD EWT teaches the tagger the words it contains. It contains almost no
  sentence-initial command verbs — 71 of its 16,622 sentences begin with one — so
  for a verb like `switch`, which appears five times in the treebank and never
  initially, the tagger has no evidence that it can head an imperative. WordNet
  has seven verb senses for it. This channel is how that reaches the model.

  ## Why both sense counts and sense frequencies

  `dim` has five verb senses and a SemCor `tag_count` of **zero** across all of
  them, against seven for its adjective senses. Frequency alone says "adjective";
  the sense inventory says the verb reading is half of what the word means. The two
  disagree often enough on exactly the words this channel exists for that both are
  encoded, with `has_frequency` marking the cases where the frequency group carries
  no information at all rather than leaving the model to read zeros as evidence.

  ## Computed from the raw token

  `features/1` takes a string and asks the lexicon. It never consults the training
  vocabulary, which is the point: `dim` is out-of-vocabulary to EWT and perfectly
  well known to WordNet, and a feature keyed on the vocabulary would lose exactly
  the words it is meant to supply.

  ## Failure

  A lexicon error raises. It cannot return zeros, because the all-zero vector is
  what a legitimately unknown token produces, and the model trains on that
  distinction — the same reason `Brain.Analysis.FeatureExtractor.WordFeatures`
  reraises rather than answering with its out-of-vocabulary value.
  """

  alias Brain.Lexicon.ClosedClass
  alias Brain.ML.Lexicon

  # WordNet's adjective satellites are adjectives; they are a position within a
  # synset cluster, not a different part of speech.
  @pos_buckets [:noun, :verb, :adj, :adv]

  @satellite_of %{adj_satellite: :adj}

  @closed_classes ClosedClass.all_classes()

  # Polysemy is scaled by this before log1p, so the busiest words in WordNet land
  # near 1.0 rather than dominating the channel. `run` has 57 senses.
  @polysemy_scale 60.0

  @doc """
  The name of every dimension, in order.

  The vector is meaningless without it: a bare list of floats cannot be inspected,
  diffed between two lexicons, or shown beside a tag on the training page.
  """
  @spec manifest() :: [atom()]
  def manifest do
    Enum.map(@pos_buckets, &:"has_#{&1}") ++
      Enum.map(@pos_buckets, &:"sense_share_#{&1}") ++
      Enum.map(@pos_buckets, &:"freq_share_#{&1}") ++
      [:has_frequency, :known_to_lexicon, :polysemy] ++
      Enum.map(@closed_classes, &:"closed_#{String.downcase(&1)}")
  end

  @doc "How many floats `features/1` returns."
  @spec width() :: pos_integer()
  def width, do: length(manifest())

  @doc """
  The lexical evidence for one token.

  An all-zero vector means the lexicon holds nothing for this token, which is
  itself the signal `known_to_lexicon` carries.
  """
  @spec features(String.t()) :: [float()]
  def features(token) when is_binary(token) do
    word = String.downcase(String.trim(token))

    senses = if word == "", do: [], else: Lexicon.senses(word)
    classes = if word == "", do: [], else: ClosedClass.classes(word)

    by_pos = Enum.group_by(senses, &bucket(&1.pos))
    counts = Map.new(@pos_buckets, fn p -> {p, length(Map.get(by_pos, p, []))} end)

    freqs =
      Map.new(@pos_buckets, fn p ->
        {p, Enum.sum(Enum.map(Map.get(by_pos, p, []), & &1.tag_count))}
      end)

    total_senses = Enum.sum(Map.values(counts))
    total_freq = Enum.sum(Map.values(freqs))

    has = Enum.map(@pos_buckets, fn p -> flag(Map.fetch!(counts, p) > 0) end)
    sense_share = Enum.map(@pos_buckets, fn p -> share(Map.fetch!(counts, p), total_senses) end)
    freq_share = Enum.map(@pos_buckets, fn p -> share(Map.fetch!(freqs, p), total_freq) end)

    known = flag(senses != [] or classes != [])
    polysemy = :math.log(1.0 + total_senses) / :math.log(1.0 + @polysemy_scale)

    closed = Enum.map(@closed_classes, fn c -> flag(c in classes) end)

    has ++
      sense_share ++
      freq_share ++
      [flag(total_freq > 0), known, min(polysemy, 1.0)] ++
      closed
  rescue
    e ->
      reraise(
        "pos_tagger lexical features: lexicon lookup failed for #{inspect(token)}. " <>
          "The tagger needs Brain.ML.Lexicon started at both train and predict time. " <>
          Exception.message(e),
        __STACKTRACE__
      )
  end

  @cache :pos_lexical_features

  @doc """
  `features/1` memoised on the lowercased token.

  Training encodes the same corpus once per epoch and EWT's vocabulary is far
  smaller than its token count, so the uncached path would repeat a few tens of
  thousands of ETS lookups per epoch to get the same answers.

  The cache is valid for as long as the loaded lexicon is: it holds what the
  lexicon said, keyed only by word. Nothing reseeds the lexicon inside a training
  or tagging run, and a model built against a different one is caught by
  `lexicon_digest/0` rather than by this.
  """
  @spec features_cached(String.t()) :: [float()]
  def features_cached(token) when is_binary(token) do
    key = String.downcase(String.trim(token))
    table = cache()

    case :ets.lookup(table, key) do
      [{^key, vector}] ->
        vector

      [] ->
        vector = features(key)
        :ets.insert(table, {key, vector})
        vector
    end
  end

  @doc "Drops the memo. For tests that change what the lexicon holds."
  @spec reset_cache() :: :ok
  def reset_cache do
    :ets.delete_all_objects(cache())
    :ok
  end

  defp cache do
    :ets.new(@cache, [:set, :public, :named_table, read_concurrency: true])
  rescue
    ArgumentError -> @cache
  end

  @doc """
  `features/1` as a keyword list, for display rather than for the model.
  """
  @spec explain(String.t()) :: [{atom(), float()}]
  def explain(token) when is_binary(token), do: Enum.zip(manifest(), features(token))

  @doc """
  A digest of the lexicon this channel is reading.

  A model's features mean what they mean because of the lexicon that produced
  them. Reseeding it changes the vector for the same token with nothing in the
  model to say so, so the digest is recorded at training time and checked at load.

  It covers the channel's own shape and the lexicon's size — words, synsets and
  morphological exceptions — deliberately excluding `load_time_ms`, which differs
  on every boot. So it catches a reseed or a changed feature layout, and does not
  catch a content change that leaves every count identical. Hashing 147,477 words
  on every load would, and is not worth its cost for a signal this is adequate
  for.
  """
  @spec lexicon_digest() :: String.t()
  def lexicon_digest do
    stats = Lexicon.stats()

    [
      width(),
      Enum.join(manifest(), ","),
      Map.get(stats, :word_count, 0),
      Map.get(stats, :synset_count, 0),
      Map.get(stats, :morph_count, 0),
      length(@closed_classes)
    ]
    |> Enum.join("|")
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> binary_part(0, 16)
  end

  defp bucket(pos), do: Map.get(@satellite_of, pos, pos)

  defp share(_part, 0), do: 0.0
  defp share(part, total), do: part / total

  defp flag(true), do: 1.0
  defp flag(false), do: 0.0
end
