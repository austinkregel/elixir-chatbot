defmodule Brain.Lexicon.Seeder do
  @moduledoc """
  Seeds the brain's own lexicon from the corpora it is built on.

  WordNet is a base to start from, not the answer. Everything derived from it
  is written into `Brain.Lexicon.UserDefined` as facts carrying
  `source: "seed:wordnet"`, so the brain can correct, extend and decay them
  later. Nothing reads WordNet for these facts at runtime.

  Seeding is idempotent — facts are upserted on their identity — so it is safe
  to run on every deploy. `mix atlas.seed` runs it.

  ## Negation

  A word negates morphologically when it is the antonym of another word **and**
  is that word plus a negative affix. The affix test is what separates negation
  from plain opposition: `hot` and `cold` are antonyms and neither negates, so
  the pair is rejected.

  Restricted to adjectives, adverbs and verbs. Nominalisations (`inability`,
  `impossibility`) are excluded: they name a negated concept rather than negate
  a clause, and admitting them would make every mention of a shortcoming a
  negation.

  Closed-class negators (`not`, `never`, `no`, `neither`) are not seeded from
  here. WordNet holds no function words, so they remain in
  `priv/knowledge/linguistic.json`.

  ## Sense usage

  How often a word is used as each entity type, counted from SemCor -- the
  sense-tagged corpus behind WordNet's per-sense `tag_count`. Each counted
  sense is assigned the entity type of its nearest anchor
  (`Brain.Lexicon.TypeAnchors`), and the counts are summed per word and type:
  `thursday` is used 13 times as a day of the week and never as a band.

  Written as `sense_usage` property facts with `source: "seed:semcor"`, one
  per word and type, the type in `ref` and the count in `value`.

  Counted senses that reach no anchor are the word's ordinary uses, written
  as `ordinary_usage` facts: `ref: "noun"` for nouns and `ref: "other_pos"`
  for verbs, adjectives and adverbs. `nice` is used 29 times as an adjective
  and never as the city.

  A word with no counted use of some kind gets no fact for it: zero is what
  an absent fact already says. These are the starting point for the running
  statistics the brain keeps from its own conversations, not the final word.

  ## Closed-class words

  WordNet counts no function words, so "of" or "the" would otherwise have no
  ordinary use at all and read as a name that happens to match (Of, a town in
  Turkey). Every word of `Brain.Lexicon.ClosedClass` gets an `ordinary_usage`
  fact with `ref: "closed_class"`, `source: "authored"`, and the count
  configured as `config :brain, :closed_class_ordinary_count` -- an estimate
  until measured frequencies replace it.
  """

  require Logger

  alias Brain.Lexicon.ClosedClass
  alias Brain.Lexicon.TypeAnchors
  alias Brain.Lexicon.UserDefined
  alias Brain.ML.Lexicon, as: WordNet

  @source "seed:wordnet"
  @semcor_source "seed:semcor"

  # The rule, not the answer: these only decide whether an antonym pair is
  # related by negation rather than by plain opposition. The words themselves
  # come from WordNet.
  @negative_prefixes ~w(un im in ir il non dis)
  @negative_suffix "less"

  # POS atoms as Brain.ML.Lexicon returns them. Nouns are excluded; see the
  # moduledoc.
  @clause_level_pos [:adj, :adj_satellite, :adv, :verb]

  @doc """
  Seeds every fact the brain derives from its base corpora.

  Returns `{:ok, %{negation: count, sense_usage: count, closed_class: count}}`.
  """
  @spec seed_all(keyword()) :: {:ok, map()}
  def seed_all(opts \\ []) do
    {:ok, negation} = seed_negation(opts)
    {:ok, sense_usage} = seed_sense_usage(opts)
    {:ok, closed_class} = seed_closed_class(opts)
    {:ok, %{negation: negation, sense_usage: sense_usage, closed_class: closed_class}}
  end

  @doc """
  Writes one `ordinary_usage` fact for every closed-class word. See the
  moduledoc.

  Returns `{:ok, count}`.
  """
  @spec seed_closed_class(keyword()) :: {:ok, non_neg_integer()}
  def seed_closed_class(opts \\ []) do
    store = Keyword.get(opts, :store, UserDefined)

    count =
      Keyword.get_lazy(opts, :count, fn -> Application.fetch_env!(:brain, :closed_class_ordinary_count) end)

    facts =
      Enum.map(ClosedClass.words(), fn word ->
        %{
          word: word,
          kind: "property",
          key: "ordinary_usage",
          ref: "closed_class",
          value: %{"count" => count},
          source: "authored"
        }
      end)

    case UserDefined.put_facts(facts, store) do
      {:ok, written} ->
        Logger.info("Lexicon.Seeder: seeded #{written} closed-class usage facts")
        {:ok, written}

      {:error, {index, changeset}} ->
        raise "Lexicon.Seeder: refusing to seed, fact #{index} is invalid: " <>
                inspect(changeset.errors)
    end
  end

  @doc """
  Writes the `sense_usage` and `ordinary_usage` property facts SemCor
  supports for every word. See the moduledoc.

  Returns `{:ok, count}`.
  """
  @spec seed_sense_usage(keyword()) :: {:ok, non_neg_integer()}
  def seed_sense_usage(opts \\ []) do
    store = Keyword.get(opts, :store, UserDefined)

    facts =
      opts
      |> derive_sense_usage()
      |> Enum.flat_map(fn {word, usage} ->
        usage_facts(word, "sense_usage", usage.types) ++
          usage_facts(word, "ordinary_usage", usage.ordinary)
      end)

    case UserDefined.put_facts(facts, store) do
      {:ok, count} ->
        Logger.info("Lexicon.Seeder: seeded #{count} SemCor sense usage facts")
        {:ok, count}

      {:error, {index, changeset}} ->
        raise "Lexicon.Seeder: refusing to seed, fact #{index} is invalid: " <>
                inspect(changeset.errors)
    end
  end

  defp usage_facts(word, key, counts) do
    Enum.map(counts, fn {ref, count} ->
      %{
        word: word,
        kind: "property",
        key: key,
        ref: ref,
        value: %{"count" => count},
        source: @semcor_source
      }
    end)
  end

  @doc """
  Returns `{word, %{types: %{entity_type => count}, ordinary: %{"noun" |
  "other_pos" => count}}}` for every word with at least one SemCor-counted
  sense, without writing anything.

  Only counted senses are classified: a sense SemCor never saw adds nothing
  to any count.
  """
  @spec derive_sense_usage(keyword()) :: [{String.t(), %{types: map(), ordinary: map()}}]
  def derive_sense_usage(opts \\ []) do
    anchors = Keyword.get_lazy(opts, :anchors, &TypeAnchors.load!/0)

    WordNet.words()
    |> Enum.sort()
    |> Enum.flat_map(fn word ->
      usage =
        word
        |> WordNet.senses()
        |> Enum.filter(&(&1.tag_count > 0))
        |> Enum.reduce(%{types: %{}, ordinary: %{}}, fn sense, acc ->
          {bucket, ref} = classify_sense(sense, anchors)
          update_in(acc, [bucket], &Map.update(&1, ref, sense.tag_count, fn n -> n + sense.tag_count end))
        end)

      if usage == %{types: %{}, ordinary: %{}}, do: [], else: [{word, usage}]
    end)
  end

  defp classify_sense(%{pos: :noun, synset_id: synset_id}, anchors) do
    case TypeAnchors.type_of_sense(synset_id, anchors) do
      {:ok, type} -> {:types, type}
      :unanchored -> {:ordinary, "noun"}
    end
  end

  defp classify_sense(_sense, _anchors), do: {:ordinary, "other_pos"}

  @doc """
  Writes one `negation` property fact for every word WordNet shows to be
  morphologically negated.

  Returns `{:ok, count}`.
  """
  @spec seed_negation(keyword()) :: {:ok, non_neg_integer()}
  def seed_negation(opts \\ []) do
    store = Keyword.get(opts, :store, UserDefined)

    facts =
      derive_negation()
      |> Enum.map(fn {word, root, affix} ->
        %{
          word: word,
          kind: "property",
          key: "negation",
          value: %{"kind" => "morphological", "root" => root, "affix" => affix},
          source: @source
        }
      end)

    case UserDefined.put_facts(facts, store) do
      {:ok, count} ->
        Logger.info("Lexicon.Seeder: seeded #{count} morphological negation facts")
        {:ok, count}

      {:error, {index, changeset}} ->
        raise "Lexicon.Seeder: refusing to seed, fact #{index} is invalid: " <>
                inspect(changeset.errors)
    end
  end

  @doc """
  Returns `{word, root, affix}` for every morphologically negated word WordNet
  knows, without writing anything.
  """
  @spec derive_negation() :: [{String.t(), String.t(), String.t()}]
  def derive_negation do
    WordNet.antonym_pairs()
    |> Enum.flat_map(fn {word, antonym} ->
      case negative_affix(word, antonym) do
        nil -> []
        affix -> if clause_level?(word), do: [{word, antonym, affix}], else: []
      end
    end)
    |> Enum.uniq_by(fn {word, _root, _affix} -> word end)
  end

  defp clause_level?(word) do
    Enum.any?(WordNet.pos(word), &(&1 in @clause_level_pos))
  end

  # Two shapes of affixal negation, anchored differently.
  #
  #   prefix: the word is its antonym plus a prefix -- "unable" / "able".
  #   suffix: the word and its antonym share a stem and the word is that stem
  #           plus "-less" -- "hopeless" / "hopeful" both sit on "hope", so the
  #           antonym is not the root and the prefix test cannot be reused.
  defp negative_affix(word, antonym) do
    cond do
      prefix = Enum.find(@negative_prefixes, &(word == &1 <> antonym)) -> prefix <> "-"
      privative_suffix?(word, antonym) -> "-" <> @negative_suffix
      true -> nil
    end
  end

  defp privative_suffix?(word, antonym) do
    case String.split(word, @negative_suffix) do
      [stem, ""] when byte_size(stem) >= 3 -> String.starts_with?(antonym, stem)
      _ -> false
    end
  end
end
