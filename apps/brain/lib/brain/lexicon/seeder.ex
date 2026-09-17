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
  """

  require Logger

  alias Brain.Lexicon.UserDefined
  alias Brain.ML.Lexicon, as: WordNet

  @source "seed:wordnet"

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

  Returns `{:ok, %{negation: count}}`.
  """
  @spec seed_all(keyword()) :: {:ok, map()}
  def seed_all(opts \\ []) do
    {:ok, negation} = seed_negation(opts)
    {:ok, %{negation: negation}}
  end

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
