defmodule Brain.Lexicon.SeederTest do
  @moduledoc """
  The seeder turns WordNet into facts the brain owns. Writes go to an isolated
  store so the global one is untouched by the sandbox rollback.
  """
  use Brain.Test.BrainCase, async: false

  alias Brain.Lexicon.Seeder
  alias Brain.Lexicon.UserDefined
  alias Brain.ML.Lexicon, as: WordNet

  setup do
    prefix = :"seeder_test_#{System.unique_integer([:positive])}"
    store = :"#{prefix}_store"
    start_supervised!({UserDefined, name: store, table_prefix: prefix}, id: store)
    {:ok, store: store}
  end

  describe "derive_negation/0" do
    setup do
      derived = Seeder.derive_negation()
      {:ok, derived: derived, by_word: Map.new(derived, fn {w, r, a} -> {w, {r, a}} end)}
    end

    test "finds words that are an antonym plus a negative affix", %{by_word: by_word} do
      assert by_word["unable"] == {"able", "un-"}
      assert by_word["impossible"] == {"possible", "im-"}
      assert by_word["incompetent"] == {"competent", "in-"}
      assert by_word["unhappy"] == {"happy", "un-"}
    end

    test "anchors -less on the shared stem, not on the antonym", %{by_word: by_word} do
      # "hopeless" is not "hopeful" plus an affix; both sit on "hope".
      assert by_word["hopeless"] == {"hopeful", "-less"}
      assert by_word["useless"] == {"useful", "-less"}
    end

    test "rejects opposition that is not negation", %{by_word: by_word} do
      # Antonyms, but neither is derived from the other by an affix.
      refute Map.has_key?(by_word, "hot")
      refute Map.has_key?(by_word, "cold")
    end

    test "excludes nominalisations, which name a negated concept rather than negate" do
      # "inability" is only a noun, so it must not be seeded even though it is
      # "ability" plus a negative prefix.
      by_word = Map.new(Seeder.derive_negation(), fn {w, r, a} -> {w, {r, a}} end)

      refute Map.has_key?(by_word, "inability")
      refute Map.has_key?(by_word, "impossibility")
    end

    test "does not invent closed-class negators", %{by_word: by_word} do
      # WordNet holds no function words, so these can only come from the
      # closed-class vocabulary.
      for word <- ~w(not never no neither nor hardly barely without) do
        refute Map.has_key?(by_word, word), "#{word} should not be derived from WordNet"
      end
    end

    test "each word appears once", %{derived: derived} do
      words = Enum.map(derived, fn {w, _r, _a} -> w end)
      assert length(words) == length(Enum.uniq(words))
    end

    test "derives a substantial vocabulary, far beyond the hand-written list", %{
      derived: derived
    } do
      assert length(derived) > 1_000
      assert length(derived) > 50 * length(Brain.LinguisticData.negation_words())
    end
  end

  describe "seed_negation/1" do
    test "writes one property fact per derived word", %{store: store} do
      {:ok, count} = Seeder.seed_negation(store: store)

      assert count == length(Seeder.derive_negation())

      assert [fact] = UserDefined.facts("unable", [kind: "property", key: "negation"], store)
      assert fact.source == "seed:wordnet"
      assert fact.value["kind"] == "morphological"
      assert fact.value["root"] == "able"
      assert fact.value["affix"] == "un-"
    end

    test "is idempotent", %{store: store} do
      {:ok, first} = Seeder.seed_negation(store: store)
      before_count = UserDefined.fact_count(store)

      {:ok, second} = Seeder.seed_negation(store: store)

      assert first == second
      assert UserDefined.fact_count(store) == before_count
    end

    test "seeds nothing for a word that only has an opposite", %{store: store} do
      {:ok, _} = Seeder.seed_negation(store: store)

      assert UserDefined.facts("cold", [key: "negation"], store) == []
      assert UserDefined.facts("hot", [key: "negation"], store) == []
    end
  end

  describe "derive_sense_usage/1" do
    setup do
      {:ok, usage: Map.new(Seeder.derive_sense_usage())}
    end

    # The SemCor count of every counted noun sense of `word` that WordNet
    # places under `anchor_word`, read from WordNet rather than written here.
    defp semcor_count_under(word, anchor_word) do
      WordNet.senses(word)
      |> Enum.filter(fn s ->
        s.pos == :noun and
          Enum.any?(WordNet.synset_ancestors(s.synset_id), fn {_sid, words, _d} -> anchor_word in words end)
      end)
      |> Enum.map(& &1.tag_count)
      |> Enum.sum()
    end

    defp semcor_count_with_pos(word, pos) do
      WordNet.senses(word)
      |> Enum.filter(&(&1.pos in pos))
      |> Enum.map(& &1.tag_count)
      |> Enum.sum()
    end

    test "counts a weekday's uses as a date and never as a band", %{usage: usage} do
      expected = semcor_count_under("thursday", "day of the week")

      assert expected > 0
      assert usage["thursday"].types == %{"sys_date" => expected}
    end

    test "counts a named city's uses as a location", %{usage: usage} do
      assert usage["paris"].types["location"] == semcor_count_under("paris", "location")
      assert usage["paris"].types["location"] > 0
    end

    test "counts uses under no anchor as ordinary, by part of speech", %{usage: usage} do
      # "nice" is used as an adjective, never (in SemCor) as the city.
      adjective_uses = semcor_count_with_pos("nice", [:adj, :adj_satellite])

      assert adjective_uses > 0
      assert usage["nice"].types == %{}
      assert usage["nice"].ordinary == %{"adj" => adjective_uses}
    end

    test "keeps a word's parts of speech apart", %{usage: usage} do
      # "light" is a noun, an adjective and a verb, each counted on its own.
      light = usage["light"].ordinary

      assert light["adj"] == semcor_count_with_pos("light", [:adj, :adj_satellite])
      assert light["verb"] == semcor_count_with_pos("light", [:verb])
      assert light["adj"] > 0 and light["verb"] > 0 and light["noun"] > 0
    end

    test "ignores senses SemCor never counted" do
      # Every Madonna sense has a count of zero, so the word derives nothing.
      assert Enum.all?(WordNet.senses("madonna"), &(&1.tag_count == 0))
      refute List.keymember?(Seeder.derive_sense_usage(), "madonna", 0)
    end

    test "every count is positive and every type is one the anchor table declares", %{usage: usage} do
      declared = Brain.Lexicon.TypeAnchors.load!() |> Map.values() |> MapSet.new()

      for {_word, %{types: types, ordinary: ordinary}} <- usage do
        assert Enum.all?(Map.values(types) ++ Map.values(ordinary), &(&1 > 0))
        assert Enum.all?(Map.keys(types), &MapSet.member?(declared, &1))
        assert Enum.all?(Map.keys(ordinary), &(&1 in ["noun", "verb", "adj", "adv"]))
      end
    end
  end

  describe "seed_sense_usage/1" do
    test "writes one fact per word and type, and per word and part of speech", %{store: store} do
      {:ok, count} = Seeder.seed_sense_usage(store: store)

      expected =
        Seeder.derive_sense_usage()
        |> Enum.map(fn {_w, u} -> map_size(u.types) + map_size(u.ordinary) end)
        |> Enum.sum()

      assert count == expected

      assert [fact] = UserDefined.facts("thursday", [key: "sense_usage"], store)
      assert fact.source == "seed:semcor"
      assert fact.kind == "property"
      assert fact.ref == "sys_date"
      assert fact.value["count"] > 0

      assert [ordinary] = UserDefined.facts("nice", [key: "ordinary_usage"], store)
      assert ordinary.ref == "adj"
    end

    test "retires its own facts that it no longer derives, and only those", %{store: store} do
      stale = %{
        word: "nice",
        kind: "property",
        key: "ordinary_usage",
        ref: "other_pos",
        value: %{"count" => 29},
        source: "seed:semcor"
      }

      other_source = %{stale | source: "derived", ref: "noun"}
      {:ok, _} = UserDefined.put_facts([stale, other_source], store)

      {:ok, _} = Seeder.seed_sense_usage(store: store)

      live = UserDefined.facts("nice", [key: "ordinary_usage"], store)
      refute Enum.any?(live, &(&1.ref == "other_pos"))
      # A fact from another source is not this seed's to retire.
      assert Enum.any?(live, &(&1.source == "derived"))

      archived = UserDefined.facts("nice", [key: "ordinary_usage", include_archived: true], store)
      assert Enum.any?(archived, &(&1.ref == "other_pos" and &1.archived))
    end

    test "is idempotent", %{store: store} do
      {:ok, first} = Seeder.seed_sense_usage(store: store)
      before_count = UserDefined.fact_count(store)

      {:ok, second} = Seeder.seed_sense_usage(store: store)

      assert first == second
      assert UserDefined.fact_count(store) == before_count
    end
  end

  describe "seed_closed_class/1" do
    test "writes one authored ordinary-use fact per closed-class word", %{store: store} do
      {:ok, count} = Seeder.seed_closed_class(store: store, count: 42)

      assert count == length(Brain.Lexicon.ClosedClass.words())

      for word <- ~w(of the all) do
        assert [fact] = UserDefined.facts(word, [key: "ordinary_usage", source: "authored"], store)
        assert fact.ref == "closed_class"
        assert fact.value == %{"count" => 42}
      end
    end

    test "the count is the configured one by default", %{store: store} do
      {:ok, _} = Seeder.seed_closed_class(store: store)
      [fact] = UserDefined.facts("of", [key: "ordinary_usage", source: "authored"], store)

      assert fact.value["count"] == Application.fetch_env!(:brain, :closed_class_ordinary_count)
    end
  end

  describe "seed_all/1" do
    test "reports what it wrote", %{store: store} do
      assert {:ok, %{negation: negation, sense_usage: sense_usage, closed_class: closed_class}} =
               Seeder.seed_all(store: store)

      assert negation > 1_000
      assert sense_usage > 1_000
      assert closed_class == length(Brain.Lexicon.ClosedClass.words())
    end
  end
end
