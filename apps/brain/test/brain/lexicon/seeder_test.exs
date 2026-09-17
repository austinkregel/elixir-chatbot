defmodule Brain.Lexicon.SeederTest do
  @moduledoc """
  The seeder turns WordNet into facts the brain owns. Writes go to an isolated
  store so the global one is untouched by the sandbox rollback.
  """
  use Brain.Test.BrainCase, async: false

  alias Brain.Lexicon.Seeder
  alias Brain.Lexicon.UserDefined

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

  describe "seed_all/1" do
    test "reports what it wrote", %{store: store} do
      assert {:ok, %{negation: count}} = Seeder.seed_all(store: store)
      assert count > 1_000
    end
  end
end
