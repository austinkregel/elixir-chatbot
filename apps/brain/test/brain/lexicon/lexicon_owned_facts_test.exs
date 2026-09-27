defmodule Brain.Lexicon.OwnedFactsTest do
  @moduledoc """
  How the brain's own facts extend WordNet's answers through `Brain.Lexicon`.

  Each test writes to an isolated store and passes it as `store:`. The global
  store is never written, because its cache would outlive the sandbox rollback.
  """
  use Brain.Test.BrainCase, async: false

  alias Brain.Lexicon
  alias Brain.Lexicon.UserDefined
  alias Brain.ML.Lexicon, as: WordNet

  setup do
    prefix = :"owned_test_#{System.unique_integer([:positive])}"
    store = :"#{prefix}_store"
    start_supervised!({UserDefined, name: store, table_prefix: prefix}, id: store)
    {:ok, store: store}
  end

  defp relation(word, key, target, extra \\ %{}) do
    %{
      word: word,
      kind: "relation",
      key: key,
      ref: target,
      value: extra,
      source: "seed:domain"
    }
  end

  describe "list-valued lookups" do
    test "owned synonyms follow WordNet's, without duplicates or the word itself", %{
      store: store
    } do
      {:ok, _} =
        UserDefined.put_facts(
          [
            relation("happy", "synonym", "chuffed"),
            relation("happy", "synonym", "glad"),
            relation("happy", "synonym", "happy")
          ],
          store
        )

      base = WordNet.synonyms("happy")
      assert "glad" in base

      assert Lexicon.synonyms("happy", nil, store: store) == base ++ ["chuffed"]
    end

    test "a POS filter applies to owned relations through their pos", %{store: store} do
      {:ok, _} =
        UserDefined.put_facts(
          [
            relation("run", "synonym", "zoom", %{"pos" => "verb"}),
            relation("run", "synonym", "streak", %{"pos" => "noun"})
          ],
          store
        )

      verbs = Lexicon.synonyms("run", :verb, store: store)
      assert "zoom" in verbs
      refute "streak" in verbs

      unfiltered = Lexicon.synonyms("run", nil, store: store)
      assert "zoom" in unfiltered and "streak" in unfiltered
    end

    test "an owned antonym is returned where WordNet has none", %{store: store} do
      assert WordNet.antonyms("smarthome") == []
      {:ok, _} = UserDefined.put_fact(relation("smarthome", "antonym", "unplugged"), store)

      assert Lexicon.antonyms("smarthome", store: store) == ["unplugged"]
    end

    test "an owned hypernym extends WordNet's", %{store: store} do
      {:ok, _} = UserDefined.put_fact(relation("weather_condition", "hypernym", "weather"), store)

      assert Lexicon.hypernyms("weather_condition", nil, store: store) == ["weather"]
    end

    test "a relation of another kind is not mixed in", %{store: store} do
      {:ok, _} = UserDefined.put_fact(relation("happy", "antonym", "glum"), store)

      refute "glum" in Lexicon.synonyms("happy", nil, store: store)
      assert "glum" in Lexicon.antonyms("happy", store: store)
    end
  end

  describe "pos/2" do
    test "includes the POS of the brain's own senses", %{store: store} do
      {:ok, :created} =
        UserDefined.add_sense("zorbl", %{pos: :noun, coarse_class: :noun_artifact}, name: store)

      assert WordNet.pos("zorbl") == []
      assert Lexicon.pos("zorbl", store: store) == [:noun]
    end
  end

  describe "expand_with_synonyms/3" do
    test "an owned synonym can reach the vocabulary", %{store: store} do
      {:ok, _} = UserDefined.put_fact(relation("zorbl", "synonym", "gadget"), store)

      assert Lexicon.expand_with_synonyms(["zorbl"], %{"gadget" => 0}, store: store) ==
               ["zorbl", "gadget"]
    end
  end

  describe "scalar lookups" do
    test "definition/2 stays WordNet's even when the brain holds facts about the word", %{
      store: store
    } do
      {:ok, _} = UserDefined.put_fact(relation("happy", "synonym", "chuffed"), store)

      assert Lexicon.definition("happy") == WordNet.definition("happy")
    end
  end

  describe "owned_facts/2" do
    test "returns the facts with their source", %{store: store} do
      {:ok, _} = UserDefined.put_fact(relation("happy", "synonym", "chuffed"), store)

      assert [fact] = Lexicon.owned_facts("happy", store: store)
      assert fact.source == "seed:domain"
      assert fact.ref == "chuffed"
    end
  end
end
