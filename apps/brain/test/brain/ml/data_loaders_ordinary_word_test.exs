defmodule Brain.ML.DataLoadersOrdinaryWordTest do
  @moduledoc """
  What makes a word an ordinary English word for the gazetteer's loaders
  (`Brain.ML.DataLoaders.ordinary_word?/2`): the owned lexicon's counts of its
  uses. Each test writes to an isolated store and passes it in; the global
  store is never written, because its cache would outlive the sandbox
  rollback.
  """
  use Brain.Test.BrainCase, async: false

  alias Brain.Lexicon.UserDefined
  alias Brain.ML.DataLoaders

  setup do
    prefix = :"ordinary_word_test_#{System.unique_integer([:positive])}"
    store = :"#{prefix}_store"
    start_supervised!({UserDefined, name: store, table_prefix: prefix}, id: store)
    {:ok, store: store}
  end

  defp usage(word, key, ref, count, source),
    do: %{word: word, kind: "property", key: key, ref: ref, value: %{"count" => count}, source: source}

  test "a word used more often as an ordinary word than as any entity type is one", %{store: store} do
    {:ok, _} =
      UserDefined.put_facts(
        [
          usage("in", "ordinary_usage", "adv", 19, "seed:semcor"),
          usage("in", "ordinary_usage", "closed_class", 1000, "authored"),
          usage("ok", "ordinary_usage", "adv", 1, "seed:semcor"),
          usage("dc", "ordinary_usage", "noun", 1, "seed:semcor"),
          usage("dc", "sense_usage", "location", 7, "seed:semcor")
        ],
        store
      )

    assert DataLoaders.ordinary_word?("in", store)
    assert DataLoaders.ordinary_word?("IN", store)
    assert DataLoaders.ordinary_word?("ok", store)

    # Used as a place seven times for every ordinary use.
    refute DataLoaders.ordinary_word?("dc", store)

    # A code the lexicon records no use of is no ordinary word.
    refute DataLoaders.ordinary_word?("tx", store)
  end
end
