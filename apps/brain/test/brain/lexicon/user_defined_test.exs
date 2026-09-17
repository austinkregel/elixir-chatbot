defmodule Brain.Lexicon.UserDefinedTest do
  @moduledoc """
  Each test runs its own store instance with its own ETS table, under the
  test's Atlas sandbox. Writes roll back with the sandbox, so using the global
  store here would leave facts cached in ETS that no longer exist in Atlas.
  """
  use Brain.Test.BrainCase, async: false

  alias Atlas.Lexicon, as: Facts
  alias Brain.Lexicon.UserDefined

  defp start_store do
    prefix = :"ud_test_#{System.unique_integer([:positive])}"
    name = :"#{prefix}_store"
    start_supervised!({UserDefined, name: name, table_prefix: prefix}, id: name)
    name
  end

  defp negation(overrides \\ %{}) do
    Map.merge(
      %{
        word: "unable",
        kind: "property",
        key: "negation",
        value: %{"kind" => "morphological", "strength" => 0.35},
        source: "seed:wordnet"
      },
      overrides
    )
  end

  defp sense(overrides \\ %{}) do
    Map.merge(%{pos: :noun, coarse_class: :noun_animal, centroid: [1.0, 0.0]}, overrides)
  end

  describe "boot" do
    test "loads the facts already in Atlas" do
      {:ok, _} = Facts.upsert_fact(negation())

      store = start_store()

      assert [fact] = UserDefined.facts("unable", [], store)
      assert fact.value["strength"] == 0.35
      assert UserDefined.fact_count(store) == 1
    end

    test "starts empty when Atlas holds nothing" do
      store = start_store()
      assert UserDefined.fact_count(store) == 0
      assert UserDefined.facts("unable", [], store) == []
    end

    test "a store that is not running fails loudly instead of reading as empty" do
      assert catch_exit(UserDefined.facts("unable", [], :no_such_lexicon_store))
    end
  end

  describe "put_fact/2 and put_facts/2" do
    test "writes to Atlas and to the cache" do
      store = start_store()

      assert {:ok, 1} = UserDefined.put_fact(negation(), store)
      assert [_] = Facts.list_facts_for_word("unable")
      assert [_] = UserDefined.facts("unable", [], store)
    end

    test "an invalid fact is rejected and nothing is cached" do
      store = start_store()

      assert {:error, changeset} = UserDefined.put_fact(negation(%{word: "Unable"}), store)
      refute changeset.valid?
      assert UserDefined.fact_count(store) == 0
      assert Facts.count_facts() == 0
    end

    test "one invalid entry in a batch writes nothing anywhere" do
      store = start_store()
      batch = [negation(), negation(%{word: "impossible", kind: "bogus"})]

      assert {:error, {1, _changeset}} = UserDefined.put_facts(batch, store)
      assert UserDefined.fact_count(store) == 0
      assert Facts.count_facts() == 0
    end

    test "rewriting a fact updates the cached copy in place" do
      store = start_store()

      {:ok, 1} = UserDefined.put_fact(negation(), store)
      {:ok, 1} = UserDefined.put_fact(negation(%{value: %{"strength" => 0.9}}), store)

      assert [fact] = UserDefined.facts("unable", [], store)
      assert fact.value == %{"strength" => 0.9}
    end
  end

  describe "facts/3" do
    test "filters by kind, key and source" do
      store = start_store()

      {:ok, _} =
        UserDefined.put_facts(
          [
            negation(),
            negation(%{source: "derived", value: %{"kind" => "none"}}),
            %{
              word: "unable",
              kind: "relation",
              key: "antonym",
              ref: "able",
              source: "seed:wordnet"
            }
          ],
          store
        )

      assert length(UserDefined.facts("unable", [], store)) == 3
      assert length(UserDefined.facts("unable", [kind: "property"], store)) == 2
      assert [_] = UserDefined.facts("unable", [kind: "relation", key: "antonym"], store)
      assert [derived] = UserDefined.facts("unable", [source: "derived"], store)
      assert derived.value == %{"kind" => "none"}
    end

    test "a learned fact and the seeded fact it contradicts are both returned" do
      store = start_store()

      {:ok, _} =
        UserDefined.put_facts([negation(), negation(%{source: "clarified", value: %{}})], store)

      sources = UserDefined.facts("unable", [key: "negation"], store) |> Enum.map(& &1.source)
      assert Enum.sort(sources) == ["clarified", "seed:wordnet"]
    end

    test "archived facts are hidden unless asked for" do
      {:ok, stored} = Facts.upsert_fact(negation())
      {:ok, _} = Facts.update_fact(stored, %{archived: true})
      store = start_store()

      assert UserDefined.facts("unable", [], store) == []
      assert [_] = UserDefined.facts("unable", [include_archived: true], store)
    end

    test "looks words up case-insensitively" do
      store = start_store()
      {:ok, _} = UserDefined.put_fact(negation(), store)

      assert [_] = UserDefined.facts("UNABLE", [], store)
    end
  end

  describe "sense view" do
    test "only words carrying a sense are entries" do
      store = start_store()
      {:ok, _} = UserDefined.put_fact(negation(), store)

      assert UserDefined.get("unable", store) == nil
      refute UserDefined.has_entry?("unable", store)
      assert UserDefined.all(store) == []
      assert UserDefined.count(store) == 0
    end

    test "a sense map carries the original fields" do
      store = start_store()
      {:ok, :created} = UserDefined.add_sense("zorbl", sense(), name: store)

      assert %{senses: [s]} = UserDefined.get("zorbl", store)
      assert s.pos == :noun
      assert s.coarse_class == :noun_animal
      assert s.centroid == [1.0, 0.0]
      assert s.source == "derived"
      assert s.frequency == 1
      assert is_integer(s.first_seen) and is_integer(s.last_seen)
      refute s.archived

      assert UserDefined.has_entry?("zorbl", store)
      assert UserDefined.count(store) == 1
      assert [{"zorbl", %{senses: [_]}}] = UserDefined.all(store)
    end
  end

  describe "add_sense/3" do
    test "creates, then updates a similar sense, then adds a distinct one" do
      store = start_store()

      assert {:ok, :created} = UserDefined.add_sense("zorbl", sense(), name: store)

      assert {:ok, :updated} =
               UserDefined.add_sense("zorbl", sense(%{centroid: [0.99, 0.01]}), name: store)

      assert {:ok, :new_sense} =
               UserDefined.add_sense("zorbl", sense(%{centroid: [0.0, 1.0]}), name: store)

      %{senses: [first, second]} = UserDefined.get("zorbl", store)
      assert first.frequency == 2
      assert second.frequency == 1
      assert second.centroid == [0.0, 1.0]
    end

    test "persists, so a fresh store sees the sense" do
      store = start_store()
      {:ok, :created} = UserDefined.add_sense("zorbl", sense(), name: store)

      assert %{senses: [_]} = UserDefined.get("zorbl", start_store())
    end

    test "requires a part of speech and a coarse class" do
      store = start_store()

      assert_raise ArgumentError, ~r/:pos/, fn ->
        UserDefined.add_sense("zorbl", %{coarse_class: :noun_animal}, name: store)
      end

      assert_raise ArgumentError, ~r/:coarse_class/, fn ->
        UserDefined.add_sense("zorbl", %{pos: :noun}, name: store)
      end
    end

    test "a sense without a centroid never merges" do
      store = start_store()
      {:ok, :created} = UserDefined.add_sense("zorbl", sense(%{centroid: nil}), name: store)

      assert {:ok, :new_sense} =
               UserDefined.add_sense("zorbl", sense(%{centroid: nil}), name: store)
    end
  end

  describe "record_observation/3" do
    test "bumps frequency and moves the centroid toward the context" do
      store = start_store()
      {:ok, :created} = UserDefined.add_sense("zorbl", sense(), name: store)

      assert {:ok, :updated} =
               UserDefined.record_observation("zorbl", [0.0, 1.0], name: store, ema_alpha: 0.5)

      %{senses: [s]} = UserDefined.get("zorbl", store)
      assert s.frequency == 2
      assert s.centroid == [0.5, 0.5]
    end

    test "reports a word with no sense" do
      store = start_store()
      assert {:error, :not_found} = UserDefined.record_observation("zorbl", [1.0], name: store)
    end
  end

  describe "decay_senses/1" do
    test "halves stale senses and archives those that fall below the threshold" do
      store = start_store()
      {:ok, :created} = UserDefined.add_sense("zorbl", sense(), name: store)
      {:ok, :created} = UserDefined.add_sense("blorp", sense(), name: store)

      [stale] = Facts.list_facts_for_word("zorbl")
      week_ago = DateTime.add(DateTime.utc_now(), -8 * 24 * 3600)
      {:ok, _} = Facts.update_fact(stale, %{frequency: 4, last_observed_at: week_ago})
      {:ok, _} = UserDefined.reload(store)

      assert UserDefined.decay_senses(name: store) == 1

      %{senses: [decayed]} = UserDefined.get("zorbl", store)
      assert decayed.frequency == 2
      refute decayed.archived

      %{senses: [fresh]} = UserDefined.get("blorp", store)
      assert fresh.frequency == 1
    end

    test "archives a sense whose frequency drops below the threshold" do
      store = start_store()
      {:ok, :created} = UserDefined.add_sense("zorbl", sense(), name: store)

      [stale] = Facts.list_facts_for_word("zorbl")
      week_ago = DateTime.add(DateTime.utc_now(), -8 * 24 * 3600)
      {:ok, _} = Facts.update_fact(stale, %{last_observed_at: week_ago})
      {:ok, _} = UserDefined.reload(store)

      assert UserDefined.decay_senses(name: store) == 1
      assert [%{archived: true, frequency: 0}] = Facts.list_facts_for_word("zorbl")
    end
  end

  describe "reload/1" do
    test "picks up facts written to Atlas directly" do
      store = start_store()
      {:ok, _} = Facts.upsert_fact(negation())

      assert UserDefined.facts("unable", [], store) == []
      assert {:ok, 1} = UserDefined.reload(store)
      assert [_] = UserDefined.facts("unable", [], store)
    end
  end
end
