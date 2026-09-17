defmodule Atlas.LexiconTest do
  use Atlas.DataCase, async: false

  alias Atlas.Lexicon
  alias Atlas.Schemas.LexiconFact

  defp fact(overrides \\ %{}) do
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

  describe "changeset validation" do
    test "accepts a well-formed fact" do
      assert LexiconFact.changeset(%LexiconFact{}, fact()).valid?
    end

    test "rejects an unknown kind" do
      changeset = LexiconFact.changeset(%LexiconFact{}, fact(%{kind: "opinion"}))
      assert errors_on(changeset)[:kind]
    end

    test "rejects a word that is not already normalised" do
      for word <- ["Unable", " unable", "unable ", ""] do
        changeset = LexiconFact.changeset(%LexiconFact{}, fact(%{word: word}))
        refute changeset.valid?, "expected #{inspect(word)} to be rejected"
      end
    end

    test "accepts seed sources and the three plain sources" do
      for source <- ["seed:wordnet", "seed:domain", "derived", "clarified", "authored"] do
        assert LexiconFact.changeset(%LexiconFact{}, fact(%{source: source})).valid?,
               "expected #{source} to be accepted"
      end
    end

    test "rejects any other source" do
      for source <- ["seed:", "wordnet", "learned", ""] do
        changeset = LexiconFact.changeset(%LexiconFact{}, fact(%{source: source}))
        refute changeset.valid?, "expected #{inspect(source)} to be rejected"
      end
    end

    test "rejects confidence outside 0..1 and negative frequency" do
      refute LexiconFact.changeset(%LexiconFact{}, fact(%{confidence: 1.5})).valid?
      refute LexiconFact.changeset(%LexiconFact{}, fact(%{confidence: -0.1})).valid?
      refute LexiconFact.changeset(%LexiconFact{}, fact(%{frequency: -1})).valid?
    end

    test "rejects a nil ref, which would slip past the unique index" do
      changeset = LexiconFact.changeset(%LexiconFact{}, fact(%{ref: nil}))
      assert errors_on(changeset)[:ref]
    end
  end

  describe "upsert_fact/1" do
    test "inserts a new fact" do
      assert {:ok, stored} = Lexicon.upsert_fact(fact())
      assert stored.word == "unable"
      assert stored.ref == ""
      assert stored.value == %{"kind" => "morphological", "strength" => 0.35}
    end

    test "writing the same identity again updates in place" do
      {:ok, first} = Lexicon.upsert_fact(fact())

      {:ok, second} =
        Lexicon.upsert_fact(fact(%{value: %{"kind" => "morphological", "strength" => 0.5}}))

      assert second.id == first.id
      assert second.value["strength"] == 0.5
      assert Lexicon.count_facts() == 1
    end

    test "a learned fact sits beside the seeded fact it contradicts" do
      {:ok, _} = Lexicon.upsert_fact(fact())
      {:ok, _} = Lexicon.upsert_fact(fact(%{source: "derived", value: %{"kind" => "none"}}))

      facts = Lexicon.list_facts_for_word("unable")
      assert length(facts) == 2
      assert Enum.map(facts, & &1.source) |> Enum.sort() == ["derived", "seed:wordnet"]
    end

    test "facts with different refs are distinct" do
      {:ok, _} =
        Lexicon.upsert_fact(%{
          word: "dog",
          kind: "relation",
          key: "hypernym",
          ref: "canine",
          value: %{},
          source: "seed:wordnet"
        })

      {:ok, _} =
        Lexicon.upsert_fact(%{
          word: "dog",
          kind: "relation",
          key: "hypernym",
          ref: "domestic animal",
          value: %{},
          source: "seed:wordnet"
        })

      assert length(Lexicon.list_facts_for_word("dog")) == 2
    end

    test "re-writing a fact does not reset its observed frequency" do
      {:ok, stored} = Lexicon.upsert_fact(fact())
      {:ok, _} = Lexicon.update_fact(stored, %{frequency: 7})
      {:ok, rewritten} = Lexicon.upsert_fact(fact())

      assert rewritten.frequency == 7
    end

    test "returns the changeset for an invalid fact" do
      assert {:error, changeset} = Lexicon.upsert_fact(fact(%{kind: "nope"}))
      assert errors_on(changeset)[:kind]
    end
  end

  describe "upsert_facts/1" do
    test "writes a batch" do
      batch = for w <- ~w(unable impossible hopeless), do: fact(%{word: w})

      assert {:ok, 3} = Lexicon.upsert_facts(batch)
      assert Lexicon.count_facts("seed:wordnet") == 3
    end

    test "is idempotent" do
      batch = for w <- ~w(unable impossible hopeless), do: fact(%{word: w})

      {:ok, _} = Lexicon.upsert_facts(batch)
      {:ok, _} = Lexicon.upsert_facts(batch)

      assert Lexicon.count_facts() == 3
    end

    test "applies schema defaults that insert_all would otherwise skip" do
      {:ok, 1} =
        Lexicon.upsert_facts([
          %{word: "unable", kind: "property", key: "negation", source: "seed:wordnet"}
        ])

      [stored] = Lexicon.list_facts_for_word("unable")
      assert stored.ref == ""
      assert stored.value == %{}
      assert stored.confidence == 1.0
      assert stored.frequency == 1
      refute stored.archived
      assert %DateTime{} = stored.inserted_at
    end

    test "one invalid entry writes nothing and names the entry" do
      batch = [fact(%{word: "unable"}), fact(%{word: "Impossible"}), fact(%{word: "hopeless"})]

      assert {:error, {1, changeset}} = Lexicon.upsert_facts(batch)
      assert errors_on(changeset)[:word]
      assert Lexicon.count_facts() == 0
    end

    test "writes more than one chunk" do
      batch = for i <- 1..2_500, do: fact(%{word: "word#{i}"})

      assert {:ok, 2_500} = Lexicon.upsert_facts(batch)
      assert Lexicon.count_facts() == 2_500
    end

    test "an empty batch writes nothing" do
      assert {:ok, 0} = Lexicon.upsert_facts([])
    end
  end

  describe "list_facts/0" do
    test "includes archived facts, so decay state survives a restart" do
      {:ok, stored} = Lexicon.upsert_fact(fact())
      {:ok, _} = Lexicon.update_fact(stored, %{archived: true})

      assert [%LexiconFact{archived: true}] = Lexicon.list_facts()
    end
  end
end
