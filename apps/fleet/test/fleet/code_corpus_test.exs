defmodule Fleet.CodeCorpusTest do
  @moduledoc """
  The code corpus as durable, world-scoped storage.

  These cover `Brain.Code.CodeGazetteer`, which lives in `brain` — they are
  here because brain's own ExUnit suite cannot boot in this tree (its
  `test_helper.exs` requires gold-standard fixtures that are absent), while
  Fleet's runs against the live stack with an Atlas sandbox. An untested
  storage layer would be worse than an oddly-placed test.

  What matters here is what changed when the corpus moved out of ETS: it
  survives the process that wrote it, re-indexing is idempotent, and one world's
  corpus cannot be seen from another.
  """
  use Fleet.FleetCase

  alias Brain.Code.CodeGazetteer

  @moduletag :integration

  defp world, do: "test-corpus-#{System.unique_integer([:positive])}"

  defp symbol(attrs \\ %{}) do
    Map.merge(
      %{
        name: "calculate_tax",
        qualified_name: "Billing.calculate_tax",
        entity_type: "code.function",
        language: :elixir,
        file_path: "lib/billing.ex",
        line: 42,
        metadata: %{arity: 2, visibility: :public}
      },
      attrs
    )
  end

  describe "storage round trip" do
    test "a symbol written is a symbol read, with its shape preserved" do
      w = world()
      assert {:ok, _} = CodeGazetteer.add_symbol(w, symbol())

      assert {:ok, [found]} = CodeGazetteer.lookup(w, "calculate_tax")
      assert found.qualified_name == "Billing.calculate_tax"
      assert found.file_path == "lib/billing.ex"
      assert found.line == 42
      # `:language` is an atom and metadata keys are atoms to every caller; the
      # move to a database must not leak through as strings.
      assert found.language == :elixir
      assert found.metadata[:arity] == 2
    end

    test "lookup is case-insensitive, as it was under ETS" do
      w = world()
      CodeGazetteer.add_symbol(w, symbol())

      assert {:ok, [_]} = CodeGazetteer.lookup(w, "CALCULATE_TAX")
      assert {:ok, _} = CodeGazetteer.lookup_qualified(w, "billing.CALCULATE_TAX")
    end

    test "a symbol missing a required field is dropped without failing its batch" do
      w = world()

      assert {:ok, 2} =
               CodeGazetteer.add_symbols(w, [
                 symbol(),
                 %{name: "nameless"},
                 symbol(%{qualified_name: "Billing.other", name: "other", line: 50})
               ])

      assert CodeGazetteer.stats(w).symbols == 2
    end
  end

  describe "re-indexing" do
    # The ETS version appended to a list, so analysing a directory twice
    # silently doubled the corpus and every count downstream.
    test "ingesting the same symbols twice leaves the corpus unchanged" do
      w = world()
      batch = [symbol(), symbol(%{name: "other", qualified_name: "Billing.other", line: 50})]

      CodeGazetteer.add_symbols(w, batch)
      assert CodeGazetteer.stats(w).symbols == 2

      CodeGazetteer.add_symbols(w, batch)
      assert CodeGazetteer.stats(w).symbols == 2
    end

    test "repeats within one batch collapse, since a batch cannot conflict with itself" do
      w = world()
      assert {:ok, 1} = CodeGazetteer.add_symbols(w, [symbol(), symbol(), symbol()])
    end

    # Identity is location. Over-deduping on name alone would erase real
    # symbols — a name legitimately recurs across files, and `use` appears once
    # per import in every module in the tree.
    test "the same name at different locations is kept as distinct symbols" do
      w = world()

      CodeGazetteer.add_symbols(w, [
        symbol(%{line: 42}),
        symbol(%{line: 99}),
        symbol(%{file_path: "lib/other.ex", line: 42})
      ])

      assert {:ok, found} = CodeGazetteer.lookup(w, "calculate_tax")
      assert match?([_, _, _], found)
    end

    test "re-indexing updates a symbol in place rather than duplicating it" do
      w = world()
      CodeGazetteer.add_symbol(w, symbol(%{metadata: %{arity: 2}}))
      CodeGazetteer.add_symbol(w, symbol(%{metadata: %{arity: 3}}))

      assert {:ok, [found]} = CodeGazetteer.lookup(w, "calculate_tax")
      assert found.metadata[:arity] == 3
    end
  end

  describe "world isolation" do
    test "one world's corpus is invisible from another" do
      a = world()
      b = world()
      CodeGazetteer.add_symbol(a, symbol())

      assert {:ok, [_]} = CodeGazetteer.lookup(a, "calculate_tax")
      assert :not_found = CodeGazetteer.lookup(b, "calculate_tax")
      assert CodeGazetteer.search(b, "calculate") == []
      assert CodeGazetteer.stats(b).symbols == 0
    end

    test "clearing a world leaves its neighbours alone" do
      a = world()
      b = world()
      CodeGazetteer.add_symbol(a, symbol())
      CodeGazetteer.add_symbol(b, symbol())

      CodeGazetteer.clear_world(a)

      assert CodeGazetteer.stats(a).symbols == 0
      assert CodeGazetteer.stats(b).symbols == 1
    end
  end

  describe "search" do
    test "matches on bare and qualified name, and respects filters" do
      w = world()

      CodeGazetteer.add_symbols(w, [
        symbol(),
        symbol(%{name: "Billing", qualified_name: "Billing", entity_type: "code.class", line: 1})
      ])

      assert match?([_, _], CodeGazetteer.search(w, "billing"))
      assert [only] = CodeGazetteer.search(w, "billing", entity_type: "code.class")
      assert only.entity_type == "code.class"
    end

    # `_` and `%` are LIKE wildcards, and code identifiers are full of
    # underscores. Unescaped, a search for "calculate_tax" would also match
    # "calculateXtax" — plausible-looking results that are simply wrong.
    test "LIKE wildcards in a query are escaped, not honoured" do
      w = world()

      CodeGazetteer.add_symbols(w, [
        symbol(%{name: "calculate_tax", qualified_name: "A.calculate_tax"}),
        symbol(%{name: "calculateXtax", qualified_name: "A.calculateXtax", line: 43})
      ])

      names = w |> CodeGazetteer.search("calculate_tax") |> Enum.map(& &1.name)
      assert names == ["calculate_tax"]
    end

    test "limit is honoured" do
      w = world()
      symbols = for i <- 1..10, do: symbol(%{qualified_name: "Billing.f#{i}", line: i})
      CodeGazetteer.add_symbols(w, symbols)

      assert match?([_, _, _], CodeGazetteer.search(w, "calculate", limit: 3))
    end
  end

  describe "relations" do
    test "round trip, deduplicated" do
      w = world()
      CodeGazetteer.add_relation(w, "A.run", :calls, "B.helper")
      CodeGazetteer.add_relation(w, "A.run", :calls, "B.helper")
      CodeGazetteer.add_relation(w, "A.run", :calls, "C.other")

      assert CodeGazetteer.get_relations(w, "A.run", :calls) == ["B.helper", "C.other"]
      assert CodeGazetteer.stats(w).relations == 2
    end

    test "an unknown relation kind is refused rather than stored" do
      w = world()
      CodeGazetteer.add_relation(w, "A.run", :teleports_to, "B.helper")

      assert CodeGazetteer.get_relations(w, "A.run", :teleports_to) == []
      assert CodeGazetteer.stats(w).relations == 0
    end
  end

  describe "stats" do
    # Derived from the rows rather than from running counters, so they cannot
    # drift from the corpus they describe.
    test "counts distinct files and languages, not writes" do
      w = world()

      CodeGazetteer.add_symbols(w, [
        symbol(%{file_path: "a.ex", line: 1}),
        symbol(%{file_path: "a.ex", line: 2, qualified_name: "Billing.two"}),
        symbol(%{file_path: "b.py", line: 1, language: :python, qualified_name: "billing.three"})
      ])

      stats = CodeGazetteer.stats(w)
      assert stats.symbols == 3
      assert stats.files == 2
      assert stats.languages == 2
      assert stats.file_set == MapSet.new(["a.ex", "b.py"])
    end

    test "an unindexed world reports zero rather than raising" do
      assert CodeGazetteer.stats(world()) == %{
               symbols: 0,
               relations: 0,
               files: 0,
               languages: 0,
               file_set: MapSet.new(),
               language_set: MapSet.new()
             }
    end
  end
end
