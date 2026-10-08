defmodule Brain.Lexicon.PragmaticFunctionTest do
  @moduledoc """
  The pragmatic-function axis: what a word *does* in discourse, as opposed to
  its part of speech (`Brain.Lexicon.ClosedClass`) or its sense (WordNet).

  These tests read `priv/knowledge/pragmatic_functions.json` itself wherever the
  answer is "whatever the file says", so seeding a new word is checked without
  editing them. Where a specific behaviour is the point — the orthogonality of
  the two axes, phrase matching, the closed inventory — the expectation is
  written out in full and asserted by equality, not by absence.
  """
  use Brain.Test.BrainCase, async: false

  alias Brain.Lexicon.ClosedClass
  alias Brain.Lexicon.PragmaticFunction, as: PF
  alias Brain.Lexicon.UserDefined

  @path Path.join(:code.priv_dir(:brain), "knowledge/pragmatic_functions.json")

  setup_all do
    {:ok, data: @path |> File.read!() |> Jason.decode!()}
  end

  setup do
    prefix = :"pragmatic_test_#{System.unique_integer([:positive])}"
    store = :"#{prefix}_store"
    start_supervised!({UserDefined, name: store, table_prefix: prefix}, id: store)
    {:ok, store: store}
  end

  defp fact(word, function, source \\ "derived") do
    %{
      word: word,
      kind: "property",
      key: "pragmatic_function",
      ref: function,
      value: %{},
      source: source
    }
  end

  describe "the declared inventory" do
    test "all_functions/0 is exactly the declared function names, sorted", %{data: data} do
      assert PF.all_functions() == data["functions"] |> Map.keys() |> Enum.sort()
    end

    test "every declared name is a function, and describes itself", %{data: data} do
      for {name, spec} <- data["functions"] do
        assert PF.function?(name), name
        assert PF.describe(name) == spec["description"], name
        assert PF.seeded_from(name) == spec["seeded_from"], name
      end
    end

    test "every declared function says where its seeds came from", %{data: data} do
      for {name, spec} <- data["functions"] do
        assert is_list(spec["seeded_from"]) and spec["seeded_from"] != [], name
        for origin <- spec["seeded_from"], do: assert(is_binary(origin) and origin != "", name)
      end
    end

    test "every declared function has seeds, and declared_entries/1 is exactly them", %{
      data: data
    } do
      for {name, _spec} <- data["functions"] do
        seeds = data["seeds"][name]
        assert is_list(seeds) and seeds != [], name
        assert PF.declared_entries(name) == Enum.sort(seeds), name
      end
    end

    test "every seeded entry maps back to the function that seeded it", %{data: data} do
      for {name, entries} <- data["seeds"], entry <- entries do
        assert name in PF.declared_functions(entry), "#{entry} should carry #{name}"
      end
    end

    test "entries/0 is exactly every seeded entry, deduplicated and sorted", %{data: data} do
      declared =
        data["seeds"] |> Map.values() |> List.flatten() |> Enum.uniq() |> Enum.sort()

      assert PF.entries() == declared
    end

    test "max_entry_tokens/0 is the longest declared entry's token count", %{data: data} do
      longest =
        data["seeds"]
        |> Map.values()
        |> List.flatten()
        |> Enum.map(&(&1 |> String.split() |> length()))
        |> Enum.max()

      assert PF.max_entry_tokens() == longest
    end

    test "phrases/0 is exactly the multi-token entries", %{data: data} do
      declared =
        data["seeds"]
        |> Map.values()
        |> List.flatten()
        |> Enum.uniq()
        |> Enum.filter(&String.contains?(&1, " "))
        |> Enum.sort()

      assert PF.phrases() == declared
    end
  end

  describe "a word carries any number of functions" do
    test "ok is a backchannel, an acknowledgment and a confirmation at once" do
      assert PF.declared_functions("ok") == ["acknowledgment", "backchannel", "confirmation"]
      assert PF.declared_functions("okay") == ["acknowledgment", "backchannel", "confirmation"]
    end

    test "lookup is case-insensitive" do
      assert PF.declared_functions("OK") == PF.declared_functions("ok")
      assert PF.declared_functions("Please") == ["politeness"]
    end

    test "an unseeded word carries nothing, which is a stated answer and not an error" do
      assert PF.declared_functions("lightbulb") == []
    end
  end

  describe "the axis is orthogonal to part of speech" do
    # This is the thesis of the module: conflating these two axes is what
    # produced sixteen overlapping word lists. Each word below has a class on
    # one axis and a function on the other, and neither answer implies the other.

    test "could is AUX by class and hedges by function" do
      assert "AUX" in ClosedClass.classes("could")
      assert PF.declared_functions("could") == ["hedge"]
    end

    test "will is AUX by class and commits by function" do
      assert "AUX" in ClosedClass.classes("will")
      assert PF.declared_functions("will") == ["commitment"]
    end

    test "if is SCONJ by class and marks a condition by function" do
      assert "SCONJ" in ClosedClass.classes("if")
      assert PF.declared_functions("if") == ["conditional"]
    end

    test "a content word has a function but no closed class" do
      assert ClosedClass.classes("please") == []
      assert PF.declared_functions("please") == ["politeness"]
    end
  end

  describe "phrase entries are matched as n-grams" do
    test "a politeness phrase and a bare politeness token both count" do
      assert PF.declared_functions_in(~w(could you please turn on the lights)) ==
               %{"hedge" => 1, "politeness" => 2}
    end

    test "two phrases in one utterance are both found" do
      assert PF.declared_functions_in(~w(hold on a bit)) ==
               %{"attention" => 1, "hedge" => 1}
    end

    test "a phrase is matched whole, not by its head" do
      assert PF.declared_functions("kind of") == ["hedge"]
      assert PF.declared_functions("kind") == []
    end

    test "'you are very kind' is not hedged — the kind-of head is not a hedge" do
      # chunk_features.ex:508 token-matches bare `kind` and `sort`, the heads of
      # the declared phrases `kind of` and `sort of`, and so scores both corpus
      # occurrences of `kind` ("you are very kind", "you're so kind") as hedges.
      # The whole map is asserted so the hedge's absence is stated rather than
      # merely unchecked.
      assert PF.declared_functions_in(~w(you are very kind)) ==
               %{"intensifier" => 1, "polar_interrogative" => 1}
    end

    test "'what kind of music' hedges on the phrase, from the n-gram not the token" do
      counts = PF.declared_functions_in(~w(what kind of music))
      assert counts["hedge"] == 1
      assert counts["interrogative"] == 1
    end

    test "each occurrence of a function counts separately" do
      assert PF.declared_functions_in(~w(maybe perhaps possibly)) == %{"hedge" => 3}
    end

    test "an utterance with no function words yields an empty map" do
      assert PF.declared_functions_in(~w(turn the lights on)) == %{}
    end
  end

  describe "the inventory is closed" do
    test "describe/1 raises on an undeclared function and names the file to edit" do
      assert_raise ArgumentError, fn -> PF.describe("hedging") end

      message =
        try do
          PF.describe("hedging")
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "not a declared pragmatic function"
      assert message =~ "apps/brain/priv/knowledge/pragmatic_functions.json"
    end

    test "seeded_from/1 and declared_entries/1 raise on an undeclared function" do
      assert_raise ArgumentError, fn -> PF.seeded_from("hedging") end
      assert_raise ArgumentError, fn -> PF.declared_entries("hedging") end
    end

    test "carries?/3 raises on an undeclared function rather than reading false", %{store: store} do
      assert PF.carries?("could", "hedge", store: store)
      assert_raise ArgumentError, fn -> PF.carries?("could", "hedging", store: store) end
    end

    test "function?/1 answers for a name without raising" do
      assert PF.function?("hedge")
      assert PF.function?("hedging") == false
    end
  end

  describe "membership is open — the owned lexicon expands a function" do
    test "an owned fact adds a function the seeds do not list", %{store: store} do
      assert PF.declared_functions("yup") == []

      {:ok, _} = UserDefined.put_facts([fact("yup", "confirmation")], store)

      assert PF.functions("yup", store: store) == ["confirmation"]
    end

    test "owned functions merge with the declared ones", %{store: store} do
      {:ok, _} = UserDefined.put_facts([fact("ok", "greeting")], store)

      assert PF.functions("ok", store: store) ==
               ["acknowledgment", "backchannel", "confirmation", "greeting"]
    end

    test "a word can own several functions, because the name is the fact's ref", %{store: store} do
      {:ok, _} =
        UserDefined.put_facts(
          [fact("righto", "acknowledgment"), fact("righto", "confirmation")],
          store
        )

      assert PF.functions("righto", store: store) == ["acknowledgment", "confirmation"]
    end

    test "an owned phrase is matched by functions_in/2 alongside declared ones", %{store: store} do
      {:ok, _} = UserDefined.put_facts([fact("no worries", "confirmation")], store)

      # `no` is seeded under both `negation` and `backchannel` — it negates in
      # "no, the other one" and continues in "no, go on" — so it contributes to
      # both counts. The owned bigram `no worries` adds `confirmation` on top,
      # over the same token. All four are candidates; nothing here resolves them.
      assert PF.functions_in(~w(no worries please), store: store) ==
               %{
                 "backchannel" => 1,
                 "confirmation" => 1,
                 "negation" => 1,
                 "politeness" => 1
               }
    end

    test "declared_functions/1 ignores owned facts, so the seed answer stays stable", %{
      store: store
    } do
      {:ok, _} = UserDefined.put_facts([fact("yup", "confirmation")], store)

      assert PF.declared_functions("yup") == []
      assert PF.functions("yup", store: store) == ["confirmation"]
    end

    test "an owned fact naming an undeclared function raises", %{store: store} do
      {:ok, _} = UserDefined.put_facts([fact("yup", "hedging")], store)

      message =
        try do
          PF.functions("yup", store: store)
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "hedging"
      assert message =~ "may add words to a function, never a new function"
    end

    test "an owned fact with no function in its ref raises", %{store: store} do
      {:ok, _} = UserDefined.put_facts([Map.put(fact("yup", ""), :ref, "")], store)

      message =
        try do
          PF.functions("yup", store: store)
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "names no function in its `ref`"
    end

    test "with no owned facts, functions/2 equals declared_functions/1", %{store: store} do
      for entry <- PF.entries() do
        assert PF.functions(entry, store: store) == PF.declared_functions(entry), entry
      end
    end
  end
end
