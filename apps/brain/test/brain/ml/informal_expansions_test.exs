defmodule Brain.ML.InformalExpansionsTest do
  use ExUnit.Case, async: false

  alias Brain.ML.InformalExpansions

  setup do
    # Ensure the agent is started for tests
    case Process.whereis(InformalExpansions) do
      nil -> InformalExpansions.start_link([])
      _pid -> :ok
    end

    :ok
  end

  describe "expand/1" do
    test "expands standard contractions" do
      assert {:ok, "i am"} = InformalExpansions.expand("i'm")
      assert {:ok, "you are"} = InformalExpansions.expand("you're")
      assert {:ok, "do not"} = InformalExpansions.expand("don't")
      assert {:ok, "cannot"} = InformalExpansions.expand("can't")
    end

    test "expands informal phonetic reductions" do
      assert {:ok, "going to"} = InformalExpansions.expand("gonna")
      assert {:ok, "want to"} = InformalExpansions.expand("wanna")
      assert {:ok, "got to"} = InformalExpansions.expand("gotta")
      assert {:ok, "kind of"} = InformalExpansions.expand("kinda")
      assert {:ok, "sort of"} = InformalExpansions.expand("sorta")
    end

    test "expands coalescence patterns" do
      assert {:ok, "did you"} = InformalExpansions.expand("didja")
      assert {:ok, "got you"} = InformalExpansions.expand("gotcha")
      assert {:ok, "what are you"} = InformalExpansions.expand("whatcha")
    end

    test "expands question compressions" do
      assert {:ok, "did you eat"} = InformalExpansions.expand("jeet")
      assert {:ok, "did you ever"} = InformalExpansions.expand("jever")
      assert {:ok, "do not know"} = InformalExpansions.expand("dunno")
    end

    test "preserves case - uppercase" do
      assert {:ok, "I AM"} = InformalExpansions.expand("I'M")
      assert {:ok, "DO NOT"} = InformalExpansions.expand("DON'T")
    end

    test "preserves case - title case" do
      assert {:ok, "I am"} = InformalExpansions.expand("I'm")
      assert {:ok, "Going to"} = InformalExpansions.expand("Gonna")
    end

    test "does not expand a contraction whose clitic has more than one meaning" do
      # "'s" is the possessive, is, has or us; "'d" is would or had.
      for word <- ~w(he's she's it's what's let's i'd she'd) do
        assert :not_found = InformalExpansions.expand(word), "#{word} was expanded"
      end
    end

    test "every contraction it expands ends in a clitic with one meaning" do
      with_clitic =
        for {informal, _} <- InformalExpansions.all_expansions(),
            {_host, clitic} <- [Brain.Lexicon.Clitics.split(informal)],
            do: {informal, clitic}

      # n't, 'm, 'll, 've and 're all appear in the dataset.
      assert with_clitic |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> length() >= 5
      assert Enum.filter(with_clitic, fn {_, clitic} -> Brain.Lexicon.Clitics.ambiguous?(clitic) end) == []
    end

    test "returns :not_found for unknown tokens" do
      assert :not_found = InformalExpansions.expand("hello")
      assert :not_found = InformalExpansions.expand("world")
      assert :not_found = InformalExpansions.expand("Jordan")
    end
  end

  describe "has_expansion?/1" do
    test "returns true for known expansions" do
      assert InformalExpansions.has_expansion?("gonna")
      assert InformalExpansions.has_expansion?("wanna")
      assert InformalExpansions.has_expansion?("i'm")
    end

    test "returns false for unknown tokens" do
      refute InformalExpansions.has_expansion?("hello")
      refute InformalExpansions.has_expansion?("Jordan")
    end

    test "is case insensitive" do
      assert InformalExpansions.has_expansion?("GONNA")
      assert InformalExpansions.has_expansion?("Gonna")
      assert InformalExpansions.has_expansion?("gonna")
    end
  end

  describe "missing data fails loudly" do
    # Stops the application's agent for the test and restarts it after, with
    # the configured data path restored.
    setup %{tmp_dir: dir} do
      ml = Application.fetch_env!(:brain, :ml)
      :ok = Supervisor.terminate_child(Brain.Supervisor, InformalExpansions)

      on_exit(fn ->
        Application.put_env(:brain, :ml, ml)
        {:ok, _} = Supervisor.restart_child(Brain.Supervisor, InformalExpansions)
      end)

      %{ml: ml, dir: dir}
    end

    @tag :tmp_dir
    test "a lookup before the agent has started raises" do
      assert_raise RuntimeError, ~r/InformalExpansions is not started/, fn -> InformalExpansions.expand("gonna") end
      assert_raise RuntimeError, ~r/not started/, fn -> Brain.ML.Tokenizer.expand_contractions("I'm gonna go") end
      assert_raise RuntimeError, ~r/not started/, fn -> InformalExpansions.all_expansions() end
      assert_raise RuntimeError, ~r/not started/, fn -> InformalExpansions.metadata() end
    end

    @tag :tmp_dir
    test "a missing, malformed or empty data file fails the start", %{ml: ml, dir: dir} do
      Application.put_env(:brain, :ml, Keyword.put(ml, :training_data_path, dir))
      Process.flag(:trap_exit, true)
      path = Path.join(dir, "informal_expansions.json")

      assert {:error, {%RuntimeError{message: message}, _}} = InformalExpansions.start_link()
      assert message =~ "cannot read #{path}"

      File.write!(path, "{not json")
      assert {:error, {%RuntimeError{message: message}, _}} = InformalExpansions.start_link()
      assert message =~ "not valid JSON"

      File.write!(path, ~s({"expansions": {}}))
      assert {:error, {%RuntimeError{message: message}, _}} = InformalExpansions.start_link()
      assert message =~ ~s(no non-empty "expansions")

      File.write!(path, ~s({"expansions": {"gonna": "going to", "he's": "he is"}}))
      assert {:error, {%RuntimeError{message: message}, _}} = InformalExpansions.start_link()
      assert message =~ ~s(expands ["he's"])
    end
  end

  describe "metadata/0" do
    test "returns loading status" do
      meta = InformalExpansions.metadata()
      assert is_boolean(meta.loaded)
      assert is_integer(meta.total_entries)
    end
  end

  describe "all_expansions/0" do
    test "returns a map of all expansions" do
      expansions = InformalExpansions.all_expansions()
      assert is_map(expansions)
      assert Map.has_key?(expansions, "gonna")
      assert Map.has_key?(expansions, "i'm")
    end
  end

  describe "integration with Tokenizer" do
    alias Brain.ML.Tokenizer

    test "tokenizer uses InformalExpansions for contraction expansion" do
      assert "I am going to the store" = Tokenizer.expand_contractions("I'm gonna the store")
      assert "What are you doing" = Tokenizer.expand_contractions("Whatcha doing")
      assert "I do not know" = Tokenizer.expand_contractions("I dunno")
    end

    test "leaves an ambiguous contraction for the tokenizer to split" do
      assert "he's been there, she'd go" = Tokenizer.expand_contractions("he's been there, she'd go")

      assert Tokenizer.tokenize_normalized("he's been there", expand_contractions: true) ==
               ["he", "'s", "been", "there"]
    end

    test "preserves proper names" do
      assert "Call me Jordan" = Tokenizer.expand_contractions("Call me Jordan")
      assert "Hi James" = Tokenizer.expand_contractions("Hi James")
    end
  end
end
