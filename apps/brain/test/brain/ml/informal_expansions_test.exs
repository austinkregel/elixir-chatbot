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

    test "preserves proper names" do
      assert "Call me Jordan" = Tokenizer.expand_contractions("Call me Jordan")
      assert "Hi James" = Tokenizer.expand_contractions("Hi James")
    end
  end
end
