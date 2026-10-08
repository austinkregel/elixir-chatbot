defmodule Brain.Lexicon.CliticsTest do
  use ExUnit.Case, async: true

  alias Brain.Lexicon.Clitics

  @table Path.join(:code.priv_dir(:brain), "knowledge/clitics.json") |> File.read!() |> Jason.decode!()

  describe "the table" do
    test "was derived from the pinned EWT train split" do
      pinned = Brain.Training.UDEWT.source()
      [input] = @table["producer"]["inputs"]

      assert input["version"] == pinned["version"]
      assert input["sha256"] == pinned["files"]["train"]["sha256"]
      assert @table["license"] == pinned["license"]
    end

    test "holds the clitics the treebank writes, apostrophes canonical" do
      assert Clitics.forms() == ["'d", "'ll", "'m", "'re", "'s", "'ve", "n't"]
    end

    test "a clitic is ambiguous exactly when the treebank gives it more than one lemma" do
      for form <- Clitics.forms() do
        assert Clitics.ambiguous?(form) == (map_size(@table["clitics"][form]["lemmas"]) > 1)
      end

      assert Enum.filter(Clitics.forms(), &Clitics.ambiguous?/1) == ["'d", "'s"]
      assert Clitics.lemmas("n't") == %{"not" => @table["clitics"]["n't"]["count"]}
    end

    test "a word that is not a clitic raises rather than reading as unambiguous" do
      assert_raise ArgumentError, ~r/not a clitic/, fn -> Clitics.ambiguous?("'em") end
      assert_raise ArgumentError, ~r/not a clitic/, fn -> Clitics.lemmas("s") end
    end
  end

  describe "apostrophes" do
    test "the variants the table lists read as the plain apostrophe" do
      assert Clitics.canonical("DON’T") == "don't"
      assert Clitics.lemmas("’S") == Clitics.lemmas("'s")
      assert Clitics.apostrophe?("’") and Clitics.apostrophe?("'")
      refute Clitics.apostrophe?("\"")
    end
  end

  describe "split/1" do
    test "splits a word into its host and the clitic it ends with" do
      assert Clitics.split("He's") == {"He", "'s"}
      assert Clitics.split("can't") == {"ca", "n't"}
      assert Clitics.split("They’ve") == {"They", "'ve"}
    end

    test "nil for a word with no clitic, or only a clitic" do
      assert Clitics.split("gonna") == nil
      assert Clitics.split("'em") == nil
      assert Clitics.split("n't") == nil
    end
  end

  describe "match/2" do
    test "finds the clitic at an apostrophe, keeping casing" do
      assert Clitics.match("don", ["t", " ", "g", "o"]) == {"do", "n", ["t"]}
      assert Clitics.match("WO", ["N", "T"]) == nil
      assert Clitics.match("WON", ["T"]) == {"WO", "N", ["T"]}
      assert Clitics.match("Sarah", ["s", "."]) == {"Sarah", "", ["s"]}
      assert Clitics.match("they", ["l", "l"]) == {"they", "", ["l", "l"]}
    end

    test "a clitic must end the word and have a host" do
      assert Clitics.match("O", ["D", "o", "n", "n", "e", "l", "l"]) == nil
      assert Clitics.match("n", ["t"]) == nil
      assert Clitics.match("rock", ["n", "'", "r", "o", "l", "l"]) == nil
    end
  end
end
