defmodule Brain.Lexicon.ClosedClassTest do
  @moduledoc """
  The closed-class vocabulary is declared data. These tests read the file
  itself, so a word added or removed there is checked without editing them.
  """
  use ExUnit.Case, async: true

  alias Brain.Lexicon.ClosedClass

  @path Path.join(:code.priv_dir(:brain), "knowledge/closed_class.json")

  setup_all do
    {:ok, data: @path |> File.read!() |> Jason.decode!()}
  end

  test "every declared word is a member, under every class it is listed in", %{data: data} do
    for {upos, words} <- data["classes"], word <- words do
      assert ClosedClass.member?(word)
      assert upos in ClosedClass.classes(word), "#{word} should be #{upos}"
    end
  end

  test "words() is exactly the declared words, sorted", %{data: data} do
    declared = data["classes"] |> Map.values() |> List.flatten() |> Enum.uniq() |> Enum.sort()
    assert ClosedClass.words() == declared
  end

  test "a word can belong to several classes" do
    assert "ADP" in ClosedClass.classes("to")
    assert "PART" in ClosedClass.classes("to")
  end

  test "the total quantifiers are exactly those declared PronType=Tot", %{data: data} do
    declared = data["features"]["PronType=Tot"]

    for word <- ClosedClass.words() do
      assert ClosedClass.total_quantifier?(word) == (word in declared), word
    end

    for word <- ~w(all every each), do: assert(ClosedClass.total_quantifier?(word))
    refute ClosedClass.total_quantifier?("the")
  end

  test "open-class words are not members, and membership ignores case" do
    refute ClosedClass.member?("light")
    assert ClosedClass.classes("light") == []
    assert ClosedClass.member?("The")
    assert ClosedClass.features("ALL") == %{"PronType" => "Tot"}
  end
end
