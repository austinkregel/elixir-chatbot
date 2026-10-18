defmodule Brain.ML.NerGoldTest do
  use ExUnit.Case, async: true

  alias Brain.ML.{EvaluationStore, NerGold}

  defp row(text, entities), do: %{"text" => text, "expected" => Enum.map(entities, fn {type, value} -> %{"type" => type, "value" => value} end)}

  describe "template!/1" do
    test "is the text with each entity replaced by its type, normalized" do
      assert NerGold.template!(row("Turn off the lights in the  Kitchen", [{"device", "lights"}, {"room", "Kitchen"}])) ==
               "turn off the <device> in the <room>"

      assert NerGold.template!(row("hello there", [])) == "hello there"
    end

    test "rows that differ only in their entities' values share one, and rows that differ around them do not" do
      kitchen = row("turn off the lights in the kitchen", [{"device", "lights"}, {"room", "kitchen"}])
      bedroom = row("turn off the radio in the living room", [{"device", "radio"}, {"room", "living room"}])
      later = row("turn off the lights in the kitchen later", [{"device", "lights"}, {"room", "kitchen"}])

      assert NerGold.template!(kitchen) == NerGold.template!(bedroom)
      refute NerGold.template!(kitchen) == NerGold.template!(later)
    end

    test "a row whose entities cannot be placed raises, as spans!/1 does" do
      assert_raise ArgumentError, ~r/no token-aligned placement/, fn -> NerGold.template!(row("play jazz", [{"genre", "rock"}])) end
    end
  end

  describe "template_folds/2" do
    test "puts the rows of one template in one fold" do
      rows = [
        row("turn off the lights in the kitchen", [{"device", "lights"}, {"room", "kitchen"}]),
        row("turn off the radio in the bedroom", [{"device", "radio"}, {"room", "bedroom"}]),
        row("turn off the heater in the hall", [{"device", "heater"}, {"room", "hall"}]),
        row("play some jazz", [{"genre", "jazz"}]),
        row("play some rock", [{"genre", "rock"}])
      ]

      [a, b, c, d, e] = NerGold.template_folds(rows, 5)
      assert a == b and b == c
      assert d == e
      assert Enum.all?([a, d], &(&1 in 0..4))
    end

    test "on the NER training split, no template's rows span two folds" do
      rows = EvaluationStore.load_gold_standard("ner", :train)
      folds = NerGold.template_folds(rows, 5)

      by_template =
        rows
        |> Enum.zip(folds)
        |> Enum.group_by(fn {row, _fold} -> NerGold.template!(row) end, &elem(&1, 1))

      assert Enum.any?(by_template, fn {_template, row_folds} -> length(row_folds) > 1 end)

      for {template, row_folds} <- by_template do
        assert length(Enum.uniq(row_folds)) == 1, "#{inspect(template)}'s rows are in folds #{inspect(Enum.uniq(row_folds))}"
      end
    end
  end
end
