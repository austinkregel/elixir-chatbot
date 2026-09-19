defmodule Brain.Lexicon.TypeAnchorsTest do
  @moduledoc """
  The anchor table is declared data: which WordNet synsets stand for which
  entity types. Loading it checks every anchor against WordNet, and a sense
  takes the type of its nearest anchor.
  """
  use ExUnit.Case, async: false

  alias Brain.Lexicon.TypeAnchors
  alias Brain.ML.Lexicon, as: WordNet

  @moduletag :tmp_dir

  @anchors_path Path.join(:code.priv_dir(:brain), "analysis/wordnet_type_anchors.json")

  defp noun_senses(word), do: Enum.filter(WordNet.senses(word), &(&1.pos == :noun))

  defp write_anchors(dir, anchors) do
    path = Path.join(dir, "anchors.json")
    File.write!(path, Jason.encode!(%{"anchors" => anchors}))
    path
  end

  describe "load!/1" do
    test "maps every declared anchor to its type" do
      declared = @anchors_path |> File.read!() |> Jason.decode!() |> Map.fetch!("anchors")
      anchors = TypeAnchors.load!()

      for {type, entries} <- declared, %{"synset_id" => sid} <- entries do
        assert anchors[sid] == type
      end

      assert map_size(anchors) == declared |> Map.values() |> List.flatten() |> length()
    end

    test "raises when WordNet no longer agrees with the word an anchor names", %{tmp_dir: dir} do
      [%{synset_id: sid} | _] = noun_senses("thursday")
      path = write_anchors(dir, %{"sys_date" => [%{"synset_id" => sid, "word" => "month"}]})

      assert_raise RuntimeError, ~r/should contain "month"/, fn -> TypeAnchors.load!(path: path) end
    end

    test "raises on an anchor that is not a WordNet synset", %{tmp_dir: dir} do
      path = write_anchors(dir, %{"sys_date" => [%{"synset_id" => 1, "word" => "day"}]})

      assert_raise RuntimeError, ~r/not a WordNet synset/, fn -> TypeAnchors.load!(path: path) end
    end

    test "raises when one synset is claimed by two types", %{tmp_dir: dir} do
      [%{synset_id: sid} | _] = noun_senses("thursday")
      entry = %{"synset_id" => sid, "word" => "thursday"}
      path = write_anchors(dir, %{"sys_date" => [entry], "music_artist" => [entry]})

      assert_raise RuntimeError, ~r/anchored to both/, fn -> TypeAnchors.load!(path: path) end
    end

    test "raises on a malformed anchor", %{tmp_dir: dir} do
      path = write_anchors(dir, %{"sys_date" => [%{"synset" => 115_188_052}]})

      assert_raise RuntimeError, ~r/malformed anchor/, fn -> TypeAnchors.load!(path: path) end
    end

    test "raises on a table with no anchors", %{tmp_dir: dir} do
      path = write_anchors(dir, %{})

      assert_raise RuntimeError, ~r/no "anchors" object/, fn -> TypeAnchors.load!(path: path) end
    end
  end

  describe "type_of_sense/3" do
    setup do
      {:ok, anchors: TypeAnchors.load!()}
    end

    test "a weekday is a date", %{anchors: anchors} do
      [sense] = noun_senses("thursday")
      assert TypeAnchors.type_of_sense(sense.synset_id, anchors) == {:ok, "sys_date"}
    end

    test "a named city reaches location through its instance edge", %{anchors: anchors} do
      types = for s <- noun_senses("paris"), do: TypeAnchors.type_of_sense(s.synset_id, anchors)
      assert {:ok, "location"} in types
    end

    test "the nearest anchor wins: a singer is a musician before a person", %{anchors: anchors} do
      # Madonna the singer reaches "musician" before "person"; Madonna the
      # mother of Jesus reaches only "person".
      types =
        "madonna"
        |> noun_senses()
        |> Enum.map(&TypeAnchors.type_of_sense(&1.synset_id, anchors))
        |> Enum.sort()

      assert types == [{:ok, "music_artist"}, {:ok, "person"}]
    end

    test "a sense under no anchor is unanchored", %{anchors: anchors} do
      # A door is a movable barrier: no anchor stands above it.
      [first | _] = noun_senses("door")
      assert TypeAnchors.type_of_sense(first.synset_id, anchors) == :unanchored
    end

    test "raises when two types are anchored at the same nearest distance" do
      # Anchor two different direct parents of one synset to two types. The
      # parents are read from WordNet, not named here.
      [person | _] = noun_senses("person")

      [a, b | _] =
        for {sid, _words, 1} <- WordNet.synset_ancestors(person.synset_id), do: sid

      anchors = %{a => "first_type", b => "second_type"}

      assert_raise RuntimeError, ~r/same distance/, fn ->
        TypeAnchors.type_of_sense(person.synset_id, anchors)
      end
    end
  end
end
