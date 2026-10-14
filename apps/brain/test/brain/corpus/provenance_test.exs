defmodule Brain.Corpus.ProvenanceTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.TypeHierarchy
  alias Brain.Corpus.{Export, Label, Provenance}
  alias Brain.ML.{EvaluationStore, NerGold}

  describe "kinds" do
    test "every kind, most genuine first" do
      assert Provenance.kinds() == ["dialogflow", "authored", "materialized", "augmented"]
    end

    test "the kinds written back into the export are the synthetic writers' and not the authored examples'" do
      assert Provenance.written_back() == ["materialized", "augmented"]
      refute Provenance.authored() in Provenance.written_back()
      assert Provenance.kind?(Provenance.authored())
    end

    test "a text found both authored and written back counts as authored, and as Dialogflow's when it is in the export" do
      assert Provenance.most_genuine(["materialized", "authored"]) == "authored"
      assert Provenance.most_genuine(["augmented", "authored", "dialogflow"]) == "dialogflow"
    end
  end

  describe "of_id!/2" do
    test "reads the kind from an export row's id" do
      assert Provenance.of_id!("orphan-0123abcd", "x.json") == "materialized"
      assert Provenance.of_id!("augmented-0123abcd", "x.json") == "augmented"
      assert Provenance.of_id!("6b8f2a54-3c1e-4d7a-9e0b-2f1c3a4b5d6e", "x.json") == "dialogflow"
      assert Provenance.of_id!(nil, "x.json") == "dialogflow"
    end

    test "raises on an id shaped like a synthetic marker with an unknown prefix, and on a non-string id" do
      assert_raise RuntimeError, ~r/unrecognised synthetic id prefix in x.json/, fn ->
        Provenance.of_id!("invented-0123abcd", "x.json")
      end

      assert_raise RuntimeError, ~r/x.json has a non-string id 7/, fn -> Provenance.of_id!(7, "x.json") end
    end
  end

  # The gold standard is rebuilt from the export, which is not tracked, and the
  # authored examples, which are. These check that the authored rows of the
  # committed gold standard are exactly what a rebuild makes of the tracked
  # files, so no rebuild can drop or relabel them.
  describe "the authored examples and the intent gold standard" do
    setup do
      authored =
        Provenance.authored_dir()
        |> Export.usersays_files()
        |> Enum.flat_map(fn path ->
          path
          |> File.read!()
          |> Jason.decode!()
          |> Enum.map(fn entry ->
            %{
              entry: entry,
              row: %{
                "text" => Export.phrase_text(entry),
                "intent" => path |> Export.label_of() |> Label.canonical(),
                "labeled_by" => Provenance.authored(),
                "source_file" => Path.basename(path)
              }
            }
          end)
        end)

      %{authored: authored, gold: EvaluationStore.load_gold_standard("intent")}
    end

    test "the directory holds authored examples, none carrying an id", %{authored: authored} do
      assert authored != []

      for %{entry: entry, row: row} <- authored do
        refute Map.has_key?(entry, "id"), "#{row["source_file"]}: #{inspect(row["text"])} carries an id"
      end
    end

    test "every authored example is a gold row labeled authored, with its file's intent and name",
         %{authored: authored, gold: gold} do
      for %{row: row} <- authored do
        assert row in gold, "#{row["source_file"]}: #{inspect(row["text"])} is not in the gold standard as #{inspect(row)}"
      end
    end

    test "every gold row labeled authored is an authored example", %{authored: authored, gold: gold} do
      rows = MapSet.new(authored, & &1.row)

      for row <- gold, row["labeled_by"] == Provenance.authored() do
        assert MapSet.member?(rows, row), "gold row #{inspect(row)} has no authored example"
      end
    end

    test "the held-out authored rows are gold rows, so they are kept out of training", %{gold: gold} do
      held_out = EvaluationStore.load_gold_standard("intent", :held_out)
      training = EvaluationStore.load_gold_standard("intent", :train)

      held_authored = Enum.filter(held_out, &(&1["labeled_by"] == Provenance.authored()))
      assert held_authored != []

      for row <- held_authored do
        assert row in gold
        refute row in training
      end
    end
  end

  # The NER gold standard's authored rows are rebuilt from the tracked
  # annotations (`mix rebuild_ner_gold`); every other row has no tracked source
  # and is kept as it stands. These check that the committed file is exactly
  # what a rebuild makes of it, so no rebuild can drop, relabel or move a row.
  describe "the authored entity annotations and the NER gold standard" do
    alias Mix.Tasks.RebuildNerGold

    setup do
      path = EvaluationStore.gold_standard_path("ner")
      %{path: path, content: File.read!(path), authored: RebuildNerGold.authored_rows!()}
    end

    test "the committed NER gold standard is what the annotations rebuild to, byte for byte", %{path: path, content: content, authored: authored} do
      assert authored != []
      assert RebuildNerGold.rebuild!(content, path, authored) == content
    end

    test "every annotation's entities are placed in its text, typed under the code root", %{authored: authored} do
      for row <- authored, span <- NerGold.spans!(row) do
        assert TypeHierarchy.is_a?(span.type, "code"), "#{inspect(row["text"])}: #{span.type} is not a code type"
      end
    end

    test "every annotation is of a text the intent gold standard files under the annotation's intent", %{authored: authored} do
      gold = EvaluationStore.load_gold_standard("intent")

      for row <- authored do
        assert Enum.any?(gold, &(Export.normalise(&1["text"]) == Export.normalise(row["text"]) and String.starts_with?(&1["intent"], "code."))),
               "#{inspect(row["text"])} is no code intent gold row"
      end
    end

    test "the held-out authored rows are gold rows kept out of training, and every entity-type set keeps training rows",
         %{authored: authored} do
      held = Enum.filter(EvaluationStore.load_gold_standard("ner", :held_out), &(&1["labeled_by"] == Provenance.authored()))
      training = EvaluationStore.load_gold_standard("ner", :train)
      assert held != []

      for row <- held do
        assert row in authored
        refute row in training
      end

      type_set = fn row -> row["expected"] |> Enum.map(& &1["type"]) |> Enum.uniq() |> Enum.sort() end
      held_sets = held |> Enum.map(type_set) |> MapSet.new()
      training_sets = for row <- training, row["labeled_by"] == Provenance.authored(), into: MapSet.new(), do: type_set.(row)

      assert MapSet.subset?(held_sets, training_sets), "a held-out entity-type set has no authored training row"
    end

    @tag :tmp_dir
    test "an annotation of a text the intent gold standard does not hold, or of a type its slot does not list, raises",
         %{tmp_dir: dir} do
      write = fn data ->
        File.write!(Path.join(dir, "code.explain_usersays_en.json"), Jason.encode!([%{"data" => data}]))
      end

      write.([%{"text" => "explain the "}, %{"alias" => "symbol", "meta" => "@function", "text" => "frobnicate"}])
      assert_raise Mix.Error, ~r/is not the text of an intent gold row filed under code.explain/, fn -> RebuildNerGold.authored_rows!(dir) end

      write.([%{"text" => "describe the function "}, %{"alias" => "symbol", "meta" => "@variable", "text" => "send_email"}])
      assert_raise Mix.Error, ~r/slot "symbol" of code.explain lists .* not "variable"/, fn -> RebuildNerGold.authored_rows!(dir) end

      write.([%{"text" => "describe the function "}, %{"alias" => "symbol", "meta" => "@function", "text" => "send_email"}])
      assert [%{"expected" => [%{"type" => "function", "value" => "send_email"}], "labeled_by" => "authored"}] = RebuildNerGold.authored_rows!(dir)
    end
  end
end
