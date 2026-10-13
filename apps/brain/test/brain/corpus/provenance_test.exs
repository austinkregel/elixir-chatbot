defmodule Brain.Corpus.ProvenanceTest do
  use ExUnit.Case, async: true

  alias Brain.Corpus.{Export, Label, Provenance}
  alias Brain.ML.EvaluationStore

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
end
