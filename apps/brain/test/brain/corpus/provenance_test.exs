defmodule Brain.Corpus.ProvenanceTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.{FreeTextCapture, SlotDetector}
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

      # Each annotation with the intent its file is filed under.
      annotations =
        for file <- Export.usersays_files(Provenance.authored_entities_dir()),
            intent = file |> Export.label_of() |> Label.canonical(),
            entry <- file |> File.read!() |> Jason.decode!(),
            do: %{intent: intent, entry: entry, text: Export.phrase_text(entry)}

      %{path: path, content: File.read!(path), authored: RebuildNerGold.authored_rows!(), relabels: RebuildNerGold.relabels!(), annotations: annotations}
    end

    test "the committed NER gold standard and its held-out split are what the annotations rebuild to, byte for byte",
         %{path: path, content: content, authored: authored, relabels: relabels} do
      assert authored != []
      assert relabels != %{}
      assert RebuildNerGold.rebuild!(content, path, authored, relabels) == content

      held_path = EvaluationStore.held_out_path("ner")
      held = File.read!(held_path)
      assert RebuildNerGold.rebuild_held_out!(held, held_path, content, path, relabels) == held
    end

    test "every relabeled row holds its annotation's entities, is labeled relabeled, and its held-out copy is the same row",
         %{relabels: relabels} do
      gold = "ner" |> EvaluationStore.load_gold_standard(:all) |> Map.new(&{&1["text"], &1})
      held = "ner" |> EvaluationStore.load_gold_standard(:held_out) |> Enum.filter(&Map.has_key?(relabels, &1["text"]))

      for {text, relabel} <- relabels do
        row = Map.fetch!(gold, text)
        assert row["labeled_by"] == Provenance.relabeled()
        spans = NerGold.spans!(row)

        for added <- relabel.added do
          assert Enum.any?(spans, &({&1.type, &1.start_pos, &1.end_pos} == {added["type"], added.start_pos, added.end_pos})),
                 "#{inspect(text)} lacks the #{added["type"]} #{inspect(added["value"])} its annotation adds"
        end
      end

      for row <- held, do: assert(row == Map.fetch!(gold, row["text"]))
    end

    # One existing row of a smart-home intent gold text, and its relabel adding
    # the device; "lights" is a type of its own that the NER type mappings
    # write as device.
    @relabel_row %{"expected" => [%{"type" => "room", "value" => "kitchen"}], "text" => "turn on the kitchen light"}

    defp write_relabel(dir, intent, data), do: File.write!(Path.join(dir, "#{intent}_usersays_en.json"), Jason.encode!([%{"data" => data}]))

    @tag :tmp_dir
    test "a relabel adds its entities to the row in place, labels it relabeled, and rebuilding the result changes nothing",
         %{tmp_dir: dir} do
      write_relabel(dir, "smarthome.device.switch.on", [%{"text" => "turn on the kitchen "}, %{"alias" => "device", "meta" => "@device", "text" => "light"}])
      relabels = RebuildNerGold.relabels!(dir, %{"lights" => "device"})

      other = %{"expected" => [], "text" => "hello there"}
      content = RebuildNerGold.encode!([other, @relabel_row])
      rebuilt = RebuildNerGold.rebuild!(content, "gold.json", [], relabels)

      assert [^other, relabeled] = Jason.decode!(rebuilt)

      assert relabeled == %{
               "expected" => [%{"type" => "room", "value" => "kitchen"}, %{"type" => "device", "value" => "light"}],
               "labeled_by" => "relabeled",
               "text" => "turn on the kitchen light"
             }

      assert RebuildNerGold.rebuild!(rebuilt, "gold.json", [], relabels) == rebuilt

      # The held-out copy of the row is relabeled with it; a held-out row of the
      # text that is not the gold standard's row raises.
      held = RebuildNerGold.encode!([@relabel_row])
      assert Jason.decode!(RebuildNerGold.rebuild_held_out!(held, "held.json", content, "gold.json", relabels)) == [relabeled]

      stale = RebuildNerGold.encode!([%{@relabel_row | "expected" => []}])

      assert_raise Mix.Error, ~r/held.json holds .* which is not the row gold.json holds/, fn ->
        RebuildNerGold.rebuild_held_out!(stale, "held.json", content, "gold.json", relabels)
      end
    end

    @tag :tmp_dir
    test "a relabel's type is checked against its intent's slot, and written as its gold label", %{tmp_dir: dir} do
      # hue.check fills its device slot with the type lights.
      write_relabel(dir, "smarthome.device.hue.check", [%{"text" => "check blue "}, %{"alias" => "device", "meta" => "@device", "text" => "lights"}, %{"text" => " hue"}])
      assert_raise Mix.Error, ~r/slot "device" of smarthome.device.hue.check lists \["lights"\], not "device"/, fn -> RebuildNerGold.relabels!(dir, %{"lights" => "device"}) end

      write_relabel(dir, "smarthome.device.hue.check", [%{"text" => "check blue "}, %{"alias" => "device", "meta" => "@lights", "text" => "lights"}, %{"text" => " hue"}])
      assert %{"check blue lights hue" => %{added: [%{"type" => "device", "value" => "lights", start_pos: 11, end_pos: 16}]}} = RebuildNerGold.relabels!(dir, %{"lights" => "device"})
    end

    @tag :tmp_dir
    test "a relabel of no row, one adding nothing, one overlapping an entity, and a relabeled row with no relabel raise", %{tmp_dir: dir} do
      content = RebuildNerGold.encode!([@relabel_row])

      write_relabel(dir, "smarthome.device.switch.on", [%{"text" => "turn on the kitchen light"}])
      assert_raise Mix.Error, ~r/must add an entity/, fn -> RebuildNerGold.relabels!(dir, %{}) end

      write_relabel(dir, "smarthome.device.switch.on", [%{"text" => "turn on the "}, %{"alias" => "device", "meta" => "@device", "text" => "kitchen light"}])
      relabels = RebuildNerGold.relabels!(dir, %{})
      assert_raise Mix.Error, ~r/overlaps the row's room "kitchen"/, fn -> RebuildNerGold.rebuild!(content, "gold.json", [], relabels) end

      write_relabel(dir, "smarthome.device.switch.on", [%{"text" => "turn on the kitchen "}, %{"alias" => "device", "meta" => "@device", "text" => "light"}])
      relabels = RebuildNerGold.relabels!(dir, %{})

      assert_raise Mix.Error, ~r/no row of gold.json outside the authored ones has that text/, fn ->
        RebuildNerGold.rebuild!(RebuildNerGold.encode!([%{"expected" => [], "text" => "hello there"}]), "gold.json", [], relabels)
      end

      relabeled = Map.put(@relabel_row, "labeled_by", "relabeled")

      assert_raise Mix.Error, ~r/labeled relabeled, which no relabeled entity annotation names/, fn ->
        RebuildNerGold.rebuild!(RebuildNerGold.encode!([relabeled]), "gold.json", [], %{})
      end
    end

    test "every annotation's entities are placed in its text, each of a type that fills its slot of the annotation's intent, " <>
           "and its free-text spans are no entities",
         %{authored: authored, annotations: annotations} do
      registry = SlotDetector.registry()
      rows = Map.new(authored, &{&1["text"], &1})

      for %{intent: intent, entry: entry, text: text} <- annotations do
        row = Map.fetch!(rows, text)
        entities = for %{"meta" => meta} = segment <- entry["data"], meta != FreeTextCapture.any_text(), do: segment

        assert Enum.map(NerGold.spans!(row), &{&1.type, &1.value}) == Enum.map(entities, &{String.trim_leading(&1["meta"], "@"), &1["text"]})

        for segment <- entities do
          types = SlotDetector.slot_entity_types!(Map.fetch!(registry, intent), segment["alias"])
          assert String.trim_leading(segment["meta"], "@") in types, "#{inspect(text)}: #{segment["meta"]} does not fill #{intent}'s #{segment["alias"]}"
        end
      end
    end

    test "every annotation is of a text the intent gold standard files under the annotation's intent", %{annotations: annotations} do
      # Filed under the intent, or listing it among its candidates, as `mix rebuild_ner_gold` checks.
      gold =
        for row <- EvaluationStore.load_gold_standard("intent"),
            intent <- [row["intent"] | Map.get(row, "candidates", [])],
            into: MapSet.new(),
            do: {Export.normalise(row["text"]), intent}

      for %{intent: intent, text: text} <- annotations do
        assert MapSet.member?(gold, {Export.normalise(text), intent}), "#{inspect(text)} is no #{intent} intent gold row"
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

    @tag :tmp_dir
    test "a free-text span names no entity, and one of a slot the intent does not declare raises", %{tmp_dir: dir} do
      write = fn data ->
        File.write!(Path.join(dir, "calendar.cancel_usersays_en.json"), Jason.encode!([%{"data" => data}]))
      end

      # calendar.cancel declares a date slot its entity_mappings does not
      # list; SlotDetector fills it with an entity of type date.
      write.([
        %{"text" => "Cancel "},
        %{"alias" => "date", "meta" => "@date", "text" => "tomorrow's"},
        %{"text" => " "},
        %{"alias" => "event", "meta" => "@sys.any", "text" => "meeting"}
      ])

      assert [%{"expected" => [%{"type" => "date", "value" => "tomorrow's"}], "text" => "Cancel tomorrow's meeting"}] = RebuildNerGold.authored_rows!(dir)

      write.([%{"text" => "Cancel the "}, %{"alias" => "reminder_text", "meta" => "@sys.any", "text" => "meeting"}])
      assert_raise Mix.Error, ~r/the free-text span's slot "reminder_text" is no slot calendar.cancel declares/, fn -> RebuildNerGold.authored_rows!(dir) end
    end
  end
end
