defmodule Brain.Analysis.FreeTextCaptureTest do
  # Points the models directory somewhere else and replaces the loaded model,
  # both global state.
  use ExUnit.Case, async: false

  alias Brain.Analysis.{FreeTextCapture, SlotDetector}
  alias Brain.Analysis.FreeTextCapture.Loader
  alias Brain.ML.EvaluationStore
  alias Mix.Tasks.TrainSlotTypePreferences

  # The tags a fixture row's words are given: what a tagger would say of them.
  @fixture_tags %{
    "to" => "PART",
    "for" => "ADP",
    "the" => "DET",
    "my" => "PRON",
    "all" => "DET",
    "that" => "DET",
    "i" => "PRON",
    "do" => "AUX",
    "n't" => "PART",
    "anymore" => "ADV",
    "up" => "ADP"
  }
  @fixture_verbs ~w(delete cancel drop remove clear need call buy feed email pay water book walk wash pick renew)

  defp fixture_tagger do
    fn texts ->
      Enum.map(texts, fn text ->
        word = String.downcase(text)
        Map.get_lazy(@fixture_tags, word, fn -> if word in @fixture_verbs, do: "VERB", else: "NOUN" end)
      end)
    end
  end

  @spans ["call mom", "buy milk", "feed the cat", "email Sarah", "pay rent", "water the plants", "book flights",
          "walk the dog", "wash the car", "call dad", "pick up the kids", "renew the passport"]
  @prefixes ["Delete the reminder to ", "Cancel the reminder to ", "Drop the reminder to "]
  @plain ["Cancel my reminder", "Clear all my reminders", "Cancel the reminder for 3pm", "Delete my reminders for tomorrow",
          "Remove all reminders", "I don't need that reminder anymore"]

  defp fixture_entries do
    spanned =
      @spans
      |> Enum.with_index()
      |> Enum.map(fn {span, k} ->
        %{"data" => [%{"text" => Enum.at(@prefixes, rem(k, 3))}, %{"text" => span, "alias" => "reminder_text", "meta" => "@sys.any"}]}
      end)

    spanned ++ Enum.map(@plain, &%{"data" => [%{"text" => &1}]})
  end

  defp text_of(entry), do: Enum.map_join(entry["data"], & &1["text"])

  defp fixture_dir!(entries) do
    dir = Path.join(System.tmp_dir!(), "free_text_capture_examples_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "reminder.cancel_usersays_en.json"), Jason.encode!(entries))
    on_exit(fn -> File.rm_rf!(dir) end)
    dir
  end

  @registry %{"reminder.cancel" => %{"required" => [], "optional" => ["reminder_text", "time", "date"]}}
  @held_out_texts ["Drop the reminder to wash the car", "Delete the reminder to email Sarah", "Clear all my reminders"]

  defp fixture_examples!(entries, opts \\ []) do
    gold = Enum.map(entries, &%{"text" => text_of(&1), "intent" => "reminder.cancel"})

    FreeTextCapture.examples!(
      registry: Keyword.get(opts, :registry, @registry),
      gold: Keyword.get(opts, :gold, gold),
      held_out: Enum.filter(gold, &(&1["text"] in @held_out_texts)),
      tagger: fixture_tagger(),
      dirs: [fixture_dir!(entries)]
    )
  end

  defp with_models_dir(fun) do
    ml = Application.get_env(:brain, :ml, [])
    dir = Path.join(System.tmp_dir!(), "free_text_capture_models_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    Application.put_env(:brain, :ml, Keyword.put(ml, :models_path, dir))

    try do
      fun.(dir)
    after
      Application.put_env(:brain, :ml, ml)
      File.rm_rf!(dir)
    end
  end

  setup_all do
    loaded = :persistent_term.get({FreeTextCapture, "en"}, nil)

    on_exit(fn ->
      case loaded do
        nil -> :persistent_term.erase({FreeTextCapture, "en"})
        model -> FreeTextCapture.put_loaded(model)
      end
    end)

    :ok
  end

  describe "examples!/1" do
    test "each annotated row is an example of its slot, its span placed on its tokens and split as the intent gold is" do
      examples = fixture_examples!(fixture_entries())

      assert length(examples) == length(fixture_entries())

      for example <- examples do
        assert example.slot == "reminder_text"
        assert example.split == if(example.text in @held_out_texts, do: :held_out, else: :train), example.text

        case example.span do
          nil ->
            assert example.text in @plain

          {i, j} ->
            first = Enum.at(example.tokens, i)
            last = Enum.at(example.tokens, j)
            assert String.slice(example.text, first.start_pos, last.end_pos - first.start_pos + 1) in @spans
        end
      end
    end

    test "the tracked annotations, in the authored intent examples and entity annotations: every span placed and split as the intent gold standard splits its rows" do
      held = "intent" |> EvaluationStore.load_gold_standard(:held_out) |> MapSet.new(&{&1["text"], &1["intent"]})

      examples =
        FreeTextCapture.examples!(
          registry: SlotDetector.registry(),
          gold: EvaluationStore.load_gold_standard("intent"),
          held_out: EvaluationStore.load_gold_standard("intent", :held_out),
          tagger: TrainSlotTypePreferences.tagger!()
        )

      assert examples |> Enum.map(& &1.slot) |> Enum.uniq() |> Enum.sort() == ["event", "reminder_text"]

      assert examples |> Enum.filter(&(&1.slot == "event")) |> Enum.map(& &1.intent) |> Enum.uniq() |> Enum.sort() ==
               ["calendar.cancel", "calendar.query", "calendar.schedule"]

      assert "reminder.create" in Enum.map(examples, & &1.intent)

      for example <- examples do
        assert example.split == if(MapSet.member?(held, {example.text, example.intent}), do: :held_out, else: :train)
      end

      for slot <- ["event", "reminder_text"], split <- [:held_out, :train] do
        assert Enum.any?(examples, &(&1.slot == slot and &1.split == split and &1.span != nil)), "no #{split} #{slot} span"
      end
    end

    test "a slot of one intent annotated in two directories raises" do
      [first, second] = Enum.split(fixture_entries(), 9) |> Tuple.to_list()
      gold = Enum.map(first ++ second, &%{"text" => text_of(&1), "intent" => "reminder.cancel"})

      assert_raise ArgumentError, ~r/reminder_text of reminder.cancel is annotated in more than one file/, fn ->
        FreeTextCapture.examples!(
          registry: @registry,
          gold: gold,
          held_out: [],
          tagger: fixture_tagger(),
          dirs: [fixture_dir!(first), fixture_dir!(second)]
        )
      end
    end

    test "a slot the registry does not declare for the intent raises" do
      registry = %{"reminder.cancel" => %{"optional" => ["time"]}}

      assert_raise ArgumentError, ~r/annotates "reminder_text", which the intent registry does not declare for reminder.cancel/, fn ->
        fixture_examples!(fixture_entries(), registry: registry)
      end
    end

    test "a row that is not a row of the intent gold standard raises" do
      assert_raise ArgumentError, ~r/is not a reminder.cancel row of the intent gold standard/, fn ->
        fixture_examples!(fixture_entries(), gold: [])
      end
    end

    test "a span holding a time raises: a time fills a slot of its own" do
      entries = [%{"data" => [%{"text" => "Cancel the reminder for "}, %{"text" => "3pm", "alias" => "reminder_text", "meta" => "@sys.any"}]}]

      assert_raise ArgumentError, ~r/holds words Brain.Temporal.Resolver reads as a date, time or length/, fn ->
        fixture_examples!(entries)
      end
    end

    test "a row holding two spans of one slot raises" do
      span = fn text -> %{"text" => text, "alias" => "reminder_text", "meta" => "@sys.any"} end
      entries = [%{"data" => [%{"text" => "Cancel "}, span.("lunch"), %{"text" => " and "}, span.("dinner")]}]

      assert_raise ArgumentError, ~r/marks 2 reminder_text spans; a row holds one/, fn -> fixture_examples!(entries) end
    end
  end

  describe "select/2" do
    test "chooses among every candidate by cross-validation, calibrates a threshold on the held-out rows, and round-trips" do
      model = FreeTextCapture.select(fixture_examples!(fixture_entries()), language: "en")

      candidates = [{:bio_lexical, nil} | Enum.map(FreeTextCapture.strengths(), &{:span_pos, &1})]
      assert model.report.cross_validation |> Map.keys() |> Enum.sort() == Enum.sort(candidates)
      assert {model.kind, model.smoothing} in candidates
      assert FreeTextCapture.slots(model) == ["reminder_text"]

      held = model.report.held_out["reminder_text"]
      threshold = model.thresholds["reminder_text"]
      assert threshold == 0.0 or threshold in Enum.map(held.rows, & &1.confidence)

      with_models_dir(fn _dir ->
        FreeTextCapture.save!(model)
        assert FreeTextCapture.read!("en") == model
      end)
    end
  end

  describe "capture/4 with the trained model" do
    # The model `mix train_free_text_capture` wrote to the models directory.
    setup do
      %{model: FreeTextCapture.read!("en")}
    end

    test "on the held-out rows, a capture is the row's gold span or nothing, and a gold span is captured", %{model: model} do
      held =
        FreeTextCapture.examples!(
          registry: SlotDetector.registry(),
          gold: EvaluationStore.load_gold_standard("intent"),
          held_out: EvaluationStore.load_gold_standard("intent", :held_out),
          tagger: TrainSlotTypePreferences.tagger!()
        )
        |> Enum.filter(&(&1.split == :held_out))

      captures =
        for example <- held do
          gold =
            with {i, j} <- example.span do
              first = Enum.at(example.tokens, i)
              last = Enum.at(example.tokens, j)
              String.slice(example.text, first.start_pos, last.end_pos - first.start_pos + 1)
            end

          case FreeTextCapture.capture(model, example.slot, example.text, []) do
            :abstain ->
              :abstain

            {:ok, %{value: value, confidence: confidence}} ->
              assert value == gold, "#{inspect(example.text)}: captured #{inspect(value)}, gold #{inspect(gold)}"
              assert confidence >= model.thresholds[example.slot]
              :captured
          end
        end

      assert :captured in captures
    end

    test "a span holds no date or time", %{model: model} do
      text = "Cancel the reminder for 3pm"
      at = :binary.match(text, "3pm") |> elem(0)

      case FreeTextCapture.capture(model, "reminder_text", text, []) do
        :abstain -> :ok
        {:ok, %{end_pos: end_pos}} -> assert end_pos < at
      end
    end

    test "a span holds no token of an entity that fills another slot", %{model: model} do
      text = "Can you remind me to call John"
      {at, length} = :binary.match(text, "John")
      assert {:ok, %{value: "call John"}} = FreeTextCapture.capture(model, "reminder_text", text, [])

      case FreeTextCapture.capture(model, "reminder_text", text, [{at, at + length - 1}]) do
        :abstain -> :ok
        {:ok, %{end_pos: end_pos}} -> assert end_pos < at
      end
    end

    test "a calendar item's event is learned from the calendar rows and captured", %{model: model} do
      assert FreeTextCapture.slots(model) == ["event", "reminder_text"]
      assert model.slots["event"].intents == ["calendar.cancel", "calendar.query", "calendar.schedule"]
      assert {:ok, %{value: "appointment"}} = FreeTextCapture.capture(model, "event", "Delete the appointment", [])
    end

    test "a slot the model does not capture raises", %{model: model} do
      assert_raise ArgumentError, ~r/not "location"/, fn -> FreeTextCapture.capture(model, "location", "in Paris", []) end
    end
  end

  describe "the loader" do
    test "boot fails without the artifact, naming the file and the task that writes it" do
      with_models_dir(fn dir ->
        error = assert_raise RuntimeError, fn -> Loader.load!() end
        assert error.message =~ "cannot read #{Path.join(dir, "free_text_capture.en.term")}"
        assert error.message =~ "mix train_free_text_capture"
      end)
    end

    test "boot fails when the model captures a slot for an intent the registry does not declare it for" do
      model = %FreeTextCapture{
        language: "en",
        slots: %{"reminder_text" => %{intents: ["weather.query"], counts: %{}}},
        kind: :span_pos,
        smoothing: 1.0,
        parameters: %{"reminder_text" => %{log_prior: 0.0, log_none: 0.0, ratios: %{}}},
        thresholds: %{"reminder_text" => 0.0},
        report: %{}
      }

      with_models_dir(fn _dir ->
        FreeTextCapture.save!(model)
        assert_raise RuntimeError, ~r/captures reminder_text for weather.query, which the intent registry does not declare/, fn -> Loader.load!() end
      end)
    end

    test "boot fails on a file that holds no model for the language" do
      with_models_dir(fn dir ->
        File.write!(Path.join(dir, "free_text_capture.en.term"), :erlang.term_to_binary(%{not: :a_model}))
        assert_raise RuntimeError, ~r/does not hold a en model/, fn -> Loader.load!() end
      end)
    end

    test "installs a model whose slots the registry declares" do
      model = %FreeTextCapture{
        language: "en",
        slots: %{"reminder_text" => %{intents: ["reminder.cancel", "reminder.check"], counts: %{}}},
        kind: :span_pos,
        smoothing: 1.0,
        parameters: %{"reminder_text" => %{log_prior: 0.0, log_none: 0.0, ratios: %{}}},
        thresholds: %{"reminder_text" => 0.0},
        report: %{}
      }

      with_models_dir(fn _dir ->
        FreeTextCapture.save!(model)
        assert Loader.load!() == :ok
        assert FreeTextCapture.loaded!("en") == model
      end)
    end
  end
end
