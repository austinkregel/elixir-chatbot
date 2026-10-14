defmodule Brain.Analysis.SlotTypePreferences.LoaderTest do
  # Points the models directory somewhere else and replaces the loaded model,
  # both global state.
  use ExUnit.Case, async: false

  alias Brain.Analysis.SlotTypePreferences
  alias Brain.Analysis.SlotTypePreferences.Loader

  defmodule Hierarchy do
    def all_types, do: ~w(location room)
    def parents_of("room"), do: ["location"]
    def parents_of(_type), do: []
  end

  setup do
    ml = Application.get_env(:brain, :ml, [])
    loaded = :persistent_term.get({SlotTypePreferences, "en"}, nil)
    dir = Path.join(System.tmp_dir!(), "slot_type_preferences_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    Application.put_env(:brain, :ml, Keyword.put(ml, :models_path, dir))

    on_exit(fn ->
      Application.put_env(:brain, :ml, ml)
      File.rm_rf!(dir)

      case loaded do
        nil -> :persistent_term.erase({SlotTypePreferences, "en"})
        model -> SlotTypePreferences.put_loaded(model)
      end
    end)

    {:ok, dir: dir}
  end

  defp model(listed) do
    examples = [
      %{
        key: 1,
        intent: "weather.query",
        spans: [%{type: "location", preceding_tag: "ADP", preceding_word: "in", numeric: false}],
        ordinary: ["NOUN"],
        gazetteer: [:entity, :ordinary]
      },
      %{
        key: 2,
        intent: "smarthome.device.switch.on",
        spans: [%{type: "room", preceding_tag: "DET", preceding_word: "the", numeric: false}],
        ordinary: ["VERB"],
        gazetteer: [:entity]
      }
    ]

    SlotTypePreferences.fit(examples,
      language: "en",
      canonical: %{},
      hierarchy: Hierarchy,
      listed: listed,
      reading_types: ["location"],
      tag_vocabulary: ["ADP", "DET", "NOUN", "VERB", :sentence_start],
      smoothing: %{
        global: 1,
        domain: 1,
        intent: 1,
        hierarchy: 0.0,
        tag_global: 1,
        tag_type: 1,
        word_global: 1,
        word_type: 1,
        numeric_global: 1,
        numeric_type: 1,
        ordinary_global: 1,
        ordinary_domain: 1,
        ordinary_intent: 1
      }
    )
  end

  defp write_raw!(dir, term), do: File.write!(Path.join(dir, "slot_type_preferences.en.term"), :erlang.term_to_binary(term))

  test "boot fails without the artifact, naming the file and the task that writes it", %{dir: dir} do
    file = Path.join(dir, "slot_type_preferences.en.term")

    error = assert_raise RuntimeError, fn -> Loader.load!() end
    assert error.message =~ "cannot read #{file}"
    assert error.message =~ "mix train_slot_type_preferences"
  end

  test "boot fails when the model does not cover a type the slot schemas list" do
    SlotTypePreferences.save!(model(["location"]))

    assert_raise RuntimeError, ~r/the slot schemas list .* which the en model was not trained with/, fn -> Loader.load!() end
  end

  test "boot fails when the file does not hold a model for the language", %{dir: dir} do
    write_raw!(dir, %{not: :a_model})

    assert_raise RuntimeError, ~r/does not hold a en model/, fn -> Loader.load!() end
  end

  test "boot fails on a model trained before the ordinary-word counts were learned", %{dir: dir} do
    write_raw!(dir, Map.delete(model(SlotTypePreferences.listed_types()), :ordinary_counts))

    error = assert_raise RuntimeError, fn -> Loader.load!() end
    assert error.message =~ "has no [:ordinary_counts]"
    assert error.message =~ "mix train_slot_type_preferences"
  end

  test "boot fails on a model whose smoothing lacks the ordinary-word pseudo-counts", %{dir: dir} do
    full = model(SlotTypePreferences.listed_types())

    for key <- [:ordinary_global, :ordinary_domain, :ordinary_intent] do
      write_raw!(dir, %{full | smoothing: Map.delete(full.smoothing, key)})

      error = assert_raise RuntimeError, fn -> Loader.load!() end
      assert error.message =~ "smoothing #{inspect(key)} is invalid"
      assert error.message =~ "mix train_slot_type_preferences"
    end
  end

  test "boot fails on a model trained before the preceding-word and numeric likelihoods were learned", %{dir: dir} do
    full = model(SlotTypePreferences.listed_types())

    for key <- [:word_counts, :numeric_counts, :word_likelihoods, :numeric_likelihoods] do
      write_raw!(dir, Map.delete(full, key))

      error = assert_raise RuntimeError, fn -> Loader.load!() end
      assert error.message =~ "has no [#{inspect(key)}]"
      assert error.message =~ "mix train_slot_type_preferences"
    end

    for key <- [:word_global, :word_type, :numeric_global, :numeric_type] do
      write_raw!(dir, %{full | smoothing: Map.delete(full.smoothing, key)})

      error = assert_raise RuntimeError, fn -> Loader.load!() end
      assert error.message =~ "smoothing #{inspect(key)} is invalid"
    end
  end

  test "boot fails on malformed ordinary-word counts", %{dir: dir} do
    full = model(SlotTypePreferences.listed_types())
    write_raw!(dir, %{full | ordinary_counts: %{global: 3}})

    assert_raise RuntimeError, ~r/malformed ordinary_counts.*mix train_slot_type_preferences/, fn -> Loader.load!() end
  end

  test "installs a model that covers every listed type" do
    saved = model(SlotTypePreferences.listed_types())
    SlotTypePreferences.save!(saved)

    assert Loader.load!() == :ok
    assert SlotTypePreferences.loaded!("en") == saved
    assert Loader.start_link() == :ignore
  end
end
