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
      %{key: 1, intent: "weather.query", spans: [%{type: "location", preceding_tag: "ADP"}], ordinary: ["NOUN"]},
      %{key: 2, intent: "smarthome.device.switch.on", spans: [%{type: "room", preceding_tag: "DET"}], ordinary: ["VERB"]}
    ]

    SlotTypePreferences.fit(examples,
      language: "en",
      canonical: %{},
      hierarchy: Hierarchy,
      listed: listed,
      tag_vocabulary: ["ADP", "DET", "NOUN", "VERB", :sentence_start],
      smoothing: %{global: 1, domain: 1, intent: 1, hierarchy: 0.0, tag_global: 1, tag_type: 1}
    )
  end

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
    File.write!(Path.join(dir, "slot_type_preferences.en.term"), :erlang.term_to_binary(%{not: :a_model}))

    assert_raise RuntimeError, ~r/does not hold a en model/, fn -> Loader.load!() end
  end

  test "installs a model that covers every listed type" do
    saved = model(SlotTypePreferences.listed_types())
    SlotTypePreferences.save!(saved)

    assert Loader.load!() == :ok
    assert SlotTypePreferences.loaded!("en") == saved
    assert Loader.start_link() == :ignore
  end
end
