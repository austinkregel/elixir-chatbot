defmodule Brain.Analysis.CodeAndPlaceTypesTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.{SlotDetector, SlotResult, TypeHierarchy}

  setup do
    if Process.whereis(TypeHierarchy) == nil, do: start_supervised!({TypeHierarchy, name: TypeHierarchy})
    :ok
  end

  defp fill(intent, type, value) do
    intent
    |> SlotDetector.detect([%{entity_type: type, value: value, confidence: 0.9}], side_effects: false)
    |> then(& &1.filled_slots)
    |> Map.new(fn {slot, filled} -> {slot, filled.value} end)
  end

  defp code_intents do
    for {intent, schema} <- SlotDetector.registry(), Map.get(schema, "domain") == "code", do: {intent, schema}
  end

  describe "code types" do
    test "a symbol is a function, a class, a method or a variable, and symbols, files and programming languages are code" do
      assert Enum.sort(TypeHierarchy.specializations("symbol")) == ~w(class function method variable)
      assert Enum.sort(TypeHierarchy.specializations("code")) == ~w(file programming_language symbol)

      for type <- ~w(function class method variable) do
        assert TypeHierarchy.parents_of(type) == ["symbol"]
        assert TypeHierarchy.is_a?(type, "code")
      end
    end

    test "a programming language is not a natural language" do
      refute TypeHierarchy.compatible?("programming_language", "language")
    end

    test "the code intents' language slots list programming languages, not natural languages" do
      assert code_intents() != []

      for {intent, schema} <- code_intents(), {slot, types} <- schema["entity_mappings"], slot == "language" do
        assert types == ["programming_language"], intent
      end
    end

    test "a natural language does not fill a code intent's language slot; a programming language does" do
      for {intent, _schema} <- code_intents() do
        refute Map.has_key?(fill(intent, "language", "French"), "language"), intent
        assert fill(intent, "programming_language", "Python")["language"] == "Python", intent
      end
    end

    test "every kind of symbol fills a code intent's symbol slot, and a file its file slot" do
      for {intent, _schema} <- code_intents(), type <- ~w(symbol function class method variable) do
        assert fill(intent, type, "calculate_tax")["symbol"] == "calculate_tax", "#{intent} #{type}"
      end

      for {intent, _schema} <- code_intents() do
        assert fill(intent, "file", "billing.ex")["file"] == "billing.ex", intent
      end
    end

    test "a search query does not fill a symbol slot" do
      refute Map.has_key?(fill("code.explain", "q", "parse"), "symbol")
    end
  end

  describe "slots that listed appliance or filepath" do
    test "no registry slot lists appliance or filepath" do
      listed =
        for {intent, schema} <- SlotDetector.registry(),
            {slot, types} <- Map.get(schema, "entity_mappings", %{}),
            type <- types,
            type in ["appliance", "filepath"],
            do: "#{intent}.#{slot}: #{type}"

      assert listed == []
    end

    test "a device entity fills the device slot of every intent whose device slot listed appliance" do
      for intent <- ["smarthome.device_set", "smarthome.switch", "device.control"] do
        assert SlotResult.get_slot_value(
                 SlotDetector.detect(intent, [%{entity_type: "device", value: "dishwasher", confidence: 0.9}], side_effects: false),
                 "device"
               ) == "dishwasher",
               intent
      end
    end
  end

  describe "place types" do
    # data/entities/ambiguous_name_location_entries_en.json gives each
    # ambiguous name a person reading and its place readings.
    test "every place reading of an ambiguous name is a location" do
      path = Path.join(Application.fetch_env!(:brain, :ml)[:training_data_path], "entities/ambiguous_name_location_entries_en.json")

      place_types =
        for entry <- path |> File.read!() |> Jason.decode!(),
            %{"entity_type" => type} <- entry["types"],
            type != "person",
            uniq: true,
            do: type

      assert Enum.sort(place_types) == ~w(borough city continent country mountain_range neighborhood river state)

      for type <- place_types do
        assert TypeHierarchy.is_a?(type, "location"), type
      end
    end

    test "a country fills knowledge.capital's country slot, which lists location" do
      assert fill("knowledge.capital", "country", "India")["country"] == "India"
    end
  end

  describe "temporal types" do
    test "every type the NER type mappings compare as date_time is a date_time" do
      mapped =
        for {type, "date_time"} <- Brain.ML.EntityExtractor.load_type_mappings(), do: type

      assert "month_name" in mapped and "weekday" in mapped

      for type <- mapped do
        assert TypeHierarchy.is_a?(type, "date_time"), type
      end
    end
  end
end
