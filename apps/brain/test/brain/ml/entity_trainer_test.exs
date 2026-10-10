defmodule Brain.ML.EntityTrainerTest do
  use ExUnit.Case, async: true

  alias Brain.ML.EntityTrainer

  # A small corpus of its own, so this tests the training system and not the
  # gold standard.
  @rows [
    %{"text" => "turn on the lights in the kitchen", "expected" => [%{"type" => "device", "value" => "lights"}, %{"type" => "room", "value" => "kitchen"}]},
    %{"text" => "turn off the lights in the bedroom", "expected" => [%{"type" => "device", "value" => "lights"}, %{"type" => "room", "value" => "bedroom"}]},
    %{"text" => "switch on the radio in the kitchen", "expected" => [%{"type" => "device", "value" => "radio"}, %{"type" => "room", "value" => "kitchen"}]},
    %{"text" => "switch off the radio in the bathroom", "expected" => [%{"type" => "device", "value" => "radio"}, %{"type" => "room", "value" => "bathroom"}]},
    %{"text" => "dim the lights in the bathroom", "expected" => [%{"type" => "device", "value" => "lights"}, %{"type" => "room", "value" => "bathroom"}]},
    %{"text" => "is the heater on in the bedroom", "expected" => [%{"type" => "device", "value" => "heater"}, %{"type" => "room", "value" => "bedroom"}]},
    %{"text" => "turn on the heater in the kitchen", "expected" => [%{"type" => "device", "value" => "heater"}, %{"type" => "room", "value" => "kitchen"}]},
    %{"text" => "what is the weather today", "expected" => [%{"type" => "date_time", "value" => "today"}]},
    %{"text" => "weather for tomorrow", "expected" => [%{"type" => "date_time", "value" => "tomorrow"}]},
    %{"text" => "hello there", "expected" => []}
  ]

  setup_all do
    {:ok, model} = EntityTrainer.train(@rows)
    {:ok, model: model}
  end

  describe "train/1" do
    test "learns the tags of its rows and a smoothing strength from them", %{model: model} do
      for tag <- ~w(O B-device B-room B-date_time), do: assert(Map.has_key?(model.tag_vocabulary, tag))
      assert is_float(model.smoothing) and model.smoothing > 0
    end

    test "tags a seen entity word as that entity", %{model: model} do
      tagged = EntityTrainer.tag(~w(turn on the lights in the bathroom), model)
      assert Enum.map(tagged, &elem(&1, 1)) == ~w(O O O B-device O O B-room)
    end

    test "no rows raises" do
      assert_raise ArgumentError, ~r/no NER rows/, fn -> EntityTrainer.train([]) end
    end
  end

  describe "fold_models/2" do
    test "gives each row a fold model trained, with the model's smoothing, on the other folds", %{model: model} do
      {folds, models} = EntityTrainer.fold_models(@rows, model)

      assert length(folds) == length(@rows)
      assert Enum.sort(Map.keys(models)) == Enum.to_list(0..4)

      for {_k, fold_model} <- models do
        assert fold_model.smoothing == model.smoothing
      end

      # "dim" is in one row only, so that row's fold model never saw it, while
      # the full model did.
      dim_row = Enum.find_index(@rows, &(&1["text"] == "dim the lights in the bathroom"))
      assert Map.has_key?(model.feature_weights, "token:dim")
      refute Map.has_key?(Map.fetch!(models, Enum.at(folds, dim_row)).feature_weights, "token:dim")
    end

    test "a model without a smoothing strength raises" do
      assert_raise ArgumentError, ~r/smoothing/, fn -> EntityTrainer.fold_models(@rows, %{tag_vocabulary: %{}}) end
    end
  end

  describe "tag/2" do
    test "each tag carries its posterior probability", %{model: model} do
      for {_token, _tag, p} <- EntityTrainer.tag(~w(dim the radio in the kitchen), model) do
        assert p > 0.0 and p <= 1.0
      end
    end

    test "an empty sequence has no tags", %{model: model} do
      assert EntityTrainer.tag([], model) == []
    end
  end

  describe "extract_entities_from_bio/1" do
    test "a B-I run is one entity, with its token positions and mean confidence" do
      tagged = [{"weather", "O", 0.9}, {"in", "O", 0.9}, {"New", "B-location", 0.8}, {"York", "I-location", 0.6}, {"today", "O", 0.9}]

      assert [entity] = EntityTrainer.extract_entities_from_bio(tagged)
      assert entity.entity_type == "location"
      assert entity.value == "New York"
      assert {entity.start_index, entity.end_index} == {2, 3}
      assert_in_delta entity.confidence, 0.7, 1.0e-9
    end

    test "consecutive entities of different types are separate" do
      tagged = [{"London", "B-location", 1.0}, {"Monday", "B-date", 1.0}]
      assert Enum.map(EntityTrainer.extract_entities_from_bio(tagged), & &1.entity_type) == ["location", "date"]
    end

    test "an entity at the end of the sequence is kept" do
      assert [%{value: "Paris"}] = EntityTrainer.extract_entities_from_bio([{"in", "O", 1.0}, {"Paris", "B-location", 1.0}])
    end

    test "no entities, and no tokens, give nothing" do
      assert EntityTrainer.extract_entities_from_bio([{"hello", "O", 1.0}]) == []
      assert EntityTrainer.extract_entities_from_bio([]) == []
    end
  end
end
