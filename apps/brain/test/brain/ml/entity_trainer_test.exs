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

  # Every room here is two words, so "room" is only ever tagged I-room: its
  # emission says I-room wherever it stands.
  @continued_rows [
    %{"text" => "dim the lights in the living room", "expected" => [%{"type" => "device", "value" => "lights"}, %{"type" => "room", "value" => "living room"}]},
    %{"text" => "turn on the radio in the dining room", "expected" => [%{"type" => "device", "value" => "radio"}, %{"type" => "room", "value" => "dining room"}]},
    %{"text" => "switch off the heater in the living room", "expected" => [%{"type" => "device", "value" => "heater"}, %{"type" => "room", "value" => "living room"}]},
    %{"text" => "is the radio on in the dining room", "expected" => [%{"type" => "device", "value" => "radio"}, %{"type" => "room", "value" => "dining room"}]},
    %{"text" => "turn off the lights in the living room", "expected" => [%{"type" => "device", "value" => "lights"}, %{"type" => "room", "value" => "living room"}]},
    %{"text" => "dim the heater in the dining room", "expected" => [%{"type" => "device", "value" => "heater"}, %{"type" => "room", "value" => "dining room"}]},
    %{"text" => "what is the weather today", "expected" => [%{"type" => "date_time", "value" => "today"}]},
    %{"text" => "hello there", "expected" => []}
  ]

  setup_all do
    {:ok, model} = EntityTrainer.train(@rows)
    {:ok, continued} = EntityTrainer.train(@continued_rows)
    {:ok, model: model, continued: continued}
  end

  # Whether every I-x tag follows B-x or I-x.
  defp valid_bio?(tags) do
    ["<START>" | tags]
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.all?(fn
      [prev, "I-" <> type] -> prev in ["B-" <> type, "I-" <> type]
      _ -> true
    end)
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
        assert fold_model.transition_smoothing == model.transition_smoothing
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

    test "a model without a transition smoothing strength raises", %{model: model} do
      assert_raise ArgumentError, ~r/transition_smoothing/, fn ->
        EntityTrainer.fold_models(@rows, Map.delete(model, :transition_smoothing))
      end
    end
  end

  describe "transitions" do
    test "are the rows' transition counts, smoothed toward how often each allowed successor follows anything",
         %{continued: model} do
      sequences = Enum.map(@continued_rows, &Brain.ML.NerGold.sequence!/1)
      pairs = Enum.flat_map(sequences, &((["<START>" | &1.tags] ++ ["<END>"]) |> Enum.chunk_every(2, 1, :discard)))
      tags = sequences |> Enum.flat_map(& &1.tags) |> Enum.uniq()
      into = Enum.frequencies_by(pairs, &List.last/1)
      s = model.transition_smoothing
      assert is_float(s) and s > 0

      # Every room opens at B-room and goes on to I-room.
      allowed_after_b_room = Enum.reject(tags ++ ["<END>"], &(String.starts_with?(&1, "I-") and &1 != "I-room"))
      n = Enum.count(pairs, &(&1 == ["B-room", "I-room"]))
      total = Enum.count(pairs, &(hd(&1) == "B-room"))
      q = Map.fetch!(into, "I-room") / (allowed_after_b_room |> Enum.map(&Map.fetch!(into, &1)) |> Enum.sum())

      assert n == 6 and total == 6
      assert_in_delta model.transition_weights["B-room"]["I-room"], (n + s * q) / (total + s), 1.0e-12

      # A transition the rows never showed but BIO allows stays possible.
      refute ["B-room", "B-device"] in pairs
      assert model.transition_weights["B-room"]["B-device"] > 0

      for {prev, row} <- model.transition_weights do
        assert_in_delta Enum.sum(Map.values(row)), 1.0, 1.0e-9, "#{prev}'s transitions sum to #{Enum.sum(Map.values(row))}"
      end
    end

    test "BIO forbids are not in the model", %{continued: model} do
      for {prev, row} <- model.transition_weights, "I-" <> type = curr <- Map.keys(row) do
        assert prev in ["B-" <> type, "I-" <> type], "#{prev} -> #{curr} is in the model"
      end

      refute Map.has_key?(model.transition_weights["<START>"], "I-room")
      refute Map.has_key?(model.transition_weights["O"], "I-room")
      refute Map.has_key?(model.transition_weights["B-device"], "I-room")
    end

    test "a model without an allowed transition raises when it decodes", %{continued: model} do
      broken = update_in(model.transition_weights["O"], &Map.delete(&1, "B-room"))

      assert_raise ArgumentError, ~r/O -> B-room, a transition BIO allows/, fn ->
        EntityTrainer.tag(~w(the dining room), broken)
      end
    end
  end

  describe "BIO decoding" do
    test "a word only ever tagged I-room is not tagged I-room where no room is open", %{continued: model} do
      for tokens <- [~w(room), ~w(clean the room), ~w(room lights)] do
        tags = tokens |> EntityTrainer.tag(model) |> Enum.map(&elem(&1, 1))
        assert valid_bio?(tags), "#{inspect(tokens)} decoded to #{inspect(tags)}"
      end

      assert ~w(O O B-room I-room) == ~w(in the dining room) |> EntityTrainer.tag(model) |> Enum.map(&elem(&1, 1))
    end

    test "never yields an invalid BIO sequence", %{continued: model} do
      vocabulary = @continued_rows |> Enum.flat_map(&String.split(&1["text"])) |> Enum.uniq()

      sentences =
        for a <- vocabulary, b <- vocabulary, c <- vocabulary, length <- [1, 2, 3] do
          Enum.take([a, b, c], length)
        end
        |> Enum.uniq()

      for tokens <- sentences do
        tagged = EntityTrainer.tag(tokens, model)
        tags = Enum.map(tagged, &elem(&1, 1))
        assert valid_bio?(tags), "#{inspect(tokens)} decoded to #{inspect(tags)}"

        for {_token, _tag, p} <- tagged, do: assert(p > 0.0 and p <= 1.0)
        assert is_list(EntityTrainer.extract_entities_from_bio(tagged))
      end
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

    test "an I- tag that continues no entity of its type raises" do
      assert_raise ArgumentError, ~r/"I-location" at token 1 \("York"\) continues no entity/, fn ->
        EntityTrainer.extract_entities_from_bio([{"in", "O", 1.0}, {"York", "I-location", 1.0}])
      end

      assert_raise ArgumentError, ~r/"I-location" at token 0 \("York"\) continues no entity/, fn ->
        EntityTrainer.extract_entities_from_bio([{"York", "I-location", 1.0}])
      end

      assert_raise ArgumentError, ~r/continues an entity of type "date"/, fn ->
        EntityTrainer.extract_entities_from_bio([{"Monday", "B-date", 1.0}, {"York", "I-location", 1.0}])
      end
    end
  end
end
