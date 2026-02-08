defmodule Brain.ML.LSTM.NERHeadTest do
  alias Nx.Random
  use ExUnit.Case, async: true

  alias Brain.ML.LSTM.NERHead
  alias Brain.ML.DataLoaders

  @moduletag :lstm

  describe "build_model/3" do
    test "builds NER head with correct dimensions" do
      input_size = 64
      num_bio_tags = 7

      model = NERHead.build_model(input_size, num_bio_tags)

      assert %Axon{} = model
    end
  end

  describe "init_params/3" do
    test "initializes parameters" do
      input_size = 64
      num_bio_tags = 7

      model = NERHead.build_model(input_size, num_bio_tags)
      params = NERHead.init_params(model, input_size)
      assert params != nil

      param_data =
        if is_struct(params, Axon.ModelState) do
          params.data
        else
          params
        end

      assert is_map(param_data)
      assert map_size(param_data) > 0
    end
  end

  describe "forward/3" do
    test "produces output with correct shape" do
      input_size = 64
      num_bio_tags = 7
      batch_size = 2
      seq_len = 10

      model = NERHead.build_model(input_size, num_bio_tags)
      params = NERHead.init_params(model, input_size)

      key = Random.key(42)

      {token_outputs, _key} =
        Random.uniform(key, shape: {batch_size, seq_len, input_size}, type: :f32)

      output = NERHead.forward(model, token_outputs, params)

      assert Nx.shape(output) == {batch_size, seq_len, num_bio_tags}

      for b <- 0..(batch_size - 1), t <- 0..(seq_len - 1) do
        sum = output[b][t] |> Nx.sum() |> Nx.to_number()
        assert_in_delta sum, 1.0, 0.01
      end
    end
  end

  describe "tag_sequence/4" do
    test "returns BIO tags for each token" do
      input_size = 64
      num_bio_tags = 3
      seq_len = 5

      model = NERHead.build_model(input_size, num_bio_tags)
      params = NERHead.init_params(model, input_size)

      bio_labels = ["O", "B-LOC", "I-LOC"]
      key = Random.key(42)
      {token_outputs, _key} = Random.uniform(key, shape: {seq_len, input_size}, type: :f32)

      result = NERHead.tag_sequence(model, token_outputs, params, bio_labels)

      assert length(result) == seq_len

      Enum.each(result, fn {idx, tag, conf} ->
        assert is_integer(idx)
        assert tag in bio_labels
        assert conf >= 0.0 and conf <= 1.0
      end)
    end
  end

  describe "extract_entities_from_tags/2" do
    test "extracts single-token entity" do
      tokens = ["what", "is", "the", "weather", "in", "London"]
      bio_tags = ["O", "O", "O", "O", "O", "B-LOC"]

      entities = NERHead.extract_entities_from_tags(tokens, bio_tags)

      assert length(entities) == 1
      assert hd(entities).text == "London"
      assert hd(entities).type == "LOC"
      assert hd(entities).start == 5
      assert hd(entities).end == 5
    end

    test "extracts multi-token entity" do
      tokens = ["play", "songs", "by", "Taylor", "Swift"]
      bio_tags = ["O", "O", "O", "B-ARTIST", "I-ARTIST"]

      entities = NERHead.extract_entities_from_tags(tokens, bio_tags)

      assert length(entities) == 1
      assert hd(entities).text == "Taylor Swift"
      assert hd(entities).type == "ARTIST"
      assert hd(entities).start == 3
      assert hd(entities).end == 4
    end

    test "extracts multiple entities" do
      tokens = ["weather", "in", "London", "on", "Monday"]
      bio_tags = ["O", "O", "B-LOC", "O", "B-DATE"]

      entities = NERHead.extract_entities_from_tags(tokens, bio_tags)

      assert length(entities) == 2

      loc = Enum.find(entities, &(&1.type == "LOC"))
      date = Enum.find(entities, &(&1.type == "DATE"))

      assert loc.text == "London"
      assert date.text == "Monday"
    end

    test "handles no entities" do
      tokens = ["hello", "world"]
      bio_tags = ["O", "O"]

      entities = NERHead.extract_entities_from_tags(tokens, bio_tags)

      assert entities == []
    end
  end

  describe "entity_type_from_bio/1" do
    test "extracts type from B- tag" do
      assert NERHead.entity_type_from_bio("B-location") == "location"
      assert NERHead.entity_type_from_bio("B-ARTIST") == "ARTIST"
    end

    test "extracts type from I- tag" do
      assert NERHead.entity_type_from_bio("I-location") == "location"
    end

    test "returns nil for O tag" do
      assert NERHead.entity_type_from_bio("O") == nil
    end
  end

  describe "entity_types_from_bio_vocab/1" do
    test "extracts unique entity types" do
      bio_labels = ["O", "B-LOC", "I-LOC", "B-ARTIST", "I-ARTIST", "B-DATE", "I-DATE"]

      types = NERHead.entity_types_from_bio_vocab(bio_labels)

      assert types == ["ARTIST", "DATE", "LOC"]
    end
  end

  describe "BIO conversion in DataLoaders" do
    test "generates BIO tags from entity annotations" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm()

      example_with_entities =
        Enum.find(examples, fn ex ->
          ex.entities != []
        end)

      if example_with_entities do
        assert is_list(example_with_entities.bio_tags)
        assert length(example_with_entities.bio_tags) == length(example_with_entities.tokens)

        Enum.each(example_with_entities.bio_tags, fn tag ->
          assert String.starts_with?(tag, "O") or
                   String.starts_with?(tag, "B-") or
                   String.starts_with?(tag, "I-")
        end)
      end
    end

    test "build_bio_vocabulary creates complete vocabulary" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm()

      {bio_to_idx, idx_to_bio} = DataLoaders.build_bio_vocabulary(examples)
      assert Map.has_key?(bio_to_idx, "O")
      assert bio_to_idx["O"] == 0

      Enum.each(bio_to_idx, fn {tag, idx} ->
        assert idx_to_bio[idx] == tag
      end)
    end
  end
end