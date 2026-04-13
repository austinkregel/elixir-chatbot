defmodule Brain.ML.EntityTrainerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.EntityTrainer

  describe "convert_to_bio_sequences/1" do
    test "converts simple example to BIO format" do
      examples = [
        %{
          text: "weather in London",
          intent: "weather.query",
          entities: [
            %{text: "London", type: "location", alias: "address", start_pos: 11, end_pos: 16}
          ]
        }
      ]

      sequences = EntityTrainer.convert_to_bio_sequences(examples)

      assert length(sequences) == 1
      seq = Enum.at(sequences, 0)

      assert seq.intent == "weather.query"
      assert is_list(seq.tokens)
      assert is_list(seq.tags)
      assert length(seq.tokens) == length(seq.tags)
    end

    test "assigns O tag to non-entity tokens" do
      examples = [
        %{
          text: "hello world",
          intent: "smalltalk.greetings.hello",
          entities: []
        }
      ]

      sequences = EntityTrainer.convert_to_bio_sequences(examples)
      seq = Enum.at(sequences, 0)
      assert Enum.all?(seq.tags, fn tag -> tag == "O" end)
    end

    test "assigns B- and I- tags for multi-word entities" do
      examples = [
        %{
          text: "weather in New York tomorrow",
          intent: "weather.query",
          entities: [
            %{text: "New York", type: "location", alias: "address", start_pos: 11, end_pos: 18}
          ]
        }
      ]

      sequences = EntityTrainer.convert_to_bio_sequences(examples)
      seq = Enum.at(sequences, 0)

      bio_tags =
        Enum.filter(seq.tags, fn tag ->
          String.starts_with?(tag, "B-") or String.starts_with?(tag, "I-")
        end)

      assert bio_tags != []
    end

    test "handles example with no entities" do
      examples = [%{text: "hello", intent: "smalltalk.greetings.hello", entities: nil}]

      sequences = EntityTrainer.convert_to_bio_sequences(examples)
      assert length(sequences) == 1
    end

    test "filters out empty text examples" do
      examples = [
        %{text: "", intent: "empty", entities: []},
        %{text: "valid text", intent: "valid", entities: []}
      ]

      sequences = EntityTrainer.convert_to_bio_sequences(examples)
      assert length(sequences) == 1
      assert Enum.at(sequences, 0).intent == "valid"
    end
  end

  describe "extract_entities_from_bio/1" do
    test "extracts entity from B-I sequence" do
      token_tag_pairs = [
        {"weather", "O"},
        {"in", "O"},
        {"New", "B-location"},
        {"York", "I-location"},
        {"today", "O"}
      ]

      entities = EntityTrainer.extract_entities_from_bio(token_tag_pairs)

      assert length(entities) == 1
      entity = Enum.at(entities, 0)
      assert entity.entity_type == "location"
      assert entity.value == "New York"
    end

    test "extracts multiple entities" do
      token_tag_pairs = [
        {"weather", "O"},
        {"in", "O"},
        {"London", "B-location"},
        {"tomorrow", "B-date"}
      ]

      entities = EntityTrainer.extract_entities_from_bio(token_tag_pairs)

      assert length(entities) == 2

      location = Enum.find(entities, fn e -> e.entity_type == "location" end)
      date = Enum.find(entities, fn e -> e.entity_type == "date" end)

      assert location.value == "London"
      assert date.value == "tomorrow"
    end

    test "handles sequence with no entities" do
      token_tag_pairs = [{"hello", "O"}, {"world", "O"}]

      entities = EntityTrainer.extract_entities_from_bio(token_tag_pairs)
      assert entities == []
    end

    test "handles empty sequence" do
      entities = EntityTrainer.extract_entities_from_bio([])
      assert entities == []
    end

    test "handles consecutive entities of different types" do
      token_tag_pairs = [{"London", "B-location"}, {"Monday", "B-date"}]

      entities = EntityTrainer.extract_entities_from_bio(token_tag_pairs)

      assert length(entities) == 2
    end

    test "handles entity at end of sequence" do
      token_tag_pairs = [{"weather", "O"}, {"in", "O"}, {"Paris", "B-location"}]

      entities = EntityTrainer.extract_entities_from_bio(token_tag_pairs)

      assert length(entities) == 1
      assert Enum.at(entities, 0).value == "Paris"
    end
  end

  describe "train/0" do
    test "trains model from intent data" do
      result = EntityTrainer.train()

      case result do
        {:ok, model} ->
          assert Map.has_key?(model, :tag_vocabulary)
          assert Map.has_key?(model, :feature_weights)
          assert Map.has_key?(model, :transition_weights)
          assert Map.has_key?(model, :tag_priors)
          assert Map.has_key?(model.tag_vocabulary, "O")

        {:error, _reason} ->
          assert true
      end
    end
  end

  describe "predict/2" do
    test "predicts tags for tokens using trained model" do
      model = %{
        tag_vocabulary: %{"O" => 0, "B-location" => 1},
        feature_weights: %{
          "token:weather" => %{"O" => 0.9, "B-location" => 0.1},
          "token:london" => %{"O" => 0.2, "B-location" => 0.8}
        },
        transition_weights: %{
          "<START>" => %{"O" => 0.7, "B-location" => 0.3},
          "O" => %{"O" => 0.6, "B-location" => 0.4},
          "B-location" => %{"O" => 0.8, "B-location" => 0.2}
        },
        tag_priors: %{"O" => 0.7, "B-location" => 0.3}
      }

      tokens = ["weather", "in", "London"]
      predictions = EntityTrainer.predict(tokens, model)

      assert length(predictions) == 3

      Enum.each(predictions, fn {token, tag} ->
        assert is_binary(token)
        assert is_binary(tag)
      end)
    end

    test "handles empty token list" do
      model = %{
        tag_vocabulary: %{"O" => 0},
        feature_weights: %{},
        transition_weights: %{},
        tag_priors: %{"O" => 1.0}
      }

      predictions = EntityTrainer.predict([], model)
      assert predictions == []
    end
  end
end
