defmodule Brain.ML.POSTaggerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.POSTagger

  describe "training" do
    test "trains model from valid sequences" do
      training_sequences = [
        %{tokens: ["I", "am", "Austin"], tags: ["PRON", "VERB", "PROPN"]},
        %{tokens: ["Hello", "there"], tags: ["INTJ", "ADV"]},
        %{tokens: ["What", "is", "the", "weather"], tags: ["PRON", "VERB", "DET", "NOUN"]}
      ]

      assert {:ok, model} = POSTagger.train(training_sequences)
      assert is_map(model)
      assert map_size(model.tag_vocabulary) > 0
      assert map_size(model.feature_weights) > 0
      assert map_size(model.transition_weights) > 0
    end

    test "returns error for empty training data" do
      assert {:error, _} = POSTagger.train([])
    end

    test "filters invalid sequences (mismatched lengths)" do
      training_sequences = [
        # Mismatched lengths
        %{tokens: ["I", "am"], tags: ["PRON"]},
        # Valid
        %{tokens: ["Hello", "world"], tags: ["INTJ", "NOUN"]}
      ]

      assert {:ok, model} = POSTagger.train(training_sequences)
      assert map_size(model.tag_vocabulary) > 0
    end

    test "normalizes tag formats (atoms and strings)" do
      training_sequences = [
        # Atoms
        %{tokens: ["I", "am"], tags: [:PRON, :VERB]},
        # Lowercase string
        %{tokens: ["Hello"], tags: ["intj"]}
      ]

      assert {:ok, model} = POSTagger.train(training_sequences)
      # All tags should be normalized to uppercase strings
      assert "PRON" in Map.keys(model.tag_vocabulary)
      assert "INTJ" in Map.keys(model.tag_vocabulary)
    end
  end

  describe "prediction" do
    setup do
      training_sequences = [
        %{tokens: ["I", "am", "Austin"], tags: ["PRON", "VERB", "PROPN"]},
        %{tokens: ["I", "am", "fine"], tags: ["PRON", "VERB", "ADJ"]},
        %{tokens: ["Hello", "there"], tags: ["INTJ", "ADV"]},
        %{tokens: ["What", "is", "the", "weather"], tags: ["PRON", "VERB", "DET", "NOUN"]},
        %{tokens: ["The", "weather", "is", "nice"], tags: ["DET", "NOUN", "VERB", "ADJ"]},
        %{tokens: ["My", "name", "is", "John"], tags: ["PRON", "NOUN", "VERB", "PROPN"]},
        %{tokens: ["You", "are", "great"], tags: ["PRON", "VERB", "ADJ"]}
      ]

      {:ok, model} = POSTagger.train(training_sequences)
      {:ok, %{model: model}}
    end

    test "predicts tags for tokens", %{model: model} do
      tokens = ["I", "am", "John"]
      predictions = POSTagger.predict(tokens, model)

      assert length(predictions) == 3

      assert Enum.all?(predictions, fn {token, tag} ->
               is_binary(token) and is_binary(tag)
             end)
    end

    test "returns empty list for empty input", %{model: model} do
      assert POSTagger.predict([], model) == []
    end

    test "predict_tags returns just tags", %{model: model} do
      tokens = ["Hello", "there"]
      tags = POSTagger.predict_tags(tokens, model)

      assert length(tags) == 2
      assert Enum.all?(tags, &is_binary/1)
    end

    test "handles unknown tokens gracefully", %{model: model} do
      tokens = ["xyzabc123", "qqqqq"]
      predictions = POSTagger.predict(tokens, model)

      # Should return some prediction even for unknown tokens
      assert length(predictions) == 2
    end
  end

  describe "model persistence" do
    setup do
      # Use a temporary path for testing
      test_path = Path.join(System.tmp_dir!(), "test_pos_model_#{:rand.uniform(10000)}.term")
      on_exit(fn -> File.rm(test_path) end)
      {:ok, %{test_path: test_path}}
    end

    test "saves and loads model", %{test_path: test_path} do
      training_sequences = [
        %{tokens: ["I", "am", "here"], tags: ["PRON", "VERB", "ADV"]}
      ]

      {:ok, original_model} = POSTagger.train(training_sequences)
      assert {:ok, ^test_path} = POSTagger.save_model(original_model, test_path)

      assert {:ok, loaded_model} = POSTagger.load_model(test_path)
      assert loaded_model.tag_vocabulary == original_model.tag_vocabulary
    end

    test "load returns error for missing file" do
      assert {:error, _} = POSTagger.load_model("/nonexistent/path/model.term")
    end
  end

  describe "valid_tags/0" do
    test "returns list of valid POS tags" do
      tags = POSTagger.valid_tags()

      assert is_list(tags)
      assert "NOUN" in tags
      assert "VERB" in tags
      assert "PRON" in tags
      assert "PROPN" in tags
    end
  end

  describe "feature extraction" do
    test "model captures capitalization patterns" do
      training_sequences = [
        %{tokens: ["austin", "is", "here"], tags: ["NOUN", "VERB", "ADV"]},
        %{tokens: ["Austin", "is", "here"], tags: ["PROPN", "VERB", "ADV"]},
        %{tokens: ["AUSTIN", "IS", "HERE"], tags: ["PROPN", "VERB", "ADV"]}
      ]

      {:ok, model} = POSTagger.train(training_sequences)

      # Feature weights should include capitalization features
      feature_keys = Map.keys(model.feature_weights)
      assert Enum.any?(feature_keys, &String.contains?(&1, "capitalized"))
    end

    test "model captures suffix patterns" do
      training_sequences = [
        %{tokens: ["running", "walking"], tags: ["VERB", "VERB"]},
        %{tokens: ["happy", "quickly"], tags: ["ADJ", "ADV"]}
      ]

      {:ok, model} = POSTagger.train(training_sequences)

      feature_keys = Map.keys(model.feature_weights)
      assert Enum.any?(feature_keys, &String.starts_with?(&1, "suffix"))
    end
  end
end
