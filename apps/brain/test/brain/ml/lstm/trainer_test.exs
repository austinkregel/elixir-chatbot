defmodule Brain.ML.LSTM.AxonTrainerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.LSTM.AxonTrainer
  alias Brain.ML.DataLoaders

  @moduletag :lstm
  @moduletag timeout: 120_000

  describe "train_intent_classifier/1" do
    @tag :slow
    test "trains a model successfully with minimal config" do
      result =
        AxonTrainer.train_intent_classifier(
          epochs: 2,
          batch_size: 16,
          max_seq_length: 30,
          max_intents: 30
        )

      assert {:ok, model} = result

      assert Map.has_key?(model, :model)
      assert Map.has_key?(model, :params)
      assert Map.has_key?(model, :vocabularies)
      assert Map.has_key?(model, :config)
      assert Map.has_key?(model, :metrics)

      assert model.metrics.final_train_accuracy > 0.05,
        "Training accuracy #{model.metrics.final_train_accuracy} should show learning progress (> 5%)"
    end
  end

  describe "classify/2" do
    @tag :slow
    test "classifies text using trained model" do
      {:ok, model} =
        AxonTrainer.train_intent_classifier(
          epochs: 2,
          batch_size: 16,
          max_seq_length: 30,
          max_intents: 30
        )

      {intent, confidence} = AxonTrainer.classify("what is the weather today", model)

      assert is_binary(intent)
      assert confidence >= 0.0 and confidence <= 1.0
    end
  end

  describe "data preparation" do
    test "load_intent_training_data_for_lstm returns properly formatted data" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm()

      assert examples != []
      first = hd(examples)
      assert Map.has_key?(first, :tokens)
      assert Map.has_key?(first, :intent)
      assert Map.has_key?(first, :entities)
      assert Map.has_key?(first, :bio_tags)

      assert is_list(first.tokens)
      assert is_binary(first.intent)
    end

    test "build_lstm_vocabulary creates proper vocabulary" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm()

      vocab = DataLoaders.build_lstm_vocabulary(examples)
      assert Map.has_key?(vocab, "<PAD>")
      assert Map.has_key?(vocab, "<UNK>")
      assert Map.has_key?(vocab, "<BOS>")
      assert Map.has_key?(vocab, "<EOS>")
      assert vocab["<PAD>"] == 0
      assert vocab["<UNK>"] == 1
      assert vocab["<BOS>"] == 2
      assert vocab["<EOS>"] == 3
    end

    test "build_intent_vocabulary creates bidirectional mappings" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm()

      {intent_to_idx, idx_to_intent} = DataLoaders.build_intent_vocabulary(examples)
      assert map_size(intent_to_idx) > 0
      assert map_size(idx_to_intent) > 0

      Enum.each(intent_to_idx, fn {intent, idx} ->
        assert idx_to_intent[idx] == intent
      end)
    end

    test "tokens_to_indices converts correctly" do
      vocab = %{"<PAD>" => 0, "<UNK>" => 1, "hello" => 2, "world" => 3}

      tokens = ["hello", "world", "unknown"]
      indices = DataLoaders.tokens_to_indices(tokens, vocab)

      assert indices == [2, 3, 1]
    end

    test "pad_sequence pads to target length" do
      indices = [1, 2, 3]
      padded = DataLoaders.pad_sequence(indices, 5, 0)
      assert padded == [1, 2, 3, 0, 0]
      truncated = DataLoaders.pad_sequence(indices, 2, 0)
      assert truncated == [1, 2]
      same = DataLoaders.pad_sequence(indices, 3, 0)
      assert same == [1, 2, 3]
    end
  end
end
