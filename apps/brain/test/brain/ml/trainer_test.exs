defmodule Brain.ML.TrainerTest do
  use ExUnit.Case, async: false
  require Nx

  alias Brain.ML.Trainer

  @moduletag :training

  setup_all do
    Brain.TestHelpers.require_services!(:ml_inference)
    :ok
  end

  describe "load_training_data/0" do
    test "returns list of {text, intent} tuples" do
      data = Trainer.load_training_data()

      assert is_list(data)
      assert data != []
      {text, intent} = hd(data)
      assert is_binary(text)
      assert is_binary(intent)
    end

    test "filters out empty texts" do
      data = Trainer.load_training_data()

      Enum.each(data, fn {text, _intent} ->
        assert String.trim(text) != ""
      end)
    end

    test "parses Dialogflow-style usersays files" do
      data = Trainer.load_training_data()
      intents = Enum.map(data, fn {_text, intent} -> intent end) |> Enum.uniq()
      assert Enum.count_until(intents, 2) > 1
    end
  end

  describe "build_tfidf_vectorizer/1" do
    test "builds vocabulary from real training data" do
      training_data = Trainer.load_training_data() |> Enum.take(100)

      vectorizer = Trainer.build_tfidf_vectorizer(training_data)

      assert is_map(vectorizer)
      assert Map.has_key?(vectorizer, :vocabulary)
      assert Map.has_key?(vectorizer, :idf_weights)
      assert Map.has_key?(vectorizer, :max_features)
      assert is_map(vectorizer.vocabulary)
      assert map_size(vectorizer.vocabulary) > 0
    end

    test "idf_weights is an Nx tensor" do
      training_data = Trainer.load_training_data() |> Enum.take(50)

      vectorizer = Trainer.build_tfidf_vectorizer(training_data)
      assert Nx.is_tensor(vectorizer.idf_weights)
    end
  end

  describe "build_gazetteer_data/1" do
    test "returns stats with entry counts" do
      stats = %{
        gazetteer_entries: 0,
        entity_types: 0
      }

      result = Trainer.build_gazetteer_data(stats)

      assert is_map(result)
      assert Map.has_key?(result, :gazetteer_entries)
      assert Map.has_key?(result, :entity_types)
      assert is_integer(result.gazetteer_entries)
      assert is_integer(result.entity_types)
    end
  end

  describe "train_intent_classifier/2 (slow)" do
    @tag :slow
    @tag timeout: 180_000
    test "returns updated stats and :ok on success" do
      stats = %{intent_samples: 0, vocab_size: 0}

      temp_dir = System.tmp_dir!()
      models_path = Path.join(temp_dir, "test_models_#{System.unique_integer()}")
      File.mkdir_p!(models_path)

      on_exit(fn ->
        File.rm_rf!(models_path)
      end)

      {updated_stats, result} = Trainer.train_intent_classifier(stats, models_path: models_path)

      assert result == :ok
      assert updated_stats.intent_samples > 0
      assert updated_stats.vocab_size > 0
      assert File.exists?(Path.join(models_path, "embedder.term"))
    end
  end

  describe "train_entity_model/2 (slow)" do
    @tag :slow
    @tag timeout: 180_000
    test "returns updated stats" do
      stats = %{entity_model_trained: false}

      temp_dir = System.tmp_dir!()
      models_path = Path.join(temp_dir, "test_entity_models_#{System.unique_integer()}")
      File.mkdir_p!(models_path)

      on_exit(fn ->
        File.rm_rf!(models_path)
      end)

      result = Trainer.train_entity_model(stats, models_path: models_path)

      assert is_map(result)
    end
  end

  describe "train_and_save/1 (integration)" do
    @tag :slow
    @tag :integration
    @tag timeout: 300_000
    test "creates all model files with temp models_path" do
      temp_dir = System.tmp_dir!()
      models_path = Path.join(temp_dir, "test_full_training_#{System.unique_integer()}")
      File.mkdir_p!(models_path)

      on_exit(fn ->
        File.rm_rf!(models_path)
      end)

      result = Trainer.train_and_save(models_path: models_path)

      assert {:ok, stats} = result
      assert is_map(stats)
      assert stats.intent_samples > 0
      assert stats.vocab_size > 0
      assert File.exists?(Path.join(models_path, "embedder.term"))
      assert File.exists?(Path.join(models_path, "gazetteer.term"))
    end
  end
end
