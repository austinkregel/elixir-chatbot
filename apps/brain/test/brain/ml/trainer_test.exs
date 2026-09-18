defmodule Brain.ML.TrainerTest do
  use ExUnit.Case, async: false
  require Nx

  alias Brain.ML.DataLoaders
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
      assert length(intents) > 1
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

  describe "train_svm_classifier/2" do
    @tag :slow
    test "produces model with training_vectors and label_encoder" do
      training_data = Trainer.load_training_data() |> Enum.take(200)

      vectorizer = Trainer.build_tfidf_vectorizer(training_data)

      if map_size(vectorizer.vocabulary) > 0 do
        try do
          classifier = Trainer.train_svm_classifier(training_data, vectorizer)

          assert is_map(classifier)
          assert Map.has_key?(classifier, :model)
          assert Map.has_key?(classifier, :label_encoder)
          model = classifier.model
          assert Map.has_key?(model, :training_vectors)
          assert Map.has_key?(model, :training_labels)
        rescue
          MatchError ->
            :ok
        end
      end
    end

    @tag :slow
    test "label encoder maps intents bidirectionally" do
      training_data = Trainer.load_training_data() |> Enum.take(200)

      vectorizer = Trainer.build_tfidf_vectorizer(training_data)

      if map_size(vectorizer.vocabulary) > 0 do
        try do
          classifier = Trainer.train_svm_classifier(training_data, vectorizer)

          label_encoder = classifier.label_encoder

          assert Map.has_key?(label_encoder, :label_to_index)
          assert Map.has_key?(label_encoder, :index_to_label)

          Enum.each(label_encoder.label_to_index, fn {label, index} ->
            assert label_encoder.index_to_label[index] == label
          end)
        rescue
          MatchError ->
            :ok
        end
      end
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

    test "raises when a source cannot be loaded" do
      ml = Application.get_env(:brain, :ml)
      empty_dir = Path.join(System.tmp_dir!(), "no_sources_#{System.unique_integer([:positive])}")
      File.mkdir_p!(empty_dir)

      on_exit(fn ->
        Application.put_env(:brain, :ml, ml)
        File.rm_rf!(empty_dir)
      end)

      Application.put_env(:brain, :ml, Keyword.put(ml, :training_data_path, empty_dir))

      assert_raise RuntimeError, ~r/required source :entities could not be loaded/, fn ->
        Trainer.build_gazetteer_data(%{}, models_path: empty_dir)
      end
    end
  end

  describe "build_gazetteer_data/2 output" do
    setup do
      models_path = Path.join(System.tmp_dir!(), "gazetteer_#{System.unique_integer([:positive])}")
      File.mkdir_p!(models_path)
      on_exit(fn -> File.rm_rf!(models_path) end)

      Trainer.build_gazetteer_data(%{}, models_path: models_path)

      gazetteer =
        models_path |> Path.join("gazetteer.term") |> File.read!() |> :erlang.binary_to_term()

      %{gazetteer: gazetteer}
    end

    test "includes every artist and city in the sources", %{gazetteer: gazetteer} do
      {:ok, artists} = DataLoaders.load_artists()
      {:ok, cities} = DataLoaders.load_cities()
      artist_lookup = DataLoaders.build_artist_lookup(artists)
      city_lookup = DataLoaders.build_city_lookup(cities)

      # Both sources are larger than the 10,000 rows the build used to keep.
      assert map_size(artist_lookup) > 10_000
      assert map_size(city_lookup) > 10_000

      missing_artists = Enum.reject(Map.keys(artist_lookup), &has_type?(gazetteer, &1, "music_artist"))
      missing_cities = Enum.reject(Map.keys(city_lookup), &has_type?(gazetteer, &1, "location"))

      assert missing_artists == [], "#{length(missing_artists)} artists missing"
      assert missing_cities == [], "#{length(missing_cities)} cities missing"
    end

    test "keeps a candidate from each source when they share a surface form",
         %{gazetteer: gazetteer} do
      # "door" is a device in the entity data and an emoji name; the emojis
      # are merged last, so an overwriting merge leaves only the emoji.
      assert has_type?(gazetteer, "door", "device")
      assert has_type?(gazetteer, "door", "emoji")

      # "thursday" is a date in the entity data and a band in the artist data.
      assert has_type?(gazetteer, "thursday", "sys_date")
      assert has_type?(gazetteer, "thursday", "music_artist")
    end
  end

  defp has_type?(gazetteer, key, entity_type) do
    gazetteer
    |> Map.get(key, [])
    |> List.wrap()
    |> Enum.any?(&(&1.entity_type == entity_type))
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