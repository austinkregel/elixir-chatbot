defmodule Brain.ML.FeatureVectorClassifierTest do
  @moduledoc """
  Unit tests for `Brain.ML.FeatureVectorClassifier`.

  This classifier is a dense centroid-based classifier that consumes
  feature vectors (`list(float())`) directly, never text. It exists
  so the ChunkProfile axis classifiers (`:intent_domain`, `:tense_class`,
  `:aspect_class`, `:urgency`, `:certainty_level`, `:coarse_semantic_class`)
  cannot be biased by proper-noun token identity in the training data.

  These tests are pure and require no loaded models. They lock the
  shape of the module's public API before implementation.
  """

  use ExUnit.Case, async: false

  alias Brain.ML.FeatureVectorClassifier

  describe "train/1" do
    test "builds a model with per-label prototype lists and a declared input_dim" do
      training = [
        {[0.0, 1.0, 0.0], "calendar"},
        {[0.1, 0.9, 0.0], "calendar"},
        {[1.0, 0.0, 0.0], "weather"},
        {[0.9, 0.1, 0.0], "weather"},
        {[0.0, 0.0, 1.0], "meta"}
      ]

      model = FeatureVectorClassifier.train(training)

      assert FeatureVectorClassifier.input_dim(model) == 3
      assert map_size(model.label_centroids) == 3
      assert Map.has_key?(model.label_centroids, "calendar")
      assert Map.has_key?(model.label_centroids, "weather")
      assert Map.has_key?(model.label_centroids, "meta")

      # A label holds a list of prototypes, k = min(4, max(1, n ÷ 10)).
      # "calendar" has 2 examples, so one prototype, of input_dim components.
      assert [calendar_prototype] = model.label_centroids["calendar"]
      assert length(calendar_prototype) == 3
    end

    test "the single prototype of a label averages its training vectors componentwise" do
      training = [
        {[0.0, 0.0, 0.0], "zeros"},
        {[1.0, 1.0, 1.0], "zeros"}
      ]

      model = FeatureVectorClassifier.train(training)

      assert model.label_centroids["zeros"] == [[0.5, 0.5, 0.5]]
    end

    test "rejects mixed-dimension training data" do
      training = [
        {[0.0, 1.0], "a"},
        {[0.0, 1.0, 0.0], "b"}
      ]

      assert_raise ArgumentError, ~r/dimension/i, fn ->
        FeatureVectorClassifier.train(training)
      end
    end

    test "empty training data produces an empty model" do
      model = FeatureVectorClassifier.train([])

      assert FeatureVectorClassifier.input_dim(model) == 0
      assert model.label_centroids == %{}
    end
  end

  describe "classify/2" do
    setup do
      training = [
        {[0.0, 1.0, 0.0], "calendar"},
        {[1.0, 0.0, 0.0], "weather"},
        {[0.0, 0.0, 1.0], "meta"}
      ]

      {:ok, model: FeatureVectorClassifier.train(training)}
    end

    test "returns the nearest centroid's label and top_k in details", %{model: model} do
      assert {:ok, "weather", confidence, details} =
               FeatureVectorClassifier.classify([0.95, 0.05, 0.0], model)

      assert is_float(confidence)
      assert confidence > 0.5
      assert is_list(details[:top_k])
      assert length(details[:top_k]) == 3
      assert hd(details[:top_k]) == {"weather", details[:top_score]}
    end

    test "returns calendar for calendar-leaning vector", %{model: model} do
      assert {:ok, "calendar", _conf, _details} =
               FeatureVectorClassifier.classify([0.1, 0.9, 0.0], model)
    end

    test "returns meta for meta-leaning vector", %{model: model} do
      assert {:ok, "meta", _conf, _details} =
               FeatureVectorClassifier.classify([0.0, 0.1, 0.9], model)
    end

    test "rejects vectors of wrong dimensionality", %{model: model} do
      assert :error == FeatureVectorClassifier.classify([1.0, 0.0], model)
    end

    test "empty model returns :error" do
      empty = FeatureVectorClassifier.train([])
      assert :error == FeatureVectorClassifier.classify([1.0, 2.0, 3.0], empty)
    end
  end

  describe "serialization compatibility with existing model loader" do
    test "model is a plain term (:erlang.term_to_binary round-trips)" do
      model = FeatureVectorClassifier.train([{[1.0, 0.0], "a"}, {[0.0, 1.0], "b"}])

      restored = model |> :erlang.term_to_binary() |> :erlang.binary_to_term()

      assert restored == model
    end

    test "model declares its kind so MicroClassifiers can dispatch" do
      model = FeatureVectorClassifier.train([{[1.0], "a"}])

      assert Map.get(model, :kind) == :feature_vector,
             "FeatureVectorClassifier models must declare :kind => :feature_vector " <>
               "so MicroClassifiers can route classify/classify_vector correctly."
    end
  end

  describe "determinism and completeness" do
    # Built from a formula, not random draws, and imbalanced (60 vs 25) with
    # enough examples per class that K-means++ runs (k > 1).
    defp imbalanced_training do
      big = for i <- 1..60, do: {[:math.sin(i), :math.cos(i), i / 60], "big"}
      small = for i <- 1..25, do: {[:math.cos(i) + 2.0, :math.sin(i), i / 25], "small"}
      big ++ small
    end

    test "the same data and seed always produce the same model" do
      first = FeatureVectorClassifier.train(imbalanced_training(), seed: 7)
      second = FeatureVectorClassifier.train(imbalanced_training(), seed: 7)

      assert first == second
    end

    test "records the seed it trained with, defaulting to the configured one" do
      assert FeatureVectorClassifier.train(imbalanced_training(), seed: 7).training_seed == 7

      assert FeatureVectorClassifier.train(imbalanced_training()).training_seed ==
               Brain.ML.TrainingSeed.get!()
    end

    test "leaves the calling process's random state untouched" do
      :rand.seed(:exsss, {1, 2, 3})
      expected = :rand.uniform()

      :rand.seed(:exsss, {1, 2, 3})
      FeatureVectorClassifier.train(imbalanced_training(), seed: 7)

      assert :rand.uniform() == expected
    end

    test "uses every example of a large class rather than dropping some to match smaller ones" do
      model = FeatureVectorClassifier.train(imbalanced_training(), seed: 7)

      # 60 examples give min(4, 60 div 10) = 4 prototypes. Undersampling the
      # class to the size of the smaller one (25) would give only 2.
      assert length(model.label_centroids["big"]) == 4
    end

    test "rejects an unknown option instead of ignoring it" do
      assert_raise ArgumentError, fn ->
        FeatureVectorClassifier.train(imbalanced_training(), balance: true)
      end
    end
  end
end
