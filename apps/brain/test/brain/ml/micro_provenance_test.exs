defmodule Brain.ML.MicroProvenanceTest do
  @moduledoc """
  Task 072's gate. These tests assert the thing that was missing: that a model
  whose training data moved underneath it **raises** rather than loading.

  The bug being guarded is specific. Six feature-vector classifiers were trained
  on vectors whose memory dimensions inverted after the AGE graph was rebuilt —
  `mem_novelty` went from 0.001 to 0.999 for the same input. The vector length
  stayed 343, so no dimension check fired, `manifest.json` recorded the hashes
  but had zero readers, and the models kept serving predictions built on a sign
  flip.
  """

  use ExUnit.Case, async: false

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.ML.MicroProvenance

  # A real classifier whose training data is on disk, so the hash is a genuine
  # one rather than a fixture. intent_full is the classifier task 072 named
  # first.
  @vector_classifier "intent_full"
  @text_classifier "personal_question"

  defp vector_model(overrides \\ %{}) do
    Map.merge(%{kind: :feature_vector, input_dim: 343, label_centroids: %{"a" => [[0.0]]}}, overrides)
  end

  defp text_model(overrides \\ %{}) do
    Map.merge(%{kind: :tfidf, label_centroids: %{"a" => [[0.0]]}}, overrides)
  end

  describe "stamp!/2" do
    test "records the training data's real SHA-256" do
      stamped = MicroProvenance.stamp!(vector_model(), @vector_classifier)

      assert [%{name: "intent_full.json", version: "repo:data/classifiers", sha256: sha}] =
               stamped.training.inputs

      assert sha =~ ~r/^[0-9a-f]{64}$/

      # Not a fixture: it must match the file on disk.
      expected =
        MicroProvenance.training_data_path(@vector_classifier)
        |> Brain.Analysis.RunProvenance.sha256_file!()

      assert sha == expected
    end

    test "records the extractor schema fingerprint for a feature-vector model" do
      stamped = MicroProvenance.stamp!(vector_model(), @vector_classifier)

      assert stamped.training.schema_fingerprint == ChunkFeatures.schema_fingerprint()
    end

    test "omits the schema fingerprint for a text model" do
      stamped = MicroProvenance.stamp!(text_model(), @text_classifier)

      refute Map.has_key?(stamped.training, :schema_fingerprint),
             "a text classifier consumes strings, not the feature vector. Recording a " <>
               "fingerprint would manufacture a false mismatch every time an unrelated " <>
               "dimension was renamed."
    end

    test "records the commit and a timestamp" do
      stamped = MicroProvenance.stamp!(vector_model(), @vector_classifier)

      assert stamped.training.git_sha =~ ~r/^[0-9a-f]{40}$/
      assert {:ok, _, _} = DateTime.from_iso8601(stamped.training.at)
    end

    test "raises when the training data is not on disk" do
      assert_raise RuntimeError, ~r/no training data for micro-classifier nonexistent_classifier/, fn ->
        MicroProvenance.stamp!(vector_model(), "nonexistent_classifier")
      end
    end
  end

  describe "stamped?/1" do
    test "distinguishes a stamped model from one that predates the gate" do
      assert MicroProvenance.stamped?(MicroProvenance.stamp!(vector_model(), @vector_classifier))
      refute MicroProvenance.stamped?(vector_model())
      refute MicroProvenance.stamped?(vector_model(%{training: %{}}))
      refute MicroProvenance.stamped?(vector_model(%{training: %{inputs: []}}))
    end
  end

  describe "check_current!/3 accepts a model that is actually current" do
    test "a freshly stamped feature-vector model passes" do
      model = MicroProvenance.stamp!(vector_model(), @vector_classifier)

      assert :ok = MicroProvenance.check_current!(model, @vector_classifier, "test/path.term")
    end

    test "a freshly stamped text model passes" do
      model = MicroProvenance.stamp!(text_model(), @text_classifier)

      assert :ok = MicroProvenance.check_current!(model, @text_classifier, "test/path.term")
    end
  end

  describe "check_current!/3 refuses a stale model" do
    test "raises when the recorded training hash is not the hash on disk" do
      # Exactly task 072: the data was regenerated after the model was trained.
      model =
        MicroProvenance.stamp!(vector_model(), @vector_classifier)
        |> put_in([:training, :inputs], [
          %{name: "intent_full.json", version: "repo:data/classifiers", sha256: String.duplicate("0", 64)}
        ])

      assert_raise RuntimeError, ~r/is stale/, fn ->
        MicroProvenance.check_current!(model, @vector_classifier, "test/path.term")
      end
    end

    test "the message names both hashes, so the mismatch is actionable" do
      model =
        MicroProvenance.stamp!(vector_model(), @vector_classifier)
        |> put_in([:training, :inputs], [
          %{name: "intent_full.json", version: "repo:data/classifiers", sha256: String.duplicate("0", 64)}
        ])

      error =
        assert_raise(RuntimeError, fn ->
          MicroProvenance.check_current!(model, @vector_classifier, "some/model.term")
        end)

      message = error.message

      assert message =~ "some/model.term"
      assert message =~ "trained on"
      assert message =~ "on disk now"
      assert message =~ "mix train_micro"
    end

    test "raises when a feature-vector model records no schema fingerprint" do
      # A model that predates the gate. Loading it would mean trusting 343
      # dimensions whose meaning cannot be established.
      model =
        vector_model()
        |> Map.put(:training, %{
          inputs: [MicroProvenance.training_input!(@vector_classifier)],
          git_sha: String.duplicate("a", 40),
          at: DateTime.utc_now() |> DateTime.to_iso8601()
        })

      assert_raise RuntimeError, ~r/records no extractor schema/, fn ->
        MicroProvenance.check_current!(model, @vector_classifier, "test/path.term")
      end
    end

    test "raises when the extractor schema fingerprint has moved" do
      # This is the drift measured on 2026-09-25: the fingerprint moved from
      # fb3ce7f1e4739cb3 to bc289842ba5ccb3d with the extractor code unchanged,
      # because feature group 23 takes its names from the AGE graph at runtime.
      model =
        MicroProvenance.stamp!(vector_model(), @vector_classifier)
        |> put_in([:training, :schema_fingerprint], "fb3ce7f1e4739cb3")

      error =
        assert_raise(RuntimeError, fn ->
          MicroProvenance.check_current!(model, @vector_classifier, "test/path.term")
        end)

      message = error.message

      assert message =~ "fb3ce7f1e4739cb3"
      assert message =~ ChunkFeatures.schema_fingerprint()
      assert message =~ "vector length is unchanged"
      assert message =~ "group 23"
    end

    test "a text model is not gated on the schema fingerprint" do
      # The inverse of the vector case: renaming a feature dimension must not
      # invalidate a classifier that never reads the vector.
      model =
        MicroProvenance.stamp!(text_model(), @text_classifier)
        |> put_in([:training, :schema_fingerprint], "0000000000000000")

      assert :ok = MicroProvenance.check_current!(model, @text_classifier, "test/path.term")
    end

    test "raises for a model with no provenance at all" do
      assert_raise RuntimeError, ~r/is stale/, fn ->
        MicroProvenance.check_current!(vector_model(), @vector_classifier, "test/path.term")
      end
    end
  end

  describe "training_data_path/1" do
    test "resolves to the umbrella's data directory, not the app's" do
      path = MicroProvenance.training_data_path(@vector_classifier)

      assert Path.basename(path) == "intent_full.json"
      assert File.regular?(path), "#{path} does not exist; cwd is #{File.cwd!()}"

      # The bug this guards: under `mix test` the cwd is apps/brain, so a
      # cwd-relative "data/classifiers" resolves to apps/brain/data/classifiers.
      refute path =~ "apps/brain/data"
    end
  end
end
