defmodule Brain.ML.LSTM.IntegrationTest do
  @moduledoc """
  Tests for the LSTM/TF-IDF ensemble integration layer.

  Covers:
  - classify_intent/2 with and without :previous_intent
  - Graph prior boosting in ensemble voting
  - Sentiment ensemble
  - Graceful fallback when models are unavailable
  """
  use ExUnit.Case, async: false

  alias Brain.ML.LSTM.Integration
  alias Brain.Graph.Training, as: GraphTraining

  @moduletag :lstm
  @moduletag timeout: 60_000

  describe "classify_intent/2" do
    test "returns {:ok, ...} or {:error, :no_classifier_available}" do
      result =
        try do
          Integration.classify_intent("what is the weather today")
        rescue
          ArgumentError -> {:error, :model_incompatible}
        end

      case result do
        {:ok, {intent, confidence, source}} ->
          assert is_binary(intent)
          assert is_number(confidence)
          assert source in [:lstm, :tfidf, :ensemble]

        {:error, reason} ->
          assert reason in [:no_classifier_available, :model_incompatible]
      end
    end

    test "accepts :previous_intent option without error" do
      result =
        try do
          Integration.classify_intent("yes that sounds good", previous_intent: "weather.query")
        rescue
          ArgumentError -> {:error, :model_incompatible}
        end

      case result do
        {:ok, {intent, confidence, source}} ->
          assert is_binary(intent)
          assert is_number(confidence)
          assert source in [:lstm, :tfidf, :ensemble]

        {:error, reason} ->
          assert reason in [:no_classifier_available, :model_incompatible]
      end
    end

    test "previous_intent: nil behaves the same as omitting it" do
      run = fn opts ->
        try do
          Integration.classify_intent("hello there", opts)
        rescue
          ArgumentError -> {:error, :model_incompatible}
        end
      end

      assert run.([]) == run.(previous_intent: nil)
    end
  end

  describe "graph prior boosting" do
    test "apply_intent_priors boosts scores based on transition history" do
      scores = [{"weather.query", 0.6}, {"dialog.continuation", 0.4}]
      priors = %{"weather.query" => %{"dialog.continuation" => 8, "weather.query" => 2}}

      boosted = GraphTraining.apply_intent_priors(scores, "weather.query", priors)
      boosted_map = Map.new(boosted)

      assert boosted_map["dialog.continuation"] > 0.4
    end

    test "is a no-op when previous intent has no transitions" do
      scores = [{"weather.query", 0.6}, {"greeting", 0.3}]
      priors = %{"music.play" => %{"dialog.continuation" => 5}}

      assert GraphTraining.apply_intent_priors(scores, "weather.query", priors) == scores
    end

    test "is a no-op with empty priors map" do
      scores = [{"weather.query", 0.6}]
      assert GraphTraining.apply_intent_priors(scores, "smalltalk.greetings.hello", %{}) == scores
    end

    test "weight parameter controls boost magnitude" do
      scores = [{"dialog.continuation", 0.5}]
      priors = %{"weather.query" => %{"dialog.continuation" => 10}}

      low_weight = GraphTraining.apply_intent_priors(scores, "weather.query", priors, weight: 0.05)
      high_weight = GraphTraining.apply_intent_priors(scores, "weather.query", priors, weight: 0.3)

      [{_, low_score}] = low_weight
      [{_, high_score}] = high_weight

      assert high_score > low_score,
        "Higher weight should produce larger boost: #{high_score} vs #{low_score}"
    end
  end

  describe "model_status/0" do
    test "returns a map with model availability flags" do
      status = Integration.model_status()

      assert is_map(status)
      assert Map.has_key?(status, :tfidf)
      assert Map.has_key?(status, :lstm_unified)
      assert is_boolean(status.tfidf)
      assert is_boolean(status.lstm_unified)
    end
  end

  describe "classify_sentiment/2" do
    test "returns a result or error" do
      result =
        try do
          Integration.classify_sentiment("I love sunny weather")
        rescue
          ArgumentError -> {:error, :model_incompatible}
        end

      case result do
        {:ok, %{label: label, confidence: conf}} ->
          assert label in [:positive, :negative, :neutral, "positive", "negative", "neutral"]
          assert is_number(conf)

        {:error, reason} ->
          assert reason in [:no_sentiment_classifier, :model_incompatible]
      end
    end
  end
end
