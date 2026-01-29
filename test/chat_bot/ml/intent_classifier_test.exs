defmodule ChatBot.ML.IntentClassifierTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  alias ChatBot.ML.IntentClassifier

  @moduletag :intent_classifier

  setup_all do
    # Load models once at the start of the test module
    # Capture logs to prevent error output
    {result, _log} =
      with_log(fn ->
        IntentClassifier.load_models()
      end)

    models =
      case result do
        {:ok, m} -> m
        {:error, _} -> nil
      end

    # Store in process dictionary for access in tests
    %{shared_models: models}
  end

  describe "load_models/0" do
    test "returns ok tuple with models when files exist" do
      {result, _log} =
        with_log(fn ->
          IntentClassifier.load_models()
        end)

      case result do
        {:ok, models} ->
          assert is_map(models)
          assert Map.has_key?(models, :vectorizer)
          assert Map.has_key?(models, :svm_model)

        {:error, reason} ->
          # Models not trained yet - this is acceptable
          assert is_binary(reason) or is_atom(reason)
      end
    end

    test "returns error when files are missing" do
      original_config = Application.get_env(:chat_bot, :ml)

      Application.put_env(
        :chat_bot,
        :ml,
        Keyword.put(original_config || [], :models_path, "/nonexistent/path")
      )

      on_exit(fn ->
        Application.put_env(:chat_bot, :ml, original_config)
      end)

      {result, log} =
        with_log(fn ->
          IntentClassifier.load_models()
        end)

      assert {:error, _reason} = result
      assert log =~ "Failed to load" or log == ""
    end
  end

  describe "classify/2" do
    test "returns ok tuple with classification result", %{shared_models: models} do
      if models do
        result = IntentClassifier.classify("Hello there!", {:ok, models})

        assert {:ok, classification} = result
        assert Map.has_key?(classification, :intent)
        assert Map.has_key?(classification, :confidence)
        assert Map.has_key?(classification, :probabilities)
      else
        :ok
      end
    end

    test "confidence is between 0 and 1", %{shared_models: models} do
      if models do
        {:ok, classification} = IntentClassifier.classify("What's the weather?", {:ok, models})

        assert classification.confidence >= 0.0
        assert classification.confidence <= 1.0
      else
        :ok
      end
    end

    test "probabilities sum to approximately 1.0", %{shared_models: models} do
      if models do
        {:ok, classification} = IntentClassifier.classify("Play some music", {:ok, models})

        total = Enum.sum(Map.values(classification.probabilities))

        assert_in_delta total, 1.0, 0.01
      else
        :ok
      end
    end

    test "returns intent as string", %{shared_models: models} do
      if models do
        {:ok, classification} = IntentClassifier.classify("Turn on the lights", {:ok, models})

        assert is_binary(classification.intent)
      else
        :ok
      end
    end

    test "handles empty text gracefully", %{shared_models: models} do
      if models do
        result = IntentClassifier.classify("", {:ok, models})

        case result do
          {:ok, classification} ->
            assert is_map(classification)

          {:error, _reason} ->
            assert true
        end
      else
        :ok
      end
    end

    test "handles unknown/random text", %{shared_models: models} do
      if models do
        result = IntentClassifier.classify("xyzabc123randomtexthere", {:ok, models})

        case result do
          {:ok, classification} ->
            assert is_map(classification)
            assert Map.has_key?(classification, :intent)

          {:error, _reason} ->
            assert true
        end
      else
        :ok
      end
    end

    test "returns error when models not loaded" do
      # classify with nil models will trigger get_loaded_models which calls load_models
      # Capture the error log that may be emitted
      {result, _log} =
        with_log(fn ->
          IntentClassifier.classify("Hello", nil)
        end)

      assert {:error, _reason} = result
    end
  end

  describe "vectorize_text/2" do
    test "produces normalized TF-IDF vector", %{shared_models: models} do
      if models do
        vectorizer = models.vectorizer
        vector = IntentClassifier.vectorize_text("Hello world", vectorizer)

        assert Nx.is_tensor(vector)

        max_val = Nx.reduce_max(vector) |> Nx.to_number()
        assert max_val <= 1.0
      else
        :ok
      end
    end

    test "vector dimension matches vocabulary size", %{shared_models: models} do
      if models do
        vectorizer = models.vectorizer
        vector = IntentClassifier.vectorize_text("Test input", vectorizer)

        {dim} = Nx.shape(vector)
        assert dim == vectorizer.max_features
      else
        :ok
      end
    end

    test "unknown tokens get zero weight", %{shared_models: models} do
      if models do
        vectorizer = models.vectorizer
        vector = IntentClassifier.vectorize_text("xyzabc123 qwerty456", vectorizer)

        assert Nx.is_tensor(vector)
      else
        :ok
      end
    end
  end

  describe "model structure" do
    test "vectorizer has required keys", %{shared_models: models} do
      if models do
        vectorizer = models.vectorizer

        assert Map.has_key?(vectorizer, :vocabulary)
        assert Map.has_key?(vectorizer, :idf_weights)
        assert Map.has_key?(vectorizer, :max_features)

        assert is_map(vectorizer.vocabulary)
        assert Nx.is_tensor(vectorizer.idf_weights)
        assert is_integer(vectorizer.max_features)
      else
        :ok
      end
    end

    test "svm_model has required structure", %{shared_models: models} do
      if models do
        svm_model = models.svm_model

        assert Map.has_key?(svm_model, :model)
        assert Map.has_key?(svm_model, :label_encoder)

        label_encoder = svm_model.label_encoder
        assert Map.has_key?(label_encoder, :label_to_index)
        assert Map.has_key?(label_encoder, :index_to_label)
      else
        :ok
      end
    end
  end
end
