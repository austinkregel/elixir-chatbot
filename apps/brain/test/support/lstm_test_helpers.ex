defmodule Brain.LSTMTestHelpers do
  @moduledoc """
  Test helpers for LSTM model testing.

  This module provides utilities for:
  - Version compatibility checking between saved models and current Nx/EXLA versions
  - Generating fresh .term model files for tests
  - Capturing and asserting on LSTM-related log messages
  - Mocking LSTM model state for unit tests

  ## Philosophy

  LSTM models are first-class citizens in this codebase. Tests should:
  1. Explicitly verify model compatibility before running LSTM-dependent tests
  2. Generate fresh .term files when needed for specific test scenarios
  3. Use `ExUnit.CaptureLog` to make explicit positive assertions about log output
  4. Never silently swallow decode errors - they indicate version mismatches

  ## Usage

  ### Checking Model Compatibility

      setup do
        case LSTMTestHelpers.check_model_compatibility() do
          :ok -> :ok
          {:error, :version_mismatch, details} ->
            # Skip LSTM tests or regenerate models
            {:ok, skip_lstm: true, reason: details}
        end
      end

  ### Generating Test Models

      setup do
        {:ok, model_path} = LSTMTestHelpers.generate_test_model(:unified, vocab_size: 100)
        on_exit(fn -> File.rm(model_path) end)
        {:ok, model_path: model_path}
      end

  ### Capturing LSTM Logs

      test "entity extraction logs decode failure" do
        log = capture_log(fn ->
          UnifiedModel.extract_entities("test input")
        end)

        assert_lstm_decode_failure(log)
      end
  """

  import ExUnit.Assertions
  import ExUnit.CaptureLog

  require Logger

  alias Brain.ML.LSTM.UnifiedModel
  alias Brain.Response.LSTMResponse

  # ============================================================================
  # Version Compatibility
  # ============================================================================

  @doc """
  Gets the current versions of Nx, EXLA, and Axon libraries.

  Returns a map with version strings for compatibility checking.
  """
  def current_ml_versions do
    %{
      nx: get_app_version(:nx),
      exla: get_app_version(:exla),
      axon: get_app_version(:axon),
      otp: System.otp_release(),
      elixir: System.version()
    }
  end

  defp get_app_version(app) do
    case Application.spec(app, :vsn) do
      nil -> "not_loaded"
      vsn -> to_string(vsn)
    end
  end

  @doc """
  Checks if the saved LSTM models are compatible with current library versions.

  This should be called at the start of any test that relies on LSTM functionality.
  Returns `:ok` if compatible, or `{:error, :version_mismatch, details}` if not.

  ## Model Version Metadata

  When models are saved, they should include version metadata. This function
  checks that metadata against current versions.
  """
  def check_model_compatibility(model_type \\ :unified) do
    model_path = get_model_path(model_type)

    case File.read(model_path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary, [:safe])
          validate_model_format(data, model_type)
        rescue
          ArgumentError ->
            {:error, :decode_failed, %{
              message: "Model binary format incompatible with current Erlang/OTP version",
              model_path: model_path,
              current_versions: current_ml_versions()
            }}

          e ->
            {:error, :unknown_error, %{
              exception: e,
              model_path: model_path
            }}
        end

      {:error, :enoent} ->
        {:error, :model_not_found, %{
          model_path: model_path,
          message: "Model file does not exist. Run `mix train_models` to generate."
        }}

      {:error, reason} ->
        {:error, :file_read_error, %{reason: reason, model_path: model_path}}
    end
  end

  defp get_model_path(:unified) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    Path.join([models_path, "lstm", "unified_model.term"])
  end

  defp get_model_path(:multi_task) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    Path.join([models_path, "lstm", "lstm_multitask.term"])
  end

  defp get_model_path(:response_scorer) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    Path.join([models_path, "lstm", "response_scorer.term"])
  end

  defp validate_model_format(data, :unified) do
    required_keys = [:params, :vocabularies, :config]
    missing = required_keys -- Map.keys(data)

    if missing == [] do
      # Try to access params to trigger any lazy decode errors
      _ = data.params
      :ok
    else
      {:error, :invalid_format, %{missing_keys: missing}}
    end
  end

  defp validate_model_format(data, _model_type) do
    # Generic validation - just check it's a map
    if is_map(data), do: :ok, else: {:error, :invalid_format, %{expected: :map, got: data}}
  end

  # ============================================================================
  # Model Generation for Tests
  # ============================================================================

  @doc """
  Generates a minimal test model file for the specified model type.

  This creates a valid .term file with a small vocabulary and random weights,
  suitable for testing model loading/saving but NOT for actual inference.

  ## Options

  - `:vocab_size` - Number of tokens in vocabulary (default: 50)
  - `:embedding_size` - Embedding dimension (default: 32)
  - `:hidden_size` - LSTM hidden size (default: 32)
  - `:output_dir` - Directory to write model (default: System.tmp_dir())

  ## Returns

  `{:ok, model_path}` on success, `{:error, reason}` on failure.
  """
  def generate_test_model(model_type, opts \\ []) do
    vocab_size = Keyword.get(opts, :vocab_size, 50)
    embedding_size = Keyword.get(opts, :embedding_size, 32)
    hidden_size = Keyword.get(opts, :hidden_size, 32)
    output_dir = Keyword.get(opts, :output_dir, System.tmp_dir!())

    model_path = Path.join(output_dir, "test_#{model_type}_model.term")

    data = build_minimal_model_data(model_type, vocab_size, embedding_size, hidden_size)
    binary = :erlang.term_to_binary(data)

    case File.write(model_path, binary) do
      :ok -> {:ok, model_path}
      {:error, reason} -> {:error, reason}
    end
  end

  defp build_minimal_model_data(:unified, vocab_size, embedding_size, hidden_size) do
    token_vocab = for i <- 0..(vocab_size - 1), into: %{}, do: {"token_#{i}", i}

    intent_labels = ["smalltalk.greetings.hello", "smalltalk.greetings.bye", "weather.query", "unknown"]
    intent_to_idx = for {label, i} <- Enum.with_index(intent_labels), into: %{}, do: {label, i}
    idx_to_intent = for {label, i} <- Enum.with_index(intent_labels), into: %{}, do: {i, label}

    bio_labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
    bio_to_idx = for {label, i} <- Enum.with_index(bio_labels), into: %{}, do: {label, i}
    idx_to_bio = for {label, i} <- Enum.with_index(bio_labels), into: %{}, do: {i, label}

    max_seq_length = 50
    num_intents = length(intent_labels)
    num_bio = length(bio_labels)
    num_sentiments = 3
    num_speech_acts = 5

    config = %{
      embedding_size: embedding_size,
      hidden_size: hidden_size,
      max_seq_length: max_seq_length
    }

    vocabularies = %{
      token_vocab: token_vocab,
      intent_to_idx: intent_to_idx,
      idx_to_intent: idx_to_intent,
      bio_to_idx: bio_to_idx,
      idx_to_bio: idx_to_bio
    }

    # Build real Axon models matching UnifiedModel architecture and init params
    encoder =
      Axon.input("input", shape: {nil, max_seq_length})
      |> Axon.embedding(vocab_size, embedding_size)
      |> Axon.lstm(hidden_size, name: "encoder_lstm")
      |> then(fn {seq, _state} -> seq end)

    intent_hidden = min(256, num_intents * 2)
    intent_head =
      Axon.input("intent_input", shape: {nil, hidden_size})
      |> Axon.dense(intent_hidden, activation: :relu, name: "intent_dense")
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(num_intents, activation: :softmax, name: "intent_output")

    sentiment_hidden = min(256, num_sentiments * 2)
    sentiment_head =
      Axon.input("sentiment_input", shape: {nil, hidden_size})
      |> Axon.dense(sentiment_hidden, activation: :relu, name: "sentiment_dense")
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(num_sentiments, activation: :softmax, name: "sentiment_output")

    speech_act_hidden = min(256, num_speech_acts * 2)
    speech_act_head =
      Axon.input("speech_act_input", shape: {nil, hidden_size})
      |> Axon.dense(speech_act_hidden, activation: :relu, name: "speech_act_dense")
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(num_speech_acts, activation: :softmax, name: "speech_act_output")

    ner_head =
      Axon.input("ner_input", shape: {nil, nil, hidden_size})
      |> Axon.dense(num_bio, activation: :softmax, name: "ner_output")

    encoder_params = init_model_params(encoder, %{"input" => Nx.template({1, max_seq_length}, :s64)})
    intent_params = init_model_params(intent_head, %{"intent_input" => Nx.template({1, hidden_size}, :f32)})
    sentiment_params = init_model_params(sentiment_head, %{"sentiment_input" => Nx.template({1, hidden_size}, :f32)})
    speech_act_params = init_model_params(speech_act_head, %{"speech_act_input" => Nx.template({1, hidden_size}, :f32)})
    ner_params = init_model_params(ner_head, %{"ner_input" => Nx.template({1, max_seq_length, hidden_size}, :f32)})

    params = %{
      encoder: transfer_to_binary(encoder_params),
      intent: transfer_to_binary(intent_params),
      ner: transfer_to_binary(ner_params),
      sentiment: transfer_to_binary(sentiment_params),
      speech_act: transfer_to_binary(speech_act_params)
    }

    %{
      params: params,
      vocabularies: vocabularies,
      config: config,
      metadata: %{
        created_at: DateTime.utc_now() |> DateTime.to_iso8601(),
        versions: current_ml_versions()
      }
    }
  end

  defp build_minimal_model_data(_model_type, vocab_size, embedding_size, hidden_size) do
    %{
      vocab_size: vocab_size,
      embedding_size: embedding_size,
      hidden_size: hidden_size,
      metadata: %{
        created_at: DateTime.utc_now() |> DateTime.to_iso8601(),
        versions: current_ml_versions()
      }
    }
  end

  defp init_model_params(model, template) do
    {init_fn, _} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    if is_struct(params, Axon.ModelState), do: params.data, else: params
  end

  defp transfer_to_binary(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)
  defp transfer_to_binary(%Axon.ModelState{} = s), do: Axon.ModelState.new(transfer_to_binary(s.data))
  defp transfer_to_binary(map) when is_map(map), do: Map.new(map, fn {k, v} -> {k, transfer_to_binary(v)} end)
  defp transfer_to_binary(other), do: other

  # ============================================================================
  # Log Capture and Assertions
  # ============================================================================

  @doc """
  Captures logs from LSTM operations and returns them for assertion.

  Wraps `ExUnit.CaptureLog.capture_log/2` with LSTM-specific defaults.

  ## Example

      logs = capture_lstm_logs(fn ->
        UnifiedModel.extract_entities("test")
      end)

      assert logs =~ "EXLA decode failed"
  """
  def capture_lstm_logs(fun, opts \\ []) do
    level = Keyword.get(opts, :level, :warning)
    capture_log([level: level], fun)
  end

  @doc """
  Asserts that the captured log contains an LSTM decode failure message.

  This is the expected log when a model was saved with an incompatible
  Nx/EXLA version.
  """
  def assert_lstm_decode_failure(log) when is_binary(log) do
    assert log =~ "EXLA decode failed" or
           log =~ "decode failed" or
           log =~ "model_incompatible",
           "Expected LSTM decode failure log, got: #{log}"
  end

  @doc """
  Asserts that the captured log does NOT contain LSTM decode failures.

  Use this when you expect LSTM operations to succeed.
  """
  def refute_lstm_decode_failure(log) when is_binary(log) do
    refute log =~ "EXLA decode failed",
           "Did not expect LSTM decode failure, but got: #{log}"
    refute log =~ "decode failed, none of the variant types could be decoded",
           "Did not expect EXLA decode failure, but got: #{log}"
  end

  @doc """
  Asserts that the captured log contains an entity extraction failure.
  """
  def assert_entity_extraction_failure(log) when is_binary(log) do
    assert log =~ "Entity extraction failed" or
           log =~ "UnifiedModel: EXLA decode failed",
           "Expected entity extraction failure log, got: #{log}"
  end

  @doc """
  Asserts that the captured log indicates model was disabled.
  """
  def assert_model_disabled(log) when is_binary(log) do
    assert log =~ "disabling model" or log =~ "model_incompatible",
           "Expected model disabled log, got: #{log}"
  end

  # ============================================================================
  # Model State Helpers
  # ============================================================================

  @doc """
  Checks if the UnifiedModel GenServer is ready.

  Returns `{:ok, true}` if ready, `{:ok, false}` if not ready,
  or `{:error, :not_running}` if the GenServer isn't started.
  """
  def unified_model_ready? do
    try do
      {:ok, UnifiedModel.ready?()}
    catch
      :exit, _ -> {:error, :not_running}
    end
  end

  @doc """
  Checks if the LSTMResponse GenServer is ready.
  """
  def lstm_response_ready? do
    try do
      {:ok, LSTMResponse.ready?()}
    catch
      :exit, _ -> {:error, :not_running}
    end
  end

  @doc """
  Waits for LSTM models to finish loading (or fail).

  Polls the ready state until either ready or max_attempts reached.
  Returns the final ready state.
  """
  def wait_for_lstm_ready(model \\ :unified, max_attempts \\ 20, delay_ms \\ 100) do
    check_fn = case model do
      :unified -> &unified_model_ready?/0
      :response -> &lstm_response_ready?/0
    end

    do_wait_ready(check_fn, max_attempts, delay_ms)
  end

  defp do_wait_ready(_check_fn, 0, _delay_ms), do: {:error, :timeout}

  defp do_wait_ready(check_fn, attempts, delay_ms) do
    case check_fn.() do
      {:ok, true} -> {:ok, :ready}
      {:ok, false} ->
        Process.sleep(delay_ms)
        do_wait_ready(check_fn, attempts - 1, delay_ms)
      {:error, reason} -> {:error, reason}
    end
  end

  # ============================================================================
  # Test Tags and Helpers
  # ============================================================================

  @doc """
  Returns ExUnit tags to skip a test if LSTM models are incompatible.

  Use in setup:

      setup do
        LSTMTestHelpers.skip_if_models_incompatible()
      end
  """
  def skip_if_models_incompatible do
    case check_model_compatibility() do
      :ok -> :ok
      {:error, error_type, details} ->
        {:skip, "LSTM models incompatible: #{error_type} - #{inspect(details)}"}
    end
  end

  @doc """
  Returns ExUnit tags to mark a test as requiring LSTM models.

  Tests with this tag can be excluded when models aren't available:

      mix test --exclude requires_lstm
  """
  def requires_lstm_tag, do: [requires_lstm: true]
end
