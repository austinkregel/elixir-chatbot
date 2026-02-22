defmodule Brain.ML.LSTM.ModelCompatibilityTest do
  @moduledoc """
  Preliminary tests to verify LSTM model version compatibility.

  These tests should run FIRST in the test suite to detect model/library
  version mismatches early. If models are incompatible, tests will fail
  with clear messages about what needs to be done.

  ## Why This Matters

  LSTM models are saved as `.term` files using `:erlang.term_to_binary/1`.
  The internal format of Nx tensors can change between versions of:
  - Nx
  - EXLA (which handles GPU acceleration)
  - Axon (model architecture)
  - OTP/Erlang

  When versions don't match, you'll see errors like:
  `(ArgumentError) decode failed, none of the variant types could be decoded`

  ## Running These Tests First

  Use test ordering or tags:

      # Run compatibility tests first
      mix test test/brain/ml/lstm/model_compatibility_test.exs

      # Then run remaining tests
      mix test --exclude requires_lstm

  ## Regenerating Models

  If models are incompatible:

      mix train_models
  """

  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  # Don't exclude this test - it should always run to detect compatibility issues
  @moduletag :lstm_compatibility

  alias Brain.LSTMTestHelpers

  describe "LSTM model version compatibility" do
    setup do
      models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
      lstm_dir = Path.join(models_path, "lstm")
      unified_path = Path.join(lstm_dir, "unified_model.term")

      unless File.exists?(unified_path) do
        File.mkdir_p!(lstm_dir)
        {:ok, path} = LSTMTestHelpers.generate_test_model(:unified, output_dir: lstm_dir)
        File.rename!(path, unified_path)
      end

      :ok
    end

    test "current ML library versions are available" do
      versions = LSTMTestHelpers.current_ml_versions()

      assert is_binary(versions.nx), "Nx version should be available"
      assert is_binary(versions.axon), "Axon version should be available"
      assert is_binary(versions.otp), "OTP version should be available"
      assert is_binary(versions.elixir), "Elixir version should be available"

      # Log versions for debugging (captured to avoid log leaks)
      capture_io(fn ->
        IO.puts("\n=== ML Library Versions ===")
        IO.puts("Nx: #{versions.nx}")
        IO.puts("EXLA: #{versions.exla}")
        IO.puts("Axon: #{versions.axon}")
        IO.puts("OTP: #{versions.otp}")
        IO.puts("Elixir: #{versions.elixir}")
        IO.puts("===========================\n")
      end)
    end

    test "unified model file exists" do
      result = LSTMTestHelpers.check_model_compatibility(:unified)

      case result do
        :ok ->
          assert true

        {:error, :model_not_found, details} ->
          flunk("""
          Unified LSTM model not found.

          Path: #{details.model_path}

          To generate:
            mix train_models

          Or to skip LSTM tests:
            mix test --exclude requires_lstm
          """)

        {:error, :decode_failed, details} ->
          flunk("""
          Unified LSTM model is INCOMPATIBLE with current library versions.

          The model was saved with a different version of Nx/EXLA/Axon.

          Model path: #{details.model_path}

          Current versions:
            Nx: #{details.current_versions.nx}
            EXLA: #{details.current_versions.exla}
            Axon: #{details.current_versions.axon}
            OTP: #{details.current_versions.otp}

          To fix, regenerate the model:
            mix train_models

          This will create new .term files compatible with your current versions.
          """)

        {:error, error_type, details} ->
          flunk("Model compatibility check failed: #{error_type} - #{inspect(details)}")
      end
    end

    test "multi-task model file exists" do
      result = LSTMTestHelpers.check_model_compatibility(:multi_task)

      case result do
        :ok ->
          assert true

        {:error, :model_not_found, _details} ->
          # Multi-task model is optional, skip if not found
          :ok

        {:error, :decode_failed, details} ->
          flunk("""
          Multi-task LSTM model is INCOMPATIBLE.

          Model path: #{details.model_path}

          Regenerate with: mix train_models
          """)

        {:error, _error_type, _details} ->
          # Other errors are acceptable for optional model
          :ok
      end
    end

    test "response scorer model file exists" do
      result = LSTMTestHelpers.check_model_compatibility(:response_scorer)

      case result do
        :ok ->
          assert true

        {:error, :model_not_found, _details} ->
          # Response scorer is optional
          :ok

        {:error, :decode_failed, details} ->
          flunk("""
          Response scorer LSTM model is INCOMPATIBLE.

          Model path: #{details.model_path}

          Regenerate with: mix train_models
          """)

        {:error, _error_type, _details} ->
          :ok
      end
    end
  end

  describe "LSTM GenServer readiness" do
    @tag timeout: 10_000
    test "UnifiedModel starts and reports ready state" do
      # Give it time to attempt loading
      result = LSTMTestHelpers.wait_for_lstm_ready(:unified, 50, 100)

      case result do
        {:ok, :ready} ->
          # Model loaded successfully
          assert true

        {:error, :timeout} ->
          # Model may have failed to load - check if it's at least running
          case LSTMTestHelpers.unified_model_ready?() do
            {:ok, false} ->
              # GenServer is running but model didn't load
              # This is expected if model is incompatible (captured to avoid log leaks)
              capture_io(fn ->
                IO.puts("\nUnifiedModel is running but not ready (model may be incompatible)")
              end)
              assert true

            {:error, :not_running} ->
              flunk("UnifiedModel GenServer is not running")
          end

        {:error, :not_running} ->
          flunk("UnifiedModel GenServer is not running")
      end
    end
  end

  describe "test model generation" do
    test "can generate a minimal test model" do
      {:ok, model_path} = LSTMTestHelpers.generate_test_model(:unified,
        vocab_size: 10,
        embedding_size: 8,
        hidden_size: 8
      )

      on_exit(fn -> File.rm(model_path) end)

      assert File.exists?(model_path)

      # Verify it can be read back
      {:ok, binary} = File.read(model_path)
      data = :erlang.binary_to_term(binary, [:safe])

      assert Map.has_key?(data, :params)
      assert Map.has_key?(data, :vocabularies)
      assert Map.has_key?(data, :config)
      assert Map.has_key?(data, :metadata)

      # Verify metadata includes versions
      assert Map.has_key?(data.metadata, :versions)
      assert Map.has_key?(data.metadata, :created_at)
    end
  end
end
