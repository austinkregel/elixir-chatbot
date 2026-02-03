defmodule Brain.Test.ModelAssertions do
  @moduledoc """
  Assertions and helpers for testing against trained ML models.

  This module provides utilities to ensure tests are actually hitting
  the production models and not silently passing with fallback behavior.

  ## Usage in Tests

      use Brain.Test.ModelAssertions

      setup do
        # Ensure models are loaded and fail fast if not
        require_models!([:tfidf, :gazetteer])
        :ok
      end

  ## Model Types

  - `:tfidf` - TF-IDF intent classifier (IntentClassifierSimple)
  - `:gazetteer` - Entity lookup tables
  - `:entities` - Entity extractor maps
  - `:unified_lstm` - Unified LSTM multi-task model
  - `:response_scorer` - LSTM response quality scorer
  - `:pos` - Part-of-speech tagger

  ## Philosophy

  Tests should fail loudly if expected models are not loaded, rather than
  silently skipping or passing with degraded behavior. This ensures CI
  catches missing or broken models.
  """

  defmacro __using__(_opts) do
    quote do
      import Brain.Test.ModelAssertions
    end
  end

  @doc """
  Checks which models are currently loaded and returns a status map.

  ## Returns

      %{
        tfidf: true,
        gazetteer: true,
        entities: true,
        unified_lstm: false,
        response_scorer: false,
        pos: false
      }
  """
  def model_status do
    %{
      tfidf: check_tfidf_loaded(),
      gazetteer: check_gazetteer_loaded(),
      entities: check_entities_loaded(),
      unified_lstm: check_unified_lstm_loaded(),
      response_scorer: check_response_scorer_loaded(),
      pos: check_pos_loaded()
    }
  end

  @doc """
  Requires specific models to be loaded. Raises with a clear error if any are missing.

  ## Examples

      # Require TF-IDF classifier for basic intent tests
      require_models!([:tfidf])

      # Require full stack for integration tests
      require_models!([:tfidf, :gazetteer, :unified_lstm])

  ## Options

    - `:allow_fallback` - If true, log a warning instead of raising (default: false)
  """
  def require_models!(model_types, opts \\ []) when is_list(model_types) do
    allow_fallback = Keyword.get(opts, :allow_fallback, false)
    status = model_status()

    missing =
      model_types
      |> Enum.filter(fn type -> Map.get(status, type) != true end)

    if length(missing) > 0 do
      message = """
      
      ============================================================
      REQUIRED MODELS NOT LOADED
      ============================================================

      The following models are required for this test but are not loaded:
        #{Enum.map(missing, &"  - #{&1}") |> Enum.join("\n")}

      Current model status:
        #{format_status(status)}

      To fix this:
        1. Run `mix train` to train all models
        2. Or run specific training tasks:
           - `mix train_models` for TF-IDF models
           - `mix train_unified` for LSTM models
           - `mix train_response` for response scorer

      If you want to skip these tests when models are unavailable,
      tag them with @tag :requires_models and exclude in test config.
      ============================================================
      """

      if allow_fallback do
        require Logger
        Logger.warning(message)
        :fallback
      else
        raise ExUnit.AssertionError, message: message
      end
    else
      :ok
    end
  end

  @doc """
  Asserts that a model classification actually used the expected model type.

  This prevents tests from passing when fallback behavior kicks in.

  ## Example

      result = IntentClassifierSimple.classify("Hello")
      assert_used_model(result, :tfidf)
  """
  def assert_used_model({:ok, result}, expected_model) do
    import ExUnit.Assertions
    
    # Check if result came from the expected model
    model_source = Map.get(result, :model_source) || Map.get(result, :source)

    if model_source do
      assert model_source == expected_model,
             "Expected result from #{expected_model}, got #{model_source}"
    else
      # If no model source is tracked, at least verify confidence is reasonable
      confidence = Map.get(result, :confidence, 0)

      assert confidence > 0.01,
             "Suspiciously low confidence (#{confidence}) suggests fallback behavior"
    end

    :ok
  end

  def assert_used_model({:error, _reason} = error, _expected_model) do
    import ExUnit.Assertions
    flunk("Model call failed: #{inspect(error)}")
  end

  @doc """
  Asserts that entity extraction returned real entities, not empty fallback.
  """
  def assert_entities_extracted(entities, min_count \\ 1) when is_list(entities) do
    import ExUnit.Assertions
    
    assert length(entities) >= min_count,
           """
           Expected at least #{min_count} entities, got #{length(entities)}.
           This may indicate entity extractor is not properly loaded.
           Entities: #{inspect(entities)}
           """
  end

  @doc """
  Returns a formatted summary of model status for logging/debugging.
  """
  def format_status(status) when is_map(status) do
    status
    |> Enum.map(fn {model, loaded} ->
      marker = if loaded, do: "✓", else: "✗"
      "#{marker} #{model}: #{if loaded, do: "loaded", else: "NOT LOADED"}"
    end)
    |> Enum.join("\n        ")
  end

  # ============================================================================
  # Private Model Checks
  # ============================================================================

  defp check_tfidf_loaded do
    try do
      Brain.ML.IntentClassifierSimple.is_loaded?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_gazetteer_loaded do
    try do
      Brain.ML.Gazetteer.is_loaded?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_entities_loaded do
    try do
      Brain.ML.EntityExtractor.is_loaded?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_unified_lstm_loaded do
    try do
      Brain.ML.LSTM.UnifiedModel.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_response_scorer_loaded do
    try do
      Brain.Response.LSTMResponse.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_pos_loaded do
    try do
      Brain.ML.POSTagger.model_exists?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end
end
