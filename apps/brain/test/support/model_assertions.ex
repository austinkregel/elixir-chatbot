defmodule Brain.Test.ModelAssertions do
  @moduledoc "Assertions and helpers for testing against trained ML models.\n\nThis module provides utilities to ensure tests are actually hitting\nthe production models and not silently passing with fallback behavior.\n\n## Usage in Tests\n\n    use Brain.Test.ModelAssertions\n\n    setup do\n      # Ensure models are loaded and fail fast if not\n      require_models!([:tfidf, :gazetteer])\n      :ok\n    end\n\n## Model Types\n\n- `:tfidf` - TF-IDF intent classifier (IntentClassifierSimple)\n- `:gazetteer` - Entity lookup tables\n- `:entities` - Entity extractor maps\n- `:unified_lstm` - Unified LSTM multi-task model\n- `:response_scorer` - LSTM response quality scorer\n- `:pos` - Part-of-speech tagger\n\n## Philosophy\n\nTests should fail loudly if expected models are not loaded, rather than\nsilently skipping or passing with degraded behavior. This ensures CI\ncatches missing or broken models.\n"

  alias Brain.ML.POSTagger
  alias Brain.Response.LSTMResponse
  alias Brain.ML.LSTM.UnifiedModel
  alias Brain.ML.EntityExtractor
  alias Brain.ML.Gazetteer
  alias Brain.ML.IntentClassifierSimple

  defmacro __using__(_opts) do
    quote do
      import Brain.Test.ModelAssertions
    end
  end

  @doc "Checks which models are currently loaded and returns a status map.\n\n## Returns\n\n    %{\n      tfidf: true,\n      gazetteer: true,\n      entities: true,\n      unified_lstm: false,\n      response_scorer: false,\n      pos: false\n    }\n"
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

  @doc "Requires specific models to be loaded. Raises with a clear error if any are missing.\n\n## Examples\n\n    # Require TF-IDF classifier for basic intent tests\n    require_models!([:tfidf])\n\n    # Require full stack for integration tests\n    require_models!([:tfidf, :gazetteer, :unified_lstm])\n\n## Options\n\n  - `:allow_fallback` - If true, log a warning instead of raising (default: false)\n"
  def require_models!(model_types, opts \\ []) when is_list(model_types) do
    allow_fallback = Keyword.get(opts, :allow_fallback, false)
    status = model_status()

    missing =
      model_types
      |> Enum.filter(fn type -> Map.get(status, type) != true end)

    if missing != [] do
      message = """

      ============================================================
      REQUIRED MODELS NOT LOADED
      ============================================================

      The following models are required for this test but are not loaded:
        #{Enum.map_join(missing, "\n", &"  - #{&1}")}

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

  @doc "Asserts that a model classification actually used the expected model type.\n\nThis prevents tests from passing when fallback behavior kicks in.\n\n## Example\n\n    result = IntentClassifierSimple.classify(\"Hello\")\n    assert_used_model(result, :tfidf)\n"
  def assert_used_model({:ok, result}, expected_model) do
    import ExUnit.Assertions
    model_source = Map.get(result, :model_source) || Map.get(result, :source)

    if model_source do
      assert model_source == expected_model,
             "Expected result from #{expected_model}, got #{model_source}"
    else
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

  @doc "Asserts that entity extraction returned real entities, not empty fallback.\n"
  def assert_entities_extracted(entities, min_count \\ 1) when is_list(entities) do
    import ExUnit.Assertions

    assert length(entities) >= min_count,
           """
           Expected at least #{min_count} entities, got #{length(entities)}.
           This may indicate entity extractor is not properly loaded.
           Entities: #{inspect(entities)}
           """
  end

  @doc "Returns a formatted summary of model status for logging/debugging.\n"
  def format_status(status) when is_map(status) do
    status
    |> Enum.map_join(
      "\n        ",
      fn {model, loaded} ->
        marker =
          if loaded do
            "✓"
          else
            "✗"
          end

        "#{marker} #{model}: #{if loaded do
          "loaded"
        else
          "NOT LOADED"
        end}"
      end
    )
  end

  defp check_tfidf_loaded do
    try do
      IntentClassifierSimple.is_loaded?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_gazetteer_loaded do
    try do
      Gazetteer.is_loaded?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_entities_loaded do
    try do
      EntityExtractor.is_loaded?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_unified_lstm_loaded do
    try do
      UnifiedModel.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_response_scorer_loaded do
    try do
      LSTMResponse.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_pos_loaded do
    try do
      POSTagger.model_exists?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end
end