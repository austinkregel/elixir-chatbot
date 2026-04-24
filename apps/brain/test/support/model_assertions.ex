defmodule Brain.Test.ModelAssertions do
  @moduledoc """
  Assertions and helpers for testing against trained ML models.

  Ensures tests hit real loaded models instead of silently degrading.

  ## Usage in Tests

      use Brain.Test.ModelAssertions

      setup do
        require_models!([:gazetteer, :micro_classifiers])
        :ok
      end

  ## Model types

  - `:gazetteer` — Gazetteer GenServer populated from disk
  - `:entities` — Entity extractor maps from trained `gazetteer.term`
  - `:pos` — POS model on disk / POSTagger
  - `:embedder` — Embedder vocabulary loaded
  - `:sentiment` — `SentimentClassifierSimple` model loaded
  - `:speech_act` — `SpeechActClassifierSimple` model loaded
  - `:micro_classifiers` — every axis in `Brain.ML.MicroClassifiers` loaded
  """

  alias Brain.ML.POSTagger
  alias Brain.ML.EntityExtractor
  alias Brain.ML.Gazetteer

  defmacro __using__(_opts) do
    quote do
      import Brain.Test.ModelAssertions
    end
  end

  @doc """
  Returns a map of model symbol → boolean (loaded / ready).
  """
  def model_status do
    %{
      gazetteer: check_gazetteer_loaded(),
      entities: check_entities_loaded(),
      pos: check_pos_loaded(),
      embedder: check_embedder_loaded(),
      sentiment: check_sentiment_loaded(),
      speech_act: check_speech_act_loaded(),
      micro_classifiers: check_micro_classifiers_loaded()
    }
  end

  @doc """
  Requires specific models to be loaded. Raises if any are missing.
  """
  def require_models!(model_types) when is_list(model_types) do
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
        1. In the Brain test suite, `test_helper.exs` runs `ModelFactory.train_and_load_test_models/0`.
        2. In dev/prod, run `mix train` (or the relevant `mix train_*` tasks).

      Tag tests with `@tag :requires_models` if you need to exclude them in a minimal CI profile.
      ============================================================
      """

      raise ExUnit.AssertionError, message: message
    else
      :ok
    end
  end

  @doc false
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

  defp check_gazetteer_loaded do
    try do
      Gazetteer.loaded?()
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

  defp check_pos_loaded do
    try do
      POSTagger.model_exists?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_embedder_loaded do
    try do
      Brain.Memory.Embedder.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_sentiment_loaded do
    try do
      Brain.ML.SentimentClassifierSimple.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_speech_act_loaded do
    try do
      Brain.ML.SpeechActClassifierSimple.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  defp check_micro_classifiers_loaded do
    try do
      Brain.ML.MicroClassifiers.ready?()
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end
end
