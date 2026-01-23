defmodule ChatBot.ML.IntentClassifierSimple do
  @moduledoc """
  Intent classifier using simple TF-IDF and nearest centroid classification.
  """

  require Logger

  def load_models do
    models_path = Application.get_env(:chat_bot, :ml)[:models_path]

    try do
      # Load classifier model
      model_path = Path.join(models_path, "classifier.term")
      model_binary = File.read!(model_path)
      model = :erlang.binary_to_term(model_binary)

      # Store in Agent for fast access
      case Agent.start_link(fn -> model end, name: __MODULE__) do
        {:ok, _pid} ->
          Logger.info("Classifier model loaded successfully", %{
            vocab_size: map_size(model.vocabulary)
          })

          {:ok, model}

        {:error, {:already_started, _pid}} ->
          # Already loaded
          model = Agent.get(__MODULE__, & &1)
          {:ok, model}

        {:error, reason} ->
          Logger.error("Failed to start Agent", %{reason: reason})
          {:error, reason}
      end
    rescue
      e ->
        Logger.error("Failed to load classifier model", %{error: inspect(e)})
        {:error, :model_not_found}
    end
  end

  @doc """
  Returns true if the classifier model is loaded.
  """
  def is_loaded? do
    Process.whereis(__MODULE__) != nil
  end

  def classify(text) do
    try do
      model = Agent.get(__MODULE__, & &1)

      case ChatBot.ML.SimpleClassifier.classify(text, model) do
        {:ok, label, score} ->
          {:ok,
           %{
             intent: label,
             confidence: score
           }}

        error ->
          error
      end
    rescue
      e ->
        Logger.error("Classification failed", %{error: inspect(e)})
        {:error, "Classification failed: #{inspect(e)}"}
    catch
      :exit, reason ->
        Logger.warning("Classification service not available", %{reason: inspect(reason)})
        # Try to load models and retry once
        case load_models() do
          {:ok, model} ->
            case ChatBot.ML.SimpleClassifier.classify(text, model) do
              {:ok, label, score} ->
                {:ok, %{intent: label, confidence: score}}

              error ->
                error
            end

          {:error, _} ->
            {:error, "Classification service not available"}
        end
    end
  end
end
