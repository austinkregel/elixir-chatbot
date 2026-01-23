defmodule ChatBot.SystemStatus do
  @moduledoc """
  Reports the status of various background systems for UI display.
  """

  alias ChatBot.Memory.{Embedder, Store}

  @doc """
  Returns a map of all system statuses.
  """
  def get_all do
    %{
      embedder: get_embedder_status(),
      memory_store: get_memory_store_status(),
      brain: get_brain_status(),
      nlp_pipeline: get_nlp_pipeline_status()
    }
  end

  @doc """
  Returns the embedder status.
  """
  def get_embedder_status do
    if Process.whereis(Embedder) do
      ready = Embedder.ready?()

      %{
        running: true,
        ready: ready,
        status: if(ready, do: :ready, else: :building_vocabulary),
        label: if(ready, do: "Ready", else: "Building vocabulary...")
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started"
      }
    end
  end

  @doc """
  Returns the memory store status.
  """
  def get_memory_store_status do
    if Process.whereis(Store) do
      stats =
        try do
          Store.stats()
        catch
          :exit, _ -> %{episode_count: 0, semantic_count: 0}
        end

      %{
        running: true,
        ready: true,
        status: :ready,
        label: "Ready",
        episodes: Map.get(stats, :episode_count, 0),
        semantics: Map.get(stats, :semantic_count, 0)
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started",
        episodes: 0,
        semantics: 0
      }
    end
  end

  @doc """
  Returns the brain status.
  """
  def get_brain_status do
    if Process.whereis(ChatBot.Brain) do
      %{
        running: true,
        ready: true,
        status: :ready,
        label: "Ready"
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started"
      }
    end
  end

  @doc """
  Returns the NLP pipeline status.
  """
  def get_nlp_pipeline_status do
    # Check if models are loaded
    classifier_ready =
      try do
        ChatBot.ML.IntentClassifierSimple.is_loaded?()
      catch
        :exit, _ -> false
      end

    gazetteer_ready =
      try do
        ChatBot.ML.Gazetteer.is_loaded?()
      catch
        :exit, _ -> false
      end

    all_ready = classifier_ready and gazetteer_ready

    %{
      running: true,
      ready: all_ready,
      status: if(all_ready, do: :ready, else: :loading),
      label: if(all_ready, do: "Ready", else: "Loading models..."),
      components: %{
        intent_classifier: classifier_ready,
        gazetteer: gazetteer_ready
      }
    }
  end

  @doc """
  Returns true if all systems are ready.
  """
  def all_ready? do
    status = get_all()

    status.embedder.ready and
      status.memory_store.ready and
      status.brain.ready and
      status.nlp_pipeline.ready
  end
end
