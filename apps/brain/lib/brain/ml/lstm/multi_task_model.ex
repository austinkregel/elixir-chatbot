defmodule Brain.ML.LSTM.MultiTaskModel do
  @moduledoc """
  Unified interface for the multi-task LSTM model.
  
  This module provides a GenServer that loads and caches the trained multi-task
  LSTM model, and exposes a simple API for performing NLP analysis (intent
  classification, entity extraction, and POS tagging) in a single call.
  
  ## Usage
  
      # The GenServer is started automatically by the application supervisor
      
      # Check if the model is ready
      MultiTaskModel.ready?()
      
      # Analyze text
      {:ok, result} = MultiTaskModel.analyze("what is the weather in London")
      # => %{
      #   intent: %{label: "weather.query", confidence: 0.92, scores: %{...}},
      #   entities: [%{text: "London", type: "location", start: 5, end: 5}],
      #   pos_tags: [{"what", "PRON"}, {"is", "AUX"}, ...],
      #   tokens: ["what", "is", "the", "weather", "in", "London"]
      # }
  
  ## Fallback Behavior
  
  If the LSTM model is not loaded (e.g., not yet trained), the module can
  optionally fall back to TF-IDF classification. This is controlled by the
  `:fallback_enabled` option.
  """
  
  use GenServer
  require Logger
  
  alias Brain.ML.LSTM.Trainer
  alias Brain.ML.DataLoaders
  
  @type analysis_result :: %{
    intent: %{label: String.t(), confidence: float(), scores: map()},
    entities: [map()],
    pos_tags: [{String.t(), String.t()}],
    tokens: [String.t()]
  }
  
  # Client API
  
  @doc """
  Start the MultiTaskModel GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end
  
  @doc """
  Check if the model is loaded and ready for inference.
  """
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end
  
  @doc """
  Get the current model status.
  """
  def status(name \\ __MODULE__) do
    GenServer.call(name, :status)
  end
  
  @doc """
  Analyze text using the multi-task model.
  
  Returns intent classification, entity extraction, and POS tagging results
  in a single call.
  
  ## Options
  - `:fallback` - Use TF-IDF fallback if LSTM not available (default: true)
  
  ## Returns
  `{:ok, result}` or `{:error, reason}`
  """
  def analyze(text, name \\ __MODULE__, opts \\ []) do
    GenServer.call(name, {:analyze, text, opts})
  end
  
  @doc """
  Classify intent only (faster than full analysis).
  """
  def classify_intent(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify_intent, text})
  end
  
  @doc """
  Extract entities only.
  """
  def extract_entities(text, name \\ __MODULE__) do
    GenServer.call(name, {:extract_entities, text})
  end
  
  @doc """
  Get POS tags only.
  """
  def get_pos_tags(text, name \\ __MODULE__) do
    GenServer.call(name, {:get_pos_tags, text})
  end
  
  @doc """
  Reload the model from disk.
  """
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload)
  end
  
  @doc """
  Train a new model (async).
  """
  def train(opts \\ [], name \\ __MODULE__) do
    GenServer.cast(name, {:train, opts})
  end
  
  # Server Callbacks
  
  @impl true
  def init(opts) do
    fallback_enabled = Keyword.get(opts, :fallback_enabled, true)
    auto_load = Keyword.get(opts, :auto_load, true)
    
    state = %{
      model: nil,
      model_type: nil,
      status: :initializing,
      fallback_enabled: fallback_enabled,
      tfidf_model: nil,
      training: false
    }
    
    if auto_load do
      send(self(), :load_model)
    end
    
    {:ok, state}
  end
  
  @impl true
  def handle_call(:ready?, _from, state) do
    ready = state.model != nil and state.status == :ready
    {:reply, ready, state}
  end
  
  @impl true
  def handle_call(:status, _from, state) do
    status = %{
      status: state.status,
      model_type: state.model_type,
      model_loaded: state.model != nil,
      fallback_enabled: state.fallback_enabled,
      fallback_loaded: state.tfidf_model != nil,
      training: state.training
    }
    {:reply, status, state}
  end
  
  @impl true
  def handle_call({:analyze, text, opts}, _from, state) do
    try do
      result = perform_analysis(text, state, opts)
      {:reply, result, state}
    rescue
      e in ArgumentError ->
        Logger.warning("MultiTaskModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | status: :degraded}}
    end
  end

  @impl true
  def handle_call({:classify_intent, text}, _from, state) do
    try do
      result = perform_intent_classification(text, state)
      {:reply, result, state}
    rescue
      e in ArgumentError ->
        Logger.warning("MultiTaskModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | status: :degraded}}
    end
  end

  @impl true
  def handle_call({:extract_entities, text}, _from, state) do
    try do
      result = perform_entity_extraction(text, state)
      {:reply, result, state}
    rescue
      e in ArgumentError ->
        Logger.warning("MultiTaskModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | status: :degraded}}
    end
  end

  @impl true
  def handle_call({:get_pos_tags, text}, _from, state) do
    try do
      result = perform_pos_tagging(text, state)
      {:reply, result, state}
    rescue
      e in ArgumentError ->
        Logger.warning("MultiTaskModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | status: :degraded}}
    end
  end
  
  @impl true
  def handle_call(:reload, _from, state) do
    new_state = load_models(%{state | status: :loading})
    {:reply, :ok, new_state}
  end
  
  @impl true
  def handle_cast({:train, opts}, state) do
    new_state = %{state | training: true, status: :training}
    
    # Start training in a Task to not block the GenServer
    Task.start(fn ->
      result = Trainer.train_multitask(opts)
      send(self(), {:training_complete, result})
    end)
    
    {:noreply, new_state}
  end
  
  @impl true
  def handle_info(:load_model, state) do
    new_state = load_models(%{state | status: :loading})
    {:noreply, new_state}
  end
  
  @impl true
  def handle_info({:training_complete, result}, state) do
    case result do
      {:ok, model} ->
        Logger.info("Multi-task training completed successfully")
        {:noreply, %{state | 
          model: model, 
          model_type: :multitask, 
          status: :ready, 
          training: false
        }}
      
      {:error, reason} ->
        Logger.error("Multi-task training failed", %{reason: reason})
        {:noreply, %{state | training: false, status: :error}}
    end
  end
  
  # Private functions
  
  defp load_models(state) do
    # Try loading in order: multitask > joint > intent-only
    case Trainer.load_multitask_model() do
      {:ok, model} ->
        Logger.info("Loaded multi-task LSTM model")
        %{state | model: model, model_type: :multitask, status: :ready}
      
      {:error, _} ->
        case Trainer.load_joint_model() do
          {:ok, model} ->
            Logger.info("Loaded joint LSTM model (intent + NER)")
            %{state | model: model, model_type: :joint, status: :ready}
          
          {:error, _} ->
            case Trainer.load_intent_model() do
              {:ok, model} ->
                Logger.info("Loaded intent-only LSTM model")
                %{state | model: model, model_type: :intent_only, status: :ready}
              
              {:error, _} ->
                Logger.warning("No LSTM model found, attempting to load TF-IDF fallback")
                load_tfidf_fallback(state)
            end
        end
    end
  end
  
  defp load_tfidf_fallback(state) do
    if state.fallback_enabled do
      # Try to load or build TF-IDF model
      case load_or_build_tfidf() do
        {:ok, tfidf_model} ->
          Logger.info("TF-IDF fallback model loaded")
          %{state | tfidf_model: tfidf_model, status: :fallback_only}
        
        {:error, reason} ->
          Logger.error("Failed to load TF-IDF fallback", %{reason: reason})
          %{state | status: :no_model}
      end
    else
      %{state | status: :no_model}
    end
  end
  
  defp load_or_build_tfidf do
    case Brain.ML.SimpleClassifier.load_model() do
      {:ok, model} -> {:ok, model}
      {:error, _} ->
        # Try to build from training data
        case DataLoaders.load_all_intents() do
          {:ok, examples} ->
            training_data = Enum.map(examples, fn ex -> {ex.text, ex.intent} end)
            model = Brain.ML.SimpleClassifier.train(training_data)
            {:ok, model}
          
          error -> error
        end
    end
  end
  
  defp perform_analysis(text, state, opts) do
    use_fallback = Keyword.get(opts, :fallback, true)
    
    cond do
      state.model != nil and state.model_type == :multitask ->
        result = Trainer.analyze_multitask(text, state.model)
        {:ok, result}
      
      state.model != nil and state.model_type == :joint ->
        result = Trainer.analyze_joint(text, state.model)
        {:ok, result}
      
      state.model != nil and state.model_type == :intent_only ->
        # Intent only - do basic analysis
        {intent, conf, scores} = Trainer.classify(text, state.model)
        tokens = Brain.ML.Tokenizer.tokenize(text)
        
        {:ok, %{
          intent: %{label: intent, confidence: conf, scores: scores},
          entities: [],
          pos_tags: [],
          tokens: tokens
        }}
      
      use_fallback and state.tfidf_model != nil ->
        # TF-IDF fallback
        {:ok, intent, conf, _details} = 
          Brain.ML.SimpleClassifier.classify_with_details(text, state.tfidf_model)
        
        tokens = Brain.ML.Tokenizer.tokenize(text)
        
        {:ok, %{
          intent: %{label: intent, confidence: conf, scores: %{}},
          entities: [],
          pos_tags: [],
          tokens: tokens,
          fallback: true
        }}
      
      true ->
        {:error, :model_not_loaded}
    end
  end
  
  defp perform_intent_classification(text, state) do
    cond do
      state.model != nil ->
        case state.model_type do
          :multitask -> 
            result = Trainer.analyze_multitask(text, state.model)
            {:ok, result.intent}
          
          :joint ->
            result = Trainer.analyze_joint(text, state.model)
            {:ok, result.intent}
          
          :intent_only ->
            {intent, conf, scores} = Trainer.classify(text, state.model)
            {:ok, %{label: intent, confidence: conf, scores: scores}}
        end
      
      state.tfidf_model != nil ->
        {:ok, intent, conf, _} = 
          Brain.ML.SimpleClassifier.classify_with_details(text, state.tfidf_model)
        {:ok, %{label: intent, confidence: conf, scores: %{}, fallback: true}}
      
      true ->
        {:error, :model_not_loaded}
    end
  end
  
  defp perform_entity_extraction(text, state) do
    cond do
      state.model != nil and state.model_type in [:multitask, :joint] ->
        result = 
          case state.model_type do
            :multitask -> Trainer.analyze_multitask(text, state.model)
            :joint -> Trainer.analyze_joint(text, state.model)
          end
        {:ok, result.entities}
      
      true ->
        {:error, :ner_not_available}
    end
  end
  
  defp perform_pos_tagging(text, state) do
    cond do
      state.model != nil and state.model_type == :multitask ->
        result = Trainer.analyze_multitask(text, state.model)
        {:ok, result.pos_tags}
      
      true ->
        {:error, :pos_not_available}
    end
  end
end
