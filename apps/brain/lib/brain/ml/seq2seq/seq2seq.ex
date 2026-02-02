defmodule Brain.ML.Seq2Seq do
  @moduledoc """
  Main API for seq2seq LSTM with Attention models.
  
  Provides high-level functions for training, generation, and model management.
  Supports world-scoped models following the existing pattern.
  """
  
  use GenServer
  require Logger
  
  alias Brain.ML.Seq2Seq.{Vocabulary, Encoder, Decoder, Trainer}
  
  @default_world_id "default"
  @pubsub Brain.PubSub
  
  # ============================================================================
  # Client API
  # ============================================================================
  
  @doc """
  Starts the seq2seq GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end
  
  @doc """
  Check if the seq2seq model is ready for a given world.
  """
  def ready?(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    
    try do
      GenServer.call(server, {:ready?, world_id}, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end
  
  @doc """
  Generate a sequence from input.
  
  ## Options
  - `:world_id` - World ID for world-scoped models (default: "default")
  - `:max_length` - Maximum generation length (default: 50)
  """
  def generate(input, opts \\ []) when is_binary(input) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    max_length = Keyword.get(opts, :max_length, 50)
    
    GenServer.call(server, {:generate, input, world_id, max_length})
  end
  
  @doc """
  Train the model for a given world.
  
  ## Options
  - `:world_id` - World ID (default: "default")
  - `:epochs` - Number of training epochs (default: 10)
  - `:batch_size` - Batch size (default: 32)
  """
  def train(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)
    
    GenServer.call(server, {:train, world_id, epochs, batch_size}, :infinity)
  end
  
  @doc """
  Load a model for a given world.
  """
  def load_model(world_id \\ @default_world_id, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:load_model, world_id}, :infinity)
  end
  
  @doc """
  Save the current model for a given world.
  """
  def save_model(world_id \\ @default_world_id, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:save_model, world_id}, :infinity)
  end
  
  @doc """
  Get status of all loaded models.
  """
  def get_status(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :get_status)
  end
  
  # ============================================================================
  # Server Callbacks
  # ============================================================================
  
  @impl true
  def init(_opts) do
    # Subscribe to world model events
    Phoenix.PubSub.subscribe(@pubsub, "world_models:status")
    
    # Models are loaded on-demand, keyed by world_id
    state = %{
      models: %{},
      vocabularies: %{},
      loading: MapSet.new()
    }
    
    # Try to load default model at startup
    send(self(), {:load_default})
    
    {:ok, state}
  end
  
  @impl true
  def handle_info({:load_default}, state) do
    case do_load_model(@default_world_id) do
      {:ok, model_data} ->
        {:noreply, put_model(state, @default_world_id, model_data)}
      
      {:error, _reason} ->
        {:noreply, state}
    end
  end
  
  @impl true
  def handle_info({:world_models_loaded, world_id, _status}, state) do
    # Reload this world's model if it was already loaded
    if Map.has_key?(state.models, world_id) do
      Logger.debug("Seq2Seq: Reloading model for world #{world_id}")
      
      case do_load_model(world_id) do
        {:ok, model_data} ->
          {:noreply, put_model(state, world_id, model_data)}
        
        {:error, _reason} ->
          {:noreply, state}
      end
    else
      {:noreply, state}
    end
  end
  
  @impl true
  def handle_info({:world_models_loading, _world_id}, state) do
    {:noreply, state}
  end
  
  @impl true
  def handle_info({:world_models_error, _world_id, _reason}, state) do
    {:noreply, state}
  end
  
  @impl true
  def handle_call({:ready?, world_id}, _from, state) do
    ready = Map.has_key?(state.models, world_id) and 
            Map.has_key?(state.vocabularies, world_id)
    {:reply, ready, state}
  end
  
  @impl true
  def handle_call({:generate, input, world_id, max_length}, _from, state) do
    case get_model(state, world_id) do
      {:ok, model_data} ->
        result = do_generate(input, model_data, max_length, world_id)
        {:reply, result, state}
      
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:train, world_id, epochs, batch_size}, _from, state) do
    if MapSet.member?(state.loading, world_id) do
      {:reply, {:error, :already_training}, state}
    else
      # Mark as loading
      new_state = %{state | loading: MapSet.put(state.loading, world_id)}
      
      # Train in background (simplified - in production would use Task)
      case Trainer.train(world_id, epochs: epochs, batch_size: batch_size) do
        {:ok, trained_model} ->
          model_data = %{
            encoder: trained_model.encoder,
            decoder: trained_model.decoder,
            params: trained_model.params,
            vocabulary: trained_model.vocabulary,
            config: trained_model.config
          }
          
          final_state = 
            new_state
            |> put_model(world_id, model_data)
            |> Map.update!(:loading, &MapSet.delete(&1, world_id))
          
          {:reply, {:ok, :trained}, final_state}
        
        {:error, reason} ->
          final_state = Map.update!(new_state, :loading, &MapSet.delete(&1, world_id))
          {:reply, {:error, reason}, final_state}
      end
    end
  end
  
  @impl true
  def handle_call({:load_model, world_id}, _from, state) do
    case do_load_model(world_id) do
      {:ok, model_data} ->
        {:reply, :ok, put_model(state, world_id, model_data)}
      
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:save_model, world_id}, _from, state) do
    case get_model(state, world_id) do
      {:ok, model_data} ->
        case do_save_model(world_id, model_data) do
          :ok -> {:reply, :ok, state}
          {:error, reason} -> {:reply, {:error, reason}, state}
        end
      
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call(:get_status, _from, state) do
    status = %{
      loaded_worlds: Map.keys(state.models),
      loading_worlds: MapSet.to_list(state.loading)
    }
    
    {:reply, status, state}
  end
  
  # ============================================================================
  # Private Functions
  # ============================================================================
  
  defp get_model(state, world_id) do
    # Try world-specific model first
    cond do
      Map.has_key?(state.models, world_id) ->
        {:ok, state.models[world_id]}
      
      world_id != @default_world_id ->
        # Fall back to default
        if Map.has_key?(state.models, @default_world_id) do
          {:ok, state.models[@default_world_id]}
        else
          {:error, :model_not_loaded}
        end
      
      true ->
        {:error, :model_not_loaded}
    end
  end
  
  defp put_model(state, world_id, model_data) do
    %{
      state |
      models: Map.put(state.models, world_id, model_data),
      vocabularies: Map.put(state.vocabularies, world_id, model_data.vocabulary)
    }
  end
  
  defp do_load_model(world_id) do
    # Determine model path
    models_path = Application.get_env(:brain, :ml)[:models_path]
    
    model_path = 
      if world_id == @default_world_id do
        Path.join(models_path, "seq2seq.axon")
      else
        world_path = Path.join(["priv", "training_worlds", world_id, "models", "seq2seq.axon"])
        if File.exists?(world_path) do
          world_path
        else
          Path.join(models_path, "seq2seq.axon")
        end
      end
    
    if File.exists?(model_path) do
      try do
        # Load serialized model
        {:ok, binary} = File.read(model_path)
        model_data = :erlang.binary_to_term(binary)
        
        # Also load vocabulary
        vocab_path = String.replace(model_path, ".axon", "_vocab.term")
        vocab_data = 
          if File.exists?(vocab_path) do
            {:ok, vocab_binary} = File.read(vocab_path)
            :erlang.binary_to_term(vocab_binary)
          else
            nil
          end
        
        model_data = Map.put(model_data, :vocabulary, vocab_data)
        {:ok, model_data}
      rescue
        e ->
          Logger.error("Failed to load seq2seq model: #{inspect(e)}")
          {:error, :load_failed}
      end
    else
      {:error, :model_not_found}
    end
  end
  
  defp do_save_model(world_id, model_data) do
    models_path = Application.get_env(:brain, :ml)[:models_path]
    
    model_path = 
      if world_id == @default_world_id do
        Path.join(models_path, "seq2seq.axon")
      else
        world_dir = Path.join(["priv", "training_worlds", world_id, "models"])
        File.mkdir_p!(world_dir)
        Path.join(world_dir, "seq2seq.axon")
      end
    
    vocab_path = String.replace(model_path, ".axon", "_vocab.term")
    
    try do
      # Save model (without vocabulary)
      model_binary = :erlang.term_to_binary(Map.delete(model_data, :vocabulary))
      File.write!(model_path, model_binary)
      
      # Save vocabulary separately
      if model_data.vocabulary do
        vocab_binary = :erlang.term_to_binary(model_data.vocabulary)
        File.write!(vocab_path, vocab_binary)
      end
      
      :ok
    rescue
      e ->
        Logger.error("Failed to save seq2seq model: #{inspect(e)}")
        {:error, :save_failed}
    end
  end
  
  defp do_generate(input, model_data, max_length, world_id) do
    # Encode input
    # TODO: Use world-scoped vocabulary once Vocabulary supports world_id
    # For now, we log the world_id for traceability
    Logger.debug("Generating for world: #{world_id}")
    {:ok, input_indices} = Vocabulary.encode(input, server: Vocabulary)
    input_tensor = Nx.tensor([input_indices])
    
    # Extract params - model_data.params is {encoder_params, decoder_params}
    {encoder_params, decoder_params} = model_data.params
    
    # Encode with encoder
    {encoder_outputs, encoder_hidden} = 
      Encoder.encode(model_data.encoder, input_tensor, encoder_params)
    
    # Initialize decoder hidden state
    decoder_hidden = encoder_hidden
    
    # Generate with decoder
    {logits, _final_hidden} = 
      Decoder.generate(
        model_data.decoder,
        decoder_params,
        encoder_outputs,
        decoder_hidden,
        max_length: max_length
      )
    
    # Get predicted tokens
    predicted = Nx.argmax(logits, axis: 2)
    predicted_list = Nx.to_list(predicted) |> List.first()
    
    # Decode to text
    {:ok, text} = Vocabulary.decode(predicted_list, server: Vocabulary)
    
    {:ok, text}
  end
end
