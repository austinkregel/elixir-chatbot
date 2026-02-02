defmodule Brain.ML.Seq2Seq.Vocabulary do
  @moduledoc """
  Vocabulary management for seq2seq models.
  
  Handles word-to-index and index-to-word mappings with special tokens:
  - SOS (Start of Sequence)
  - EOS (End of Sequence)
  - PAD (Padding)
  - UNK (Unknown/Out-of-vocabulary)
  """
  
  use GenServer
  require Logger
  
  @special_tokens %{
    sos: "<SOS>",
    eos: "<EOS>",
    pad: "<PAD>",
    unk: "<UNK>"
  }
  
  @default_vocab_size 10_000
  
  # ============================================================================
  # Client API
  # ============================================================================
  
  @doc """
  Starts the vocabulary GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end
  
  @doc """
  Build vocabulary from a list of texts.
  """
  def build_vocabulary(texts, opts \\ []) when is_list(texts) do
    server = Keyword.get(opts, :server, __MODULE__)
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    GenServer.call(server, {:build_vocabulary, texts, vocab_size}, :infinity)
  end
  
  @doc """
  Encode text to a list of token indices.
  """
  def encode(text, opts \\ []) when is_binary(text) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:encode, text})
  end
  
  @doc """
  Decode a list of token indices back to text.
  """
  def decode(indices, opts \\ []) when is_list(indices) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:decode, indices})
  end
  
  @doc """
  Get vocabulary size.
  """
  def size(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :size)
  end
  
  @doc """
  Check if vocabulary is ready.
  """
  def ready?(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    try do
      GenServer.call(server, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end
  
  @doc """
  Get special token index.
  """
  def get_special_token(token_name, opts \\ []) when is_atom(token_name) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:get_special_token, token_name})
  end
  
  @doc """
  Load vocabulary from a saved model.
  """
  def load(vocab_map, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:load, vocab_map})
  end
  
  @doc """
  Export vocabulary for persistence.
  """
  def export(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :export)
  end
  
  # ============================================================================
  # Server Callbacks
  # ============================================================================
  
  @impl true
  def init(_opts) do
    {:ok, %{
      word_to_index: %{},
      index_to_word: %{},
      ready: false,
      vocab_size: 0
    }}
  end
  
  @impl true
  def handle_call({:build_vocabulary, texts, vocab_size}, _from, state) do
    # Log if we're rebuilding an existing vocabulary
    if state.ready do
      Logger.info("Rebuilding vocabulary (previous size: #{state.vocab_size})")
    end
    
    Logger.info("Building vocabulary from #{length(texts)} texts (max size: #{vocab_size})")
    
    # Tokenize all texts
    alias Brain.ML.Tokenizer
    all_tokens = 
      texts
      |> Enum.flat_map(&Tokenizer.tokenize_normalized/1)
      |> Enum.reject(&(&1 == ""))
    
    # Count frequencies
    frequencies = Enum.frequencies(all_tokens)
    
    # Sort by frequency and take top N
    top_words = 
      frequencies
      |> Enum.sort_by(fn {_word, count} -> -count end)
      |> Enum.take(vocab_size - 4)  # Reserve space for special tokens
    
    # Build mappings starting with special tokens
    word_to_index = 
      @special_tokens
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {{token_name, token}, idx}, acc ->
        Logger.debug("Registering special token :#{token_name} as '#{token}' at index #{idx}")
        Map.put(acc, token, idx)
      end)
    
    # Add regular words
    {word_to_index, index_to_word} = 
      top_words
      |> Enum.with_index(4)  # Start after special tokens
      |> Enum.reduce({word_to_index, %{}}, fn {{word, _count}, idx}, {w2i, i2w} ->
        {
          Map.put(w2i, word, idx),
          Map.put(i2w, idx, word)
        }
      end)
    
    # Add special tokens to index_to_word
    index_to_word = 
      @special_tokens
      |> Enum.with_index()
      |> Enum.reduce(index_to_word, fn {{_name, token}, idx}, acc ->
        Map.put(acc, idx, token)
      end)
    
    new_state = %{
      word_to_index: word_to_index,
      index_to_word: index_to_word,
      ready: true,
      vocab_size: map_size(word_to_index)
    }
    
    Logger.info("Vocabulary built: #{new_state.vocab_size} tokens")
    {:reply, {:ok, new_state.vocab_size}, new_state}
  end
  
  @impl true
  def handle_call({:encode, text}, _from, state) do
    if not state.ready do
      {:reply, {:error, :not_ready}, state}
    else
      alias Brain.ML.Tokenizer
      tokens = Tokenizer.tokenize_normalized(text)
      unk_idx = Map.get(state.word_to_index, @special_tokens.unk, 0)
      
      indices = Enum.map(tokens, fn token ->
        Map.get(state.word_to_index, token, unk_idx)
      end)
      
      {:reply, {:ok, indices}, state}
    end
  end
  
  @impl true
  def handle_call({:decode, indices}, _from, state) do
    if not state.ready do
      {:reply, {:error, :not_ready}, state}
    else
      # Filter out special tokens and padding
      pad_idx = Map.get(state.word_to_index, @special_tokens.pad, 0)
      eos_idx = Map.get(state.word_to_index, @special_tokens.eos, 1)
      sos_idx = Map.get(state.word_to_index, @special_tokens.sos, 2)
      
      words = 
        indices
        |> Enum.reject(fn idx -> idx in [pad_idx, eos_idx, sos_idx] end)
        |> Enum.map(fn idx ->
          Map.get(state.index_to_word, idx, @special_tokens.unk)
        end)
        |> Enum.reject(&(&1 in Map.values(@special_tokens)))
      
      text = Enum.join(words, " ")
      {:reply, {:ok, text}, state}
    end
  end
  
  @impl true
  def handle_call(:size, _from, state) do
    {:reply, state.vocab_size, state}
  end
  
  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end
  
  @impl true
  def handle_call({:get_special_token, token_name}, _from, state) do
    if not state.ready do
      {:reply, {:error, :not_ready}, state}
    else
      token = Map.get(@special_tokens, token_name)
      idx = Map.get(state.word_to_index, token)
      {:reply, {:ok, idx}, state}
    end
  end
  
  @impl true
  def handle_call({:load, vocab_map}, _from, _state) do
    word_to_index = Map.get(vocab_map, :word_to_index, vocab_map["word_to_index"] || %{})
    index_to_word = Map.get(vocab_map, :index_to_word, vocab_map["index_to_word"] || %{})
    
    new_state = %{
      word_to_index: word_to_index,
      index_to_word: index_to_word,
      ready: true,
      vocab_size: map_size(word_to_index)
    }
    
    {:reply, :ok, new_state}
  end
  
  @impl true
  def handle_call(:export, _from, state) do
    export_data = %{
      word_to_index: state.word_to_index,
      index_to_word: state.index_to_word,
      vocab_size: state.vocab_size
    }
    
    {:reply, {:ok, export_data}, state}
  end
end
