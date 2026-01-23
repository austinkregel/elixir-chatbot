defmodule ChatBot.Memory.Embedder do
  @moduledoc """
  Embedding utilities for the cognitive memory system.

  Produces TF-IDF based embeddings for text that can be used in
  similarity search. This replaces the byte frequency approach
  from the Rust implementation with a more semantically meaningful
  TF-IDF vectorization.

  The embedder maintains a vocabulary and IDF weights that are
  built from training data and used to consistently embed new text.
  """

  use GenServer

  alias ChatBot.ML.Tokenizer

  require Logger

  @default_vocab_size 2000
  @min_word_frequency 2

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Build vocabulary and IDF weights from a list of texts.
  This should be called during training.
  """
  def build_vocabulary(texts) when is_list(texts) do
    GenServer.call(__MODULE__, {:build_vocabulary, texts}, :infinity)
  end

  @doc """
  Embed text into a TF-IDF vector.
  Returns a list of floats representing the embedding.
  """
  def embed(text) when is_binary(text) do
    GenServer.call(__MODULE__, {:embed, text})
  end

  @doc """
  Compute cosine similarity between two embedding vectors.
  Returns a float in [-1, 1].
  """
  def cosine_similarity(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b) do
    compute_cosine_similarity(vec_a, vec_b)
  end

  @doc """
  Check if the embedder is initialized with vocabulary.
  Uses a short timeout to avoid blocking if embedder is busy.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc """
  Get the current vocabulary size.
  """
  def vocabulary_size do
    GenServer.call(__MODULE__, :vocabulary_size)
  end

  @doc """
  Load a pre-built vocabulary and IDF weights.
  """
  def load_model(model) when is_map(model) do
    GenServer.call(__MODULE__, {:load_model, model})
  end

  @doc """
  Export the current vocabulary and IDF weights for persistence.
  """
  def export_model do
    GenServer.call(__MODULE__, :export_model)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    {:ok,
     %{
       vocabulary: %{},
       idf_weights: %{},
       ready: false
     }}
  end

  @impl true
  def handle_call({:build_vocabulary, texts}, _from, state) do
    Logger.info("Building embedder vocabulary from #{length(texts)} texts")

    # Tokenize all texts
    all_tokens =
      texts
      |> Enum.flat_map(&tokenize/1)
      |> Enum.frequencies()
      |> Enum.filter(fn {_word, count} -> count >= @min_word_frequency end)
      |> Enum.sort_by(fn {_word, count} -> -count end)
      |> Enum.take(@default_vocab_size)
      |> Enum.map(fn {word, _count} -> word end)

    vocabulary =
      all_tokens
      |> Enum.with_index()
      |> Enum.into(%{})

    # Calculate IDF weights
    num_docs = length(texts)

    idf_weights =
      vocabulary
      |> Enum.map(fn {word, _idx} ->
        doc_freq = Enum.count(texts, fn text -> word in tokenize(text) end)
        idf = :math.log(num_docs / max(doc_freq, 1))
        {word, idf}
      end)
      |> Enum.into(%{})

    Logger.info("Embedder vocabulary built: #{map_size(vocabulary)} words")

    new_state = %{
      state
      | vocabulary: vocabulary,
        idf_weights: idf_weights,
        ready: true
    }

    {:reply, {:ok, map_size(vocabulary)}, new_state}
  end

  @impl true
  def handle_call({:embed, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:embed, text}, _from, state) do
    vector = vectorize(text, state.vocabulary, state.idf_weights)
    {:reply, {:ok, vector}, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  @impl true
  def handle_call(:vocabulary_size, _from, state) do
    {:reply, map_size(state.vocabulary), state}
  end

  @impl true
  def handle_call({:load_model, model}, _from, _state) do
    new_state = %{
      vocabulary: Map.get(model, :vocabulary, %{}),
      idf_weights: Map.get(model, :idf_weights, %{}),
      ready: true
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:export_model, _from, state) do
    model = %{
      vocabulary: state.vocabulary,
      idf_weights: state.idf_weights
    }

    {:reply, {:ok, model}, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp tokenize(text) do
    Tokenizer.tokenize_normalized(text, min_length: 2)
  end

  defp vectorize(text, vocabulary, idf_weights) do
    tokens = tokenize(text)
    token_freq = Enum.frequencies(tokens)

    # Build TF-IDF vector
    vector =
      vocabulary
      |> Enum.sort_by(fn {_word, idx} -> idx end)
      |> Enum.map(fn {word, _idx} ->
        tf = Map.get(token_freq, word, 0)
        idf = Map.get(idf_weights, word, 0.0)
        tf * idf
      end)

    # Normalize the vector
    normalize_vector(vector)
  end

  defp normalize_vector(vector) do
    magnitude = :math.sqrt(Enum.reduce(vector, 0, fn val, acc -> acc + val * val end))

    if magnitude > 0 do
      Enum.map(vector, &(&1 / magnitude))
    else
      vector
    end
  end

  defp compute_cosine_similarity(vec1, vec2) do
    if length(vec1) != length(vec2) or length(vec1) == 0 do
      0.0
    else
      dot_product =
        Enum.zip(vec1, vec2)
        |> Enum.reduce(0, fn {a, b}, acc -> acc + a * b end)

      mag1 = :math.sqrt(Enum.reduce(vec1, 0, fn val, acc -> acc + val * val end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0, fn val, acc -> acc + val * val end))

      if mag1 > 0 and mag2 > 0 do
        dot_product / (mag1 * mag2)
      else
        0.0
      end
    end
  end
end
