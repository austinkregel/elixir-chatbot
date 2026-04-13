defmodule Brain.Response.ChunkCompatibility do
  @moduledoc """
  Learns and scores chunk compatibility for template blending.

  Chunks that appear together in the same template are considered compatible.
  The compatibility score is based on:
  1. Co-occurrence frequency in templates
  2. Embedding similarity (semantic compatibility)

  ## Usage

      # Learn compatibility from existing templates
      ChunkCompatibility.learn(all_chunks_by_template)

      # Check if two chunks are compatible
      ChunkCompatibility.compatible?(chunk_a, chunk_b)
      # => true

      # Get compatibility score
      ChunkCompatibility.score(chunk_a, chunk_b)
      # => 0.85
  """

  use GenServer
  require Logger

  @compatibility_threshold 0.3

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Learns chunk compatibility from templates.

  Takes a list of templates where each template is a list of chunks.
  Chunks in the same template are considered compatible.
  """
  def learn(templates_as_chunk_lists) when is_list(templates_as_chunk_lists) do
    GenServer.call(__MODULE__, {:learn, templates_as_chunk_lists}, 30_000)
  end

  @doc """
  Checks if two chunks are compatible (can be blended together).
  """
  def compatible?(chunk_a, chunk_b) do
    score(chunk_a, chunk_b) >= @compatibility_threshold
  end

  @doc """
  Gets the compatibility score between two chunks.
  Returns a float between 0.0 and 1.0.
  """
  def score(chunk_a, chunk_b) do
    GenServer.call(__MODULE__, {:score, chunk_a, chunk_b})
  catch
    :exit, _ -> fallback_score(chunk_a, chunk_b)
  end

  @doc """
  Gets all chunks that are compatible with the given chunk.
  """
  def find_compatible(chunk, all_chunks) do
    Enum.filter(all_chunks, fn other ->
      other.text != chunk.text and compatible?(chunk, other)
    end)
  end

  @doc """
  Gets the learned co-occurrence matrix.
  """
  def get_cooccurrence_matrix do
    GenServer.call(__MODULE__, :get_matrix)
  catch
    :exit, _ -> %{}
  end

  @doc """
  Checks if the compatibility store is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    {:ok,
     %{
       ready: true,
       cooccurrence: %{},
       type_cooccurrence: %{}
     }}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call({:learn, templates_as_chunk_lists}, _from, state) do
    Logger.info("Learning chunk compatibility from #{length(templates_as_chunk_lists)} templates")

    {cooccurrence, type_cooccurrence} = build_cooccurrence_matrices(templates_as_chunk_lists)

    Logger.info("Learned #{map_size(cooccurrence)} chunk pairs, #{map_size(type_cooccurrence)} type pairs")

    {:reply, :ok, %{state | cooccurrence: cooccurrence, type_cooccurrence: type_cooccurrence}}
  end

  def handle_call({:score, chunk_a, chunk_b}, _from, state) do
    score = calculate_score(chunk_a, chunk_b, state)
    {:reply, score, state}
  end

  def handle_call(:get_matrix, _from, state) do
    {:reply, state.cooccurrence, state}
  end

  # ============================================================================
  # Co-occurrence Learning
  # ============================================================================

  defp build_cooccurrence_matrices(templates_as_chunk_lists) do
    # Build pairwise co-occurrence for both specific chunks and chunk types
    {chunk_pairs, type_pairs} =
      templates_as_chunk_lists
      |> Enum.reduce({%{}, %{}}, fn chunk_list, {chunk_acc, type_acc} ->
        # Get all pairs of chunks in this template
        pairs = for a <- chunk_list, b <- chunk_list, a.text != b.text, do: {a, b}

        # Count chunk text pairs
        chunk_acc =
          Enum.reduce(pairs, chunk_acc, fn {a, b}, acc ->
            key = normalize_pair_key(a.text, b.text)
            Map.update(acc, key, 1, &(&1 + 1))
          end)

        # Count type pairs
        type_acc =
          Enum.reduce(pairs, type_acc, fn {a, b}, acc ->
            key = normalize_type_pair_key(a.type, b.type)
            Map.update(acc, key, 1, &(&1 + 1))
          end)

        {chunk_acc, type_acc}
      end)

    # Normalize scores to 0-1 range
    max_chunk_count = chunk_pairs |> Map.values() |> Enum.max(fn -> 1 end)
    max_type_count = type_pairs |> Map.values() |> Enum.max(fn -> 1 end)

    normalized_chunks =
      Map.new(chunk_pairs, fn {k, v} -> {k, v / max_chunk_count} end)

    normalized_types =
      Map.new(type_pairs, fn {k, v} -> {k, v / max_type_count} end)

    {normalized_chunks, normalized_types}
  end

  defp normalize_pair_key(text_a, text_b) do
    # Sort alphabetically to ensure consistent key regardless of order
    if text_a <= text_b do
      {text_a, text_b}
    else
      {text_b, text_a}
    end
  end

  defp normalize_type_pair_key(type_a, type_b) do
    # Sort types to ensure consistent key
    types = Enum.sort([type_a, type_b])
    {Enum.at(types, 0), Enum.at(types, 1)}
  end

  # ============================================================================
  # Scoring
  # ============================================================================

  defp calculate_score(chunk_a, chunk_b, state) do
    # Score is a weighted combination of:
    # 1. Direct co-occurrence (if we've seen these exact chunks together)
    # 2. Type co-occurrence (if these types commonly appear together)
    # 3. Embedding similarity (semantic compatibility)

    direct_score = get_direct_cooccurrence(chunk_a, chunk_b, state)
    type_score = get_type_cooccurrence(chunk_a, chunk_b, state)
    embedding_score = get_embedding_similarity(chunk_a, chunk_b)

    # Weights for combining scores
    direct_weight = if direct_score > 0, do: 0.5, else: 0.0
    type_weight = 0.3
    embedding_weight = 0.2

    total_weight = direct_weight + type_weight + embedding_weight

    if total_weight > 0 do
      (direct_score * direct_weight + type_score * type_weight + embedding_score * embedding_weight) /
        total_weight
    else
      embedding_score
    end
  end

  defp get_direct_cooccurrence(chunk_a, chunk_b, state) do
    key = normalize_pair_key(chunk_a.text, chunk_b.text)
    Map.get(state.cooccurrence, key, 0.0)
  end

  defp get_type_cooccurrence(chunk_a, chunk_b, state) do
    key = normalize_type_pair_key(chunk_a.type, chunk_b.type)
    Map.get(state.type_cooccurrence, key, 0.0)
  end

  defp get_embedding_similarity(chunk_a, chunk_b) do
    if chunk_a.embedding && chunk_b.embedding do
      cosine_similarity(chunk_a.embedding, chunk_b.embedding)
    else
      0.0
    end
  end

  defp fallback_score(chunk_a, chunk_b) do
    # Fallback when GenServer not available: use type-based heuristics
    type_compatibility = %{
      {:greeting, :body} => 0.8,
      {:greeting, :offer} => 0.7,
      {:greeting, :acknowledgment} => 0.6,
      {:acknowledgment, :body} => 0.8,
      {:acknowledgment, :clarification} => 0.7,
      {:body, :offer} => 0.8,
      {:body, :clarification} => 0.7,
      {:body, :closing} => 0.6,
      {:offer, :closing} => 0.5,
      {:clarification, :offer} => 0.4
    }

    key = normalize_type_pair_key(chunk_a.type, chunk_b.type)
    Map.get(type_compatibility, key, 0.3)
  end

  # ============================================================================
  # Similarity
  # ============================================================================

  defp cosine_similarity(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    if length(vec1) != length(vec2) do
      0.0
    else
      dot = Enum.zip(vec1, vec2) |> Enum.reduce(0.0, fn {a, b}, sum -> sum + a * b end)
      mag1 = :math.sqrt(Enum.reduce(vec1, 0.0, fn x, sum -> sum + x * x end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0.0, fn x, sum -> sum + x * x end))

      if mag1 == 0.0 or mag2 == 0.0, do: 0.0, else: dot / (mag1 * mag2)
    end
  end

  defp cosine_similarity(_, _), do: 0.0
end
