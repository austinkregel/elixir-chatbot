defmodule ChatBot.Response.TemplateBlender do
  @moduledoc """
  Generates novel responses by blending chunks from multiple templates.

  Template blending enables the system to compose responses from reusable
  parts learned from existing templates. This is particularly useful when:
  - No single template perfectly matches the context
  - The query spans multiple intents (e.g., greeting + question)
  - Need to combine acknowledgment with substantive content

  ## How It Works

  1. **Chunk Pool**: Maintains a pool of embedded chunks segmented from templates
  2. **Context Filtering**: Filters chunks that match the current context
  3. **Chunk Selection**: Selects the best chunk for each needed type
  4. **Composition**: Combines selected chunks into a coherent response

  ## Response Flow

  The blender determines which chunk types to include based on context:

  | Context                 | Chunk Flow                          |
  |-------------------------|-------------------------------------|
  | Greeting + question     | greeting → body → offer             |
  | Introduction            | greeting → acknowledgment           |
  | Missing slots           | acknowledgment → clarification      |
  | Farewell                | body → closing                      |

  ## Usage

      TemplateBlender.blend("Hi, I need weather info", %{
        speech_acts: [:greeting, :request],
        entities: [%{entity_type: "topic", value: "weather"}],
        missing_slots: ["location"]
      })
      # => {:ok, "Hello! For weather information, what location would you like?"}
  """

  use GenServer
  require Logger

  alias ChatBot.Response.{ChunkSegmenter, ChunkCompatibility, TemplateStore}
  alias ChatBot.Memory.Embedder

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Generates a novel response by blending chunks from multiple templates.

  ## Parameters
  - `query_text` - The original user query
  - `context` - Map with speech_acts, entities, filled_slots, missing_slots, etc.
  - `opts` - Options (e.g., `force_types: [:greeting, :body]`)

  ## Returns
  - `{:ok, response}` - Successfully blended response
  - `{:error, reason}` - Failed to generate response
  """
  def blend(query_text, context, opts \\ []) do
    GenServer.call(__MODULE__, {:blend, query_text, context, opts}, 10_000)
  catch
    :exit, _ -> blend_fallback(query_text, context, opts)
  end

  @doc """
  Checks if the blender is ready with a chunk pool.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Initializes the chunk pool from template store.
  Call this after TemplateStore is loaded.
  """
  def initialize_pool do
    GenServer.call(__MODULE__, :initialize_pool, 30_000)
  end

  @doc """
  Gets statistics about the chunk pool.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  catch
    :exit, _ -> %{ready: false}
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Initialize pool lazily
    send(self(), :initialize_pool_async)

    {:ok,
     %{
       ready: false,
       chunks_by_type: %{},
       all_chunks: [],
       initialized: false
     }}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call(:initialize_pool, _from, state) do
    new_state = do_initialize_pool(state)
    {:reply, :ok, new_state}
  end

  def handle_call({:blend, query_text, context, opts}, _from, state) do
    if state.ready do
      result = do_blend(query_text, context, opts, state)
      {:reply, result, state}
    else
      {:reply, {:error, :not_ready}, state}
    end
  end

  def handle_call(:stats, _from, state) do
    stats = %{
      ready: state.ready,
      total_chunks: length(state.all_chunks),
      chunks_by_type:
        state.chunks_by_type
        |> Enum.map(fn {type, chunks} -> {type, length(chunks)} end)
        |> Map.new()
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_info(:initialize_pool_async, state) do
    # Wait for TemplateStore to be ready
    if TemplateStore.ready?() do
      new_state = do_initialize_pool(state)
      {:noreply, new_state}
    else
      # Retry later
      Process.send_after(self(), :initialize_pool_async, 1000)
      {:noreply, state}
    end
  end

  # ============================================================================
  # Pool Initialization
  # ============================================================================

  defp do_initialize_pool(state) do
    Logger.info("TemplateBlender: Initializing chunk pool from templates...")

    # Get all templates from TemplateStore
    intents = TemplateStore.list_intents()

    templates_by_intent =
      intents
      |> Enum.map(fn intent ->
        templates = TemplateStore.get_templates(intent)
        {intent, templates}
      end)
      |> Enum.filter(fn {_, templates} -> length(templates) > 0 end)
      |> Map.new()

    # Segment all templates into chunks
    all_chunks = ChunkSegmenter.segment_all(templates_by_intent)

    # Group chunks by type for fast lookup
    chunks_by_type = Enum.group_by(all_chunks, & &1.type)

    # Learn chunk compatibility (if compatibility server is running)
    if ChunkCompatibility.ready?() do
      # Group chunks by source template for learning
      templates_as_chunk_lists =
        templates_by_intent
        |> Enum.flat_map(fn {intent, templates} ->
          Enum.map(templates, fn template ->
            ChunkSegmenter.segment(template, intent)
          end)
        end)
        |> Enum.filter(fn chunks -> length(chunks) > 1 end)

      ChunkCompatibility.learn(templates_as_chunk_lists)
    end

    Logger.info(
      "TemplateBlender: Pool ready with #{length(all_chunks)} chunks across #{map_size(chunks_by_type)} types"
    )

    %{
      state
      | ready: true,
        chunks_by_type: chunks_by_type,
        all_chunks: all_chunks,
        initialized: true
    }
  end

  # ============================================================================
  # Blending Logic
  # ============================================================================

  defp do_blend(query_text, context, opts, state) do
    # Step 1: Determine which chunk types we need
    chunk_flow = determine_chunk_flow(context, opts)

    # Step 2: Get query embedding for similarity ranking
    query_embedding =
      case Embedder.embed(query_text) do
        {:ok, embedding} -> embedding
        _ -> nil
      end

    # Step 3: Select best chunk for each type
    selected_chunks =
      chunk_flow
      |> Enum.map(fn type ->
        chunks = Map.get(state.chunks_by_type, type, [])
        best = select_best_chunk(chunks, query_embedding, context)
        {type, best}
      end)
      |> Enum.filter(fn {_type, chunk} -> chunk != nil end)

    # Step 4: Compose response from selected chunks
    case compose_response(selected_chunks, context) do
      {:ok, response} when response != "" ->
        {:ok, response}

      _ ->
        {:error, :no_suitable_chunks}
    end
  end

  defp determine_chunk_flow(context, opts) do
    # Override with forced types if specified
    if force_types = Keyword.get(opts, :force_types) do
      force_types
    else
      infer_chunk_flow(context)
    end
  end

  defp infer_chunk_flow(context) do
    speech_acts = Map.get(context, :speech_acts, [])
    missing_slots = Map.get(context, :missing_slots, [])

    has_greeting = :greeting in speech_acts or :expressive in speech_acts
    has_question = :question in speech_acts or :directive in speech_acts or :request in speech_acts
    has_farewell = :farewell in speech_acts or :closing in speech_acts
    needs_clarification = length(missing_slots) > 0

    cond do
      # Greeting + question with missing info
      has_greeting and has_question and needs_clarification ->
        [:greeting, :body, :clarification]

      # Greeting + question
      has_greeting and has_question ->
        [:greeting, :body, :offer]

      # Just greeting (introduction)
      has_greeting ->
        [:greeting, :acknowledgment]

      # Question with missing info
      has_question and needs_clarification ->
        [:acknowledgment, :clarification]

      # Just question
      has_question ->
        [:body, :offer]

      # Farewell
      has_farewell ->
        [:body, :closing]

      # Default
      true ->
        [:body]
    end
  end

  defp select_best_chunk(chunks, query_embedding, context) when is_list(chunks) do
    if length(chunks) == 0 do
      nil
    else
      chunks
      |> score_chunks(query_embedding, context)
      |> Enum.sort_by(fn {_chunk, score} -> -score end)
      |> List.first()
      |> case do
        {chunk, _score} -> chunk
        nil -> nil
      end
    end
  end

  defp score_chunks(chunks, query_embedding, context) do
    Enum.map(chunks, fn chunk ->
      # Score based on:
      # 1. Semantic similarity to query
      # 2. Relevance to context (entities, intent)

      semantic_score =
        if query_embedding && chunk.embedding do
          cosine_similarity(query_embedding, chunk.embedding)
        else
          0.0
        end

      context_score = score_context_relevance(chunk, context)

      # Weighted combination
      total_score = semantic_score * 0.6 + context_score * 0.4

      {chunk, total_score}
    end)
  end

  defp score_context_relevance(chunk, context) do
    # Check if chunk's source intent matches any context signals
    entities = Map.get(context, :entities, [])
    target_intent = Map.get(context, :intent)

    intent_match =
      if chunk.source_intent && target_intent do
        if chunk.source_intent == target_intent do
          1.0
        else
          # Partial match for related intents
          if related_intents?(chunk.source_intent, target_intent), do: 0.5, else: 0.0
        end
      else
        0.0
      end

    # Check if chunk mentions any entity types present
    entity_match =
      entities
      |> Enum.any?(fn entity ->
        entity_type = entity[:entity_type] || entity["entity_type"] || ""
        String.contains?(String.downcase(chunk.text), String.downcase(entity_type))
      end)

    entity_score = if entity_match, do: 0.3, else: 0.0

    intent_match * 0.7 + entity_score * 0.3
  end

  defp related_intents?(intent_a, intent_b) do
    # Check if intents share a common prefix (e.g., "weather.query" and "weather.info")
    parts_a = String.split(intent_a, ".")
    parts_b = String.split(intent_b, ".")

    if length(parts_a) > 0 and length(parts_b) > 0 do
      List.first(parts_a) == List.first(parts_b)
    else
      false
    end
  end

  defp compose_response(selected_chunks, context) do
    # Join chunks with appropriate connectors
    texts =
      selected_chunks
      |> Enum.map(fn {_type, chunk} -> chunk.text end)

    if length(texts) > 0 do
      response = join_with_flow(texts)

      # Substitute any slot placeholders
      entities = Map.get(context, :entities, [])
      final_response = TemplateStore.substitute_slots(response, entities)

      {:ok, final_response}
    else
      {:error, :empty}
    end
  end

  defp join_with_flow(texts) do
    # Join text chunks with appropriate spacing
    texts
    |> Enum.filter(&(is_binary(&1) and String.length(&1) > 0))
    |> Enum.join(" ")
  end

  # ============================================================================
  # Fallback (when GenServer not available)
  # ============================================================================

  defp blend_fallback(_query_text, _context, _opts) do
    {:error, :blender_not_available}
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
