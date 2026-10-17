defmodule Brain.Response.TemplateBlender do
  @moduledoc "Generates novel responses by blending chunks from multiple templates.\n\nTemplate blending enables the system to compose responses from reusable\nparts learned from existing templates. This is particularly useful when:\n- No single template perfectly matches the context\n- The query spans multiple intents (e.g., greeting + question)\n- Need to combine acknowledgment with substantive content\n\n## How It Works\n\n1. **Chunk Pool**: Maintains a pool of embedded chunks segmented from templates\n2. **Context Filtering**: Filters chunks that match the current context\n3. **Chunk Selection**: Selects the best chunk for each needed type\n4. **Composition**: Combines selected chunks into a coherent response\n\n## Response Flow\n\nThe blender determines which chunk types to include based on context:\n\n| Context                 | Chunk Flow                          |\n|-------------------------|-------------------------------------|\n| Greeting + question     | greeting → body → offer             |\n| Introduction            | greeting → acknowledgment           |\n| Missing slots           | acknowledgment → clarification      |\n| Farewell                | body → closing                      |\n\n## Usage\n\n    TemplateBlender.blend(world_id, \"Hi, I need weather info\", %{\n      speech_acts: [:greeting, :request],\n      entities: [%{entity_type: \"topic\", value: \"weather\"}],\n      missing_slots: [\"location\"]\n    })\n    # => {:ok, \"Hello! For weather information, what location would you like?\"}\n\n## Worlds\n\nChunks and queries are embedded by a world's embedder (`Brain.Worlds.Embedder`), so the\nchunk pool is per world: a world's pool is segmented from the templates the\nfirst time the world blends, and `initialize_pool/1` rebuilds it. When the\nworld's vocabulary is rebuilt, or the world is destroyed, its pool is forgotten\n(`forget_world/1`) and its next blend segments it with the world's vocabulary.\nA world whose embedder cannot embed raises, and so does a chunk embedding whose\nlength differs from the query's. A rebuild of the world's vocabulary calls the\nblender, so the blender never waits for one: while the vocabulary is being\nrebuilt it replies `{:error, :vocabulary_building}`, and `blend/4` and\n`initialize_pool/1` wait in the caller for the rebuild to finish\n(`Brain.Worlds.Embedder.await_ready/2`) and ask again.\n"

  alias Brain.Response
  use GenServer
  require Logger

  alias Response.{ChunkSegmenter, ChunkCompatibility, TemplateStore}
  alias Brain.Worlds.Embedder, as: WorldEmbedder
  alias Brain.ML.Tokenizer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Generates a novel response by blending chunks from the world's pool.\n\n## Parameters\n- `world_id` - The world whose embedder embeds the chunks and the query\n- `query_text` - The original user query\n- `context` - Map with speech_acts, entities, filled_slots, missing_slots, etc.\n- `opts` - Options (e.g., `force_types: [:greeting, :body]`)\n\n## Returns\n- `{:ok, response}` - Successfully blended response\n- `{:error, reason}` - Failed to generate response\n"
  def blend(world_id, query_text, context, opts \\ []) when is_binary(world_id) do
    awaiting_rebuild(world_id, fn ->
      GenServer.call(__MODULE__, {:blend, world_id, query_text, context, opts}, 30_000)
    end)
  end

  # A rebuild of the world's vocabulary calls the blender (`forget_world/1`), so
  # the blender never waits for one: it replies `{:error, :vocabulary_building}`,
  # and the caller waits here for the rebuild to finish and asks again.
  defp awaiting_rebuild(world_id, call) do
    case call.() do
      {:error, :vocabulary_building} ->
        :ok = WorldEmbedder.await_ready(world_id)
        awaiting_rebuild(world_id, call)

      result ->
        result
    end
  end

  @doc "Checks if the blender is ready: the template store it segments has loaded.\nEach world's chunk pool is built on its first blend.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Rebuilds the world's chunk pool from the template store.\n"
  def initialize_pool(world_id) when is_binary(world_id) do
    awaiting_rebuild(world_id, fn -> GenServer.call(__MODULE__, {:initialize_pool, world_id}, 30_000) end)
  end

  @doc """
  Forgets the world's chunk pool, for a world whose vocabulary was rebuilt or
  that is destroyed (`Brain.EmbeddingCaches.forget_world/1`); the world's next
  blend segments it with the world's vocabulary. A blender that is not running
  holds no pool, so there is nothing to forget.
  """
  @spec forget_world(String.t()) :: :ok
  def forget_world(world_id) when is_binary(world_id) do
    if GenServer.whereis(__MODULE__) do
      GenServer.call(__MODULE__, {:forget_world, world_id}, 30_000)
    end

    :ok
  end

  @doc "Gets statistics about the chunk pools.\n"
  def stats do
    GenServer.call(__MODULE__, :stats)
  catch
    :exit, _ -> %{ready: false}
  end

  @impl true
  def init(_opts) do
    send(self(), :await_template_store)

    {:ok, %{ready: false, pools: %{}}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call({:initialize_pool, world_id}, _from, state) do
    case build_pool(world_id) do
      {:ok, pool} -> {:reply, :ok, %{state | pools: Map.put(state.pools, world_id, pool)}}
      {:error, :vocabulary_building} = building -> {:reply, building, state}
    end
  end

  def handle_call({:forget_world, world_id}, _from, state) do
    {:reply, :ok, %{state | pools: Map.delete(state.pools, world_id)}}
  end

  def handle_call({:blend, world_id, query_text, context, opts}, _from, state) do
    if state.ready do
      case pool_for(state, world_id) do
        {:ok, pool, state} -> {:reply, do_blend(world_id, query_text, context, opts, pool), state}
        {:error, :vocabulary_building} = building -> {:reply, building, state}
      end
    else
      {:reply, {:error, :not_ready}, state}
    end
  end

  def handle_call(:stats, _from, state) do
    pools = Map.values(state.pools)

    stats = %{
      ready: state.ready,
      worlds: map_size(state.pools),
      total_chunks: pools |> Enum.map(&length(&1.all_chunks)) |> Enum.sum(),
      chunks_by_type:
        pools
        |> Enum.flat_map(fn pool -> Enum.map(pool.chunks_by_type, fn {type, chunks} -> {type, length(chunks)} end) end)
        |> Enum.reduce(%{}, fn {type, count}, acc -> Map.update(acc, type, count, &(&1 + count)) end)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_info(:await_template_store, state) do
    if TemplateStore.ready?() do
      {:noreply, %{state | ready: true}}
    else
      Process.send_after(self(), :await_template_store, 1000)
      {:noreply, state}
    end
  end

  # The world's pool, segmented when the world has none. While the world's
  # vocabulary is being rebuilt nothing can be segmented, and this is
  # `{:error, :vocabulary_building}`: the blender is a process the rebuild
  # calls, so it never waits here.
  defp pool_for(state, world_id) do
    case Map.fetch(state.pools, world_id) do
      {:ok, pool} ->
        {:ok, pool, state}

      :error ->
        with {:ok, pool} <- build_pool(world_id) do
          {:ok, pool, %{state | pools: Map.put(state.pools, world_id, pool)}}
        end
    end
  end

  defp build_pool(world_id) do
    Logger.info("TemplateBlender: Initializing chunk pool for world #{world_id} from templates...")
    intents = TemplateStore.list_intents()

    templates_by_intent =
      intents
      |> Enum.map(fn intent ->
        templates = TemplateStore.get_templates(intent)
        {intent, templates}
      end)
      |> Enum.filter(fn {_, templates} -> templates != [] end)
      |> Map.new()

    with {:ok, templates_as_chunk_lists} <- segment_templates(world_id, templates_by_intent) do
      all_chunks = List.flatten(templates_as_chunk_lists)
      chunks_by_type = Enum.group_by(all_chunks, & &1.type)

      if ChunkCompatibility.ready?() do
        templates_as_chunk_lists
        |> Enum.filter(fn chunks -> length(chunks) > 1 end)
        |> ChunkCompatibility.learn()
      end

      Logger.info(
        "TemplateBlender: Pool for world #{world_id} ready with #{length(all_chunks)} chunks across #{map_size(chunks_by_type)} types"
      )

      {:ok, %{chunks_by_type: chunks_by_type, all_chunks: all_chunks}}
    end
  end

  # Each template's chunks, in the world's embedder, without waiting for a
  # rebuild (`ChunkSegmenter.try_segment/3`).
  defp segment_templates(world_id, templates_by_intent) do
    templates_by_intent
    |> Enum.flat_map(fn {intent, templates} -> Enum.map(templates, &{intent, &1}) end)
    |> Enum.reduce_while({:ok, []}, fn {intent, template}, {:ok, acc} ->
      case ChunkSegmenter.try_segment(world_id, template, intent) do
        {:ok, chunks} -> {:cont, {:ok, [chunks | acc]}}
        {:error, :vocabulary_building} = building -> {:halt, building}
      end
    end)
    |> case do
      {:ok, chunk_lists} -> {:ok, Enum.reverse(chunk_lists)}
      building -> building
    end
  end

  # While the world's vocabulary is being rebuilt the query cannot be embedded,
  # and this is `{:error, :vocabulary_building}`, which `blend/4` waits out.
  defp do_blend(world_id, query_text, context, opts, pool) do
    case WorldEmbedder.embed(world_id, query_text) do
      {:ok, query_embedding} ->
        blend_chunks(query_embedding, context, opts, pool)

      {:error, :vocabulary_building} = building ->
        building

      {:error, reason} ->
        raise "TemplateBlender: world #{inspect(world_id)}'s embedder cannot embed: #{inspect(reason)}"
    end
  end

  defp blend_chunks(query_embedding, context, opts, pool) do
    chunk_flow = determine_chunk_flow(context, opts)

    selected_chunks =
      chunk_flow
      |> Enum.map(fn type ->
        chunks = Map.get(pool.chunks_by_type, type, [])
        best = select_best_chunk(chunks, query_embedding, context)
        {type, best}
      end)
      |> Enum.filter(fn {_type, chunk} -> chunk != nil end)

    case compose_response(selected_chunks, context) do
      {:ok, response} when response != "" ->
        {:ok, response}

      _ ->
        {:error, :no_suitable_chunks}
    end
  end

  defp determine_chunk_flow(context, opts) do
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

    has_question =
      :question in speech_acts or :directive in speech_acts or :request in speech_acts

    has_farewell = :farewell in speech_acts or :closing in speech_acts
    needs_clarification = missing_slots != []

    cond do
      has_greeting and has_question and needs_clarification ->
        [:greeting, :body, :clarification]

      has_greeting and has_question ->
        [:greeting, :body, :offer]

      has_greeting ->
        [:greeting, :acknowledgment]

      has_question and needs_clarification ->
        [:acknowledgment, :clarification]

      has_question ->
        [:body, :offer]

      has_farewell ->
        [:body, :closing]

      true ->
        [:body]
    end
  end

  defp select_best_chunk(chunks, query_embedding, context) when is_list(chunks) do
    if chunks == [] do
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
      semantic_score = FourthWall.Math.cosine_similarity!(query_embedding, chunk.embedding)
      context_score = score_context_relevance(chunk, context)
      total_score = semantic_score * 0.6 + context_score * 0.4

      {chunk, total_score}
    end)
  end

  defp score_context_relevance(chunk, context) do
    entities = Map.get(context, :entities, [])
    target_intent = Map.get(context, :intent)

    intent_match =
      if chunk.source_intent && target_intent do
        if chunk.source_intent == target_intent do
          1.0
        else
          if related_intents?(chunk.source_intent, target_intent) do
            0.5
          else
            0.0
          end
        end
      else
        0.0
      end

    chunk_words =
      chunk.text |> Tokenizer.tokenize_normalized(min_length: 2) |> MapSet.new()

    entity_match =
      entities
      |> Enum.any?(fn entity ->
        entity_type = entity[:entity_type] || entity["entity_type"] || ""
        et_words = entity_type |> Tokenizer.tokenize_normalized(min_length: 2)
        et_words != [] and Enum.any?(et_words, &MapSet.member?(chunk_words, &1))
      end)

    entity_score =
      if entity_match do
        0.3
      else
        0.0
      end

    intent_match * 0.7 + entity_score * 0.3
  end

  defp related_intents?(intent_a, intent_b) do
    Brain.Analysis.IntentUtils.same_domain_prefix?(intent_a, intent_b)
  end

  defp compose_response(selected_chunks, context) do
    texts =
      selected_chunks
      |> Enum.map(fn {_type, chunk} -> chunk.text end)

    if texts != [] do
      response = join_with_flow(texts)
      entities = Map.get(context, :entities, [])
      final_response = TemplateStore.substitute_slots(response, entities)

      {:ok, final_response}
    else
      {:error, :empty}
    end
  end

  defp join_with_flow(texts) do
    texts
    |> Enum.filter(&(is_binary(&1) and String.length(&1) > 0))
    |> Enum.join(" ")
  end
end
