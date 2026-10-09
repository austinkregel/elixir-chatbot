defmodule Brain.Response.SemanticFactRetriever do
  @moduledoc "Semantic fact retrieval using TF-IDF embeddings.\n\nUnlike keyword matching, this module:\n- Embeds facts and queries using the same TF-IDF space\n- Uses cosine similarity to find semantically related facts\n- Handles synonyms and related concepts naturally\n- Works with learned facts that may have different phrasing\n\nThe TF-IDF space is a world's embedder (`World.Embedder`), so the index is\nper world: a world's index embeds the FactDatabase's facts with that world's\nembedder, is built the first time the world searches, and is rebuilt with\n`rebuild_index/1` after facts change. When the world's vocabulary is rebuilt,\nor the world is destroyed, its index is forgotten (`forget_world/1`) and its\nnext search builds it with the world's vocabulary. A world whose embedder\ncannot embed raises, and so does a fact embedding whose length differs from\nthe query's. A rebuild of the world's vocabulary calls the retriever, so the\nretriever never waits for one: while the vocabulary is being rebuilt it\nreplies `{:error, :vocabulary_building}`, and `search/3` and `rebuild_index/1`\nwait in the caller for the rebuild to finish (`World.Embedder.await_ready/2`)\nand ask again.\n\n## Example\n\n    # Query: \"What is an earthquake?\"\n    # Finds: \"earthquake causes: The shaking of the ground causes damage to buildings.\"\n    # Even though \"What is\" != \"causes\"\n"

  alias Brain.ML
  use GenServer
  require Logger

  # World.Embedder is in a sibling umbrella app that depends on :brain.
  @compile {:no_warn_undefined, World.Embedder}
  alias World.Embedder, as: WorldEmbedder
  alias Brain.FactDatabase

  @ets_table :semantic_fact_index
  @similarity_threshold 0.3
  @max_results 5

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Searches the world's index for facts semantically similar to the query.\n\nReturns facts ranked by cosine similarity to the query embedding. The world's\nindex is built first when the world has none.\n\n## Options\n  - `:threshold` - Minimum similarity score (default: 0.3)\n  - `:limit` - Maximum results (default: 5)\n  - `:category` - Filter by category before semantic search\n"
  @spec search(String.t(), String.t(), keyword()) :: [map()]
  def search(world_id, query, opts \\ []) when is_binary(world_id) and is_binary(query) do
    awaiting_rebuild(world_id, fn -> GenServer.call(__MODULE__, {:search, world_id, query, opts}, 60_000) end)
  end

  @doc "Rebuilds the world's semantic index from current FactDatabase contents.\nCall after adding new facts.\n"
  @spec rebuild_index(String.t()) :: :ok
  def rebuild_index(world_id) when is_binary(world_id) do
    awaiting_rebuild(world_id, fn -> GenServer.call(__MODULE__, {:rebuild_index, world_id}, 60_000) end)
  end

  # A rebuild of the world's vocabulary calls the retriever (`forget_world/1`),
  # so the retriever never waits for one: it replies
  # `{:error, :vocabulary_building}`, and the caller waits here for the rebuild
  # to finish and asks again.
  defp awaiting_rebuild(world_id, call) do
    case call.() do
      {:error, :vocabulary_building} ->
        :ok = WorldEmbedder.await_ready(world_id)
        awaiting_rebuild(world_id, call)

      result ->
        result
    end
  end

  @doc """
  Forgets the world's index, for a world whose vocabulary was rebuilt or that
  is destroyed (`Brain.EmbeddingCaches.forget_world/1`); the world's next
  search builds it with the world's vocabulary. A retriever that is not
  running holds no index, so there is nothing to forget.
  """
  @spec forget_world(String.t()) :: :ok
  def forget_world(world_id) when is_binary(world_id) do
    if GenServer.whereis(__MODULE__) do
      GenServer.call(__MODULE__, {:forget_world, world_id}, 60_000)
    end

    :ok
  end

  @doc "Checks if the retriever is running. Each world's index is built on its first search.\n"
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Gets stats about the semantic indexes: the worlds indexed, and their facts.\n"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @impl true
  def init(_opts) do
    :ets.new(@ets_table, [:named_table, :set, :public, read_concurrency: true])

    {:ok, %{indexes: %{}, ready: true}}
  end

  @impl true
  def handle_call({:search, world_id, query, opts}, _from, state) do
    case ensure_index(state, world_id) do
      {:ok, state} ->
        case do_search(world_id, query, opts) do
          {:ok, results} ->
            record_semantic_retrieval(length(results))
            {:reply, results, state}

          {:error, :vocabulary_building} = building ->
            {:reply, building, state}
        end

      {{:error, :vocabulary_building} = building, state} ->
        {:reply, building, state}
    end
  end

  @impl true
  def handle_call({:rebuild_index, world_id}, _from, state) do
    case do_rebuild_index(state, world_id) do
      {:ok, state} -> {:reply, :ok, state}
      {{:error, :vocabulary_building} = building, state} -> {:reply, building, state}
    end
  end

  @impl true
  def handle_call({:forget_world, world_id}, _from, state) do
    :ets.match_delete(@ets_table, {{world_id, :_}, :_, :_})
    {:reply, :ok, %{state | indexes: Map.delete(state.indexes, world_id)}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      worlds_indexed: map_size(state.indexes),
      indexed_count: state.indexes |> Map.values() |> Enum.map(& &1.indexed_count) |> Enum.sum(),
      last_indexed:
        state.indexes
        |> Map.values()
        |> Enum.map(& &1.last_indexed)
        |> Enum.max(DateTime, fn -> nil end),
      ready: state.ready
    }

    {:reply, stats, state}
  end

  defp record_semantic_retrieval(result_count) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(Brain.Metrics.Aggregator, {:record_semantic_retrieval, result_count})
    end
  end

  defp ensure_index(state, world_id) do
    if Map.has_key?(state.indexes, world_id), do: {:ok, state}, else: do_rebuild_index(state, world_id)
  end

  # While the world's vocabulary is being rebuilt from its episodes no fact can
  # be embedded, so the world is left unindexed, and this is
  # `{{:error, :vocabulary_building}, state}`: the caller waits for the rebuild
  # and asks again, which builds it.
  defp do_rebuild_index(state, world_id) do
    Logger.info("Building semantic fact index for world #{world_id}...")
    :ets.match_delete(@ets_table, {{world_id, :_}, :_, :_})
    facts = FactDatabase.query(limit: 10_000)

    indexed =
      Enum.reduce_while(facts, {:ok, 0}, fn fact, {:ok, count} ->
        case world_embedding(world_id, build_search_text(fact)) do
          {:ok, embedding} ->
            :ets.insert(@ets_table, {{world_id, fact.id}, fact, embedding})
            {:cont, {:ok, count + 1}}

          :vocabulary_building ->
            {:halt, :vocabulary_building}
        end
      end)

    case indexed do
      {:ok, indexed_count} ->
        Logger.info("Semantic fact index built", world_id: world_id, indexed_count: indexed_count)

        entry = %{indexed_count: indexed_count, last_indexed: DateTime.utc_now()}
        {:ok, %{state | indexes: Map.put(state.indexes, world_id, entry)}}

      :vocabulary_building ->
        :ets.match_delete(@ets_table, {{world_id, :_}, :_, :_})
        {{:error, :vocabulary_building}, %{state | indexes: Map.delete(state.indexes, world_id)}}
    end
  end

  defp do_search(world_id, query, opts) do
    threshold = Keyword.get(opts, :threshold, @similarity_threshold)
    limit = Keyword.get(opts, :limit, @max_results)
    category = Keyword.get(opts, :category)

    case world_embedding(world_id, query) do
      {:ok, query_embedding} ->
        results =
          :ets.match_object(@ets_table, {{world_id, :_}, :_, :_})
          |> Enum.map(fn {_key, fact, fact_embedding} ->
            similarity = FourthWall.Math.cosine_similarity!(query_embedding, fact_embedding)
            {fact, similarity}
          end)
          |> Enum.filter(fn {fact, similarity} ->
            similarity >= threshold and
              (is_nil(category) or fact.category == category)
          end)
          |> Enum.sort_by(fn {_, similarity} -> -similarity end)
          |> Enum.take(limit)
          |> Enum.map(fn {fact, similarity} ->
            %{
              fact: fact,
              similarity: similarity
            }
          end)

        {:ok, results}

      :vocabulary_building ->
        {:error, :vocabulary_building}
    end
  end

  # A world's embedding comes from the world's embedder and nowhere else. A world
  # whose embedder has no copy, no vocabulary, or no table raises.
  defp world_embedding(world_id, text) do
    case WorldEmbedder.embed(world_id, text) do
      {:ok, embedding} ->
        {:ok, embedding}

      {:error, :vocabulary_building} ->
        :vocabulary_building

      {:error, reason} ->
        raise "SemanticFactRetriever: world #{inspect(world_id)}'s embedder cannot embed: #{inspect(reason)}"
    end
  end

  defp build_search_text(fact) do
    entity_content = extract_content_words(fact.entity)
    fact_content = fact.fact
    "#{entity_content} #{fact_content}"
  end

  defp extract_content_words(text) when is_binary(text) do
    alias ML.{Tokenizer, POSTagger}
    content_tags = ~w(NOUN PROPN VERB ADJ ADV NUM)

    tokens = Tokenizer.tokenize_words(text)

    case POSTagger.load_model() do
      {:ok, model} ->
        tags = POSTagger.predict_tags(tokens, model)

        Enum.zip(tokens, tags)
        |> Enum.filter(fn {_token, tag} -> tag in content_tags end)
        |> Enum.map_join(
          " ",
          fn {token, _tag} -> token end
        )

      {:error, _} ->
        text
    end
  end

  defp extract_content_words(text) do
    to_string(text)
  end
end
