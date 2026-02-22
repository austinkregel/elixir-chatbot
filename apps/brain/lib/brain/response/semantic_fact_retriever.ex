defmodule Brain.Response.SemanticFactRetriever do
  @moduledoc "Semantic fact retrieval using TF-IDF embeddings.\n\nUnlike keyword matching, this module:\n- Embeds facts and queries using the same TF-IDF space\n- Uses cosine similarity to find semantically related facts\n- Handles synonyms and related concepts naturally\n- Works with learned facts that may have different phrasing\n\n## Example\n\n    # Query: \"What is an earthquake?\"\n    # Finds: \"earthquake causes: The shaking of the ground causes damage to buildings.\"\n    # Even though \"What is\" != \"causes\"\n"

  alias Brain.ML
  use GenServer
  require Logger

  alias Brain.Memory.Embedder
  alias Brain.FactDatabase

  @ets_table :semantic_fact_index
  @similarity_threshold 0.3
  @max_results 5

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Searches for facts semantically similar to the query.\n\nReturns facts ranked by cosine similarity to the query embedding.\n\n## Options\n  - `:threshold` - Minimum similarity score (default: 0.3)\n  - `:limit` - Maximum results (default: 5)\n  - `:category` - Filter by category before semantic search\n"
  @spec search(String.t(), keyword()) :: [map()]
  def search(query, opts \\ []) when is_binary(query) do
    GenServer.call(__MODULE__, {:search, query, opts}, 10_000)
  end

  @doc "Rebuilds the semantic index from current FactDatabase contents.\nCall after adding new facts.\n"
  @spec rebuild_index() :: :ok
  def rebuild_index do
    GenServer.call(__MODULE__, :rebuild_index, 60_000)
  end

  @doc "Checks if the semantic index is ready.\n"
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Gets stats about the semantic index.\n"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @impl true
  def init(_opts) do
    :ets.new(@ets_table, [:named_table, :set, :public, read_concurrency: true])
    send(self(), :build_initial_index)

    {:ok, %{indexed_count: 0, last_indexed: nil, ready: false}}
  end

  @impl true
  def handle_info(:build_initial_index, state) do
    if Embedder.ready?() do
      new_state = do_rebuild_index(state)
      {:noreply, new_state}
    else
      Process.send_after(self(), :build_initial_index, 2000)
      {:noreply, state}
    end
  end

  @impl true
  def handle_call({:search, query, opts}, _from, state) do
    if state.ready and Embedder.ready?() do
      results = do_search(query, opts)
      record_semantic_retrieval(length(results))
      {:reply, results, state}
    else
      record_semantic_retrieval(0)
      {:reply, [], state}
    end
  end

  @impl true
  def handle_call(:rebuild_index, _from, state) do
    new_state = do_rebuild_index(state)
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      indexed_count: state.indexed_count,
      last_indexed: state.last_indexed,
      ready: state.ready
    }

    {:reply, stats, state}
  end

  defp record_semantic_retrieval(result_count) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(Brain.Metrics.Aggregator, {:record_semantic_retrieval, result_count})
    end
  end

  defp do_rebuild_index(state) do
    Logger.info("Building semantic fact index...")
    :ets.delete_all_objects(@ets_table)
    facts = FactDatabase.query(limit: 10_000)

    indexed_count =
      facts
      |> Enum.reduce(0, fn fact, count ->
        search_text = build_search_text(fact)

        case Embedder.embed(search_text) do
          {:ok, embedding} ->
            :ets.insert(@ets_table, {fact.id, fact, embedding})
            count + 1

          {:error, _} ->
            count
        end
      end)

    Logger.info("Semantic fact index built", indexed_count: indexed_count)

    %{state | indexed_count: indexed_count, last_indexed: DateTime.utc_now(), ready: true}
  end

  defp do_search(query, opts) do
    threshold = Keyword.get(opts, :threshold, @similarity_threshold)
    limit = Keyword.get(opts, :limit, @max_results)
    category = Keyword.get(opts, :category)

    case Embedder.embed(query) do
      {:ok, query_embedding} ->
        :ets.tab2list(@ets_table)
        |> Enum.map(fn {_id, fact, fact_embedding} ->
          similarity = cosine_similarity(query_embedding, fact_embedding)
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

      {:error, _} ->
        []
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

  defp cosine_similarity(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    if length(vec1) != length(vec2) do
      0.0
    else
      dot_product = Enum.zip(vec1, vec2) |> Enum.reduce(0.0, fn {a, b}, acc -> acc + a * b end)
      magnitude1 = :math.sqrt(Enum.reduce(vec1, 0.0, fn x, acc -> acc + x * x end))
      magnitude2 = :math.sqrt(Enum.reduce(vec2, 0.0, fn x, acc -> acc + x * x end))

      if magnitude1 == 0.0 or magnitude2 == 0.0 do
        0.0
      else
        dot_product / (magnitude1 * magnitude2)
      end
    end
  end

  defp cosine_similarity(_, _) do
    0.0
  end
end
