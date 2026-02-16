defmodule Brain.Memory.VectorIndex do
  @moduledoc "Vector index for approximate nearest neighbor search.\n\nPorted from the Rust cognitive_memory_system NaiveIndex.\n\nThis implementation uses ETS for storage and performs linear search\nover stored embeddings. Sufficient for small datasets (<10k items).\nFor larger datasets, could be extended with HNSW or other ANN algorithms.\n"

  require Logger

  @type embedding :: [float()]
  @type search_result :: {String.t(), float()}

  @doc "Create a new vector index backed by an ETS table.\nReturns the table reference.\n"
  def new(name \\ nil) do
    table_name = name || generate_table_name()
    :ets.new(table_name, [:set, :public, read_concurrency: true])
  end

  defp generate_table_name do
    id = :erlang.unique_integer([:positive])
    String.to_atom("memory_vector_index_#{id}")
  end

  @doc "Insert a vector with its ID into the index.\n"
  def insert(table, id, embedding) when is_list(embedding) do
    :ets.insert(table, {id, embedding})
    :ok
  end

  @doc "Search for the top `k` vectors most similar to the query.\nReturns a list of {id, similarity} tuples sorted by similarity descending.\n"
  def search(table, query_embedding, k) when is_list(query_embedding) and is_integer(k) do
    search_all(table, query_embedding, k)
  end

  @doc "Search for all vectors similar to the query, returning top k.\nReturns a list of {id, similarity} tuples sorted by similarity descending.\n\nThis is useful when IDs are composite (e.g., {world_id, episode_id}) and\nfiltering is needed after the similarity search.\n"
  def search_all(table, query_embedding, k) when is_list(query_embedding) do
    table
    |> :ets.tab2list()
    |> Enum.map(fn {id, embedding} ->
      similarity = cosine_similarity(query_embedding, embedding)
      {id, similarity}
    end)
    |> Enum.sort_by(fn {_id, sim} -> -sim end)
    |> Enum.take(k)
  end

  @doc "Remove a vector by ID from the index.\n"
  def delete(table, id) do
    :ets.delete(table, id)
    :ok
  end

  @doc "Get the count of vectors in the index.\n"
  def count(table) do
    :ets.info(table, :size)
  end

  @doc "Clear all vectors from the index.\n"
  def clear(table) do
    :ets.delete_all_objects(table)
    :ok
  end

  @doc "Get a specific embedding by ID.\n"
  def get(table, id) do
    case :ets.lookup(table, id) do
      [{^id, embedding}] -> {:ok, embedding}
      [] -> {:error, :not_found}
    end
  end

  @doc "Check if an ID exists in the index.\n"
  def exists?(table, id) do
    :ets.member(table, id)
  end

  @doc "Export all vectors as a list of {id, embedding} tuples.\n"
  def export(table) do
    :ets.tab2list(table)
  end

  @doc "Import vectors from a list of {id, embedding} tuples.\n"
  def import(table, vectors) when is_list(vectors) do
    Enum.each(vectors, fn {id, embedding} ->
      insert(table, id, embedding)
    end)

    :ok
  end

  @doc "Compute cosine similarity between two vectors.\nReturns a value in [-1, 1].\n"
  def cosine_similarity(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b),
    do: FourthWall.Math.cosine_similarity(vec_a, vec_b)

  @doc "Compute euclidean distance between two vectors.\n"
  def euclidean_distance(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b) do
    if length(vec_a) != length(vec_b) do
      :infinity
    else
      sum_sq =
        Enum.zip(vec_a, vec_b)
        |> Enum.reduce(0.0, fn {a, b}, acc -> acc + (a - b) * (a - b) end)

      :math.sqrt(sum_sq)
    end
  end

  @doc "Compute the mean of multiple vectors (for centroid calculation).\n"
  def mean_vector(vectors) when is_list(vectors) do
    if vectors == [] do
      []
    else
      vec_length = length(List.first(vectors))
      count = length(vectors)

      for i <- 0..(vec_length - 1) do
        sum =
          Enum.reduce(vectors, 0.0, fn vec, acc ->
            acc + Enum.at(vec, i, 0.0)
          end)

        sum / count
      end
    end
  end
end
