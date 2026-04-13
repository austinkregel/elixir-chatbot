defmodule Atlas.Graph.Cypher do
  @moduledoc """
  Low-level Cypher query execution through Apache AGE.

  AGE extends PostgreSQL with openCypher support. Cypher queries
  are embedded in SQL via the `cypher()` function from the `ag_catalog` schema.

  All queries go through `Ecto.Adapters.SQL.query/3` so they participate
  in Ecto's connection pool and (in tests) the SQL sandbox.
  """

  alias Atlas.Graph.Types.{Vertex, Edge}

  @doc """
  Execute a Cypher query against a named graph.

  Returns `{:ok, results}` or `{:error, reason}`.

  ## Parameters
    - repo: The Ecto repo (usually Atlas.Repo)
    - graph_name: Name of the AGE graph (e.g., "knowledge_graph")
    - query: openCypher query string
    - params: Map of query parameters (embedded as literals -- AGE does not support $-params in all contexts)

  ## Examples

      Cypher.execute(Atlas.Repo, "knowledge_graph", "MATCH (n) RETURN n LIMIT 10")

      Cypher.execute(Atlas.Repo, "user_graph",
        "MATCH (u:User {name: '\#{name}'}) RETURN u")
  """
  def execute(repo, graph_name, query, _params \\ %{}) do
    col_count = count_return_columns(query)
    sql = build_sql(graph_name, query, col_count)

    case Ecto.Adapters.SQL.query(repo, sql, []) do
      {:ok, %{rows: rows, columns: columns}} ->
        {:ok, parse_results(rows, columns)}

      {:error, %Postgrex.Error{postgres: %{message: message}}} ->
        {:error, message}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Execute a Cypher query, raising on error.
  """
  def execute!(repo, graph_name, query, params \\ %{}) do
    case execute(repo, graph_name, query, params) do
      {:ok, results} -> results
      {:error, reason} -> raise "Cypher query failed: #{inspect(reason)}"
    end
  end

  defp build_sql(graph_name, query, col_count) do
    col_defs =
      1..col_count
      |> Enum.map(fn i -> "col#{i} agtype" end)
      |> Enum.join(", ")

    """
    SELECT * FROM cypher('#{graph_name}', $$
      #{query}
    $$) AS (#{col_defs})
    """
  end

  # Count the number of expressions in the RETURN clause.
  # Handles nested function calls like `count(n)`, `id(a)`, `properties(n)`.
  defp count_return_columns(query) do
    # Find the RETURN clause (case-insensitive), taking the last one if multiple
    case Regex.run(~r/\bRETURN\s+(.+?)(?:\s+ORDER\s+|\s+LIMIT\s+|\s+SKIP\s+|\s*$)/is, query) do
      [_, return_expr] ->
        # Count top-level commas (not inside parentheses) to determine column count
        return_expr
        |> String.graphemes()
        |> Enum.reduce({0, 1}, fn
          "(", {depth, count} -> {depth + 1, count}
          ")", {depth, count} -> {max(depth - 1, 0), count}
          ",", {0, count} -> {0, count + 1}
          _, acc -> acc
        end)
        |> elem(1)

      nil ->
        1
    end
  end

  @doc """
  Parse raw AGE result rows into Elixir structs.

  AGE returns results as `agtype` which Postgrex sees as text.
  We parse the JSON-like agtype format into our Vertex/Edge structs.
  """
  def parse_results(rows, _columns) do
    Enum.map(rows, fn row ->
      Enum.map(row, &parse_agtype/1)
    end)
  end

  defp parse_agtype(nil), do: nil

  defp parse_agtype(value) when is_binary(value) do
    cond do
      String.contains?(value, "::vertex") ->
        parse_vertex(value)

      String.contains?(value, "::edge") ->
        parse_edge(value)

      true ->
        parse_scalar(value)
    end
  end

  defp parse_agtype(value), do: value

  defp parse_vertex(raw) do
    json_str = String.replace(raw, "::vertex", "")

    case Jason.decode(json_str) do
      {:ok, %{"id" => id, "label" => label, "properties" => props}} ->
        %Vertex{id: id, label: label, properties: props}

      _ ->
        raw
    end
  end

  defp parse_edge(raw) do
    json_str = String.replace(raw, "::edge", "")

    case Jason.decode(json_str) do
      {:ok, %{"id" => id, "start_id" => start_id, "end_id" => end_id, "label" => label, "properties" => props}} ->
        %Edge{id: id, start_id: start_id, end_id: end_id, label: label, properties: props}

      _ ->
        raw
    end
  end

  defp parse_scalar(value) do
    case Jason.decode(value) do
      {:ok, decoded} -> decoded
      _ -> value
    end
  end
end
