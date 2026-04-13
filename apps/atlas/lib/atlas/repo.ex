defmodule Atlas.Repo do
  use Ecto.Repo,
    otp_app: :atlas,
    adapter: Ecto.Adapters.Postgres

  @doc """
  Called by Postgrex on each new connection from the pool.
  Loads the Apache AGE extension and sets the search path so
  the `cypher()` SQL function is available without per-query setup.

  Silently skips if AGE is not yet installed (e.g. during initial
  migration on a fresh database).
  """
  def load_age(conn) do
    case Postgrex.query(conn, "LOAD 'age'", []) do
      {:ok, _} ->
        Postgrex.query!(conn, "SET search_path = ag_catalog, \"$user\", public", [])

      {:error, _} ->
        :ok
    end
  end
end
