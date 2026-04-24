defmodule Atlas.Repo do
  use Ecto.Repo,
    otp_app: :atlas,
    adapter: Ecto.Adapters.Postgres

  @doc """
  Called by Postgrex on each new connection from the pool in dev/prod.

  Loads the Apache AGE extension and sets the search path so the
  `cypher()` SQL function is available without per-query setup.

  Fails loud on unexpected errors. The narrow `:undefined_file` case
  (AGE shared library not present in `$libdir`) is treated as
  recoverable so that a Repo pointed at a non-AGE Postgres image
  (e.g., during a release migration window or a managed-Postgres
  fallback) can still come up. With the `mix atlas.bootstrap_age`
  task and the Docker initdb script in place, the extension itself
  must already exist before this callback runs.
  """
  def load_age(conn) do
    case Postgrex.query(conn, "LOAD 'age'", []) do
      {:ok, _} ->
        Postgrex.query!(conn, ~s(SET search_path = "$user", public, ag_catalog), [])

      {:error, %Postgrex.Error{postgres: %{code: :undefined_file}}} ->
        :ok

      {:error, reason} ->
        raise "Atlas.Repo.load_age failed unexpectedly: #{inspect(reason)}"
    end
  end

  @doc """
  Test-environment `after_connect` callback.

  Pins the dedicated `atlas_test` schema first on the search path so
  every Atlas table (including `schema_migrations`) lives in a single
  schema isolated from dev/prod. AGE is loaded in front of `public`
  if available, but never in front of `atlas_test`.

  Falls back to a search path without `ag_catalog` if `LOAD 'age'`
  fails — test correctness for non-graph code does not depend on AGE
  being loaded, and `atlas_test` is what matters for migration
  resolution.
  """
  def load_age_test(conn) do
    case Postgrex.query(conn, "LOAD 'age'", []) do
      {:ok, _} ->
        Postgrex.query!(
          conn,
          ~s(SET search_path = atlas_test, ag_catalog, "$user", public),
          []
        )

      {:error, _} ->
        Postgrex.query!(
          conn,
          ~s(SET search_path = atlas_test, "$user", public),
          []
        )
    end
  end
end
