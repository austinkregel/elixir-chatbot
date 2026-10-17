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

  # The session-level advisory lock a test run holds on its database, keyed
  # by this name; and where the VM keeps the connection that holds it.
  @test_run_lock_key :erlang.phash2("Atlas.Repo test run")
  @test_run_claim {__MODULE__, :test_run_claim}

  @doc """
  Claims the test database for the calling VM's test run, for as long as the
  VM runs, and raises when another run holds it.

  Each test runs in a sandbox transaction that is rolled back, but two runs on
  one database still share its unique keys: a row one run's open transaction
  writes, such as the node-wide metric window `Brain.Metrics.Aggregator`
  writes every second, holds up the other run's write of the same key until
  that transaction ends. The held-up write keeps its test's sandbox
  connection, so the test's own queries are dropped from the queue and the
  write is cancelled when the test ends, in tests that have nothing to do
  with metrics. So a run takes its database for itself: a session-level
  advisory lock, on a connection of its own outside the pool, held until the
  VM exits. A run on a database another run holds raises here, naming that
  run; `MIX_TEST_PARTITION` gives a run a database of its own.

  The connection does not reconnect: losing it would drop the lock, so it
  stops instead, and takes the calling process, the run, with it.

  `label` names the run (its app) in the connection's `application_name`,
  which a refused run reports. A VM that already holds the claim, as when the
  umbrella runs one app's suite after another, keeps it.
  """
  @spec claim_test_database!(String.t()) :: :ok
  def claim_test_database!(label) when is_binary(label) do
    case :persistent_term.get(@test_run_claim, nil) do
      conn when is_pid(conn) ->
        if Process.alive?(conn) do
          :ok
        else
          raise "Atlas.Repo: this test run's claim on its database was lost: the connection " <>
                  "holding it (#{inspect(conn)}) is gone"
        end

      nil ->
        claim_test_database_now!(label)
    end
  end

  defp claim_test_database_now!(label) do
    config = config()

    {:ok, conn} =
      config
      |> Keyword.take([:hostname, :port, :socket, :socket_dir, :username, :password, :database, :ssl])
      |> Keyword.merge(
        parameters: [application_name: "#{label} test run, OS pid #{System.pid()}"],
        backoff_type: :stop,
        max_restarts: 0,
        sync_connect: true
      )
      |> Postgrex.start_link()

    case Postgrex.query!(conn, "SELECT pg_try_advisory_lock($1)", [@test_run_lock_key]) do
      %{rows: [[true]]} ->
        :persistent_term.put(@test_run_claim, conn)
        :ok

      %{rows: [[false]]} ->
        %{rows: holders} =
          Postgrex.query!(
            conn,
            """
            SELECT a.application_name, a.backend_start
            FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid
            WHERE l.locktype = 'advisory' AND l.granted AND l.classid = 0 AND l.objsubid = 1
              AND l.objid::bigint = $1
              AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())
            """,
            [@test_run_lock_key]
          )

        raise "Atlas.Repo: the test database #{inspect(config[:database])} is in use by another test run " <>
                "(#{Enum.map_join(holders, ", ", fn [name, since] -> "#{inspect(name)}, since #{since}" end)}). " <>
                "Two runs on one database hold up each other's writes of the same keys, and their tests " <>
                "fail on time. Wait for that run to finish, or give this one a database of its own with " <>
                "MIX_TEST_PARTITION (prepared with `MIX_TEST_PARTITION=<n> MIX_ENV=test mix test.prepare`)."
    end
  end
end
