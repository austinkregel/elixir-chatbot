defmodule Atlas.DataCase do
  @moduledoc """
  This module defines the setup for tests requiring
  access to the Atlas data layer.

  You may define functions here to be used as helpers in
  your tests.

  Finally, if the test case interacts with the database,
  we enable the SQL sandbox so changes are rolled back
  at the end of every test.

  ## Starting from an empty table

  The test database is prepared by `mix test.prepare` with a seeded lexicon
  that most suites read, so a test's tables are not empty to begin with. A
  test that must start from nothing — one that counts whole tables, say —
  declares it:

      @moduletag :blank_slate

  The tables are emptied inside that test's own transaction, so the seed is
  back for the next test.
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      alias Atlas.Repo

      import Ecto
      import Ecto.Changeset
      import Ecto.Query
      import Atlas.DataCase
    end
  end

  setup tags do
    Atlas.DataCase.setup_sandbox(tags)
    :ok
  end

  @doc """
  Sets up the sandbox based on the test tags.
  """
  def setup_sandbox(tags) do
    pid = Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: not tags[:async])

    # AGE requires LOAD and search_path to be set per-connection.
    # In test, atlas_test must come first so all Atlas tables resolve there;
    # ag_catalog follows so cypher() and AGE graphs remain accessible.
    Ecto.Adapters.SQL.query!(Atlas.Repo, "LOAD 'age'", [])

    Ecto.Adapters.SQL.query!(
      Atlas.Repo,
      ~s(SET search_path = atlas_test, ag_catalog, "$user", public),
      []
    )

    if tags[:blank_slate], do: blank_slate!()

    on_exit(fn -> Ecto.Adapters.SQL.Sandbox.stop_owner(pid) end)
  end

  @doc """
  Empties every table of the test schema but the migration log, inside the
  caller's sandbox transaction, so it is rolled back with the test. For a test
  that asserts on whole tables and must not see the prepared seed.
  """
  def blank_slate! do
    %{rows: rows} =
      Ecto.Adapters.SQL.query!(
        Atlas.Repo,
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'atlas_test' AND table_type = 'BASE TABLE'
          AND table_name <> 'schema_migrations'
        """,
        []
      )

    tables = Enum.map_join(rows, ", ", fn [name] -> ~s("atlas_test"."#{name}") end)

    # TRUNCATE is transactional in Postgres, so this is undone with the test.
    if tables != "" do
      Ecto.Adapters.SQL.query!(Atlas.Repo, "TRUNCATE #{tables} RESTART IDENTITY CASCADE", [])
    end

    :ok
  end

  @doc """
  A helper that transforms changeset errors into a map of messages.

      assert {:error, changeset} = Accounts.create_user(%{password: "short"})
      assert "should be at least 8 character(s)" in errors_on(changeset).password
      assert %{password: ["should be at least 8 character(s)"]} = errors_on(changeset)
  """
  def errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {message, opts} ->
      Regex.replace(~r"%{(\w+)}", message, fn _, key ->
        opts |> Keyword.get(String.to_existing_atom(key), key) |> to_string()
      end)
    end)
  end
end
