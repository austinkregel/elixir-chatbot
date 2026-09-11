defmodule Atlas.DataCase do
  @moduledoc """
  Case template for tests that touch the Atlas data layer.

  Opens an Ecto SQL sandbox per test (shared when the test is not `async`) so
  every change rolls back at the end. Because Atlas runs on Apache AGE, each
  checked-out connection also loads the `age` extension and pins `search_path`
  to `atlas_test, ag_catalog, ...` so both plain tables and cypher graphs
  resolve. `errors_on/1` is provided for asserting on changeset errors.
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

    on_exit(fn -> Ecto.Adapters.SQL.Sandbox.stop_owner(pid) end)
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
