{:ok, _} = Application.ensure_all_started(:atlas)

# Run pending migrations for the test database
migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")

try do
  Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true)
rescue
  e in Postgrex.Error ->
    if e.postgres.code == :duplicate_table do
      :ok
    else
      reraise e, __STACKTRACE__
    end
catch
  :exit, {:shutdown, {_, _, %Postgrex.Error{postgres: %{code: :duplicate_table}}}} ->
    :ok

  :exit, reason ->
    exit(reason)
end

Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)

ExUnit.start()
