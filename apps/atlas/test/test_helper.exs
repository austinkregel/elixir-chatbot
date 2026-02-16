{:ok, _} = Application.ensure_all_started(:atlas)

# Run pending migrations for the test database
migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")

Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true)

Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)

ExUnit.start()
