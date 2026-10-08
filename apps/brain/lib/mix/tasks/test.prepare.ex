defmodule Mix.Tasks.Test.Prepare do
  @shortdoc "Prepare the test database: Apache AGE, the atlas_test schema, migrations and the seeded lexicon"
  @moduledoc """
  Everything the test suites need in the database, written once, deliberately,
  outside any test.

      MIX_ENV=test mix test.prepare

  `mix test` from the umbrella root runs this first, so it is rarely invoked
  by hand. It is idempotent.

  1. Bootstraps Apache AGE and creates the `atlas_test` schema.
  2. Runs the Atlas migrations with that prefix.
  3. Seeds the brain's lexicon (WordNet/SemCor facts and the closed class),
     which every suite reads and no suite should have to write.

  The suites never commit: `Atlas.Repo`'s pool stays in the Sandbox's
  `:manual` mode, so each test runs in a transaction that is rolled back, and
  a write from a process that owns no connection raises rather than
  persisting. This task is the one place that writes for real, so it starts
  the repo with an ordinary pool instead of putting the Sandbox into `:auto`.
  """

  use Mix.Task

  @requirements ["app.config"]

  @impl Mix.Task
  def run(args) do
    {_opts, _, invalid} = OptionParser.parse(args, strict: [])
    if invalid != [], do: Mix.raise("test.prepare: unknown options #{inspect(invalid)}")

    unless Mix.env() == :test do
      Mix.raise("test.prepare prepares the test database; run it with MIX_ENV=test")
    end

    sandbox_config = Application.fetch_env!(:atlas, Atlas.Repo)

    try do
      use_ordinary_pool!(sandbox_config)

      {:ok, _} = Application.ensure_all_started(:atlas)

      Mix.Task.run("atlas.bootstrap_age")
      Atlas.Repo.query!("CREATE SCHEMA IF NOT EXISTS atlas_test", [])

      :atlas
      |> Application.app_dir("priv/repo/migrations")
      |> then(&Ecto.Migrator.run(Atlas.Repo, &1, :up, all: true, prefix: "atlas_test"))

      # The prepared database is exactly this: the migrated schema and the
      # seeded lexicon. Everything else starts empty, so a run never inherits
      # rows an earlier one left behind, and the seed is what today's rules
      # produce rather than the union of every seed ever run.
      empty_tables!()
      empty_graphs!()

      # The seeder writes through the lexicon store, so the brain has to be up.
      {:ok, _} = Application.ensure_all_started(:brain)

      {:ok, counts} = Brain.Lexicon.Seeder.seed_all()
      {:ok, loaded} = Brain.Lexicon.UserDefined.reload()

      Mix.shell().info(
        "test.prepare: atlas_test migrated; lexicon seeded " <>
          "(#{counts.negation} negation facts, #{Brain.Test.Database.count_lexicon_facts()} rows, #{loaded} loaded)"
      )
    after
      # `mix test` runs in this same VM after the root alias calls this task,
      # and its test_helper must start the repo itself, with the Sandbox pool.
      # Leave nothing of the ordinary pool behind.
      Application.stop(:brain)
      Application.stop(:atlas)
      Application.put_env(:atlas, Atlas.Repo, sandbox_config)
    end
  end

  # Every table of the test schema but the migration log. `schema_migrations`
  # stays: the migrations were just run and the log records that.
  defp empty_tables! do
    %{rows: rows} =
      Atlas.Repo.query!(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'atlas_test' AND table_type = 'BASE TABLE'
          AND table_name <> 'schema_migrations'
        """,
        []
      )

    case Enum.map(rows, fn [name] -> ~s("atlas_test"."#{name}") end) do
      [] -> :ok
      tables -> Atlas.Repo.query!("TRUNCATE #{Enum.join(tables, ", ")} RESTART IDENTITY CASCADE", [])
    end

    Mix.shell().info("test.prepare: emptied #{length(rows)} tables")
  end

  # AGE keeps each graph in its own schema; dropping and recreating is how a
  # graph is emptied. The vertices and edges the pipeline wrote during earlier
  # runs live here.
  defp empty_graphs! do
    %{rows: rows} = Atlas.Repo.query!("SELECT name::text FROM ag_catalog.ag_graph", [])

    for [name] <- rows do
      Atlas.Repo.query!("SELECT ag_catalog.drop_graph($1, true)", [name])
      Atlas.Repo.query!("SELECT ag_catalog.create_graph($1)", [name])
    end

    Mix.shell().info("test.prepare: emptied #{length(rows)} graphs")
  end

  # The Sandbox pool the test config declares refuses checkouts without an
  # owner, and this task writes for real.
  defp use_ordinary_pool!(sandbox_config) do
    Application.put_env(:atlas, Atlas.Repo, Keyword.put(sandbox_config, :pool, DBConnection.ConnectionPool))
  end
end
