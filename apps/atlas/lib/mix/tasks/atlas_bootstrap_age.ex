defmodule Mix.Tasks.Atlas.BootstrapAge do
  use Mix.Task

  @shortdoc "Install Apache AGE extension and create required graphs"
  @moduledoc """
  Bootstraps Apache AGE in the configured Atlas database. Idempotent.

  Opens a raw Postgrex connection (no `after_connect` callback) so it
  works on a fresh database where the extension does not yet exist.

  Run once after database creation in any environment whose Postgres
  was not initialized via the Docker initdb script (CI, prod managed
  Postgres, externally-provisioned databases).
  """

  @graphs ~w(
    knowledge_graph
    user_graph
    semantic_graph
    conversation_graph
    epistemic_graph
    pos_graph
  )

  @impl Mix.Task
  def run(_args) do
    Mix.Task.run("app.config")
    {:ok, _} = Application.ensure_all_started(:postgrex)

    repo_config =
      :atlas
      |> Application.get_env(Atlas.Repo, [])
      |> Keyword.drop([:after_connect, :pool, :pool_size, :types])

    {:ok, conn} = Postgrex.start_link(repo_config)

    try do
      Postgrex.query!(conn, "CREATE EXTENSION IF NOT EXISTS age", [])
      Postgrex.query!(conn, "LOAD 'age'", [])
      Postgrex.query!(conn, ~s(SET search_path = ag_catalog, "$user", public), [])

      Enum.each(@graphs, fn name ->
        Postgrex.query!(
          conn,
          """
          DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = '#{name}')
            THEN PERFORM ag_catalog.create_graph('#{name}'); END IF;
          END $$;
          """,
          []
        )
      end)
    after
      GenServer.stop(conn)
    end

    Mix.shell().info(
      "Atlas: AGE bootstrapped (extension + #{length(@graphs)} graphs)"
    )
  end
end
