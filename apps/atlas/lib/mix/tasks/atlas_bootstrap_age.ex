defmodule Mix.Tasks.Atlas.BootstrapAge do
  use Mix.Task

  @shortdoc "Install Apache AGE extension and create required graphs"
  @moduledoc """
  Bootstraps Apache AGE in the configured Atlas database. Idempotent.

  Creates the shared, expression-level graphs (`Atlas.Graph.shared_graphs/0`)
  and the "default" world's graphs (`Atlas.Graph.world_graphs/1`). Every
  other world's graphs are created when the world is
  (`Atlas.Graph.ensure_world_graphs!/1`).

  Refuses while an unscoped graph of a per-world kind (`knowledge_graph`, ...)
  exists: that holds data from before graphs were per world. Move it into the
  "default" world first with `mix atlas.move_graphs_to_world --world default`.

  Opens a raw Postgrex connection (no `after_connect` callback) so it
  works on a fresh database where the extension does not yet exist.

  Run once after database creation in any environment whose Postgres
  was not initialized via the Docker initdb script (CI, prod managed
  Postgres, externally-provisioned databases).
  """

  @default_world "default"

  @impl Mix.Task
  def run(_args) do
    Mix.Task.run("app.config")
    {:ok, _} = Application.ensure_all_started(:postgrex)

    repo_config =
      :atlas
      |> Application.get_env(Atlas.Repo, [])
      |> Keyword.drop([:after_connect, :pool, :pool_size, :types])

    {:ok, conn} = Postgrex.start_link(repo_config)

    graphs = Atlas.Graph.shared_graphs() ++ Atlas.Graph.world_graphs(@default_world)

    try do
      Postgrex.query!(conn, "CREATE EXTENSION IF NOT EXISTS age", [])
      Postgrex.query!(conn, "LOAD 'age'", [])
      Postgrex.query!(conn, ~s(SET search_path = ag_catalog, "$user", public), [])

      refuse_unscoped_graphs!(conn)

      Enum.each(graphs, fn name ->
        unless graph_exists?(conn, name) do
          Postgrex.query!(conn, "SELECT ag_catalog.create_graph($1)", [name])
        end
      end)
    after
      GenServer.stop(conn)
    end

    Mix.shell().info("Atlas: AGE bootstrapped (extension + #{length(graphs)} graphs)")
  end

  defp refuse_unscoped_graphs!(conn) do
    case Enum.filter(Atlas.Graph.world_graph_kinds(), &graph_exists?(conn, &1)) do
      [] ->
        :ok

      unscoped ->
        Mix.raise(
          "atlas.bootstrap_age: unscoped graphs #{inspect(unscoped)} hold data from before graphs " <>
            "were per world. Move them into the default world first: " <>
            "mix atlas.move_graphs_to_world --world #{@default_world}"
        )
    end
  end

  defp graph_exists?(conn, name) do
    %{rows: [[count]]} =
      Postgrex.query!(conn, "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = $1::name", [name])

    count > 0
  end
end
