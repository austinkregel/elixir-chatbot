defmodule Mix.Tasks.Atlas.MoveGraphsToWorld do
  use Mix.Task

  @shortdoc "Move the unscoped knowledge/user/semantic/conversation/epistemic graphs into a world"
  @moduledoc """
  Graphs of the per-world kinds used to be one graph each, shared by every
  world: `knowledge_graph`, `user_graph`, `semantic_graph`,
  `conversation_graph`, `epistemic_graph`. Each world now has its own
  (`Atlas.Graph.world_graph/2`). This task gives the data in those unscoped
  graphs to one world, explicitly:

      mix atlas.move_graphs_to_world --world default

  Each unscoped graph is renamed with AGE's `alter_graph(..., 'RENAME', ...)`
  to the world's graph of its kind, so `knowledge_graph` becomes
  `knowledge_graph__default`. Nothing is copied or deleted; the graph keeps
  its vertices, edges and ids under its new name. `pos_graph` is shared by
  every world and is not touched.

  A kind with no unscoped graph is skipped. If the world already has a graph
  of a kind that also has an unscoped one, the task raises before renaming
  anything: both hold data, and a rename cannot merge them.

  Run it before `mix atlas.bootstrap_age`, which refuses while unscoped graphs
  exist.
  """

  @impl Mix.Task
  def run(args) do
    {opts, _rest, invalid} = OptionParser.parse(args, strict: [world: :string])
    if invalid != [], do: Mix.raise("atlas.move_graphs_to_world: unknown options #{inspect(invalid)}")

    world_id =
      Keyword.get(opts, :world) ||
        Mix.raise("atlas.move_graphs_to_world needs --world <id>, the world the unscoped graphs belong to")

    Mix.Task.run("app.config")
    {:ok, _} = Application.ensure_all_started(:atlas)

    Enum.each(Atlas.Graph.move_unscoped_graphs_to_world!(world_id), fn
      {kind, :moved} ->
        Mix.shell().info("  #{kind} -> #{Atlas.Graph.world_graph(kind, world_id)}")

      {kind, :no_unscoped_graph} ->
        Mix.shell().info("  #{kind}: no unscoped graph, nothing to move")
    end)

    Mix.shell().info("atlas.move_graphs_to_world: done for world #{inspect(world_id)}")
  end
end
