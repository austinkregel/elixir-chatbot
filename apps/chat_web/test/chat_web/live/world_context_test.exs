defmodule ChatWeb.WorldContextTest do
  @moduledoc """
  The world-context on_mount hook, exercised against the Fleet console (the only
  LiveView left in this Fleet-management workbench). The hook supplies the world
  context assigns every page mount depends on; per-world data isolation is covered
  directly at the fleet layer (`Fleet.MindWorldIsolationTest`).
  """
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  alias World.Manager, as: WorldManager

  setup do
    ensure_pubsub_started()
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(World.Manager)
    ensure_started(World.ModelRegistry)
    :ok
  end

  test "the console mounts with the world context the hook supplies", %{conn: conn} do
    # app_shell requires @current_world_id; a successful mount proves the on_mount
    # hook populated the world-context assigns.
    {:ok, _view, html} = live(conn, "/fleet")
    assert html =~ "Fleet"
  end

  test "the world manager lists the default world", %{conn: _conn} do
    worlds = WorldManager.list_worlds()
    assert is_list(worlds)
    assert Enum.any?(worlds, fn w -> w.id == "default" end)
  end
end
