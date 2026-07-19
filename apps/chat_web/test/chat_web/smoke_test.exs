defmodule ChatWeb.SmokeTest do
  @moduledoc """
  Smoke test: the Fleet console renders without crashing at both its routes. This
  is a Fleet-management workbench — the console is the whole UI.
  """
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  @moduletag :smoke

  setup do
    ensure_pubsub_started()
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(World.Manager)
    :ok
  end

  for path <- ["/", "/fleet"] do
    test "GET #{path} renders the Fleet console", %{conn: conn} do
      {:ok, _view, html} = live(conn, unquote(path))
      assert html =~ "Fleet"
      assert html =~ "Commission"
      assert html =~ "Chain of command"
    end
  end
end
