defmodule ChatWeb.Admin.FleetLiveTest do
  use ChatWeb.ConnCase
  import Phoenix.LiveViewTest

  test "the fleet console mounts (connected) and renders", %{conn: conn} do
    {:ok, _view, html} = live(conn, "/fleet")
    assert html =~ "Crew roster"
    assert html =~ "Commission"
    assert html =~ "phx-submit=\"commission\""
  end
end
