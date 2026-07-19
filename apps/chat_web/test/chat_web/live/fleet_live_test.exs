defmodule ChatWeb.Admin.FleetLiveTest do
  use ChatWeb.ConnCase
  import Phoenix.LiveViewTest

  test "the fleet console mounts (connected) and renders", %{conn: conn} do
    {:ok, _view, html} = live(conn, "/fleet")
    assert html =~ "Chain of command"
    assert html =~ "Commission"
    assert html =~ "phx-submit=\"commission\""
    # The commission form confers a billet, and the billet selector lists the
    # command postings (their standing authorities are derived, not hand-checked).
    assert html =~ ~s(name="rank")
    assert html =~ "First Officer (XO)"
    assert html =~ "Security Officer"
  end
end
