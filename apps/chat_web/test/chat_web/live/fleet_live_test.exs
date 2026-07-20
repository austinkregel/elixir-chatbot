defmodule ChatWeb.Admin.FleetLiveTest do
  use ChatWeb.ConnCase
  import Phoenix.LiveViewTest

  test "the fleet console mounts (connected) and renders", %{conn: conn} do
    {:ok, view, html} = live(conn, "/fleet")
    assert html =~ "Chain of command"
    # Commissioning is a deliberate act behind a focused dialog, not an
    # always-open form crowding the bridge.
    assert html =~ "Commission officer"
    assert html =~ "phx-click=\"open_commission\""
    refute html =~ "phx-submit=\"commission\""

    # Opening the dialog surfaces the commission form. It confers a billet, and
    # the billet selector lists the command postings (their standing authorities
    # are derived, not hand-checked).
    html = render_click(view, "open_commission")
    assert html =~ "phx-submit=\"commission\""
    assert html =~ ~s(name="rank")
    assert html =~ "First Officer (XO)"
    assert html =~ "Security Officer"
  end
end
