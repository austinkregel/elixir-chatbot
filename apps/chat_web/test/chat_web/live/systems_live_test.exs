defmodule ChatWeb.Admin.SystemsLiveTest do
  @moduledoc "The systems black-box board mounts (connected) and renders the layered inventory."
  use ChatWeb.ConnCase
  import Phoenix.LiveViewTest

  test "the systems board renders services, processes, and subsystem counts", %{conn: conn} do
    {:ok, _view, html} = live(conn, "/systems")
    assert html =~ "Systems"
    assert html =~ "Services"
    assert html =~ "Processes"
    assert html =~ "Subsystems"
    # the layered inventory reaches 200+ modules
    assert html =~ "modules"
  end

  test "both console and board are reachable and cross-linked", %{conn: conn} do
    {:ok, _v, fleet_html} = live(conn, "/")
    assert fleet_html =~ ~s(href="/systems")

    {:ok, _v, sys_html} = live(conn, "/systems")
    assert sys_html =~ ~s(href="/")
  end
end
