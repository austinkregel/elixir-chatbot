defmodule ChatWeb.PageControllerTest do
  use ChatWeb.ConnCase

  describe "home redirect" do
    test "GET / redirects to /chat", %{conn: conn} do
      conn = get(conn, "/")
      assert redirected_to(conn) == "/chat"
    end
  end

  describe "legacy route redirects" do
    test "GET /admin redirects to /settings", %{conn: conn} do
      conn = get(conn, "/admin")
      assert redirected_to(conn) == "/settings"
    end

    test "GET /worlds redirects to /explorer", %{conn: conn} do
      conn = get(conn, "/worlds")
      assert redirected_to(conn) == "/explorer"
    end

    test "GET /worlds/:world_id/entities redirects to /explorer", %{conn: conn} do
      conn = get(conn, "/worlds/test_world/entities")
      assert redirected_to(conn) == "/explorer"
    end
  end

  describe "ops dashboard redirect" do
    test "GET /ops/dashboard redirects to /dashboard", %{conn: conn} do
      conn = get(conn, "/ops/dashboard")
      assert redirected_to(conn) == "/dashboard"
    end
  end
end
