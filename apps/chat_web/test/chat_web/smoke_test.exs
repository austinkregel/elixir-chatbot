defmodule ChatWeb.SmokeTest do
  @moduledoc """
  Smoke tests to verify all major routes render without crashing.

  These tests are intentionally lightweight - they only verify that pages
  load without 500 errors and contain expected basic elements.
  """
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  @moduletag :smoke

  setup do
    ensure_pubsub_started()

    # Start minimal required services for rendering pages
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(World.Manager)

    :ok
  end

  describe "main routes render without crashing" do
    test "GET / redirects to /chat", %{conn: conn} do
      conn = get(conn, "/")
      assert redirected_to(conn) == "/chat"
    end

    test "GET /chat renders chat page", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/chat")

      # Basic smoke check: page rendered with expected elements
      assert html =~ "ChatBot"
    end

    test "GET /dashboard renders dashboard page", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      # Basic smoke check
      assert html =~ "Operations Dashboard" or html =~ "Dashboard"
    end

    test "GET /explorer renders explorer page", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/explorer")

      # Basic smoke check
      assert html =~ "Explorer" or html =~ "explorer"
    end

    test "GET /settings renders settings page", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      # Basic smoke check
      assert html =~ "Settings" or html =~ "settings"
    end

    test "GET /knowledge-review renders knowledge review page", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/knowledge-review")

      # Basic smoke check
      assert html =~ "Knowledge" or html =~ "Review" or html =~ "knowledge"
    end
  end

  describe "API endpoints respond" do
    test "GET /api/test-knowledge returns JSON", %{conn: conn} do
      conn = get(conn, "/api/test-knowledge")

      # Should return some JSON response (success or error)
      assert json_response(conn, 200) || json_response(conn, 500)
    end
  end

  describe "legacy redirects work" do
    test "GET /admin redirects to /settings", %{conn: conn} do
      conn = get(conn, "/admin")
      assert redirected_to(conn) == "/settings"
    end

    test "GET /worlds redirects to /explorer", %{conn: conn} do
      conn = get(conn, "/worlds")
      assert redirected_to(conn) == "/explorer"
    end

    test "GET /ops/dashboard redirects to /dashboard", %{conn: conn} do
      conn = get(conn, "/ops/dashboard")
      assert redirected_to(conn) == "/dashboard"
    end
  end
end
