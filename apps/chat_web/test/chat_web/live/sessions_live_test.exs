defmodule ChatWeb.SessionsLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.KnowledgeStore)
    :ok
  end

  describe "mount" do
    test "mounts successfully with list view", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/sessions")
      assert html =~ "Session" or html =~ "Learning" or html =~ "Research"
    end

    test "displays session list page", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/sessions")
      assert is_binary(html)
      assert String.length(html) > 100
    end
  end

  describe "refresh" do
    test "handles periodic refresh", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/sessions")
      send(view.pid, :refresh)
      Process.sleep(50)
      html = render(view)
      assert is_binary(html)
    end
  end
end
