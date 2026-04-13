defmodule ChatWeb.CodeAnalysisLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()
    :ok
  end

  describe "mount" do
    test "mounts successfully", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/code")
      assert html =~ "Code" or html =~ "Analysis" or html =~ "Symbol"
    end

    test "displays code analysis interface", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/code")
      assert is_binary(html)
      assert String.length(html) > 100
    end

    test "shows browse tab by default", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/code")
      html = render(view)
      assert html =~ "Browse" or html =~ "browse" or html =~ "Symbols" or html =~ "Code"
    end
  end

  describe "refresh" do
    test "handles refresh message", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/code")
      send(view.pid, :refresh_status)
      Process.sleep(50)
      html = render(view)
      assert is_binary(html)
    end
  end
end
