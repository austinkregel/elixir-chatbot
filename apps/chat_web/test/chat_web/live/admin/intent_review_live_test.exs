defmodule ChatWeb.Admin.IntentReviewLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()
    :ok
  end

  describe "mount" do
    test "mounts successfully", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/intent-review")
      assert html =~ "Intent" or html =~ "Review" or html =~ "Candidates"
    end

    test "displays intent review interface", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/intent-review")
      assert is_binary(html)
      assert String.length(html) > 100
    end
  end
end
