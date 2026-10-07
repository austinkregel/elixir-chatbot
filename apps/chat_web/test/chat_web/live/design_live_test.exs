defmodule ChatWeb.DesignLiveTest do
  @moduledoc """
  The `/design` gallery renders every token family and every shared component
  state, in a light panel and a dark panel, inside the app shell.
  """
  use ChatWeb.ConnCase, async: false

  import Phoenix.LiveViewTest

  test "renders inside the app shell with its nav item active", %{conn: conn} do
    {:ok, _view, html} = live(conn, "/design")

    assert html =~ "phx-change=\"switch_world\""
    assert html =~ ~r/<a[^>]*href="\/design"[^>]*aria-current="page"/
  end

  test "renders every section in both themes", %{conn: conn} do
    {:ok, view, html} = live(conn, "/design")

    for title <- [
          "Color tokens",
          "Type",
          "Marks",
          "Value origin",
          "What a score means",
          "Candidate versus resolved",
          "Reach of an action",
          "Verdict",
          "Availability and status",
          "Job progress",
          "Badges",
          "Buttons",
          "Tabs, toggle and inputs",
          "Alerts, flash, cards and headers",
          "Table and list",
          "Harness, rendered from sample calls"
        ] do
      assert html =~ title
    end

    sections = view |> element("section", "Color tokens") |> render()
    assert sections =~ ~s(data-theme="light")
    assert sections =~ ~s(data-theme="dark")
  end

  test "the harness samples render every origin, every verdict and a raised call", %{conn: conn} do
    {:ok, _view, html} = live(conn, "/design")

    for label <- ["your input", "declared", "fallback default", "no data — stand-in", "source unreadable"] do
      assert html =~ label
    end

    for verdict <- ["pass", "fail", "error", "pending"] do
      assert html =~ ~s(data-verdict="#{verdict}")
    end

    assert html =~ "The subsystem raised"
    assert html =~ "a sample raise, rendered as the result it would be"
  end

  test "handles a world change without crashing", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    send(view.pid, {:world_context_changed, "default"})

    assert render(view) =~ "Design Language"
  end
end
