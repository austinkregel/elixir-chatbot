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
          "Execute confirmation",
          "Verdict",
          "Empty states",
          "Regression gate",
          "Pagination",
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

  test "the new components render in both themes", %{conn: conn} do
    {:ok, view, html} = live(conn, "/design")

    for kind <- ["observed_clean", "not_observable", "not_instrumented", "could_not_ask"] do
      assert html =~ ~s(data-empty="#{kind}")
    end

    for reach <- ["local", "shared", "device"] do
      assert html =~ ~s(data-reach="#{reach}")
    end

    for kind <- ["completeness", "belief_confidence", "match_confidence", "reranked_confidence", "mapped_confidence", "cosine_similarity", "distance", "unestablished"] do
      assert html =~ ~s(data-score="#{kind}")
    end

    assert html =~ "no confidence · intent inferred from speech act"
    assert html =~ "data-spinner"
    assert html =~ ~s(aria-current="page")
    assert html =~ "Showing 51–100 of 1,204"
    assert html =~ "data-coverage"
    assert html =~ ~s(data-origin="unavailable")
    refute html =~ "outline-reach-device"

    section = view |> element("section", "Empty states") |> render()
    assert section =~ ~s(data-theme="light")
    assert section =~ ~s(data-theme="dark")
  end

  test "the plain empty panel shows its own words with no mark", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    for theme <- ["light", "dark"] do
      plain = view |> element(~s([data-theme="#{theme}"] [data-empty="plain"])) |> render()

      assert plain =~ "No saved cases for this subsystem yet"
      refute plain =~ "data-mark"
    end
  end

  test "the worded not-run verdict reads as pending with its own words", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    section = view |> element("section#verdict") |> render()

    assert section =~ ~r/data-verdict="pending"[^>]*>(?:(?!<\/span>).)*gate not set/s
  end

  test "the regression gate renders each of its states in both themes", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    for theme <- ["light", "dark"] do
      not_set = view |> element("#design-gate-not_set-#{theme}") |> render()
      assert not_set =~ ~s(data-gate="not_set")
      assert not_set =~ ~s(data-verdict="pending")
      assert not_set =~ "gate not set"

      pass = view |> element("#design-gate-pass-#{theme}") |> render()
      assert pass =~ ~s(data-gate="pass")
      assert pass =~ ~s(data-verdict="pass")
      assert pass =~ "−0.4 pts against baseline · allows 2"

      fail = view |> element("#design-gate-fail-#{theme}") |> render()
      assert fail =~ ~s(data-gate="fail")
      assert fail =~ ~s(data-verdict="fail")
      assert fail =~ "−4.8 pts against baseline · allows 2"
      assert fail =~ "5 new unknown, errored or not-loaded predictions"

      unmeasured = view |> element("#design-gate-canary_not_measured-#{theme}") |> render()
      assert unmeasured =~ ~s(data-gate="pass")
      assert unmeasured =~ ~s(data-gate-canary="not_measured")
      assert unmeasured =~ "canary not measured"
      refute unmeasured =~ "new unknown, errored or not-loaded predictions"
    end
  end

  test "a stat tile carries the gate's verdict in its verdict slot", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    for theme <- ["light", "dark"] do
      assert has_element?(view, ~s(#design-kpi-gate-pass-#{theme} [data-kpi-verdict] [data-gate="pass"]))
      assert has_element?(view, ~s(#design-kpi-gate-not-set-#{theme} [data-kpi-verdict] [data-gate="not_set"]))
    end
  end

  test "pagination with one page disables Prev and Next and shows the one page", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    bar = ~s(nav[aria-label="One page, light"])

    assert has_element?(view, ~s(#{bar} button[aria-current="page"][phx-value-page="1"]))
    assert has_element?(view, ~s(#{bar} button[disabled]), "Prev")
    assert has_element?(view, ~s(#{bar} button[disabled]), "Next")
    refute has_element?(view, ~s(#{bar} [data-page-gap]))
    assert render(view) =~ "Showing 1–12 of 12"
  end

  test "pagination with a few pages shows every page and no gap", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    bar = ~s(nav[aria-label="A few pages, light"])

    for page <- 1..4 do
      assert has_element?(view, ~s(#{bar} button[phx-value-page="#{page}"]))
    end

    refute has_element?(view, ~s(#{bar} [data-page-gap]))
    assert has_element?(view, ~s(#{bar} button[aria-current="page"][phx-value-page="2"]))
  end

  test "pagination with many pages windows the numbers with gaps", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    bar = ~s(nav[aria-label="Many pages, with gaps, light"])
    numbers = view |> element(bar) |> render()

    for page <- [1, 29, 30, 31, 61] do
      assert numbers =~ ~s(phx-value-page="#{page}")
    end

    refute numbers =~ ~s(phx-value-page="2")
    assert length(String.split(numbers, "data-page-gap")) - 1 == 2
    assert has_element?(view, ~s(#{bar} button[aria-current="page"][phx-value-page="30"]))

    view |> element(~s(#{bar} button[phx-value-page="61"]), "61") |> render_click()

    assert has_element?(view, ~s(#{bar} button[aria-current="page"][phx-value-page="61"]))
    assert has_element?(view, ~s(#{bar} button[disabled]), "Next")
    assert length(String.split(view |> element(bar) |> render(), "data-page-gap")) - 1 == 1
  end

  test "an execute confirmation opens in place, and Cancel closes it", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    refute has_element?(view, "#design-confirm-shared-light")

    html = view |> element("#design-trigger-shared-light") |> render_click()
    assert has_element?(view, "#design-confirm-shared-light[role=dialog]")
    assert html =~ "reach-ring border-reach-shared"
    refute has_element?(view, "#design-confirm-shared-dark")

    view |> element("#design-confirm-shared-light-cancel") |> render_click()
    refute has_element?(view, "#design-confirm-shared-light")
  end

  test "confirming a shared write closes the panel and says nothing was written", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    view |> element("#design-trigger-shared-dark") |> render_click()
    html = view |> element("#design-confirm-shared-dark-confirm") |> render_click()

    refute has_element?(view, "#design-confirm-shared-dark")
    assert html =~ "The sample wrote nothing."
  end

  test "a failed device confirmation stays open and shows the failure", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    view |> element("#design-trigger-device-light") |> render_click()
    html = view |> element("#design-confirm-device-light-confirm") |> render_click()

    assert has_element?(view, "#design-confirm-device-light")
    assert html =~ "Nothing was sent to light.kitchen"
  end

  test "the form confirmation submits its fields", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    view |> element("#design-trigger-form-light") |> render_click()

    html =
      view
      |> form("#design-confirm-form-light-form", %{"topic" => "astronomy"})
      |> render_submit()

    assert html =~ "Confirmed design-confirm-form-light with topic &quot;astronomy&quot;"
    refute has_element?(view, "#design-confirm-form-light")
  end

  test "?confirm=open opens every confirmation", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design?confirm=open")

    for kind <- ["shared", "device", "local", "form"], theme <- ["light", "dark"] do
      assert has_element?(view, "#design-confirm-#{kind}-#{theme}[role=dialog]")
    end
  end

  test "the sample pagination moves the selected segment", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    assert has_element?(view, ~s(nav[aria-label="Sample pages, light"] button[aria-current="page"]), "2")

    html =
      view
      |> element(~s(nav[aria-label="Sample pages, light"] button[phx-value-page="3"]), "3")
      |> render_click()

    assert html =~ "Showing 101–150 of 1,204"
    assert has_element?(view, ~s(nav[aria-label="Sample pages, light"] button[aria-current="page"]), "3")
    refute has_element?(view, ~s(nav[aria-label="Sample pages, light"] button[aria-current="page"]), "2")
  end

  test "handles a world change without crashing", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/design")

    send(view.pid, {:world_context_changed, "default"})

    assert render(view) =~ "Design Language"
  end
end
