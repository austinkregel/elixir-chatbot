defmodule ChatWeb.AccuracyLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()
    :ok
  end

  describe "mount" do
    test "mounts with the page heading", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/accuracy")
      assert html =~ "Accuracy Dashboard"
    end

    test "displays a tab for each task and the optimizer", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/accuracy")

      for tab <- ~w(intent ner sentiment speech_act optimizer) do
        assert has_element?(view, ~s([role="tab"][phx-value-tab="#{tab}"])), "missing the #{tab} tab"
      end
    end

    test "the intent tab is selected by default", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/accuracy")

      assert has_element?(view, ~s([role="tab"][phx-value-tab="intent"][aria-selected="true"]))
      refute has_element?(view, ~s([role="tab"][phx-value-tab="ner"][aria-selected="true"]))
      assert view |> element("#run-evaluation-reach") |> render() =~ "Intent evaluation"
    end
  end

  describe "tab switching" do
    test "clicking a task tab selects it and points Run Evaluation at that task", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/accuracy")

      view |> element(~s([phx-click="switch_tab"][phx-value-tab="ner"])) |> render_click()

      assert has_element?(view, ~s([role="tab"][phx-value-tab="ner"][aria-selected="true"]))
      assert has_element?(view, ~s([role="tab"][phx-value-tab="intent"][aria-selected="false"]))
      assert view |> element("#run-evaluation-reach") |> render() =~ "NER evaluation"
    end
  end

  describe "macro-F1 tile" do
    setup do
      Atlas.Repo.insert!(%Atlas.Schemas.EvaluationResult{
        task: "intent",
        accuracy: 0.287,
        macro_f1: 0.412,
        weighted_f1: 0.301,
        total_examples: 4870,
        per_class: %{}
      })

      :ok
    end

    test "carries the gate's verdict inside the tile", %{conn: conn} do
      {:ok, view, html} = live(conn, "/accuracy")

      verdict = view |> element("[data-kpi-verdict]") |> render()

      assert verdict =~ "data-gate="
      assert length(String.split(html, "data-gate=")) - 1 == 1,
             "the gate verdict should appear once, in the macro-F1 tile's verdict slot"

      assert html =~ "macro-F1 · 4870 examples"
    end
  end
end
