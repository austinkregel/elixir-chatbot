defmodule ChatWeb.POSTrainingLiveTest do
  @moduledoc """
  The POS training page: recorded runs are listed and plotted, and the form
  refuses what is not a run. Starting and cancelling runs is covered at the
  training server (`Brain.ML.TrainingServerTest`).
  """
  use ChatWeb.ConnCase, async: false

  import Phoenix.LiveViewTest

  alias Brain.Training.POSRuns

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    ml = Application.fetch_env!(:brain, :ml)
    Application.put_env(:brain, :ml, Keyword.put(ml, :training_runs_path, tmp_dir))
    on_exit(fn -> Application.put_env(:brain, :ml, ml) end)
    :ok
  end

  test "with no runs, says so", %{conn: conn} do
    {:ok, _view, html} = live(conn, "/training/pos")

    assert html =~ "POS Training"
    assert html =~ "No runs yet."
  end

  test "lists a recorded run and plots its curve", %{conn: conn} do
    run = POSRuns.run!(%{sentences: 40, max_epochs: 2, patience: nil, seed: 5}, id: "page-run")

    {:ok, view, html} = live(conn, "/training/pos")

    assert html =~ "page-run"
    assert html =~ "40 sentences, 2 epochs, no early stop, seed 5"
    assert html =~ "<polyline"
    assert has_element?(view, "#run-page-run option[value='epoch-2']")
    assert html =~ "#{Float.round(run["best"]["dev_accuracy"] * 100, 2)}% @ #{run["best"]["epoch"]}"

    html = view |> element("#run-page-run input[type=checkbox]") |> render_click()
    assert html =~ "Pick a run with at least two epochs to plot."
  end

  test "refuses a form that is not a run", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/training/pos")

    html = view |> form("#pos-run-form", %{"run" => %{"max_epochs" => "lots"}}) |> render_submit()
    assert html =~ "max epochs must be a whole number"

    html = view |> form("#pos-run-form", %{"run" => %{"max_epochs" => "0"}}) |> render_submit()
    assert html =~ "max_epochs is missing or invalid"
  end
end
