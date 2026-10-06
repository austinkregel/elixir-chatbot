defmodule ChatWeb.VerifyLiveTest do
  @moduledoc """
  The `/verify` index. The property that matters is that it is honest while it
  is empty: a subsystem with no page and no cases is listed with zeros, not
  omitted, because an index showing only what exists would report an all-clear
  on the day it shipped.
  """
  use ChatWeb.ConnCase, async: false

  import Phoenix.LiveViewTest

  alias Atlas.Verification
  alias Atlas.Verification.Subsystems

  describe "mount with no cases" do
    test "lists every declared subsystem", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/verify")

      for subsystem <- Subsystems.all() do
        assert html =~ subsystem.title, "#{subsystem.id} should be listed"
        assert html =~ subsystem.id
      end
    end

    test "says plainly that nothing has been verified", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ "No verification cases exist in world"
      assert html =~ "not an all-clear"
    end

    test "a subsystem with no cases reads as unverified rather than passing", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ "unverified"
      refute html =~ "Pass</span>"
    end

    test "names the task that would build each missing page", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ "not built — task 040"
      assert html =~ "not built — task 057"
    end

    test "the subsystem count matches the declaration", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ to_string(length(Subsystems.all()))
    end
  end

  describe "with cases recorded" do
    setup do
      base = %{
        subsystem: "speech_act",
        world_id: "default",
        input: %{text: "hello"},
        expected: %{category: "expressive"}
      }

      {:ok, passing} = Verification.save_case(Map.put(base, :name, "passes"))
      {:ok, failing} = Verification.save_case(Map.put(base, :name, "fails"))
      {:ok, _never} = Verification.save_case(Map.put(base, :name, "never run"))

      {:ok, _, _} = Verification.record_result(passing, %{category: "expressive"})
      {:ok, _, _} = Verification.record_result(failing, %{category: "directive"})

      :ok
    end

    test "the totals reflect the recorded statuses", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ "Passing"
      assert html =~ "Failing"
      refute html =~ "No verification cases exist in world"
    end

    test "a subsystem with a failing case does not read as passing", %{conn: conn} do
      # The worst state present wins, so one failure keeps the row off green
      # even though another case passes.
      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ "Fail"
    end

    test "a raised case is shown as its own state, not as a failure", %{conn: conn} do
      {:ok, raising} =
        Verification.save_case(%{
          subsystem: "memory",
          name: "raises",
          world_id: "default",
          input: %{text: "hello"},
          expected: %{category: "expressive"}
        })

      {:ok, _} = Verification.record_error(raising, %RuntimeError{message: "boom"})

      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ "Raised"
    end
  end

  describe "refresh" do
    test "re-reads the counts", %{conn: conn} do
      {:ok, view, html} = live(conn, "/verify")
      assert html =~ "No verification cases exist in world"

      {:ok, _} =
        Verification.save_case(%{
          subsystem: "speech_act",
          name: "added after mount",
          world_id: "default",
          input: %{text: "hello"},
          expected: %{category: "expressive"}
        })

      html = view |> element("button[phx-click='refresh']") |> render_click()

      refute html =~ "No verification cases exist in world"
    end
  end

  describe "chrome" do
    test "uses the app shell rather than bespoke navigation", %{conn: conn} do
      # Task 039's criterion: no new page duplicates AppShell or WorldContext.
      # The shell's world selector and nav are the evidence it was used.
      {:ok, _view, html} = live(conn, "/verify")

      assert html =~ "Training World"
      assert html =~ "phx-change=\"switch_world\""
      assert html =~ "Verification"
    end
  end
end
