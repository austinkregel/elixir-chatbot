defmodule ChatWeb.Admin.KnowledgeReviewLiveTest do
  alias Brain.Knowledge.Types
  alias Brain.Knowledge
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  alias Knowledge.{ReviewQueue, SourceReliability, LearningCenter}
  alias Types.{Finding, SourceInfo, ReviewCandidate}

  setup do
    ensure_pubsub_started()

    case Task.Supervisor.start_link(name: Brain.Knowledge.AgentSupervisor) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
    end

    ensure_started(SourceReliability)
    ensure_started(ReviewQueue)
    ensure_started(LearningCenter)
    ReviewQueue.clear()

    :ok
  end

  describe "mount" do
    test "renders empty state when no candidates", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "Knowledge Review Queue"
      assert html =~ "No pending candidates"
    end

    test "renders pending candidates", %{conn: conn} do
      candidate = build_test_candidate("France", "Paris is the capital of France")
      ReviewQueue.add(candidate)

      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "Paris is the capital of France"
      assert html =~ "France"
    end

    test "displays stats bar", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "Pending"
      assert html =~ "Approved Today"
      assert html =~ "Rejected Today"
    end
  end

  describe "approve action" do
    test "approving removes candidate from list", %{conn: conn} do
      candidate = build_test_candidate("Test", "Test claim")
      ReviewQueue.add(candidate)

      {:ok, view, _html} = live(conn, "/knowledge-review")
      view |> element("button[phx-click=approve]") |> render_click()
      html = render(view)
      refute html =~ "Test claim"
    end

    test "approving updates stats", %{conn: conn} do
      candidate = build_test_candidate("Test", "Test claim")
      ReviewQueue.add(candidate)

      {:ok, view, _html} = live(conn, "/knowledge-review")

      view |> element("button[phx-click=approve]") |> render_click()
      stats = ReviewQueue.stats()
      assert stats.approved_today >= 1
    end
  end

  describe "reject action" do
    test "rejecting removes candidate from list", %{conn: conn} do
      candidate = build_test_candidate("Test", "Reject this claim")
      ReviewQueue.add(candidate)

      {:ok, view, _html} = live(conn, "/knowledge-review")

      view |> element("button[phx-click=reject]") |> render_click()

      html = render(view)
      refute html =~ "Reject this claim"
    end
  end

  describe "defer action" do
    test "deferring removes candidate from pending list", %{conn: conn} do
      candidate = build_test_candidate("Test", "Defer this claim")
      ReviewQueue.add(candidate)

      {:ok, view, _html} = live(conn, "/knowledge-review")

      view |> element("button[phx-click=defer]") |> render_click()

      html = render(view)
      refute html =~ "Defer this claim"
    end
  end

  describe "bulk actions" do
    test "bulk approve button is disabled when nothing selected", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "Approve Selected (0)"
      assert html =~ "disabled"
    end

    test "selecting candidates enables bulk actions", %{conn: conn} do
      candidate = build_test_candidate("Test", "Bulk test claim")
      ReviewQueue.add(candidate)

      {:ok, view, _html} = live(conn, "/knowledge-review")
      view |> element("input[phx-click=toggle_select]") |> render_click()

      html = render(view)
      assert html =~ "Approve Selected (1)"
    end

    test "bulk approve processes multiple candidates", %{conn: _conn} do
      candidates =
        for i <- 1..3 do
          c = build_test_candidate("Entity#{i}", "Claim #{i}")
          ReviewQueue.add(c)
          c
        end

      ids = Enum.map(candidates, & &1.id)
      {:ok, count} = ReviewQueue.bulk_approve(ids)

      assert count == 3
      pending = ReviewQueue.get_pending()
      assert pending == []
    end
  end

  describe "start session modal" do
    test "clicking start session shows modal", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/knowledge-review")

      view |> element("button", "Start Learning Session") |> render_click()

      html = render(view)
      assert html =~ "Topic to research"
    end

    test "starting session with topic creates session", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/knowledge-review")
      view |> element("button", "Start Learning Session") |> render_click()

      view
      |> form("form[phx-submit=start_session]", %{topic: "European capitals"})
      |> render_submit()

      sessions = LearningCenter.list_sessions()
      assert Enum.any?(sessions, &(&1.topic == "European capitals"))
    end
  end

  describe "source display" do
    test "shows source reliability badge", %{conn: conn} do
      source =
        SourceInfo.new("https://wikipedia.org/wiki/Test",
          reliability_score: 0.85,
          trust_tier: :verified,
          bias_rating: :center
        )

      finding = Finding.new("Test claim", "Test Entity", source)
      candidate = ReviewCandidate.new(finding, aggregate_confidence: 0.8)
      ReviewQueue.add(candidate)

      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "wikipedia.org"
      assert html =~ "85"
    end

    test "shows corroboration count", %{conn: conn} do
      source1 = SourceInfo.new("https://source1.com/article")
      source2 = SourceInfo.new("https://source2.com/article")

      finding = Finding.new("Corroborated claim", "Entity", source1)

      candidate =
        ReviewCandidate.new(finding, corroborating_sources: [source2], aggregate_confidence: 0.9)

      ReviewQueue.add(candidate)

      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "Sources:"
      assert html =~ "2"
    end

    test "shows contradiction warning", %{conn: conn} do
      source = SourceInfo.new("https://example.com/article")
      finding = Finding.new("Conflicting claim", "Entity", source)

      candidate =
        ReviewCandidate.new(finding,
          existing_contradictions: [%{object: "Existing belief about Entity"}]
        )

      ReviewQueue.add(candidate)

      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "Contradicts existing belief"
    end
  end

  describe "confidence display" do
    test "shows confidence as progress bar", %{conn: conn} do
      candidate = build_test_candidate("Test", "High confidence claim", confidence: 0.95)
      ReviewQueue.add(candidate)

      {:ok, _view, html} = live(conn, "/knowledge-review")

      assert html =~ "95"
      assert html =~ "progress"
    end
  end

  defp build_test_candidate(entity, claim, opts \\ []) do
    confidence = Keyword.get(opts, :confidence, 0.7)
    domain = Keyword.get(opts, :domain, "test-source.com")

    source =
      SourceInfo.new("https://#{domain}/article", reliability_score: 0.8, trust_tier: :verified)

    finding = Finding.new(claim, entity, source, entity_type: "location", confidence: confidence)

    ReviewCandidate.new(finding, aggregate_confidence: confidence)
  end
end