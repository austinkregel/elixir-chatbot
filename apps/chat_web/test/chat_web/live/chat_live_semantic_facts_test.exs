defmodule ChatWeb.ChatLiveSemanticFactsTest do
  @moduledoc """
  The inspector's memory tab shows the semantic facts retrieved for a message,
  which of them was the turn's recurring pattern, and who used it.
  """
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers, only: [start_brain_services: 0]

  alias ChatWeb.ChatLive

  @pattern_id "6f1c2a9e-3b7d-4c1e-9a52-1d0e8f7b6a43"
  @stated_id "0b8e4d21-7c5a-4f3e-8d19-5e6a2c4b9f10"

  setup do
    start_brain_services()
    :ok
  end

  defp progress(message_id, step, data) do
    Map.merge(data, %{step: step, message_id: message_id, conversation_id: "conv_semantic_facts"})
  end

  defp retrieved(message_id) do
    progress(message_id, :semantic_facts_retrieved, %{
      world_id: "default",
      recurring_pattern_id: @pattern_id,
      semantic_facts: [
        %{
          semantic_fact_id: @pattern_id,
          similarity: 0.82,
          text: "is it sunny in paris | is it sunny in london",
          action: "conversation",
          evidence_count: 3
        },
        %{semantic_fact_id: @stated_id, similarity: 0.41, text: "paris weather", action: nil, evidence_count: 0}
      ]
    })
  end

  defp used(message_id, used_by) do
    progress(message_id, :semantic_fact_used, %{
      world_id: "default",
      semantic_fact_id: @pattern_id,
      used_by: used_by,
      similarity: 0.82,
      reply: "It's usually sunny there.",
      reply_count: 2,
      evidence_count: 3
    })
  end

  defp open_memory_tab(view, message_id) do
    render_click(view, "select_message", %{"message_id" => message_id})
    render_click(view, "set_dev_tab", %{"message_id" => message_id, "tab" => "memory"})
  end

  test "lists the retrieved facts, marks the recurring pattern, and names the use the turn reported", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_semantic_facts"

    send(view.pid, {:analysis_progress, retrieved(message_id)})
    # A turn reports its response's use of the pattern once.
    send(view.pid, {:analysis_progress, used(message_id, :synthesizer)})
    # The episode query arriving after the facts keeps them.
    send(view.pid, {:analysis_progress, progress(message_id, :memory_query, %{match_count: 0, top_similarity: 0.0, matches: []})})

    open_memory_tab(view, message_id)

    pattern = view |> element("#semantic-fact-#{@pattern_id}") |> render()
    assert pattern =~ "is it sunny in paris | is it sunny in london"
    assert pattern =~ "action: conversation"
    assert pattern =~ "3 evidence episode(s)"
    assert pattern =~ "recurring pattern"
    assert pattern =~ "It&#39;s usually sunny there."
    assert length(String.split(pattern, "used by synthesizer")) == 2
    refute pattern =~ "used by realization packet"

    stated = view |> element("#semantic-fact-#{@stated_id}") |> render()
    assert stated =~ "paris weather"
    refute stated =~ "recurring pattern"
    refute stated =~ "used by"
  end

  test "lists every use reported, without merging them", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_semantic_facts_reported_twice"

    send(view.pid, {:analysis_progress, retrieved(message_id)})
    send(view.pid, {:analysis_progress, used(message_id, :realization_packet)})
    send(view.pid, {:analysis_progress, used(message_id, :realization_packet)})

    open_memory_tab(view, message_id)

    pattern = view |> element("#semantic-fact-#{@pattern_id}") |> render()
    assert length(String.split(pattern, "used by realization packet")) == 3
  end

  test "says so when no semantic facts were retrieved", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_no_semantic_facts"

    send(view.pid, {:analysis_progress, progress(message_id, :semantic_facts_retrieved, %{world_id: "default", recurring_pattern_id: nil, semantic_facts: []})})

    open_memory_tab(view, message_id)

    assert view |> element("#memory-semantic-facts") |> render() =~ "No semantic facts retrieved"
  end

  describe "the progress steps" do
    test "describe the facts retrieved and the pattern's use" do
      assert ChatLive.stage_detail(retrieved("m")) == "2 semantic fact(s), recurring pattern found"
      assert ChatLive.stage_detail(used("m", :realization_packet)) == "recurring pattern used by realization packet"
    end
  end
end
