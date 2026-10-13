defmodule ChatWeb.ChatLiveUnresolvedSpansTest do
  @moduledoc """
  The inspector's entities tab shows, apart from a chunk's entities, the spans
  entity extraction could not decide: each span's text, the types its measured
  readings supported, every reading with its source, and why it was left
  undecided.
  """
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers, only: [start_brain_services: 0]

  alias ChatWeb.ChatLive

  setup do
    start_brain_services()
    :ok
  end

  defp progress(message_id, step, data) do
    Map.merge(data, %{step: step, message_id: message_id, conversation_id: "conv_unresolved_spans", chunk_index: 0})
  end

  # As `Brain.Analysis.Pipeline` reports a chunk's extraction: one decided
  # entity, and one span whose readings the data could not decide between.
  defp extracted(message_id, unresolved_spans) do
    progress(message_id, :entities_extracted, %{
      entity_count: 1,
      entities: [%{type: "city", value: "Paris", confidence: 0.91, source: :gazetteer}],
      unresolved_spans: unresolved_spans
    })
  end

  @prince %{
    value: "Prince",
    start_pos: 9,
    end_pos: 14,
    reason: :insufficient_evidence,
    common_type: nil,
    candidate_types: [%{type: "artist", score: 0.41}, %{type: "person", score: 0.38}],
    readings: [
      %{source: :gazetteer, entity_type: "music_artist", raw_confidence: 0.82, reliability: 0.6},
      %{source: :pos_tagger_propn, entity_type: "person", raw_confidence: 0.65, reliability: 0.59},
      %{source: :location_hint, entity_type: "emoji", raw_confidence: 0.7, reliability: nil}
    ]
  }

  defp open_entities_tab(view, message_id) do
    render_click(view, "select_message", %{"message_id" => message_id})
    render_click(view, "set_dev_tab", %{"message_id" => message_id, "tab" => "entities"})
  end

  test "shows an undecided span's text, candidate types, readings with their sources, and reason", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_unresolved_spans"

    send(view.pid, {:analysis_progress, progress(message_id, :chunk_start, %{chunk_text: "I played Prince in Paris", chunk_length: 24})})
    send(view.pid, {:analysis_progress, extracted(message_id, [@prince])})

    open_entities_tab(view, message_id)

    span = view |> element("#unresolved-span-0-9") |> render()
    assert span =~ "&quot;Prince&quot;"
    assert span =~ "reason: insufficient_evidence"
    assert span =~ "shared type: none"
    assert span =~ "artist"
    assert span =~ "person"
    assert span =~ "gazetteer"
    assert span =~ "music_artist"
    assert span =~ "pos_tagger_propn"
    assert span =~ "location_hint"
    assert span =~ "emoji"
    assert span =~ "unmeasured"
    assert span =~ "span reliability"

    # The decided entity is listed among the entities, not the unresolved spans.
    unresolved = view |> element("#unresolved-spans-0") |> render()
    refute unresolved =~ "Paris"
  end

  test "says so when no span was left undecided", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_no_unresolved_spans"

    send(view.pid, {:analysis_progress, extracted(message_id, [])})

    open_entities_tab(view, message_id)

    assert view |> element("#unresolved-spans-0") |> render() =~ "No span was left undecided"
  end

  test "a span with no measured reading lists no candidate types, and names the type its readings share", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_unmeasured_span"
    span = %{@prince | reason: :no_measured_reading, candidate_types: [], common_type: "person"}

    send(view.pid, {:analysis_progress, extracted(message_id, [span])})

    open_entities_tab(view, message_id)

    html = view |> element("#unresolved-span-0-9") |> render()
    assert html =~ "reason: no_measured_reading"
    assert html =~ "none measured"
    assert html =~ "shared type: person"
  end

  test "the extraction step counts the undecided spans beside the entities" do
    assert ChatLive.stage_detail(extracted("m", [@prince])) == "1 entity(ies), 1 unresolved span(s)"
    assert ChatLive.stage_detail(extracted("m", [])) == "1 entity(ies)"
  end
end
