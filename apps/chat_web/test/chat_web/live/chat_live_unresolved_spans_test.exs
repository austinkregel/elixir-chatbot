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

  # As `Brain.Analysis.Pipeline` reports pass 2's attempt on the chunk's
  # unresolved spans once the intent is known.
  defp resolution(message_id, outcomes, resolved_count) do
    progress(message_id, :unresolved_spans_resolved, %{
      intent: "music.play",
      resolved_count: resolved_count,
      outcomes: outcomes
    })
  end

  @resolved %{
    value: "Prince",
    start_pos: 9,
    end_pos: 14,
    outcome: :resolved,
    entity_type: "music_artist",
    score: 1.9,
    margin: 0.8,
    evidence: [%{type: "artist", log_ratio: 1.4}, %{type: "person", log_ratio: -0.6}]
  }

  test "shows which unresolved spans pass 2 resolved, as what, and the intent's evidence", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_span_resolved"

    send(view.pid, {:analysis_progress, extracted(message_id, [@prince])})
    send(view.pid, {:analysis_progress, resolution(message_id, [@resolved], 1)})

    open_entities_tab(view, message_id)

    html = view |> element("#span-outcome-0-9") |> render()
    assert html =~ "Pass 2 (music.play)"
    assert html =~ "resolved by intent"
    assert html =~ "music_artist"
    assert html =~ "Intent evidence"
    assert html =~ "artist"
    assert html =~ "intent log-ratio"
  end

  test "shows a span pass 2 left undecided, and one it did not weigh again with why", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_span_undecided"
    other = %{@prince | start_pos: 20, end_pos: 24, value: "Paris"}

    outcomes = [
      %{@resolved | outcome: :undecided} |> Map.delete(:entity_type) |> Map.put(:best_type, "artist") |> Map.put(:margin, :infinity),
      %{value: "Paris", start_pos: 20, end_pos: 24, outcome: :not_redecided, reason: :no_measured_reading}
    ]

    send(view.pid, {:analysis_progress, extracted(message_id, [@prince, other])})
    send(view.pid, {:analysis_progress, resolution(message_id, outcomes, 0)})

    open_entities_tab(view, message_id)

    undecided = view |> element("#span-outcome-0-9") |> render()
    assert undecided =~ "still undecided, leading type"
    assert undecided =~ "no competitor"

    not_weighed = view |> element("#span-outcome-0-20") |> render()
    assert not_weighed =~ "not weighed again: no_measured_reading"
  end

  test "shows the entities pass 2 retyped with the chunk's profile", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")
    message_id = "msg_retyped"

    send(view.pid, {:analysis_progress, extracted(message_id, [])})

    send(
      view.pid,
      {:analysis_progress,
       progress(message_id, :entities_retyped, %{
         profile_domain: :smarthome,
         changes: [%{value: "Paris", start_pos: 20, end_pos: 24, from: "location", to: "room", disambiguation_source: :context_analysis}]
       })}
    )

    open_entities_tab(view, message_id)

    html = view |> element("#retyped-entities-0") |> render()
    assert html =~ "Retyped with the smarthome profile"
    assert html =~ "location"
    assert html =~ "room"
  end

  test "the pass 2 steps say how many spans were resolved and entities retyped" do
    assert ChatLive.stage_detail(resolution("m", [@resolved], 1)) == "1 of 1 unresolved span(s) resolved by intent"

    assert ChatLive.stage_detail(progress("m", :entities_retyped, %{profile_domain: :music, changes: []})) ==
             "0 retyped with the music profile"
  end

  test "the extraction step counts the undecided spans beside the entities" do
    assert ChatLive.stage_detail(extracted("m", [@prince])) == "1 entity(ies), 1 unresolved span(s)"
    assert ChatLive.stage_detail(extracted("m", [])) == "1 entity(ies)"
  end
end
