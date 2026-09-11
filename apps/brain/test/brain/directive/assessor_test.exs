defmodule Brain.Directive.AssessorTest do
  @moduledoc """
  The directive decision table, exercised on hand-built analyses so the rules
  are tested in isolation from the pipeline that normally produces them.

  These run without the ComprehensionAssessor GenServer on purpose: the
  assessor must record an unavailable check as `:unavailable` rather than
  treating it as a pass, and that honesty is itself under test.
  """
  use ExUnit.Case, async: true

  alias Brain.Analysis.{ChunkAnalysis, InternalModel, SlotResult}
  alias Brain.Directive.{Assessment, Assessor}

  defp chunk(attrs \\ %{}) do
    base = %ChunkAnalysis{
      chunk_index: 0,
      text: "analyse the sector",
      intent: nil,
      confidence: 0.8,
      slots: %SlotResult{},
      epistemic_status: :unchecked
    }

    struct(base, attrs)
  end

  defp model(chunks) do
    %InternalModel{raw_input: "analyse the sector", chunks: [], analyses: chunks}
  end

  describe "comply" do
    test "a well-formed cognitive directive complies" do
      a = Assessor.assess(model([chunk()]))

      assert a.verdict == :comply
      assert a.reasons == []
      assert a.capability == :cognitive
      assert Assessment.primary_reason_tag(a) == nil
    end

    test "an unregistered intent is treated as a cognitive task, not an incapacity" do
      a = Assessor.assess(model([chunk(%{intent: "unknown"})]))

      assert a.verdict == :comply
      assert a.capability == :cognitive
    end
  end

  describe "unavailable checks are recorded, never silently passed" do
    test "comprehension is marked unavailable when the assessor is not running" do
      a = Assessor.assess(model([chunk()]))

      assert a.checks[:comprehension] == :unavailable
      assert a.comprehension[:verdict] == nil
    end

    test "an empty analysis list skips comprehension rather than passing it" do
      a = Assessor.assess(model([]))

      assert a.checks[:comprehension] == :skipped
    end
  end

  describe "slots" do
    test "missing required slots are advisory, with deterministic prompts" do
      slots = %SlotResult{
        schema_name: "weather.query",
        missing_required: ["location"],
        all_required_filled: false
      }

      a = Assessor.assess(model([chunk(%{intent: "weather.query", slots: slots})]))

      assert a.slots.missing == ["location"]
      assert [prompt] = a.slots.prompts
      assert is_binary(prompt) and prompt != ""

      # Advisory, not gating: the intent this was derived from is a classifier
      # guess, and real directives are routinely misclassified into registered
      # intents (see Assessor's moduledoc).
      assert {:missing_slots, ["location"]} in a.advisories
      refute {:missing_slots, ["location"]} in a.reasons
    end

    test "filled slots record the check as run without a reason" do
      slots = %SlotResult{filled_slots: %{"location" => %{value: "here"}}}
      a = Assessor.assess(model([chunk(%{intent: "weather.query", slots: slots})]))

      assert a.verdict == :comply
      assert a.checks[:slots] == :ran
      assert a.slots.filled == %{"location" => %{value: "here"}}
    end
  end

  describe "premises" do
    test "a contradicted chunk clarifies rather than refusing" do
      c =
        chunk(%{
          epistemic_status: :contradicted,
          related_beliefs: [%{subject: "reactor", predicate: "is", object: "offline"}]
        })

      a = Assessor.assess(model([c]))

      assert a.verdict == :clarify
      assert Assessment.primary_reason_tag(a) == :contradicted_premise
      assert [%{text: "analyse the sector"}] = a.premises
    end

    test "a string epistemic_status is tolerated" do
      a = Assessor.assess(model([chunk(%{epistemic_status: "contradicted"})]))

      assert a.verdict == :clarify
    end
  end

  describe "capability" do
    # weather.query resolves to a registered service, so the capability check
    # genuinely applies to it rather than short-circuiting as :cognitive.
    @capability_intent "weather.query"

    test "a capability gap never blocks the order" do
      a = Assessor.assess(model([chunk(%{intent: @capability_intent})]), world_id: "test")

      assert a.verdict == :comply

      case a.capability do
        {:external, :incapable} ->
          assert {:incapable_action, @capability_intent} in a.advisories
          refute {:incapable_action, @capability_intent} in a.reasons

        {:external, :capable} ->
          assert a.advisories == []

        :cognitive ->
          # The capability check could not run (vault down); it must have said so.
          assert a.checks[:capability] == :unavailable
      end
    end

    test "an unavailable capability check is recorded rather than assumed" do
      a = Assessor.assess(model([chunk(%{intent: @capability_intent})]), world_id: "test")

      assert a.checks[:capability] in [:ran, :unavailable]
    end
  end

  describe "report payload" do
    test "is flat and JSON-encodable" do
      c =
        chunk(%{
          intent: "weather.query",
          epistemic_status: :contradicted,
          related_beliefs: [%{subject: "s", predicate: "p", object: "o"}]
        })

      payload = model([c]) |> Assessor.assess() |> Assessment.to_report_payload()

      assert payload.verdict == "clarify"
      assert payload.rule == "contradicted_premise"
      assert payload.intent == "weather.query"
      assert is_list(payload.reasons)
      assert payload.checks["comprehension"] == "unavailable"
      assert {:ok, _json} = Jason.encode(payload)
    end
  end

  describe "task frames are carried, not gated on" do
    test "frames ride along without changing the verdict" do
      c = chunk(%{event_frames: [%{verb: "analyse"}], srl_frames: [%{arg0: "you"}]})
      a = Assessor.assess(model([c]))

      assert a.verdict == :comply
      assert match?([_], a.task_frames.events)
      assert match?([_], a.task_frames.srl_triples)
    end
  end
end
