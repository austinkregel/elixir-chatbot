defmodule Brain.Response.RefinementLoopMemoryTest do
  @moduledoc """
  A refined turn realizes its response on each pass, and a pass realized from
  the turn's recurring pattern draws on it each time. The loop settles on one
  pass, so the response's use of the pattern is reported once, for that pass:
  `[:chat_bot, :memory, :semantic_fact, :used]` fires once when the pass the
  turn settled on drew on the pattern, and not at all otherwise.
  """
  use ExUnit.Case, async: false

  alias Brain.Analysis.{ChunkAnalysis, InternalModel, SpeechActResult}
  alias Brain.Memory.RecurringPattern
  alias Brain.Memory.Types.SemanticFact
  alias Brain.Response.RefinementLoop

  # The world the test suite runs in, seeded by `mix test.models`; the loop
  # evaluates each pass by analyzing its response there. Side effects are off,
  # so nothing is written to it.
  @world_id "default"

  # An intent with no domain knowledge: whatever the synthesizer realizes
  # comes from memory.
  @intent "completely_unknown.intent"

  setup do
    test_pid = self()
    handler = "refinement-loop-memory-test-#{System.unique_integer([:positive])}"

    :ok =
      :telemetry.attach(
        handler,
        [:chat_bot, :memory, :semantic_fact, :used],
        fn _event, _measurements, metadata, _config -> send(test_pid, {:used, metadata}) end,
        nil
      )

    on_exit(fn -> :telemetry.detach(handler) end)

    pattern = %RecurringPattern{
      world_id: "test_refinement_loop_memory",
      fact: SemanticFact.new("is it sunny in lyon | is it sunny in lyon", [], ["ep1", "ep2", "ep3"], []),
      similarity: 0.8,
      reply: "It's usually sunny in $location.",
      reply_count: 3,
      evidence_count: 3
    }

    {:ok, pattern: pattern}
  end

  # Every `used` event the loop emitted. Telemetry handlers run in the emitting
  # process, so they are all in the mailbox once the loop returns.
  defp uses do
    receive do
      {:used, metadata} -> [metadata | uses()]
    after
      0 -> []
    end
  end

  test "a refined turn reports its response's use of the recurring pattern once", %{pattern: pattern} do
    # A low-confidence question, which the evaluator is not expected to settle
    # on in one pass; the assertion below fails loudly if it does.
    analysis = %ChunkAnalysis{
      chunk_index: 0,
      text: "is it sunny in lyon",
      speech_act: SpeechActResult.new(:directive, :question, 0.3, is_question: true),
      intent: @intent,
      entities: [%{entity_type: "location", value: "Lyon"}],
      sentiment: %{label: :neutral, confidence: 0.5},
      confidence: 0.2,
      response_strategy: :can_respond,
      epistemic_status: :unchecked
    }

    model = %InternalModel{raw_input: analysis.text, analyses: [analysis], overall_strategy: :can_respond}
    unified_context = %{memory: %{recurring_pattern: pattern, similar_episodes: []}}

    assert {:ok, response, metadata} =
             RefinementLoop.generate(model,
               world_id: @world_id,
               unified_context: unified_context,
               max_iterations: 3,
               side_effects: false
             )

    assert metadata.iterations > 1,
           "the turn settled in one pass, so it shows nothing about a refined turn: #{inspect(metadata.score)}"

    # The pass the turn settled on drew on the pattern when the synthesizer
    # realized it; the lattice draws on no pattern, and a turn whose silence is
    # preferred has no response to report.
    settled_on_pattern =
      response != nil and Enum.all?(metadata.primitives, &(&1.source == :synthesizer_template))

    fact_id = pattern.fact.id

    if settled_on_pattern do
      assert [%{semantic_fact_id: ^fact_id, used_by: :synthesizer}] = uses()
    else
      assert uses() == []
    end
  end
end
