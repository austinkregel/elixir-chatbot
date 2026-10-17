defmodule Brain.Metrics.MissedEventsTest do
  @moduledoc """
  An event `Brain.Metrics.Aggregator` cannot count is never dropped
  silently, and never costs a handler its attachment: `:telemetry` detaches
  a handler for good the first time it raises, so the handlers never raise.

  The event is logged as an error and counted as missed, by the handler
  that sent it and why (`Brain.Metrics.Aggregator.get_missed_events/0`),
  where the dashboard shows it: when the aggregator is not running, when
  Atlas refuses the write, when the event lacks what it is counted by, and
  when it was waiting in the mailbox as the aggregator stopped. Counting goes
  on once the aggregator is back, and the misses held in memory meanwhile
  are written to Atlas by the next aggregator.
  """
  use Brain.MetricsCase, async: false
  import ExUnit.CaptureLog

  alias Brain.Memory.RecurringPattern
  alias Brain.Memory.Types.SemanticFact

  @used_event [:chat_bot, :memory, :semantic_fact, :used]
  @used_handler "chatbot-semantic-fact-used"

  setup do
    world_id = "test_missed_events_#{System.unique_integer([:positive])}"
    fact = SemanticFact.new("is it sunny in paris", [], ["ep1", "ep2"], [])

    pattern = %RecurringPattern{
      world_id: world_id,
      fact: fact,
      similarity: 0.82,
      reply: "It's sunny there.",
      reply_count: 2,
      evidence_count: 2
    }

    {:ok, world_id: world_id, pattern: pattern}
  end

  test "an event sent while the aggregator is not running is logged and counted as missed, the handler stays attached, and counting goes on once it is back",
       %{world_id: world_id, pattern: pattern} do
    before = missed(@used_handler, :aggregator_not_running)

    log =
      capture_log(fn ->
        while_aggregator_down(fn -> :ok = RecurringPattern.used(pattern, :synthesizer, []) end)
      end)

    assert log =~ "[error]"
    assert log =~ "from #{@used_handler} (aggregator_not_running)"
    assert attached?(@used_event, @used_handler)
    assert missed(@used_handler, :aggregator_not_running) == before + 1

    :ok = RecurringPattern.used(pattern, :realization_packet, [])
    :ok = counted()

    assert Aggregator.get_semantic_fact_metrics(world_id).used == %{synthesizer: 0, realization_packet: 1}

    # The aggregator that started again wrote the miss to Atlas.
    assert [%{values: values}] = Atlas.Metrics.windows(:node, "telemetry", 1)
    assert values["missed:#{@used_handler}:aggregator_not_running"] >= 1
    assert missed(@used_handler, :aggregator_not_running) == before + 1
  end

  test "an event Atlas refuses to write is logged and counted as missed, and the aggregator goes on counting" do
    aggregator = Process.whereis(Aggregator)
    handler = "chatbot-jtms-justify-stop"
    before = missed(handler, :atlas_write_failed)

    # The window holds the span's shortest duration as a sum, so keeping the
    # smaller of it and the next one is refused by the database.
    :ok =
      Atlas.Metrics.write!([%{scope: :node, family: "telemetry", values: [{"duration:jtms_justify:min_us", :sum, 7}]}])

    log =
      capture_log(fn ->
        :ok = Brain.Telemetry.span(:jtms_justify, %{}, fn -> :ok end)
        :ok = counted()
      end)

    assert log =~ "[error]"
    assert log =~ "from #{handler} (atlas_write_failed)"
    assert Process.whereis(Aggregator) == aggregator
    assert attached?([:chat_bot, :epistemic, :jtms_justify, :stop], handler)
    assert missed(handler, :atlas_write_failed) == before + 1

    :ok = Brain.Telemetry.span(:belief_operation, %{}, fn -> :ok end)
    :ok = counted()
    assert Aggregator.get_metric(:belief_operation).count == 1
  end

  test "an event without what it is counted by is logged and counted as missed, and the handler stays attached" do
    aggregator = Process.whereis(Aggregator)
    before = missed(@used_handler, :malformed_event)

    log =
      capture_log(fn ->
        # No world to count it in.
        :ok = :telemetry.execute(@used_event, %{similarity: 0.82}, %{used_by: :synthesizer})
        :ok = counted()
      end)

    assert log =~ "from #{@used_handler} (malformed_event)"
    assert log =~ ":world_id"
    assert Process.whereis(Aggregator) == aggregator
    assert attached?(@used_event, @used_handler)
    assert missed(@used_handler, :malformed_event) == before + 1
  end

  test "an event waiting in the mailbox when the aggregator stops is counted as it stops, and the next aggregator counts on",
       %{world_id: world_id, pattern: pattern} do
    before = missed(@used_handler, :aggregator_stopped)
    aggregator = Process.whereis(Aggregator)

    # Suspended, it takes no event from its mailbox until it is stopped.
    :ok = :sys.suspend(aggregator)
    :ok = RecurringPattern.used(pattern, :synthesizer, [])

    _restarted = restart_aggregator!()

    assert missed(@used_handler, :aggregator_stopped) == before
    assert Aggregator.get_semantic_fact_metrics(world_id).used_total == 1

    :ok = RecurringPattern.used(pattern, :synthesizer, [])
    :ok = counted()
    assert Aggregator.get_semantic_fact_metrics(world_id).used_total == 2
  end

  test "an event still waiting once the aggregator has used its time to count what waits is logged and counted as missed",
       %{world_id: world_id, pattern: pattern} do
    before = missed(@used_handler, :aggregator_stopped)

    log =
      capture_log(fn ->
        while_aggregator_down(fn ->
          # No time to count what waits: everything waiting is missed.
          {:ok, pid} = GenServer.start(Aggregator, [drain_ms: 0], name: Aggregator)
          :ok = :sys.suspend(pid)
          :ok = RecurringPattern.used(pattern, :synthesizer, [])
          :ok = GenServer.stop(pid)
        end)
      end)

    assert log =~ "from #{@used_handler} (aggregator_stopped)"
    assert missed(@used_handler, :aggregator_stopped) == before + 1
    assert Aggregator.get_semantic_fact_metrics(world_id).used_total == 0
  end

  test "its supervisor gives it the time it takes to stop, so a full mailbox is counted, never killed uncounted" do
    assert Aggregator.child_spec([]).shutdown == 15_000

    source = "record_fact_retrieval"
    before = missed(source, :aggregator_stopped)
    aggregator = Process.whereis(Aggregator)
    ref = Process.monitor(aggregator)
    sent = 20_000

    :ok = :sys.suspend(aggregator)
    for _ <- 1..sent, do: :ok = Aggregator.record_fact_retrieval(1)

    # Stopped by its supervisor, which kills it once its shutdown time is up.
    {stop_us, :ok} = :timer.tc(fn -> while_aggregator_down(fn -> :ok end) end)

    assert_received {:DOWN, ^ref, :process, ^aggregator, reason}
    assert reason == :shutdown
    assert stop_us < 15_000_000

    counted_in_atlas = Aggregator.get_enrichment_metrics().fact_queries
    assert counted_in_atlas + (missed(source, :aggregator_stopped) - before) == sent
  end

  test "misses from one sender for one reason are logged once, then together, and every one is counted" do
    handler = "chatbot-code-gazetteer-add-stop"
    event = [:chat_bot, :code, :gazetteer, :add, :stop]
    before = missed(handler, :aggregator_not_running)

    log =
      capture_log(fn ->
        while_aggregator_down(fn ->
          for _ <- 1..100, do: :ok = :telemetry.execute(event, %{duration: 1_000}, %{})
        end)

        :ok = report_missed()
      end)

    assert missed(handler, :aggregator_not_running) == before + 100
    # The first is logged as it happens; the 99 after it, in the aggregator's report.
    assert lines_with(log, "from #{handler} (aggregator_not_running)") == 2
    assert log =~ "missed 1 event(s) from #{handler} (aggregator_not_running)"
    assert log =~ "99 from #{handler} (aggregator_not_running)"
  end
end
