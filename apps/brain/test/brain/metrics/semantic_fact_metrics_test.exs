defmodule Brain.Metrics.SemanticFactMetricsTest do
  @moduledoc """
  `Brain.Metrics.Aggregator` counts the semantic-fact events per world: the
  turns that retrieved facts, how many found a recurring pattern, who built a
  response from it, and how similar the retrieved facts and used patterns were.
  The events reach it through the handlers `Brain.Telemetry.attach_handlers/0`
  attaches, as they do in the running application.

  The counts are kept in Atlas, in measurement windows, so they survive the
  aggregator restarting, and a new window can be started to compare one
  stretch of responses with the next. Every count a test makes is written
  inside its sandbox transaction and rolled back with it.
  """
  use Brain.MetricsCase, async: false

  alias Brain.Memory.RecurringPattern
  alias Brain.Memory.Types.SemanticFact

  setup do
    unique = System.unique_integer([:positive])
    {:ok, world_id: "test_semantic_fact_metrics_#{unique}", other_world_id: "test_semantic_fact_metrics_other_#{unique}"}
  end

  defp pattern(fact, similarity, world_id) do
    %RecurringPattern{
      world_id: world_id,
      fact: fact,
      similarity: similarity,
      reply: "It's sunny there.",
      reply_count: 2,
      evidence_count: 2
    }
  end

  # A turn: the facts retrieved for it, the pattern found among them, and
  # whoever built its response from the pattern.
  defp turn(world_id, retrieved, pattern, used_by) do
    :ok = RecurringPattern.retrieved(retrieved, pattern, world_id, [])
    if used_by, do: :ok = RecurringPattern.used(pattern, used_by, [])
  end

  defp counted(world_id) do
    :ok = counted()
    Aggregator.get_semantic_fact_metrics(world_id)
  end

  test "counts retrievals, patterns found, uses by who used them, and similarities, per world",
       %{world_id: world_id, other_world_id: other_world_id} do
    sunny = SemanticFact.new("is it sunny in paris", [], ["ep1", "ep2"], [])
    weather = SemanticFact.new("paris weather", [], [], [])

    turn(world_id, [{sunny, 0.82}, {weather, 0.41}], pattern(sunny, 0.82, world_id), :synthesizer)
    turn(world_id, [{sunny, 0.75}], pattern(sunny, 0.75, world_id), :realization_packet)
    turn(world_id, [{weather, 0.35}], nil, nil)
    turn(other_world_id, [{sunny, 0.91}], pattern(sunny, 0.91, other_world_id), :synthesizer)

    metrics = counted(world_id)

    assert metrics.retrievals == 3
    assert metrics.facts_retrieved == 4
    assert metrics.patterns_found == 2
    assert metrics.pattern_found_rate == 66.7
    assert metrics.used == %{synthesizer: 1, realization_packet: 1}
    assert metrics.used_total == 2

    assert metrics.retrieved_similarity == [
             %{from: 0.3, to: 0.4, count: 1},
             %{from: 0.4, to: 0.5, count: 1},
             %{from: 0.7, to: 0.8, count: 1},
             %{from: 0.8, to: 0.9, count: 1}
           ]

    assert metrics.used_similarity == [%{from: 0.7, to: 0.8, count: 1}, %{from: 0.8, to: 0.9, count: 1}]

    other = counted(other_world_id)
    assert other.retrievals == 1
    assert other.patterns_found == 1
    assert other.pattern_found_rate == 100.0
    assert other.used == %{synthesizer: 1, realization_packet: 0}
    assert other.retrieved_similarity == [%{from: 0.9, to: 1.0, count: 1}]
  end

  test "a turn that retrieved no facts is still a retrieval", %{world_id: world_id} do
    turn(world_id, [], nil, nil)

    metrics = counted(world_id)
    assert metrics.retrievals == 1
    assert metrics.facts_retrieved == 0
    assert metrics.patterns_found == 0
    assert metrics.pattern_found_rate == 0.0
    assert metrics.retrieved_similarity == []
  end

  test "the top bucket holds a similarity of 1.0, and a negative similarity has a bucket of its own",
       %{world_id: world_id} do
    fact = SemanticFact.new("is it sunny in paris", [], [], [])
    turn(world_id, [{fact, 1.0}, {fact, -0.05}], nil, nil)

    assert counted(world_id).retrieved_similarity == [
             %{from: -0.1, to: 0.0, count: 1},
             %{from: 0.9, to: 1.0, count: 1}
           ]
  end

  test "a world nothing was counted for has no window, no rate and no distribution", %{world_id: world_id} do
    assert counted(world_id) == %{
             window: nil,
             retrievals: 0,
             facts_retrieved: 0,
             patterns_found: 0,
             pattern_found_rate: nil,
             used: %{synthesizer: 0, realization_packet: 0},
             used_total: 0,
             retrieved_similarity: [],
             used_similarity: []
           }

    assert Aggregator.get_previous_semantic_fact_metrics(world_id) == nil
  end

  test "the counts survive the aggregator restarting, and later events add to them", %{world_id: world_id} do
    sunny = SemanticFact.new("is it sunny in paris", [], ["ep1", "ep2"], [])
    turn(world_id, [{sunny, 0.82}], pattern(sunny, 0.82, world_id), :synthesizer)
    before_restart = counted(world_id)
    assert before_restart.retrievals == 1
    assert before_restart.used == %{synthesizer: 1, realization_packet: 0}
    assert %DateTime{} = before_restart.window.started_at

    _restarted = restart_aggregator!()

    assert Aggregator.get_semantic_fact_metrics(world_id) == before_restart

    turn(world_id, [{sunny, 0.85}], pattern(sunny, 0.85, world_id), :realization_packet)
    after_restart = counted(world_id)

    assert after_restart.window == before_restart.window
    assert after_restart.retrievals == 2
    assert after_restart.patterns_found == 2
    assert after_restart.used == %{synthesizer: 1, realization_packet: 1}
    assert after_restart.retrieved_similarity == [%{from: 0.8, to: 0.9, count: 2}]
  end

  test "starting a new window keeps the ended window's counts, and every event sent before it counts in it",
       %{world_id: world_id, other_world_id: other_world_id} do
    sunny = SemanticFact.new("is it sunny in paris", [], ["ep1", "ep2"], [])
    turn(world_id, [{sunny, 0.82}], pattern(sunny, 0.82, world_id), :synthesizer)
    turn(other_world_id, [{sunny, 0.82}], nil, nil)

    # Sent straight after the events, without waiting for them to be counted.
    window = Aggregator.start_semantic_fact_window(world_id)

    previous = Aggregator.get_previous_semantic_fact_metrics(world_id)
    assert previous.retrievals == 1
    assert previous.used == %{synthesizer: 1, realization_packet: 0}
    assert previous.window.ended_at == window.started_at

    current = Aggregator.get_semantic_fact_metrics(world_id)
    assert current.window == window
    assert window.ended_at == nil
    assert current.retrievals == 0
    assert current.used_total == 0

    turn(world_id, [{sunny, 0.91}], nil, nil)
    assert counted(world_id).retrievals == 1
    assert Aggregator.get_previous_semantic_fact_metrics(world_id).retrievals == 1

    # The other world's window goes on.
    assert counted(other_world_id).retrievals == 1
    assert Aggregator.get_previous_semantic_fact_metrics(other_world_id) == nil
  end
end
