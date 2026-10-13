defmodule Brain.Metrics.SemanticFactMetricsTest do
  @moduledoc """
  `Brain.Metrics.Aggregator` counts the semantic-fact events per world: the
  turns that retrieved facts, how many found a recurring pattern, who built a
  response from it, and how similar the retrieved facts and used patterns were.
  The events reach it through the handlers `Brain.Telemetry.attach_handlers/0`
  attaches, as they do in the running application.
  """
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Memory.RecurringPattern
  alias Brain.Memory.Types.SemanticFact
  alias Brain.Metrics.Aggregator

  setup do
    {:ok, _apps} = Application.ensure_all_started(:telemetry)
    # Running already when the application is up; started here otherwise.
    ensure_started(Aggregator)
    assert Process.whereis(Aggregator)
    :ok = Brain.Telemetry.attach_handlers()

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

  # The aggregator handles its messages in order, so once it answers a call
  # made after the events, it has counted them.
  defp counted(world_id) do
    _state = :sys.get_state(Aggregator)
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

  test "a world nothing was counted for has no rate and no distribution", %{world_id: world_id} do
    assert counted(world_id) == %{
             retrievals: 0,
             facts_retrieved: 0,
             patterns_found: 0,
             pattern_found_rate: nil,
             used: %{synthesizer: 0, realization_packet: 0},
             used_total: 0,
             retrieved_similarity: [],
             used_similarity: []
           }
  end
end
