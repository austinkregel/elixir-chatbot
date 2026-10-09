defmodule Brain.Response.ContentSpecifierTest do
  @moduledoc """
  A narrative content primitive is filled with the beliefs held in the world the
  response is realized in (`:world_id`), read through
  `BeliefStore.query_beliefs/2`'s `{:ok, beliefs}`.
  """

  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.ChunkAnalysis
  alias Brain.Response.{ContentSpecifier, Primitive}

  setup do
    ensure_started(World.Manager)
    setup_world_sandbox()

    {:ok, with_belief} = create_test_world("content_specifier_with_belief")
    {:ok, without_belief} = create_test_world("content_specifier_without_belief")

    {:ok, belief_id} = Brain.Epistemic.BeliefStore.add_belief(with_belief.id, "dogs", :are, "animals")

    {:ok, with_belief: with_belief.id, without_belief: without_belief.id, belief_id: belief_id}
  end

  test "a narrative primitive holds the beliefs of the named world, and only those",
       %{with_belief: with_belief, without_belief: without_belief, belief_id: belief_id} do
    narrative = Primitive.new(:content, :narrative)

    [specified] = ContentSpecifier.specify([narrative], %ChunkAnalysis{}, world_id: with_belief)
    assert Enum.map(specified.content.beliefs, & &1.id) == [belief_id]
    assert specified.content.belief_count == 1

    [specified] = ContentSpecifier.specify([narrative], %ChunkAnalysis{}, world_id: without_belief)
    assert specified.content.beliefs == []
    assert specified.content.belief_count == 0
  end

  test "a narrative primitive without a world raises" do
    assert_raise KeyError, ~r/:world_id/, fn ->
      ContentSpecifier.specify([Primitive.new(:content, :narrative)], %ChunkAnalysis{}, [])
    end
  end
end
