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
    # Creating a world and adding a belief write to Atlas, in this process and
    # in the allowed workers.
    owner = Brain.Test.AtlasSandbox.checkout_allowed!()
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    setup_world_sandbox()

    # A world's id names its AGE graphs, which allow 63 bytes with the longest
    # kind's prefix, so 43 for the id: these names leave room for any unique
    # integer `create_test_world/2` appends.
    {:ok, with_belief} = create_test_world("with_belief")
    {:ok, without_belief} = create_test_world("without_belief")

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

  # The self-knowledge reply for a turn with no user: nothing is known of them
  # (primitives.json's "empty_knowledge"), and the lattice says it as it is.
  test "a narrative primitive states the self-knowledge reply, which the lattice words",
       %{without_belief: world} do
    narrative = Primitive.new(:content, :narrative, %{chunk_index: 0})
    [specified] = ContentSpecifier.specify([narrative], %ChunkAnalysis{chunk_index: 0}, world_id: world)

    assert specified.content.narrative in [
               "I don't have much information about you yet. We're just getting to know each other.",
               "I haven't learned much about you so far. Is there anything you'd like to share?",
               "We haven't really talked much yet, so I don't have much to go on.",
               "I'm still getting to know you. We haven't shared much yet."
             ]

    assert specified.content.content_complete

    assert {:ok, _rendered, reply} =
             Brain.Response.PhraseLatticeRealizer.realize([specified],
               world_id: world,
               analysis: %ChunkAnalysis{chunk_index: 0},
               analyses: [%ChunkAnalysis{chunk_index: 0}]
             )

    assert reply == specified.content.narrative
  end

  # A factual primitive states the first statement known for the turn, the one
  # that answers a polar question first, as it was stored; it is never given
  # a statement nothing retrieved.
  test "a factual primitive states the known statement, and none when nothing is known", %{without_belief: world} do
    fact = %Brain.FactDatabase.Fact{id: "geo_001", entity: "France", fact: "The capital of France is Paris", category: "geography"}
    known = %ChunkAnalysis{chunk_index: 0, related_facts: [fact]}

    [specified] = ContentSpecifier.specify([Primitive.new(:content, :factual)], known, world_id: world)
    assert specified.content.fact == "The capital of France is Paris"
    assert specified.content.facts == ["The capital of France is Paris"]
    assert specified.content.content_complete

    [specified] = ContentSpecifier.specify([Primitive.new(:content, :factual)], %ChunkAnalysis{chunk_index: 0}, world_id: world)
    refute Map.has_key?(specified.content, :fact)
    refute specified.content.content_complete
  end

  test "a narrative primitive without a world raises" do
    assert_raise KeyError, ~r/:world_id/, fn ->
      ContentSpecifier.specify([Primitive.new(:content, :narrative)], %ChunkAnalysis{}, [])
    end
  end
end
