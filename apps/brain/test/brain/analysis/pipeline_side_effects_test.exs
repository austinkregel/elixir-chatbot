defmodule Brain.Analysis.PipelineSideEffectsTest do
  @moduledoc """
  The analysis pipeline writes what it learned about a turn (beliefs extracted
  from events, graph data, feedback statistics) unless its caller passes
  `side_effects: false`.

  Evaluation and training-data runs push whole corpora through the pipeline
  and pass `false`. When belief extraction ignored that option, every corpus
  utterance became a belief about the user ("user requests turn lights"), so
  these tests require that a run with `false` writes nothing, and that a run
  with the default, and a live conversation turn, still write.
  """

  use Brain.Test.GraphCase, async: false

  import Ecto.Query

  alias Brain.Analysis.{LearningStore, Pipeline}
  alias Brain.Epistemic.BeliefStore

  # An imperative with an object. Belief extraction turns its event into
  # `user requests "turn lights"`, the shape of the rows the leak produced.
  @directive "Turn off the lights in the kitchen"

  setup do
    {:ok, user_id: "side_effects_#{System.unique_integer([:positive])}"}
  end

  describe "a run with side_effects: false writes nothing" do
    test "analyze_chunk, as the evaluation tasks call it" do
      assert_writes_nothing(fn -> Pipeline.analyze_chunk(@directive, side_effects: false) end)
    end

    test "process" do
      assert_writes_nothing(fn -> Pipeline.process(@directive, side_effects: false) end)
    end

    test "with a user_id", %{user_id: user_id} do
      Pipeline.analyze_chunk(@directive, user_id: user_id, side_effects: false)
      Pipeline.process(@directive, user_id: user_id, side_effects: false)

      assert beliefs_for(user_id) == []
      assert atlas_belief_count(user_id) == 0
    end
  end

  describe "a run with side effects writes" do
    test "process with no side_effects option extracts a belief for the user", %{user_id: user_id} do
      Pipeline.process(@directive, user_id: user_id)

      assert Enum.any?(beliefs_for(user_id), &(&1.predicate == :requests)),
             "Expected a :requests belief for #{user_id}, got: #{inspect(beliefs_for(user_id))}"

      assert atlas_belief_count(user_id) > 0
    end

    test "process with side_effects: true extracts a belief for the user", %{user_id: user_id} do
      Pipeline.process(@directive, user_id: user_id, side_effects: true)

      assert Enum.any?(beliefs_for(user_id), &(&1.predicate == :requests)),
             "Expected a :requests belief for #{user_id}, got: #{inspect(beliefs_for(user_id))}"

      assert atlas_belief_count(user_id) > 0
    end

    test "a live conversation turn extracts a belief for the user", %{user_id: user_id} do
      {:ok, conversation_id} = Brain.create_conversation()

      assert {:ok, %{response: _}} = Brain.evaluate(conversation_id, @directive, user_id: user_id)

      assert Enum.any?(beliefs_for(user_id), &(&1.predicate == :requests)),
             "Expected the conversation turn to store a :requests belief for #{user_id}, " <>
               "got: #{inspect(beliefs_for(user_id))}"

      assert atlas_belief_count(user_id) > 0
    end
  end

  describe "the option is validated" do
    test "a value other than true or false raises" do
      assert_raise ArgumentError, ~r/:side_effects must be true or false/, fn ->
        Pipeline.analyze_chunk(@directive, side_effects: :yes)
      end

      assert_raise ArgumentError, ~r/:side_effects must be true or false/, fn ->
        Pipeline.process(@directive, side_effects: nil)
      end
    end
  end

  # Belief extraction runs in-process under test (`pipeline_belief_extraction_sync`)
  # and Atlas writes are synchronous (`atlas_sync_mode`), so every write the run
  # makes has landed by the time it returns. The feedback statistic is a cast;
  # the `get_stats/0` call behind it is served only after the cast is handled.
  defp assert_writes_nothing(run) do
    beliefs_before = BeliefStore.stats().total_beliefs
    atlas_before = Atlas.Repo.aggregate(Atlas.Schemas.Belief, :count)
    stats_before = LearningStore.get_stats()
    {:ok, graph_before} = count_nodes("knowledge_graph")

    run.()

    assert BeliefStore.stats().total_beliefs == beliefs_before
    assert Atlas.Repo.aggregate(Atlas.Schemas.Belief, :count) == atlas_before
    assert LearningStore.get_stats() == stats_before
    assert {:ok, ^graph_before} = count_nodes("knowledge_graph")
  end

  defp beliefs_for(user_id) do
    {:ok, beliefs} = BeliefStore.query_beliefs(user_id: user_id)
    beliefs
  end

  defp atlas_belief_count(user_id) do
    Atlas.Repo.aggregate(from(b in Atlas.Schemas.Belief, where: b.user_id == ^user_id), :count)
  end
end
