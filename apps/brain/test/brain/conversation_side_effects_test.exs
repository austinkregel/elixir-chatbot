defmodule Brain.ConversationSideEffectsTest do
  @moduledoc """
  A conversation created with `side_effects: false` (a benchmark's, for one)
  runs its turns end to end but writes nothing: no beliefs, user-model facts,
  episodes, learner memory, graph nodes or feedback statistics, and no service
  call that changes state.

  A benchmark used to time `Brain.evaluate` on ordinary conversations, and
  every timed turn wrote its input into the stores like a real user's.
  """

  use Brain.Test.GraphCase, async: false

  alias Brain.Analysis.LearningStore
  alias Brain.Epistemic.BeliefStore
  alias Brain.Services.Dispatcher

  # Inputs that reach the turn's writes: a self-referential statement (user
  # facts, beliefs), an imperative with an object (event beliefs), and a
  # question about a place (entities, slots).
  @inputs [
    "My name is Dallas and I live in Chicago",
    "Turn off the lights in the kitchen",
    "What's the weather like in Seattle?"
  ]

  @atlas_schemas [
    Atlas.Schemas.Belief,
    Atlas.Schemas.Episode,
    Atlas.Schemas.UserModel,
    Atlas.Schemas.SemanticFact,
    Atlas.Schemas.LearnedFact,
    Atlas.Schemas.IntentReviewCandidate
  ]

  @graphs ~w(conversation_graph knowledge_graph epistemic_graph user_graph semantic_graph)

  setup do
    {:ok, user_id: "conversation_side_effects_#{System.unique_integer([:positive])}"}
  end

  describe "a conversation created with side_effects: false" do
    test "writes nothing across its turns", %{user_id: user_id} do
      before = snapshot()

      {:ok, conversation_id} = Brain.create_conversation(side_effects: false)

      for input <- @inputs do
        assert {:ok, %{response: _}} = Brain.evaluate(conversation_id, input, user_id: user_id)
      end

      assert snapshot() == before
    end
  end

  describe "a conversation created with the default" do
    test "still writes what its turns learn", %{user_id: user_id} do
      before = snapshot()

      {:ok, conversation_id} = Brain.create_conversation()

      for input <- @inputs do
        assert {:ok, %{response: _}} = Brain.evaluate(conversation_id, input, user_id: user_id)
      end

      after_turns = snapshot()

      assert after_turns.atlas[Atlas.Schemas.Belief] > before.atlas[Atlas.Schemas.Belief]
      assert after_turns.graphs["conversation_graph"] > before.graphs["conversation_graph"]
    end
  end

  describe "the option is validated" do
    test "create_conversation raises on a value other than true or false" do
      assert_raise ArgumentError, ~r/:side_effects must be true or false/, fn ->
        Brain.create_conversation(side_effects: :no)
      end
    end

    test "evaluate does not take side_effects" do
      {:ok, conversation_id} = Brain.create_conversation()

      assert_raise ArgumentError, ~r/set for the whole conversation/, fn ->
        Brain.evaluate(conversation_id, "Hello", side_effects: false)
      end
    end
  end

  describe "service calls with side effects off" do
    test "a device action is refused before the service is called" do
      assert Brain.Services.HomeAssistant.writes?("smarthome.switch")

      assert {:error, :side_effects_off} =
               Dispatcher.dispatch("smarthome.switch", %{device: "kitchen lights"}, %{side_effects: false})
    end

    test "a read-only intent is not refused" do
      refute Brain.Services.HomeAssistant.writes?("smarthome.device_check")

      refute Dispatcher.dispatch("smarthome.device_check", %{device: "kitchen lights"}, %{side_effects: false}) ==
               {:error, :side_effects_off}
    end
  end

  # Brain.get_status/0 is served after the :process_learning_queue message a
  # turn sends itself, so any episode a turn queued has been stored by then.
  # Belief extraction and Atlas writes are synchronous under test, and the
  # LearningStore and MemoryStore calls are served after any earlier casts.
  defp snapshot do
    %{name: persona} = Brain.get_status()

    %{
      beliefs: BeliefStore.stats().total_beliefs,
      atlas: Map.new(@atlas_schemas, &{&1, Atlas.Repo.aggregate(&1, :count)}),
      graphs:
        Map.new(@graphs, fn graph ->
          {:ok, count} = count_nodes(graph)
          {graph, count}
        end),
      learning: LearningStore.get_stats(),
      persona_memory: length(Brain.MemoryStore.get_memory_window(persona, 1_000_000))
    }
  end
end
