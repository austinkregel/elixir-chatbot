defmodule Brain.Memory.RecurringPatternTest do
  @moduledoc """
  Which retrieved semantic fact becomes the turn's recurring pattern, the reply
  it carries, and how its retrieval and use are reported.
  """
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Memory.{RecurringPattern, Store}
  alias Brain.Memory.Types.SemanticFact
  alias Brain.Worlds.Embedder, as: WorldEmbedder

  @words ~w(weather paris rain sunny umbrella)
  @model %{vocabulary: @words |> Enum.with_index() |> Map.new(), idf_weights: Map.new(@words, &{&1, 1.0})}

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    # Running already when the application is up; started here otherwise.
    ensure_started(Store)
    ensure_started({Phoenix.PubSub, name: Brain.PubSub})
    assert Process.whereis(Store) && Process.whereis(Brain.PubSub)
    :ok = WorldEmbedder.init()

    world_id = "test_recurring_pattern_#{System.unique_integer([:positive])}"
    :ok = WorldEmbedder.load_model(world_id, @model)
    on_exit(fn -> WorldEmbedder.clear(world_id) end)

    {:ok, world_id: world_id}
  end

  # A fact over episodes that got these replies, as consolidation builds one.
  defp fact_over(replies, world_id) do
    ids =
      Enum.map(replies, fn reply ->
        {:ok, id} = Store.add_episode("is it sunny in paris", "conversation", reply, [], world_id: world_id)
        id
      end)

    SemanticFact.new("is it sunny in paris", [], ids, [], action: "conversation")
  end

  describe "find/2" do
    test "takes the most similar qualifying fact, with the reply its episodes most often got", %{world_id: world_id} do
      pattern = fact_over(["It's sunny there.", "Take an umbrella.", "It's sunny there."], world_id)
      weaker = fact_over(["Paris is lovely."], world_id)

      assert %RecurringPattern{} =
               found = RecurringPattern.find([{weaker, 0.72}, {pattern, 0.75}], world_id)

      assert found.fact.id == pattern.id
      assert found.world_id == world_id
      assert found.similarity == 0.75
      assert found.reply == "It's sunny there."
      assert found.reply_count == 2
      assert found.evidence_count == 3
    end

    test "between replies given equally often, takes the one listed first", %{world_id: world_id} do
      fact = fact_over(["Take an umbrella.", "It's sunny there."], world_id)

      assert %RecurringPattern{reply: "Take an umbrella.", reply_count: 1} =
               RecurringPattern.find([{fact, 0.8}], world_id)
    end

    test "passes over a fact below the threshold, one with no evidence, and one whose episodes got no reply",
         %{world_id: world_id} do
      below = fact_over(["It's sunny there."], world_id)
      unevidenced = SemanticFact.new("is it sunny in paris", [], [], [])
      unanswered = fact_over(["", ""], world_id)
      answered = fact_over(["Take an umbrella."], world_id)

      threshold = RecurringPattern.similarity_threshold()

      facts = [{below, threshold - 0.01}, {unevidenced, 0.99}, {unanswered, 0.95}, {answered, threshold}]

      assert %RecurringPattern{reply: "Take an umbrella."} = found = RecurringPattern.find(facts, world_id)
      assert found.fact.id == answered.id

      assert RecurringPattern.find([{below, threshold - 0.01}, {unevidenced, 0.99}, {unanswered, 0.95}], world_id) == nil
      assert RecurringPattern.find([], world_id) == nil
    end

    test "raises on a fact that names episodes the world does not hold", %{world_id: world_id} do
      missing_id = Ecto.UUID.generate()
      fact = SemanticFact.new("is it sunny in paris", [], [missing_id], [])

      error = assert_raise RuntimeError, fn -> RecurringPattern.find([{fact, 0.9}], world_id) end
      assert error.message =~ inspect({:semantic_evidence_missing, fact.id, [missing_id]})
    end
  end

  describe "retrieved/4, used/3 and report_use/2" do
    setup do
      test_pid = self()
      handler = "recurring-pattern-test-#{System.unique_integer([:positive])}"

      :ok =
        :telemetry.attach_many(
          handler,
          [[:chat_bot, :memory, :semantic_fact, :retrieved], [:chat_bot, :memory, :semantic_fact, :used]],
          fn event, measurements, metadata, _config -> send(test_pid, {:telemetry, event, measurements, metadata}) end,
          nil
        )

      on_exit(fn -> :telemetry.detach(handler) end)

      :ok = Phoenix.PubSub.subscribe(Brain.PubSub, "brain:analysis")
      progress = %{conversation_id: "conv_#{System.unique_integer([:positive])}", message_id: "msg_1"}

      {:ok, progress: progress}
    end

    test "report the facts retrieved for a turn and the pattern found among them",
         %{world_id: world_id, progress: progress} do
      fact = fact_over(["It's sunny there.", "It's sunny there."], world_id)
      other = SemanticFact.new("a fact of its own", [], [], [])
      retrieved = [{fact, 0.81234}, {other, 0.4}]
      pattern = RecurringPattern.find(retrieved, world_id)

      :ok = RecurringPattern.retrieved(retrieved, pattern, world_id, progress: progress)

      fact_id = fact.id
      other_id = other.id

      assert_receive {:telemetry, [:chat_bot, :memory, :semantic_fact, :retrieved], %{count: 2},
                      %{world_id: ^world_id, semantic_fact_ids: [^fact_id, ^other_id], recurring_pattern_id: ^fact_id}}

      assert_receive {:analysis_progress, %{step: :semantic_facts_retrieved} = step}
      assert step.recurring_pattern_id == fact_id

      assert [
               %{semantic_fact_id: ^fact_id, similarity: 0.812, text: "is it sunny in paris", action: "conversation", evidence_count: 2},
               %{semantic_fact_id: ^other_id, similarity: 0.4, text: "a fact of its own", action: nil, evidence_count: 0}
             ] = step.semantic_facts
    end

    test "report the turn's response built from the pattern, and by whom", %{world_id: world_id, progress: progress} do
      fact = fact_over(["It's sunny there."], world_id)
      pattern = RecurringPattern.find([{fact, 0.9}], world_id)

      :ok = RecurringPattern.used(pattern, :synthesizer, progress: progress)

      fact_id = fact.id

      assert_receive {:telemetry, [:chat_bot, :memory, :semantic_fact, :used],
                      %{similarity: 0.9, evidence_count: 1, reply_count: 1},
                      %{world_id: ^world_id, semantic_fact_id: ^fact_id, used_by: :synthesizer}}

      assert_receive {:analysis_progress,
                      %{step: :semantic_fact_used, semantic_fact_id: ^fact_id, used_by: :synthesizer, reply: "It's sunny there."}}
    end

    test "report through telemetry alone for a turn with no progress", %{world_id: world_id} do
      fact = fact_over(["It's sunny there."], world_id)
      pattern = RecurringPattern.find([{fact, 0.9}], world_id)

      :ok = RecurringPattern.used(pattern, :realization_packet, [])

      assert_receive {:telemetry, [:chat_bot, :memory, :semantic_fact, :used], _measurements, %{used_by: :realization_packet}}
      refute_receive {:analysis_progress, _}, 50
    end

    test "report a response's use of the pattern as a realizer hands it back, once",
         %{world_id: world_id, progress: progress} do
      fact = fact_over(["It's sunny there."], world_id)
      pattern = RecurringPattern.find([{fact, 0.9}], world_id)
      fact_id = fact.id

      :ok = RecurringPattern.report_use({pattern, :realization_packet}, progress: progress)

      assert_receive {:telemetry, [:chat_bot, :memory, :semantic_fact, :used], _measurements,
                      %{semantic_fact_id: ^fact_id, used_by: :realization_packet}}

      assert_receive {:analysis_progress, %{step: :semantic_fact_used, used_by: :realization_packet}}
      refute_receive {:telemetry, [:chat_bot, :memory, :semantic_fact, :used], _, _}, 50
      refute_receive {:analysis_progress, %{step: :semantic_fact_used}}, 50
    end

    test "report nothing for a response that drew on no pattern", %{progress: progress} do
      :ok = RecurringPattern.report_use(nil, progress: progress)

      refute_receive {:telemetry, [:chat_bot, :memory, :semantic_fact, :used], _, _}, 50
      refute_receive {:analysis_progress, %{step: :semantic_fact_used}}, 50
    end
  end
end
