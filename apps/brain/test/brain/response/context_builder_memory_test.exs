defmodule Brain.Response.ContextBuilderMemoryTest do
  @moduledoc """
  The memory part of the unified context: the semantic facts retrieved for the
  turn's chunks and the recurring pattern found among them, which the
  response draws on ahead of the similar episodes.
  """
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.{ChunkAnalysis, ContextAccumulator, InternalModel}
  alias Brain.Memory.{RecurringPattern, Store}
  alias Brain.Memory.Types.SemanticFact
  alias Brain.Response.ContextBuilder
  alias Brain.Worlds.Embedder, as: WorldEmbedder

  @words ~w(weather paris sunny)
  @model %{vocabulary: @words |> Enum.with_index() |> Map.new(), idf_weights: Map.new(@words, &{&1, 1.0})}

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    # Running already when the application is up; started here otherwise.
    ensure_started(Store)
    ensure_started({Phoenix.PubSub, name: Brain.PubSub})
    assert Process.whereis(Store) && Process.whereis(Brain.PubSub)
    :ok = WorldEmbedder.init()

    world_id = "test_context_builder_memory_#{System.unique_integer([:positive])}"
    :ok = WorldEmbedder.load_model(world_id, @model)
    on_exit(fn -> WorldEmbedder.clear(world_id) end)

    {:ok, world_id: world_id}
  end

  defp analysis(index, text, semantics, episodes) do
    %ChunkAnalysis{
      chunk_index: index,
      text: text,
      response_strategy: :can_respond,
      accumulated_context: %ContextAccumulator{relevant_semantics: semantics, relevant_episodes: episodes}
    }
  end

  test "carries each retrieved fact once, most similar first, and the turn's recurring pattern",
       %{world_id: world_id} do
    ids =
      for reply <- ["It's sunny there.", "It's sunny there."] do
        {:ok, id} = Store.add_episode("is it sunny in paris", "conversation", reply, [], world_id: world_id)
        id
      end

    {:ok, episode} = Store.get_episode(hd(ids), world_id: world_id)

    recurring = SemanticFact.new("is it sunny in paris | is it sunny in paris", [], ids, [], action: "conversation")
    stated = SemanticFact.new("paris weather", [], [], [])

    # Two chunks retrieved the recurring fact, at different similarities.
    analyses = [
      analysis(0, "is it sunny in paris", [{recurring, 0.74}, {stated, 0.5}], [{episode, 0.9}]),
      analysis(1, "sunny paris", [{recurring, 0.82}], [{episode, 0.9}])
    ]

    :ok = Phoenix.PubSub.subscribe(Brain.PubSub, "brain:analysis")
    progress = %{conversation_id: "conv_#{System.unique_integer([:positive])}", message_id: "msg_1"}

    context =
      ContextBuilder.build(%InternalModel{raw_input: "is it sunny in paris. sunny paris", analyses: analyses},
        world_id: world_id,
        side_effects: false,
        progress: progress
      )

    recurring_id = recurring.id
    stated_id = stated.id

    assert [{%SemanticFact{id: ^recurring_id}, 0.82}, {%SemanticFact{id: ^stated_id}, 0.5}] =
             context.memory.semantic_facts

    assert %RecurringPattern{fact: %{id: ^recurring_id}, similarity: 0.82, reply: "It's sunny there.", reply_count: 2} =
             context.memory.recurring_pattern

    assert_receive {:analysis_progress, %{step: :semantic_facts_retrieved, recurring_pattern_id: ^recurring_id}}
  end

  test "carries no pattern when no retrieved fact is close enough", %{world_id: world_id} do
    stated = SemanticFact.new("paris weather", [], [], [])

    context =
      ContextBuilder.build(
        %InternalModel{raw_input: "is it sunny", analyses: [analysis(0, "is it sunny", [{stated, 0.3}], [])]},
        world_id: world_id,
        side_effects: false
      )

    assert [{%SemanticFact{}, 0.3}] = context.memory.semantic_facts
    assert context.memory.recurring_pattern == nil
  end
end
