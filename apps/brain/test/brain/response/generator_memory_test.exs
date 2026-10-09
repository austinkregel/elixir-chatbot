defmodule Brain.Response.GeneratorMemoryTest do
  @moduledoc """
  The Generator builds its own context: it retrieves the world's semantic
  facts with the turn's similar episodes, finds the recurring pattern among
  them, and the synthesizer answers from that pattern ahead of any single
  episode.
  """
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  # World.Embedder is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Embedder}

  alias Brain.Memory.{Store, VectorIndex}
  alias Brain.Memory.Types.SemanticFact
  alias Brain.Response.Generator
  alias World.Embedder, as: WorldEmbedder

  @words ~w(weather paris sunny rain)
  @model %{vocabulary: @words |> Enum.with_index() |> Map.new(), idf_weights: Map.new(@words, &{&1, 1.0})}

  # An intent with no domain knowledge: what is synthesized comes from memory.
  @intent "completely_unknown.intent"

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    # Running already when the application is up; started here otherwise. The
    # Generator's enrichment reads the world's service credentials.
    ensure_started(Store)
    ensure_started(Brain.Services.CredentialVault)
    assert Process.whereis(Store) && Process.whereis(Brain.Services.CredentialVault)
    :ok = WorldEmbedder.init()

    world_id = "test_generator_memory_#{System.unique_integer([:positive])}"
    :ok = WorldEmbedder.load_model(world_id, @model)
    on_exit(fn -> WorldEmbedder.clear(world_id) end)

    test_pid = self()
    handler = "generator-memory-test-#{System.unique_integer([:positive])}"

    :ok =
      :telemetry.attach_many(
        handler,
        [[:chat_bot, :memory, :semantic_fact, :retrieved], [:chat_bot, :memory, :semantic_fact, :used]],
        fn event, _measurements, metadata, _config -> send(test_pid, {event, metadata}) end,
        nil
      )

    on_exit(fn -> :telemetry.detach(handler) end)

    {:ok, world_id: world_id}
  end

  test "answers from the world's recurring pattern ahead of a single episode that matches the turn as well",
       %{world_id: world_id} do
    pattern_ids =
      for reply <- ["It is sunny in paris.", "It is sunny in paris."] do
        {:ok, id} = Store.add_episode("sunny paris weather", "conversation", reply, [], world_id: world_id)
        id
      end

    # A single episode with the same words and another reply.
    {:ok, _single} = Store.add_episode("sunny paris weather", "conversation", "Bring an umbrella.", [], world_id: world_id)

    center =
      pattern_ids
      |> Enum.map(fn id -> elem(Store.get_episode(id, world_id: world_id), 1).embedding end)
      |> VectorIndex.mean_vector()

    fact = SemanticFact.new("sunny paris weather | sunny paris weather", center, pattern_ids, [], action: "conversation")
    {:ok, fact_id} = Store.add_semantic(fact, world_id: world_id)

    assert {:ok, "It is sunny in paris.", :synthesized} =
             Generator.generate(@intent, [], "sunny paris weather", world_id: world_id, side_effects: false)

    assert_receive {[:chat_bot, :memory, :semantic_fact, :retrieved], %{world_id: ^world_id, recurring_pattern_id: ^fact_id}}
    assert_receive {[:chat_bot, :memory, :semantic_fact, :used], %{semantic_fact_id: ^fact_id, used_by: :synthesizer}}
  end

  test "answers from the single episode when the world has no semantic fact", %{world_id: world_id} do
    {:ok, _single} = Store.add_episode("sunny paris weather", "conversation", "Bring an umbrella.", [], world_id: world_id)

    assert {:ok, "Bring an umbrella.", :synthesized} =
             Generator.generate(@intent, [], "sunny paris weather", world_id: world_id, side_effects: false)

    assert_receive {[:chat_bot, :memory, :semantic_fact, :retrieved], %{world_id: ^world_id, semantic_fact_ids: [], recurring_pattern_id: nil}}
    refute_receive {[:chat_bot, :memory, :semantic_fact, :used], _}, 50
  end
end
