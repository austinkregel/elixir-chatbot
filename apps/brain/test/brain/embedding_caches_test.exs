defmodule Brain.EmbeddingCachesTest do
  @moduledoc """
  Every cache of vectors embedded with a world's vocabulary
  (`Brain.EmbeddingCaches`) is forgotten when the world's vocabulary is rebuilt
  and when the world is destroyed, and is built again with the world's
  vocabulary on its next use.

  A cache that was not forgotten would compare vectors of the old
  vocabulary's length with a query embedded with the new one, which raises
  (`FourthWall.Math.cosine_similarity!/2`), so a next use that succeeds and
  holds vectors of the new length shows the cache was rebuilt.

  The worlds are `test_` worlds, each seeded in memory with the trained
  embedder vocabulary when it is created, and cleaned up after the test.
  """

  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.EmbeddingCaches
  alias Brain.Memory.Store
  alias Brain.Response.{ChunkSegmenter, SemanticFactRetriever, TemplateBlender, TemplateStore}
  alias Brain.Worlds
  alias Brain.Worlds.Embedder

  # A sentence the heuristics classify as :body, so the world's chunk type
  # centroids classify it.
  @body_sentence "The weather forecast shows sunny skies."
  @context %{speech_acts: [:greeting], entities: []}

  setup tags do
    # A rebuild reads the world's episodes from Atlas and writes their
    # embeddings back, inside this test's sandbox transaction.
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    start_world_test_services()
    setup_world_sandbox()

    :ok
  end

  describe "a rebuild of the world's vocabulary" do
    test "forgets every embedding cache of the world, each rebuilt with the new vocabulary on its next use" do
      {:ok, world} = create_test_world("embedding_caches_rebuild")
      trained_size = map_size(Embedder.read!(Embedder.trained_path(), :trained).vocabulary)
      intent = ranked_intent!()

      build_every_cache(world.id, intent)
      counts = cache_counts()

      # Episodes the rebuild builds its vocabulary from: every word occurs at
      # least twice, so the rebuilt vocabulary is far smaller than the seeded one.
      for text <- [
            "the weather in paris is sunny",
            "the weather in london is rainy",
            "is the weather in paris rainy",
            "is the weather in london sunny"
          ] do
        {:ok, _id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
      end

      assert {:ok, vocabulary_size} = Embedder.build_vocabulary(world.id, force: true)
      assert vocabulary_size != trained_size

      assert cache_counts() == %{
               fact_indexes: counts.fact_indexes - 1,
               chunk_pools: counts.chunk_pools - 1,
               template_embeddings: counts.template_embeddings - 1
             }

      build_every_cache(world.id, intent)
      assert cache_counts() == counts

      assert [%{type: :body, embedding: embedding}] = ChunkSegmenter.segment(world.id, @body_sentence)
      assert length(embedding) == vocabulary_size
    end
  end

  describe "destroying the world" do
    test "forgets every embedding cache of the world" do
      {:ok, world} = create_test_world("embedding_caches_destroy")
      intent = ranked_intent!()

      build_every_cache(world.id, intent)
      counts = cache_counts()

      :ok = Worlds.stop!(world.id)

      assert cache_counts() == %{
               fact_indexes: counts.fact_indexes - 1,
               chunk_pools: counts.chunk_pools - 1,
               template_embeddings: counts.template_embeddings - 1
             }

      # Give the destroyed world's id a vocabulary of another length: nothing
      # cached for the world before it was destroyed may be compared with it.
      on_exit(fn ->
        Embedder.clear(world.id)
        EmbeddingCaches.forget_world(world.id)
      end)

      :ok =
        Embedder.load_model(world.id, %{
          vocabulary: %{"weather" => 0, "sunny" => 1},
          idf_weights: %{"weather" => 1.0, "sunny" => 1.0}
        })

      build_every_cache(world.id, intent)
      assert cache_counts() == counts

      assert [%{type: :body, embedding: embedding}] = ChunkSegmenter.segment(world.id, @body_sentence)
      assert length(embedding) == 2
    end
  end

  describe "a use of the world's caches while its vocabulary is rebuilt" do
    test "waits for the rebuild, which the caches never hold up, and is answered in the rebuilt vocabulary" do
      {:ok, world} = create_test_world("embedding_caches_during_rebuild")
      intent = ranked_intent!()

      # Hold the rebuild while it reads the world's episodes: the world cannot
      # embed, and none of its caches is built yet.
      store = Process.whereis(Store)
      :ok = :sys.suspend(store)
      rebuild = Task.async(fn -> Embedder.build_vocabulary(world.id, force: true) end)

      eventually(
        fn -> Embedder.get_status(world.id).phase end,
        fn phase -> phase == :loading_episodes end
      )

      uses = [
        Task.async(fn -> SemanticFactRetriever.search(world.id, "What is the weather like?") end),
        Task.async(fn -> ChunkSegmenter.segment(world.id, @body_sentence) end),
        Task.async(fn -> TemplateBlender.blend(world.id, "Hello there", @context) end),
        Task.async(fn -> TemplateStore.get_best_template(world.id, intent, "hello", %{}) end)
      ]

      # Each is waiting for the rebuild, not answered without the vocabulary.
      for use <- uses, do: assert(Task.yield(use, 200) == nil)

      # The rebuild forgets every cache of the world (a call into each of them)
      # and finishes: no cache is held up by a caller waiting for it.
      :ok = :sys.resume(store)
      assert {:ok, :kept} = Task.await(rebuild)

      [facts, chunks, blended, template] = Enum.map(uses, &Task.await/1)
      assert is_list(facts)
      assert [%{type: :body, embedding: embedding}] = chunks
      assert {:ok, embedding} == Embedder.embed(world.id, @body_sentence)
      assert {:ok, _response} = blended
      assert {:ok, _text} = template
    end
  end

  # Uses every embedding cache in the world: each is built on its first use,
  # and each use compares a fresh query with what the cache holds.
  defp build_every_cache(world_id, intent) do
    assert is_list(SemanticFactRetriever.search(world_id, "What is the weather like?"))
    assert [%{type: :body}] = ChunkSegmenter.segment(world_id, @body_sentence)
    assert {:ok, _response} = TemplateBlender.blend(world_id, "Hello there", @context)
    assert {:ok, _text} = TemplateStore.get_best_template(world_id, intent, "hello", %{})
  end

  defp cache_counts do
    %{
      fact_indexes: SemanticFactRetriever.stats().worlds_indexed,
      chunk_pools: TemplateBlender.stats().worlds,
      template_embeddings: TemplateStore.stats().worlds_embedded
    }
  end

  # An intent with at least two templates that match any context, so ranking
  # by similarity in the world's embedder is what chooses between them.
  defp ranked_intent! do
    assert TemplateStore.ready?()
    assert TemplateBlender.ready?()

    intent =
      Enum.find(TemplateStore.list_intents(), fn intent ->
        intent
        |> TemplateStore.get_structured_templates()
        |> Enum.count(&(&1.condition in [nil, ""])) >= 2
      end)

    assert intent != nil, "templates.json holds no intent with two unconditional templates"
    intent
  end
end
