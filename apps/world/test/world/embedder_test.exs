defmodule World.EmbedderTest do
  @moduledoc """
  A world embeds with its own copy of the trained embedder vocabulary, seeded
  when `World.Manager` creates it. A world with no copy raises naming the
  world: there is no fallback to the trained vocabulary or to another world.
  """
  # A rebuild reads the world's episodes from Atlas and writes their
  # embeddings back, inside this test's sandbox transaction (`World.AtlasCase`).
  use World.AtlasCase, async: false
  import Brain.TestHelpers

  alias Brain.Memory.Store
  alias Brain.Memory.Types.SemanticFact
  alias Brain.Memory.VectorIndex
  alias World.Embedder
  alias World.Manager, as: WorldManager

  setup do
    start_world_test_services()
    setup_world_sandbox()

    eventually(
      fn -> WorldManager.ready?() end,
      fn ready -> ready end,
      50,
      20
    )

    :ok
  end

  # A world id with nothing loaded for it and no directory on disk.
  defp unseeded_world_id, do: "test_unseeded_embedder_#{System.unique_integer([:positive])}"

  describe "get_status/1" do
    test "reports a world with nothing loaded as not initialized" do
      assert Embedder.get_status(unseeded_world_id()) == %{
               ready: false,
               phase: :not_initialized,
               phase_label: "Not initialized",
               vocabulary_size: 0,
               built_at: nil,
               episode_count: 0
             }
    end
  end

  describe "embed/2" do
    test "raises naming a world that has no embedder copy" do
      world_id = unseeded_world_id()

      assert_raise RuntimeError, ~r/world "#{world_id}" has no embedder vocabulary/, fn ->
        Embedder.embed(world_id, "hello")
      end
    end

    test "embeds in a world created with its seeded copy" do
      {:ok, world} = create_test_world("embedder_seeded")

      trained = Embedder.read!(Embedder.trained_path(), :trained)
      assert map_size(trained.vocabulary) > 0

      assert {:ok, embedding} = Embedder.embed(world.id, "hello there, how is the weather")
      assert length(embedding) == map_size(trained.vocabulary)
      assert Enum.all?(embedding, &is_float/1)

      assert Embedder.ready?(world.id)
      assert Embedder.get_status(world.id).vocabulary_size == map_size(trained.vocabulary)
    end

    test "raises once the world that held the copy is destroyed" do
      {:ok, world} = create_test_world("embedder_destroyed")
      assert {:ok, _embedding} = Embedder.embed(world.id, "hello")

      :ok = WorldManager.destroy(world.id)

      assert_raise RuntimeError, ~r/world "#{world.id}" has no embedder vocabulary/, fn ->
        Embedder.embed(world.id, "hello")
      end
    end
  end

  describe "build_vocabulary/2 on a world with no episodes" do
    @text "hello there, how is the weather"

    test "keeps the seeded vocabulary loaded for the world" do
      {:ok, world} = create_test_world("embedder_empty_rebuild")
      assert {:ok, seeded_embedding} = Embedder.embed(world.id, @text)

      assert {:ok, :kept} = Embedder.build_vocabulary(world.id, force: true)

      assert Embedder.ready?(world.id)
      assert Embedder.embed(world.id, @text) == {:ok, seeded_embedding}
    end

    test "keeps the seeded copy of a world whose vocabulary was never loaded" do
      {:ok, world} = create_test_world("embedder_empty_rebuild_unloaded", mode: :persistent)
      assert Embedder.get_status(world.id).phase == :not_initialized
      trained = Embedder.read!(Embedder.trained_path(), :trained)

      assert {:ok, :kept} = Embedder.build_vocabulary(world.id)

      assert Embedder.ready?(world.id)
      assert Embedder.get_status(world.id).vocabulary_size == map_size(trained.vocabulary)
      assert {:ok, embedding} = Embedder.embed(world.id, @text)
      assert length(embedding) == map_size(trained.vocabulary)
    end
  end

  describe "an episode written while the vocabulary is rebuilt" do
    test "is stored, then embedded and indexed when the rebuild finishes" do
      {:ok, world} = create_test_world("embedder_rebuild_queue")

      # Episodes the rebuild builds its vocabulary from: every word occurs at
      # least twice, so each one is in the rebuilt vocabulary.
      for text <- [
            "the weather in paris is sunny",
            "the weather in london is rainy",
            "is the weather in paris rainy",
            "is the weather in london sunny"
          ] do
        {:ok, _id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
      end

      queued_text = "rainy weather in paris"
      store = Process.whereis(Store)

      # Hold the store so the episode's write is waiting in its mailbox ahead
      # of the rebuild's read of the world's episodes: the store then handles
      # the write while the vocabulary is being rebuilt.
      :ok = :sys.suspend(store)

      write =
        Task.async(fn ->
          Store.add_episode(queued_text, "weather", "", [], world_id: world.id)
        end)

      eventually(
        fn -> store |> Process.info(:messages) |> elem(1) end,
        fn messages ->
          Enum.any?(messages, &match?({:"$gen_call", _, {:add_episode, ^queued_text, _, _, _, _}}, &1))
        end
      )

      rebuild = Task.async(fn -> Embedder.build_vocabulary(world.id, force: true) end)

      eventually(
        fn -> Embedder.get_status(world.id).phase end,
        fn phase -> phase == :loading_episodes end
      )

      :ok = :sys.resume(store)

      assert {:ok, queued_id} = Task.await(write)
      assert {:ok, vocabulary_size} = Task.await(rebuild)
      assert vocabulary_size > 0

      {:ok, episode} = Store.get_episode(queued_id, world_id: world.id)
      assert length(episode.embedding) == vocabulary_size
      assert {:ok, episode.embedding} == Embedder.embed(world.id, queued_text)

      {:ok, results} = Store.query_similar(queued_text, 5, world_id: world.id, rerank: false)
      assert [{%{id: ^queued_id}, _similarity} | _] = results
    end
  end

  describe "an episode or semantic fact stored before a rebuild" do
    test "is re-embedded with the rebuilt vocabulary and found after it" do
      {:ok, world} = create_test_world("embedder_rebuild_reembeds")
      trained_size = map_size(Embedder.read!(Embedder.trained_path(), :trained).vocabulary)

      # Every word occurs at least twice, so each one is in the rebuilt
      # vocabulary, which is far smaller than the seeded one.
      [paris_sunny, _london_rainy, paris_rainy, _london_sunny] =
        texts = [
          "the weather in paris is sunny",
          "the weather in london is rainy",
          "is the weather in paris rainy",
          "is the weather in london sunny"
        ]

      [paris_sunny_id, _, paris_rainy_id, _] =
        ids =
        Enum.map(texts, fn text ->
          {:ok, id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
          id
        end)

      episodes = Enum.map(ids, fn id -> elem(Store.get_episode(id, world_id: world.id), 1) end)
      assert Enum.all?(episodes, &(length(&1.embedding) == trained_size))

      paris = Enum.filter(episodes, &(&1.id in [paris_sunny_id, paris_rainy_id]))

      fact =
        SemanticFact.new(
          "the weather in paris",
          VectorIndex.mean_vector(Enum.map(paris, & &1.embedding)),
          [paris_sunny_id, paris_rainy_id],
          ["weather"]
        )

      {:ok, fact_id} = Store.add_semantic(fact, world_id: world.id)

      assert {:ok, vocabulary_size} = Embedder.build_vocabulary(world.id, force: true)
      assert vocabulary_size != trained_size

      for {id, text} <- Enum.zip(ids, texts) do
        {:ok, episode} = Store.get_episode(id, world_id: world.id)
        assert {:ok, episode.embedding} == Embedder.embed(world.id, text)
      end

      # Each episode is found by its own text: embedded with the rebuilt
      # vocabulary, the query and the stored embedding are the same vector.
      for {id, text} <- [{paris_sunny_id, paris_sunny}, {paris_rainy_id, paris_rainy}] do
        assert {:ok, [{%{id: ^id}, similarity}]} = Store.query_similar(text, 1, world_id: world.id, rerank: false)
        assert_in_delta similarity, 1.0, 1.0e-9
      end

      # The fact is the mean of its evidence episodes' rebuilt embeddings.
      rebuilt_paris =
        Enum.map([paris_sunny_id, paris_rainy_id], fn id ->
          {:ok, episode} = Store.get_episode(id, world_id: world.id)
          episode.embedding
        end)

      {:ok, stored_fact} = Store.get_semantic(fact_id, world_id: world.id)
      assert stored_fact.embedding == VectorIndex.mean_vector(rebuilt_paris)
      assert length(stored_fact.embedding) == vocabulary_size

      assert {:ok, [{%{id: ^fact_id}, fact_similarity}]} =
               Store.query_semantic("the weather in paris", 1, world_id: world.id, rerank: false)

      assert fact_similarity > 0.0
    end

    test "a fact with no evidence episodes is re-embedded from its own text and found by it" do
      {:ok, world} = create_test_world("embedder_rebuild_textual_fact")

      for text <- ["the weather in paris is sunny", "is the weather in paris sunny"] do
        {:ok, _id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
      end

      {:ok, fact_id} = Store.add_semantic(SemanticFact.new("the weather in paris", [], [], []), world_id: world.id)

      assert {:ok, vocabulary_size} = Embedder.build_vocabulary(world.id, force: true)

      {:ok, stored_fact} = Store.get_semantic(fact_id, world_id: world.id)
      assert length(stored_fact.embedding) == vocabulary_size
      assert {:ok, stored_fact.embedding} == Embedder.embed(world.id, "the weather in paris")

      assert {:ok, [{%{id: ^fact_id}, similarity}]} =
               Store.query_semantic("the weather in paris", 1, world_id: world.id, rerank: false)

      assert_in_delta similarity, 1.0, 1.0e-9
    end

    test "a fact whose evidence episodes are not in the world stops the rebuild, naming it" do
      {:ok, world} = create_test_world("embedder_rebuild_missing_evidence")

      for text <- ["the weather in paris is sunny", "is the weather in paris sunny"] do
        {:ok, _id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
      end

      {:ok, embedding} = Embedder.embed(world.id, "the weather in paris")
      missing_id = Ecto.UUID.generate()
      fact = SemanticFact.new("the weather in paris", embedding, [missing_id], [])
      {:ok, fact_id} = Store.add_semantic(fact, world_id: world.id)

      error = assert_raise RuntimeError, fn -> Embedder.build_vocabulary(world.id, force: true) end
      assert error.message =~ inspect({:semantic_evidence_missing, fact_id, [missing_id]})
    end
  end

  describe "await_ready/2" do
    test "returns at once for a world whose embedder is ready" do
      {:ok, world} = create_test_world("embedder_await_ready")
      assert {:ok, _embedding} = Embedder.embed(world.id, "hello")

      assert :ok = Embedder.await_ready(world.id, 0)
    end

    test "a caller waiting for a rebuild proceeds when the rebuild finishes" do
      {:ok, world} = create_test_world("embedder_await_rebuild")
      assert {:ok, embedding} = Embedder.embed(world.id, "hello there")

      resume = hold_rebuild!(world.id)
      assert {:error, :vocabulary_building} = Embedder.embed(world.id, "hello there")

      waiter = Task.async(fn -> Embedder.await_ready(world.id, 5_000) end)
      assert Task.yield(waiter, 200) == nil

      rebuild = resume.()

      assert {:ok, :kept} = Task.await(rebuild)
      assert :ok = Task.await(waiter)
      assert Embedder.embed(world.id, "hello there") == {:ok, embedding}
    end

    test "raises naming the world and the timeout when the rebuild does not finish in time" do
      {:ok, world} = create_test_world("embedder_await_timeout")
      assert {:ok, _embedding} = Embedder.embed(world.id, "hello")

      resume = hold_rebuild!(world.id)

      error = assert_raise RuntimeError, fn -> Embedder.await_ready(world.id, 50) end
      assert error.message =~ "world #{inspect(world.id)}'s vocabulary rebuild did not finish within 50 ms"
      assert error.message =~ "Loading episodes"

      assert {:ok, :kept} = Task.await(resume.())
    end

    test "raises with the failure when the rebuild it waits for fails" do
      {:ok, world} = create_test_world("embedder_await_failed")

      for text <- ["the weather in paris is sunny", "is the weather in paris sunny"] do
        {:ok, _id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
      end

      {:ok, embedding} = Embedder.embed(world.id, "the weather in paris")
      missing_id = Ecto.UUID.generate()
      fact = SemanticFact.new("the weather in paris", embedding, [missing_id], [])
      {:ok, _fact_id} = Store.add_semantic(fact, world_id: world.id)

      resume = hold_rebuild!(world.id, fn -> catch_error(Embedder.build_vocabulary(world.id, force: true)) end)
      waiter = Task.async(fn -> catch_error(Embedder.await_ready(world.id, 5_000)) end)
      assert Task.yield(waiter, 200) == nil

      assert %RuntimeError{} = Task.await(resume.())
      assert %RuntimeError{message: message} = Task.await(waiter)
      assert message =~ "world #{inspect(world.id)}'s vocabulary rebuild failed"
      assert message =~ "semantic_evidence_missing"

      assert Embedder.get_status(world.id).phase == :rebuild_failed
      assert Embedder.embed(world.id, "the weather in paris") == {:error, :rebuild_failed}
      assert_raise RuntimeError, ~r/last vocabulary rebuild failed/, fn -> Embedder.await_ready(world.id, 0) end
    end
  end

  describe "Brain.Knowledge.Corroborator during a rebuild" do
    test "waits for the rebuild and compares in the rebuilt vocabulary, never by token overlap" do
      {:ok, world} = create_test_world("embedder_corroborator_waits")

      # The world's vocabulary knows only "paris", so both claims embed as the
      # same vector (similarity 1.0); their token overlap ({paris, capital}
      # against {paris, city}) would be 1/3.
      :ok = Embedder.load_model(world.id, %{vocabulary: %{"paris" => 0}, idf_weights: %{"paris" => 1.0}})

      resume = hold_rebuild!(world.id)

      compare = Task.async(fn -> Brain.Knowledge.Corroborator.compare_claims(world.id, "Paris capital", "Paris city") end)
      assert Task.yield(compare, 200) == nil

      assert {:ok, :kept} = Task.await(resume.())
      assert {:ok, 1.0} = Task.await(compare)
    end
  end

  describe "Brain.Memory.Store queries during a rebuild" do
    test "wait for the rebuild and search the world's memory re-embedded with the rebuilt vocabulary" do
      {:ok, world} = create_test_world("embedder_store_query_waits")

      # Every word occurs at least twice, so each one is in the rebuilt
      # vocabulary.
      [paris_sunny, _paris_rainy] = texts = ["the weather in paris is sunny", "is the weather in paris rainy"]

      [paris_sunny_id, _] =
        ids =
        Enum.map(texts, fn text ->
          {:ok, id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
          id
        end)

      episodes = Enum.map(ids, fn id -> elem(Store.get_episode(id, world_id: world.id), 1) end)
      fact = SemanticFact.new("the weather in paris", VectorIndex.mean_vector(Enum.map(episodes, & &1.embedding)), ids, [])
      {:ok, fact_id} = Store.add_semantic(fact, world_id: world.id)

      resume = hold_rebuild!(world.id)

      similar = Task.async(fn -> Store.query_similar(paris_sunny, 1, world_id: world.id, rerank: false) end)
      semantic = Task.async(fn -> Store.query_semantic("the weather in paris", 1, world_id: world.id, rerank: false) end)
      assert Task.yield(similar, 200) == nil
      assert Task.yield(semantic, 0) == nil

      assert {:ok, vocabulary_size} = Task.await(resume.())

      assert {:ok, [{%{id: ^paris_sunny_id} = episode, similarity}]} = Task.await(similar)
      assert length(episode.embedding) == vocabulary_size
      assert_in_delta similarity, 1.0, 1.0e-9

      assert {:ok, [{%{id: ^fact_id} = stored_fact, fact_similarity}]} = Task.await(semantic)
      assert length(stored_fact.embedding) == vocabulary_size
      assert fact_similarity > 0.0
    end

    test "raise with the failure when the rebuild they wait for fails" do
      {:ok, world} = create_test_world("embedder_store_query_failed")

      for text <- ["the weather in paris is sunny", "is the weather in paris sunny"] do
        {:ok, _id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
      end

      {:ok, embedding} = Embedder.embed(world.id, "the weather in paris")
      fact = SemanticFact.new("the weather in paris", embedding, [Ecto.UUID.generate()], [])
      {:ok, _fact_id} = Store.add_semantic(fact, world_id: world.id)

      resume = hold_rebuild!(world.id, fn -> catch_error(Embedder.build_vocabulary(world.id, force: true)) end)

      query =
        Task.async(fn ->
          catch_error(Store.query_similar("the weather in paris", 1, world_id: world.id, rerank: false))
        end)

      assert Task.yield(query, 200) == nil

      assert %RuntimeError{} = Task.await(resume.())
      assert %RuntimeError{message: message} = Task.await(query)
      assert message =~ "world #{inspect(world.id)}'s vocabulary rebuild failed"
    end
  end

  describe "the speech-act memory voter during a rebuild" do
    alias Brain.Analysis.SpeechActClassifier

    test "waits for the rebuild, however long it takes, and votes from the world's memory" do
      {:ok, world} = create_test_world("embedder_speech_act_waits")

      for text <- ["is the weather in paris sunny", "is the weather in paris rainy"] do
        {:ok, _id} = Store.add_episode(text, "weather", "", ["question"], world_id: world.id)
      end

      resume = hold_rebuild!(world.id)

      analysis = Task.async(fn -> SpeechActClassifier.analyze("is the weather in paris sunny", world_id: world.id) end)

      # Longer than the half second the vote was once given before it was
      # dropped.
      assert Task.yield(analysis, 800) == nil

      assert {:ok, _vocabulary_size} = Task.await(resume.())

      assert %{memory_analysis: %{similar_count: count, category: category}} = Task.await(analysis)
      assert count > 0
      assert category != nil
    end

    test "raises with the failure when the rebuild it waits for fails" do
      {:ok, world} = create_test_world("embedder_speech_act_failed")

      for text <- ["the weather in paris is sunny", "is the weather in paris sunny"] do
        {:ok, _id} = Store.add_episode(text, "weather", "", [], world_id: world.id)
      end

      {:ok, embedding} = Embedder.embed(world.id, "the weather in paris")
      fact = SemanticFact.new("the weather in paris", embedding, [Ecto.UUID.generate()], [])
      {:ok, _fact_id} = Store.add_semantic(fact, world_id: world.id)

      resume = hold_rebuild!(world.id, fn -> catch_error(Embedder.build_vocabulary(world.id, force: true)) end)

      analysis =
        Task.async(fn ->
          catch_error(SpeechActClassifier.analyze("is the weather in paris sunny", world_id: world.id))
        end)

      assert Task.yield(analysis, 200) == nil

      assert %RuntimeError{} = Task.await(resume.())
      assert %RuntimeError{message: message} = Task.await(analysis)
      assert message =~ "world #{inspect(world.id)}'s vocabulary rebuild failed"
    end
  end

  # Starts a rebuild of the world's vocabulary and holds it while it reads the
  # world's episodes, with the store suspended. Returns a function that lets
  # the rebuild go on and returns its task.
  defp hold_rebuild!(world_id, rebuild \\ nil) do
    rebuild = rebuild || fn -> Embedder.build_vocabulary(world_id, force: true) end
    store = Process.whereis(Store)
    :ok = :sys.suspend(store)
    task = Task.async(rebuild)

    eventually(
      fn -> Embedder.get_status(world_id).phase end,
      fn phase -> phase == :loading_episodes end
    )

    fn ->
      :ok = :sys.resume(store)
      task
    end
  end

  describe "cosine_similarity/2" do
    test "is 1.0 for a vector with itself" do
      assert_in_delta Embedder.cosine_similarity([0.6, 0.8], [0.6, 0.8]), 1.0, 1.0e-9
    end

    test "is 0.0 for orthogonal vectors" do
      assert Embedder.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    end

    test "is 0.0 for vectors of different lengths or empty vectors" do
      assert Embedder.cosine_similarity([1.0, 0.0], [1.0]) == 0.0
      assert Embedder.cosine_similarity([], []) == 0.0
    end
  end
end
