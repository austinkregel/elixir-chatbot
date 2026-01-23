defmodule ChatBot.Memory.StoreTest do
  use ExUnit.Case, async: false

  alias ChatBot.Memory.{Store, Embedder}
  alias ChatBot.Memory.Types.SemanticFact

  setup do
    # Start embedder and build vocabulary
    case Process.whereis(Embedder) do
      nil ->
        {:ok, _} = Embedder.start_link()

      pid ->
        GenServer.stop(pid)
        {:ok, _} = Embedder.start_link()
    end

    texts = [
      "hello world",
      "goodbye world",
      "how are you",
      "what is the weather",
      "play some music"
    ]

    Embedder.build_vocabulary(texts)

    # Start or restart store with temp persistence path
    case Process.whereis(Store) do
      nil ->
        Store.start_link(
          persistence_path: "/tmp/test_memory_store_#{:rand.uniform(100_000)}.term"
        )

      pid ->
        GenServer.stop(pid)

        Store.start_link(
          persistence_path: "/tmp/test_memory_store_#{:rand.uniform(100_000)}.term"
        )
    end

    # Clear any existing data
    Store.clear()

    :ok
  end

  describe "add_episode" do
    test "adds an episode and returns its id" do
      {:ok, id} = Store.add_episode("hello world", "greeting", "hi there", ["greeting"])

      assert is_binary(id)
      assert String.length(id) == 32
    end

    test "episode can be retrieved by id" do
      {:ok, id} = Store.add_episode("hello world", "greeting", "hi there", ["greeting"])

      {:ok, episode} = Store.get_episode(id)

      assert episode.state == "hello world"
      assert episode.action == "greeting"
      assert episode.outcome == "hi there"
      assert episode.tags == ["greeting"]
    end

    test "returns error for non-existent id" do
      assert {:error, :not_found} = Store.get_episode("nonexistent")
    end
  end

  describe "query_similar" do
    test "finds similar episodes" do
      {:ok, _} = Store.add_episode("hello world", "greeting", "", ["greeting"])
      {:ok, _} = Store.add_episode("hello there", "greeting", "", ["greeting"])
      {:ok, _} = Store.add_episode("goodbye world", "farewell", "", ["farewell"])

      {:ok, results} = Store.query_similar("hello friend", 3)

      assert length(results) == 3
      # Results should be tuples of {episode, similarity}
      [{ep1, sim1} | _] = results
      assert is_struct(ep1)
      assert is_float(sim1)
    end

    test "returns empty list when no episodes exist" do
      {:ok, results} = Store.query_similar("hello", 5)
      assert results == []
    end
  end

  describe "query_by_tags" do
    test "finds episodes with matching tags" do
      {:ok, _} = Store.add_episode("hello", "greeting", "", ["greeting", "casual"])
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", ["farewell"])
      {:ok, _} = Store.add_episode("hi there", "greeting", "", ["greeting", "formal"])

      {:ok, results} = Store.query_by_tags(["greeting"])

      assert length(results) == 2
      assert Enum.all?(results, fn ep -> "greeting" in ep.tags end)
    end

    test "returns empty list when no tags match" do
      {:ok, _} = Store.add_episode("hello", "greeting", "", ["greeting"])

      {:ok, results} = Store.query_by_tags(["nonexistent"])
      assert results == []
    end
  end

  describe "semantic facts" do
    test "adds and retrieves semantic facts" do
      fact = SemanticFact.new("greeting pattern", [0.1, 0.2, 0.3], ["ep1"], ["greeting"])
      {:ok, id} = Store.add_semantic(fact)

      {:ok, retrieved} = Store.get_semantic(id)

      assert retrieved.representation == "greeting pattern"
      assert retrieved.tags == ["greeting"]
    end

    test "queries semantic facts by similarity" do
      # Need matching embedding size
      {:ok, embedding} = Embedder.embed("hello world")

      fact = SemanticFact.new("greeting pattern", embedding, ["ep1"], ["greeting"])
      {:ok, _} = Store.add_semantic(fact)

      {:ok, results} = Store.query_semantic("hello there", 5)

      assert length(results) == 1
      [{semantic, _sim}] = results
      assert semantic.representation == "greeting pattern"
    end
  end

  describe "link_episode_to_semantic" do
    test "links an episode to a semantic fact" do
      {:ok, ep_id} = Store.add_episode("hello", "greeting", "", [])
      {:ok, episode} = Store.get_episode(ep_id)
      assert episode.semantic_id == nil

      :ok = Store.link_episode_to_semantic(ep_id, "sem_123")

      {:ok, updated} = Store.get_episode(ep_id)
      assert updated.semantic_id == "sem_123"
    end

    test "returns error for non-existent episode" do
      assert {:error, :not_found} = Store.link_episode_to_semantic("fake", "sem")
    end
  end

  describe "all_episodes and all_semantics" do
    test "returns all episodes" do
      {:ok, _} = Store.add_episode("a", "b", "c", [])
      {:ok, _} = Store.add_episode("d", "e", "f", [])

      {:ok, episodes} = Store.all_episodes()

      assert length(episodes) == 2
    end

    test "returns all semantic facts" do
      {:ok, embedding} = Embedder.embed("hello")

      fact1 = SemanticFact.new("fact1", embedding, [], [])
      fact2 = SemanticFact.new("fact2", embedding, [], [])
      {:ok, _} = Store.add_semantic(fact1)
      {:ok, _} = Store.add_semantic(fact2)

      {:ok, semantics} = Store.all_semantics()

      assert length(semantics) == 2
    end
  end

  describe "stats" do
    test "returns store statistics" do
      {:ok, _} = Store.add_episode("hello", "greeting", "", [])
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", [])

      stats = Store.stats()

      assert stats.episode_count == 2
      assert stats.semantic_count == 0
      assert stats.episode_index_size == 2
    end
  end

  describe "clear" do
    test "removes all data" do
      {:ok, _} = Store.add_episode("hello", "greeting", "", [])
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", [])

      :ok = Store.clear()

      {:ok, episodes} = Store.all_episodes()
      assert episodes == []
    end
  end
end
