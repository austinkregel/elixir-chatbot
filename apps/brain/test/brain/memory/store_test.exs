defmodule Brain.Memory.StoreTest do
  alias Brain.Memory
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Memory.{Store, Embedder}
  alias Brain.Memory.Types.SemanticFact

  setup do
    ensure_pubsub_started()
    ensure_started(Embedder)

    texts = [
      "hello world",
      "goodbye world",
      "how are you",
      "what is the weather",
      "play some music"
    ]

    Embedder.build_vocabulary(texts)

    ensure_started(
      {Store, persistence_path: "/tmp/test_memory_store_#{:rand.uniform(100_000)}.term"}
    )

    Store.clear()

    :ok
  end

  describe "add_episode" do
    test "adds an episode and returns its id" do
      {:ok, id} = Store.add_episode("hello world", "smalltalk.greetings.hello", "hi there", ["smalltalk.greetings.hello"])

      assert is_binary(id)
      assert String.length(id) == 36
    end

    test "episode can be retrieved by id" do
      {:ok, id} = Store.add_episode("hello world", "smalltalk.greetings.hello", "hi there", ["smalltalk.greetings.hello"])

      {:ok, episode} = Store.get_episode(id)

      assert episode.state == "hello world"
      assert episode.action == "smalltalk.greetings.hello"
      assert episode.outcome == "hi there"
      assert episode.tags == ["smalltalk.greetings.hello"]
    end

    test "returns error for non-existent id" do
      assert {:error, :not_found} = Store.get_episode("nonexistent")
    end
  end

  describe "query_similar" do
    test "finds similar episodes" do
      {:ok, _} = Store.add_episode("hello world", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"])
      {:ok, _} = Store.add_episode("hello there", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"])
      {:ok, _} = Store.add_episode("goodbye world", "farewell", "", ["farewell"])

      {:ok, results} = Store.query_similar("hello friend", 3)

      assert length(results) == 3
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
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello", "casual"])
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", ["farewell"])
      {:ok, _} = Store.add_episode("hi there", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello", "formal"])

      {:ok, results} = Store.query_by_tags(["smalltalk.greetings.hello"])

      assert length(results) == 2
      assert Enum.all?(results, fn ep -> "smalltalk.greetings.hello" in ep.tags end)
    end

    test "returns empty list when no tags match" do
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"])

      {:ok, results} = Store.query_by_tags(["nonexistent"])
      assert results == []
    end
  end

  describe "semantic facts" do
    test "adds and retrieves semantic facts" do
      fact = SemanticFact.new("greeting pattern", [0.1, 0.2, 0.3], ["ep1"], ["smalltalk.greetings.hello"])
      {:ok, id} = Store.add_semantic(fact)

      {:ok, retrieved} = Store.get_semantic(id)

      assert retrieved.representation == "greeting pattern"
      assert retrieved.tags == ["smalltalk.greetings.hello"]
    end

    test "queries semantic facts by similarity" do
      {:ok, embedding} = Embedder.embed("hello world")

      fact = SemanticFact.new("greeting pattern", embedding, ["ep1"], ["smalltalk.greetings.hello"])
      {:ok, _} = Store.add_semantic(fact)

      {:ok, results} = Store.query_semantic("hello there", 5)

      assert length(results) == 1
      [{semantic, _sim}] = results
      assert semantic.representation == "greeting pattern"
    end
  end

  describe "link_episode_to_semantic" do
    test "links an episode to a semantic fact" do
      {:ok, ep_id} = Store.add_episode("hello", "smalltalk.greetings.hello", "", [])
      {:ok, episode} = Store.get_episode(ep_id)
      assert episode.semantic_id == nil

      sem_id = Ecto.UUID.generate()
      :ok = Store.link_episode_to_semantic(ep_id, sem_id)

      {:ok, updated} = Store.get_episode(ep_id)
      assert updated.semantic_id == sem_id
    end

    test "returns error for non-existent episode" do
      fake_id = Ecto.UUID.generate()
      assert {:error, :not_found} = Store.link_episode_to_semantic(fake_id, Ecto.UUID.generate())
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
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", [])
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", [])

      stats = Store.stats()

      assert stats.episode_count == 2
      assert stats.semantic_count == 0
      assert stats.episode_index_size == 2
    end
  end

  describe "clear" do
    test "removes all data" do
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", [])
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", [])

      :ok = Store.clear()

      {:ok, episodes} = Store.all_episodes()
      assert episodes == []
    end
  end
end
