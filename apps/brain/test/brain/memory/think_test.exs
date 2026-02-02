defmodule Brain.Memory.ThinkTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Memory.{Think, Store, Embedder}

  setup do
    # PubSub is started globally in test_helper.exs
    ensure_pubsub_started()

    # Start the embedder under ExUnit supervision
    ensure_started(Embedder)

    texts = [
      "hello world",
      "goodbye world",
      "how are you",
      "what is the weather",
      "play some music"
    ]

    Embedder.build_vocabulary(texts)

    # Start the store under ExUnit supervision with unique path per test
    ensure_started({Store, persistence_path: "/tmp/test_think_#{:rand.uniform(100_000)}.term"})

    Store.clear()

    :ok
  end

  describe "think(:add_episode)" do
    test "adds an episode and returns the id" do
      {:ok, {:episode_added, id}} =
        Think.think(:add_episode, %{
          state: "hello world",
          action: "greeting",
          outcome: "hi there",
          tags: ["greeting"]
        })

      assert is_binary(id)
      assert String.length(id) == 32
    end

    test "works with minimal params" do
      {:ok, {:episode_added, id}} = Think.think(:add_episode, %{state: "hello"})

      assert is_binary(id)
    end
  end

  describe "think(:query_chat)" do
    test "queries for similar episodes" do
      # Add some episodes first
      Think.think(:add_episode, %{state: "hello world", action: "greeting", tags: ["greeting"]})
      Think.think(:add_episode, %{state: "goodbye world", action: "farewell", tags: ["farewell"]})

      {:ok, {:chat_results, results}} = Think.think(:query_chat, %{input: "hello friend", k: 5})

      assert is_list(results)
      # Results are {id, similarity} tuples
      if length(results) > 0 do
        [{id, sim} | _] = results
        assert is_binary(id)
        assert is_float(sim)
      end
    end

    test "returns empty results for no matches" do
      {:ok, {:chat_results, results}} = Think.think(:query_chat, %{input: "hello", k: 5})

      assert results == []
    end
  end

  describe "think(:query_semantic)" do
    test "queries semantic facts" do
      {:ok, {:semantic_results, results}} =
        Think.think(:query_semantic, %{input: "hello", k: 5})

      assert is_list(results)
    end
  end

  describe "think(:consolidate)" do
    test "consolidates episodes into semantic facts" do
      # Add similar episodes
      Think.think(:add_episode, %{state: "hello world", action: "greeting", tags: ["greeting"]})
      Think.think(:add_episode, %{state: "hello there", action: "greeting", tags: ["greeting"]})
      Think.think(:add_episode, %{state: "hi friend", action: "greeting", tags: ["greeting"]})

      {:ok, {:consolidated, count}} =
        Think.think(:consolidate, %{threshold: 0.3, min_size: 2})

      assert is_integer(count)
      assert count >= 0
    end
  end

  describe "think(:stats)" do
    test "returns store statistics" do
      Think.think(:add_episode, %{state: "hello", action: "greeting"})
      Think.think(:add_episode, %{state: "goodbye", action: "farewell"})

      {:ok, {:stats, stats}} = Think.think(:stats)

      assert is_map(stats)
      assert stats.episode_count == 2
    end
  end

  describe "think(:clear)" do
    test "clears all memory" do
      Think.think(:add_episode, %{state: "hello", action: "greeting"})

      {:ok, :cleared} = Think.think(:clear)

      {:ok, {:stats, stats}} = Think.think(:stats)
      assert stats.episode_count == 0
    end
  end

  describe "think(:persist)" do
    test "persists memory to disk" do
      Think.think(:add_episode, %{state: "hello", action: "greeting"})

      result = Think.think(:persist)

      assert result == {:ok, :persisted}
    end
  end
end
