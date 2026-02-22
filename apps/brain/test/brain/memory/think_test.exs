defmodule Brain.Memory.ThinkTest do
  alias Brain.Memory
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Memory.{Think, Store, Embedder}

  setup _context do
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
    ensure_started({Store, persistence_path: "/tmp/test_think_#{:rand.uniform(100_000)}.term"})

    Store.clear()

    :ok
  end

  describe "think(:add_episode)" do
    test "adds an episode and returns the id" do
      {:ok, {:episode_added, id}} =
        Think.think(:add_episode, %{
          state: "hello world",
          action: "smalltalk.greetings.hello",
          outcome: "hi there",
          tags: ["smalltalk.greetings.hello"]
        })

      assert is_binary(id)
      assert String.length(id) == 36
    end

    test "works with minimal params" do
      {:ok, {:episode_added, id}} = Think.think(:add_episode, %{state: "hello"})

      assert is_binary(id)
    end
  end

  describe "think(:query_chat)" do
    test "queries for similar episodes" do
      Think.think(:add_episode, %{state: "hello world", action: "smalltalk.greetings.hello", tags: ["smalltalk.greetings.hello"]})
      Think.think(:add_episode, %{state: "goodbye world", action: "smalltalk.greetings.bye", tags: ["smalltalk.greetings.bye"]})

      {:ok, {:chat_results, results}} = Think.think(:query_chat, %{input: "hello friend", k: 5})

      assert is_list(results)

      if results != [] do
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
      Think.think(:add_episode, %{state: "hello world", action: "smalltalk.greetings.hello", tags: ["smalltalk.greetings.hello"]})
      Think.think(:add_episode, %{state: "hello there", action: "smalltalk.greetings.hello", tags: ["smalltalk.greetings.hello"]})
      Think.think(:add_episode, %{state: "hi friend", action: "smalltalk.greetings.hello", tags: ["smalltalk.greetings.hello"]})

      {:ok, {:consolidated, count}} =
        Think.think(:consolidate, %{threshold: 0.3, min_size: 2})

      assert is_integer(count)
      assert count >= 0
    end
  end

  describe "think(:stats)" do
    test "returns store statistics" do
      Think.think(:add_episode, %{state: "hello", action: "smalltalk.greetings.hello"})
      Think.think(:add_episode, %{state: "goodbye", action: "smalltalk.greetings.bye"})

      {:ok, {:stats, stats}} = Think.think(:stats)

      assert is_map(stats)
      assert stats.episode_count == 2
    end
  end

  describe "think(:clear)" do
    test "clears all memory" do
      Think.think(:add_episode, %{state: "hello", action: "smalltalk.greetings.hello"})

      {:ok, :cleared} = Think.think(:clear)

      {:ok, {:stats, stats}} = Think.think(:stats)
      assert stats.episode_count == 0
    end
  end

  describe "think(:persist)" do
    test "persists memory to disk" do
      Think.think(:add_episode, %{state: "hello", action: "smalltalk.greetings.hello"})

      result = Think.think(:persist)

      assert result == {:ok, :persisted}
    end
  end
end
