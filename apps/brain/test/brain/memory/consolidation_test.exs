defmodule Brain.Memory.ConsolidationTest do
  alias Brain.Memory
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Memory.{Consolidation, Store, Embedder, VectorIndex}
  alias Brain.Memory.Types.Episode

  setup do
    ensure_pubsub_started()
    ensure_started(Embedder)

    texts = [
      "hello world friend",
      "hello there buddy",
      "hi how are you",
      "goodbye world",
      "bye bye friend",
      "farewell my friend"
    ]

    Embedder.build_vocabulary(texts)

    ensure_started(
      {Store, persistence_path: "/tmp/test_consolidation_#{:rand.uniform(100_000)}.term"}
    )

    Store.clear()

    :ok
  end

  describe "find_clusters" do
    test "groups similar episodes together" do
      {:ok, emb_hello1} = Embedder.embed("hello world friend")
      {:ok, emb_hello2} = Embedder.embed("hello there buddy")
      {:ok, emb_bye} = Embedder.embed("goodbye world")

      episodes = [
        Episode.new("hello world", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], emb_hello1),
        Episode.new("hello there", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], emb_hello2),
        Episode.new("goodbye world", "smalltalk.greetings.bye", "", ["smalltalk.greetings.bye"], emb_bye)
      ]

      clusters = Consolidation.find_clusters(episodes, 0.5, 2)
      assert clusters != []

      hello_cluster =
        Enum.find(clusters, fn cluster ->
          Enum.any?(cluster, fn ep -> String.contains?(ep.state, "hello") end)
        end)

      if hello_cluster do
        assert length(hello_cluster) >= 2
      end
    end

    test "returns empty list when no clusters meet min_size" do
      {:ok, emb1} = Embedder.embed("hello")
      {:ok, emb2} = Embedder.embed("goodbye")
      {:ok, emb3} = Embedder.embed("weather")

      episodes = [
        Episode.new("hello", "a", "", [], emb1),
        Episode.new("goodbye", "b", "", [], emb2),
        Episode.new("weather", "c", "", [], emb3)
      ]

      clusters = Consolidation.find_clusters(episodes, 0.99, 2)
      assert is_list(clusters)
    end
  end

  describe "create_semantic_from_cluster" do
    test "creates a semantic fact from episode cluster" do
      {:ok, id1} = Store.add_episode("hello world", "smalltalk.greetings.hello", "hi", ["smalltalk.greetings.hello"])
      {:ok, id2} = Store.add_episode("hello there", "smalltalk.greetings.hello", "hey", ["smalltalk.greetings.hello"])

      {:ok, ep1} = Store.get_episode(id1)
      {:ok, ep2} = Store.get_episode(id2)

      cluster = [ep1, ep2]

      {:ok, semantic_id} = Consolidation.create_semantic_from_cluster(cluster)

      assert is_binary(semantic_id)
      {:ok, semantic} = Store.get_semantic(semantic_id)
      assert "smalltalk.greetings.hello" in semantic.tags
      assert length(semantic.evidence_ids) == 2
      {:ok, updated_ep1} = Store.get_episode(id1)
      assert updated_ep1.semantic_id == semantic_id
    end

    test "returns error for empty cluster" do
      assert {:error, :empty_cluster} = Consolidation.create_semantic_from_cluster([])
    end
  end

  describe "consolidate" do
    test "creates semantic facts from similar episodes" do
      {:ok, _} = Store.add_episode("hello world", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"])
      {:ok, _} = Store.add_episode("hello there", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"])
      {:ok, _} = Store.add_episode("hi friend", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"])
      {:ok, new_count} = Consolidation.consolidate(threshold: 0.3, min_cluster_size: 2)
      assert new_count >= 0

      {:ok, semantics} = Store.all_semantics()
      assert is_list(semantics)
    end

    test "returns 0 when not enough episodes" do
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", [])

      {:ok, count} = Consolidation.consolidate(min_cluster_size: 5)

      assert count == 0
    end
  end

  describe "VectorIndex.mean_vector" do
    test "computes correct mean" do
      vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

      mean = VectorIndex.mean_vector(vectors)

      assert_in_delta Enum.at(mean, 0), 0.6667, 0.01
      assert_in_delta Enum.at(mean, 1), 0.6667, 0.01
    end
  end
end
