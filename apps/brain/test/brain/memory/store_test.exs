defmodule Brain.Memory.StoreTest do
  alias Brain.Memory
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  # World.Embedder is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Embedder}

  alias Memory.Store
  alias Brain.Memory.Types.{Episode, SemanticFact}
  alias World.Embedder, as: WorldEmbedder

  # The world the test suite runs in, seeded by `mix test.models`. The store
  # embeds with this world's embedder, so embeddings a test builds by hand come
  # from the same one.
  @world_id "default"

  setup do
    ensure_pubsub_started()

    ensure_started(
      {Store, persistence_path: "/tmp/test_memory_store_#{:rand.uniform(100_000)}.term"}
    )

    Store.clear()

    # The database rows roll back with the test, but the store's index and
    # record cache outlive it: a fact left there names evidence episodes no
    # world holds, and recalling it for a later turn raises
    # (`Brain.Memory.RecurringPattern.find/2`). Runs before the sandbox owner
    # stops, so the store can still reach Atlas.
    on_exit(fn -> Store.clear() end)

    :ok
  end

  describe "add_episode" do
    test "adds an episode and returns its id" do
      {:ok, id} = Store.add_episode("hello world", "smalltalk.greetings.hello", "hi there", ["smalltalk.greetings.hello"], world_id: @world_id)

      assert is_binary(id)
      assert String.length(id) == 36
    end

    test "episode can be retrieved by id" do
      {:ok, id} = Store.add_episode("hello world", "smalltalk.greetings.hello", "hi there", ["smalltalk.greetings.hello"], world_id: @world_id)

      {:ok, episode} = Store.get_episode(id, world_id: @world_id)

      assert episode.state == "hello world"
      assert episode.action == "smalltalk.greetings.hello"
      assert episode.outcome == "hi there"
      assert episode.tags == ["smalltalk.greetings.hello"]
    end

    test "returns error for non-existent id" do
      assert {:error, :not_found} = Store.get_episode("nonexistent", world_id: @world_id)
    end
  end

  describe "query_similar" do
    test "finds similar episodes" do
      {:ok, _} = Store.add_episode("hello world", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], world_id: @world_id)
      {:ok, _} = Store.add_episode("hello there", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], world_id: @world_id)
      {:ok, _} = Store.add_episode("goodbye world", "farewell", "", ["farewell"], world_id: @world_id)

      {:ok, results} = Store.query_similar("hello friend", 3, world_id: @world_id)

      assert length(results) == 3
      [{ep1, sim1} | _] = results
      assert is_struct(ep1)
      assert is_float(sim1)
    end

    test "returns empty list when no episodes exist" do
      {:ok, results} = Store.query_similar("hello", 5, world_id: @world_id)
      assert results == []
    end
  end

  describe "a query in one world beside another world's similar memories" do
    setup do
      ensure_started(World.Manager)
      setup_world_sandbox()
    end

    @query "what is the weather in paris today"

    test "query_similar still returns the world's own episode" do
      {:ok, world} = create_test_world("store_scope_own")
      {:ok, crowd} = create_test_world("store_scope_crowd")

      # The other world's episodes match the query exactly, so each ranks
      # above this world's own, which shares only part of it.
      for _ <- 1..20 do
        {:ok, _} = Store.add_episode(@query, "weather", "", [], world_id: crowd.id)
      end

      {:ok, own_id} = Store.add_episode("the weather in paris", "weather", "", [], world_id: world.id)

      assert {:ok, [{_crowd_episode, crowd_similarity}]} =
               Store.query_similar(@query, 1, world_id: crowd.id, rerank: false)

      assert {:ok, [{%{id: ^own_id}, own_similarity}]} =
               Store.query_similar(@query, 1, world_id: world.id, rerank: false)

      assert crowd_similarity > own_similarity
    end

    test "query_semantic still returns the world's own fact" do
      {:ok, world} = create_test_world("store_scope_own_fact")
      {:ok, crowd} = create_test_world("store_scope_crowd_fact")

      {:ok, crowd_embedding} = WorldEmbedder.embed(crowd.id, @query)

      for _ <- 1..20 do
        {:ok, _} = Store.add_semantic(SemanticFact.new(@query, crowd_embedding, [], []), world_id: crowd.id)
      end

      {:ok, own_embedding} = WorldEmbedder.embed(world.id, "the weather in paris")
      own = SemanticFact.new("the weather in paris", own_embedding, [], [])
      {:ok, own_id} = Store.add_semantic(own, world_id: world.id)

      assert {:ok, [{_crowd_fact, crowd_similarity}]} =
               Store.query_semantic(@query, 1, world_id: crowd.id, rerank: false)

      assert {:ok, [{%{id: ^own_id}, own_similarity}]} =
               Store.query_semantic(@query, 1, world_id: world.id, rerank: false)

      assert crowd_similarity > own_similarity
    end
  end

  describe "query_by_tags" do
    test "finds episodes with matching tags" do
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello", "casual"], world_id: @world_id)
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", ["farewell"], world_id: @world_id)
      {:ok, _} = Store.add_episode("hi there", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello", "formal"], world_id: @world_id)

      {:ok, results} = Store.query_by_tags(["smalltalk.greetings.hello"], world_id: @world_id)

      assert length(results) == 2
      assert Enum.all?(results, fn ep -> "smalltalk.greetings.hello" in ep.tags end)
    end

    test "returns empty list when no tags match" do
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], world_id: @world_id)

      {:ok, results} = Store.query_by_tags(["nonexistent"], world_id: @world_id)
      assert results == []
    end
  end

  describe "semantic facts" do
    test "adds and retrieves semantic facts" do
      fact = SemanticFact.new("greeting pattern", [0.1, 0.2, 0.3], ["ep1"], ["smalltalk.greetings.hello"])
      {:ok, id} = Store.add_semantic(fact, world_id: @world_id)

      {:ok, retrieved} = Store.get_semantic(id, world_id: @world_id)

      assert retrieved.representation == "greeting pattern"
      assert retrieved.tags == ["smalltalk.greetings.hello"]
    end

    test "queries semantic facts by similarity" do
      {:ok, embedding} = WorldEmbedder.embed(@world_id, "hello world")

      fact = SemanticFact.new("greeting pattern", embedding, ["ep1"], ["smalltalk.greetings.hello"])
      {:ok, _} = Store.add_semantic(fact, world_id: @world_id)

      {:ok, results} = Store.query_semantic("hello there", 5, world_id: @world_id)

      assert length(results) == 1
      [{semantic, _sim}] = results
      assert semantic.representation == "greeting pattern"
    end
  end

  describe "link_episode_to_semantic" do
    test "links an episode to a semantic fact" do
      {:ok, ep_id} = Store.add_episode("hello", "smalltalk.greetings.hello", "", [], world_id: @world_id)
      {:ok, episode} = Store.get_episode(ep_id, world_id: @world_id)
      assert episode.semantic_id == nil

      sem_id = Ecto.UUID.generate()
      :ok = Store.link_episode_to_semantic(ep_id, sem_id, world_id: @world_id)

      {:ok, updated} = Store.get_episode(ep_id, world_id: @world_id)
      assert updated.semantic_id == sem_id
    end

    test "returns error for non-existent episode" do
      fake_id = Ecto.UUID.generate()
      assert {:error, :not_found} = Store.link_episode_to_semantic(fake_id, Ecto.UUID.generate(), world_id: @world_id)
    end
  end

  describe "all_episodes and all_semantics" do
    test "returns all episodes" do
      {:ok, _} = Store.add_episode("a", "b", "c", [], world_id: @world_id)
      {:ok, _} = Store.add_episode("d", "e", "f", [], world_id: @world_id)

      {:ok, episodes} = Store.all_episodes(world_id: @world_id)

      assert length(episodes) == 2
    end

    test "returns all semantic facts" do
      {:ok, embedding} = WorldEmbedder.embed(@world_id, "hello")

      fact1 = SemanticFact.new("fact1", embedding, [], [])
      fact2 = SemanticFact.new("fact2", embedding, [], [])
      {:ok, _} = Store.add_semantic(fact1, world_id: @world_id)
      {:ok, _} = Store.add_semantic(fact2, world_id: @world_id)

      {:ok, semantics} = Store.all_semantics(world_id: @world_id)

      assert length(semantics) == 2
    end
  end

  describe "stats" do
    test "returns store statistics" do
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", [], world_id: @world_id)
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", [], world_id: @world_id)

      stats = Store.stats()

      assert stats.episode_count == 2
      assert stats.semantic_count == 0
      assert stats.episode_index_size == 2
    end
  end

  describe "clear" do
    test "removes all data" do
      {:ok, _} = Store.add_episode("hello", "smalltalk.greetings.hello", "", [], world_id: @world_id)
      {:ok, _} = Store.add_episode("goodbye", "farewell", "", [], world_id: @world_id)

      :ok = Store.clear()

      {:ok, episodes} = Store.all_episodes(world_id: @world_id)
      assert episodes == []
    end
  end

  describe "rerank option" do
    test "query_similar with rerank: false returns results without entity extraction" do
      {:ok, _} = Store.add_episode("hello world", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], world_id: @world_id)
      {:ok, _} = Store.add_episode("hello there", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], world_id: @world_id)

      {:ok, results} = Store.query_similar("hello friend", 3, world_id: @world_id, rerank: false)

      assert is_list(results)
    end

    test "query_similar with rerank: false skips rerank even when config enables it" do
      {:ok, _} = Store.add_episode("hello world", "smalltalk.greetings.hello", "", ["smalltalk.greetings.hello"], world_id: @world_id)

      original_config = Application.get_env(:brain, :kg_signals, [])
      Application.put_env(:brain, :kg_signals, Keyword.put(original_config, :memory_rerank, true))

      {:ok, results} = Store.query_similar("hello friend", 3, world_id: @world_id, rerank: false)

      Application.put_env(:brain, :kg_signals, original_config)

      assert is_list(results)
    end

    test "query_semantic with rerank: false returns results" do
      {:ok, embedding} = WorldEmbedder.embed(@world_id, "hello world")
      fact = SemanticFact.new("greeting", embedding, [], ["smalltalk.greetings.hello"])
      {:ok, _} = Store.add_semantic(fact, world_id: @world_id)

      {:ok, results} = Store.query_semantic("hello there", 5, world_id: @world_id, rerank: false)

      assert is_list(results)
    end
  end

  describe "entity_names on episodes" do
    test "add_episode populates entity_names field" do
      {:ok, id} = Store.add_episode("hello world", "smalltalk.greetings.hello", "hi there", ["smalltalk.greetings.hello"], world_id: @world_id)
      {:ok, episode} = Store.get_episode(id, world_id: @world_id)

      assert is_list(episode.entity_names)
    end

    test "an episode's entity_names hold no span entity extraction could not decide" do
      for {text, undecided} <- undecided_span_cases(@world_id, 3) do
        {:ok, id} = Store.add_episode(text, "conversation", "", [], world_id: @world_id)
        {:ok, episode} = Store.get_episode(id, world_id: @world_id)

        names = MapSet.new(episode.entity_names, &String.downcase/1)

        for span <- undecided do
          refute MapSet.member?(names, String.downcase(span.value)),
                 "#{inspect(span.value)} of #{inspect(text)} became an episode entity name: #{inspect(episode.entity_names)}"
        end
      end
    end

    test "add_episode_direct preserves pre-populated entity_names" do
      episode = Episode.new(
        "hello world",
        "smalltalk.greetings.hello",
        "hi there",
        ["smalltalk.greetings.hello"],
        []
      )
      episode = %{episode | entity_names: ["world"]}

      {:ok, id} = Store.add_episode_direct(episode, world_id: @world_id)
      {:ok, retrieved} = Store.get_episode(id, world_id: @world_id)

      assert retrieved.entity_names == ["world"]
    end
  end

  describe "a world is required" do
    test "add_episode raises without :world_id" do
      assert_raise ArgumentError, ~r/add_episode requires a :world_id option/, fn ->
        Store.add_episode("hello", "smalltalk.greetings.hello", "", [], [])
      end
    end

    test "add_episode raises on an empty :world_id" do
      assert_raise ArgumentError, ~r/add_episode requires a :world_id option/, fn ->
        Store.add_episode("hello", "smalltalk.greetings.hello", "", [], world_id: "")
      end
    end

    test "add_episode_direct raises without :world_id" do
      episode = Episode.new("hello", "smalltalk.greetings.hello", "", [], [])

      assert_raise ArgumentError, ~r/add_episode_direct requires a :world_id option/, fn ->
        Store.add_episode_direct(episode, [])
      end
    end

    test "add_event_episode raises without :world_id" do
      event = %{action: %{lemma: "play"}, object: %{text: "jazz"}}

      assert_raise ArgumentError, ~r/add_event_episode requires a :world_id option/, fn ->
        Store.add_event_episode(event, %{response: "Playing jazz"}, [])
      end
    end

    test "query_similar raises without :world_id" do
      assert_raise ArgumentError, ~r/query_similar requires a :world_id option/, fn ->
        Store.query_similar("hello", [])
      end
    end

    test "query_by_tags and the event queries raise without :world_id" do
      assert_raise ArgumentError, ~r/query_by_tags requires a :world_id option/, fn ->
        Store.query_by_tags(["smalltalk.greetings.hello"], [])
      end

      assert_raise ArgumentError, ~r/query_by_tags requires a :world_id option/, fn ->
        Store.query_events_by_action("play", [])
      end

      assert_raise ArgumentError, ~r/query_by_tags requires a :world_id option/, fn ->
        Store.query_events_by_object("jazz", [])
      end

      assert_raise ArgumentError, ~r/query_by_tags requires a :world_id option/, fn ->
        Store.query_events_by_actor("user", [])
      end
    end

    test "get_episode raises without :world_id" do
      assert_raise ArgumentError, ~r/get_episode requires a :world_id option/, fn ->
        Store.get_episode(Ecto.UUID.generate(), [])
      end
    end

    test "add_semantic raises without :world_id" do
      fact = SemanticFact.new("greeting pattern", [0.1, 0.2, 0.3], [], [])

      assert_raise ArgumentError, ~r/add_semantic requires a :world_id option/, fn ->
        Store.add_semantic(fact, [])
      end
    end

    test "query_semantic raises without :world_id" do
      assert_raise ArgumentError, ~r/query_semantic requires a :world_id option/, fn ->
        Store.query_semantic("hello", [])
      end
    end

    test "get_semantic raises without :world_id" do
      assert_raise ArgumentError, ~r/get_semantic requires a :world_id option/, fn ->
        Store.get_semantic(Ecto.UUID.generate(), [])
      end
    end

    test "link_episode_to_semantic raises without :world_id" do
      assert_raise ArgumentError, ~r/link_episode_to_semantic requires a :world_id option/, fn ->
        Store.link_episode_to_semantic(Ecto.UUID.generate(), Ecto.UUID.generate(), [])
      end
    end

    test "all_episodes and all_semantics raise without :world_id" do
      assert_raise ArgumentError, ~r/all_episodes requires a :world_id option/, fn ->
        Store.all_episodes([])
      end

      assert_raise ArgumentError, ~r/all_semantics requires a :world_id option/, fn ->
        Store.all_semantics([])
      end
    end
  end
end
