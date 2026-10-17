defmodule Brain.Memory.SemanticFactSearchTest do
  @moduledoc """
  How a semantic fact is found: one with evidence episodes by their center,
  the mean of their embeddings; one with none by an embedding of its own text.
  A vocabulary rebuild re-embeds each the same way.

  Each test has a world of its own, embedding with a small vocabulary loaded
  for it, so every vector here can be worked out by hand.
  """
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  # World.Embedder is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Embedder}

  alias Brain.Memory.{Store, VectorIndex}
  alias Brain.Memory.Types.SemanticFact
  alias World.Embedder, as: WorldEmbedder

  @words ~w(weather paris london rain sunny hello friend there)
  @model %{
    vocabulary: @words |> Enum.with_index() |> Map.new(),
    idf_weights: Map.new(@words, &{&1, 1.0})
  }

  # The vocabulary a rebuild leaves: fewer words, in another order.
  @rebuilt_words ~w(rain paris hello friend weather)
  @rebuilt_model %{
    vocabulary: @rebuilt_words |> Enum.with_index() |> Map.new(),
    idf_weights: Map.new(@rebuilt_words, &{&1, 1.0})
  }

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    # Running already when the application is up; started here otherwise.
    ensure_started(Store)
    assert Process.whereis(Store)
    :ok = WorldEmbedder.init()

    world_id = "test_semantic_search_#{System.unique_integer([:positive])}"
    :ok = WorldEmbedder.load_model(world_id, @model)

    on_exit(fn -> WorldEmbedder.clear(world_id) end)

    {:ok, world_id: world_id}
  end

  defp add_episodes(texts, world_id) do
    Enum.map(texts, fn text ->
      {:ok, id} = Store.add_episode(text, "weather", "", [], world_id: world_id)
      {:ok, episode} = Store.get_episode(id, world_id: world_id)
      episode
    end)
  end

  defp embed!(world_id, text) do
    {:ok, embedding} = WorldEmbedder.embed(world_id, text)
    embedding
  end

  describe "a semantic fact with no evidence episodes" do
    test "is found by an embedding of its own text, whatever embedding it carried", %{world_id: world_id} do
      fact = SemanticFact.new("hello friend", [0.5, 0.5], [], [])
      {:ok, fact_id} = Store.add_semantic(fact, world_id: world_id)

      {:ok, stored} = Store.get_semantic(fact_id, world_id: world_id)
      assert stored.embedding == embed!(world_id, "hello friend")

      assert {:ok, [{%{id: ^fact_id}, similarity}]} =
               Store.query_semantic("hello friend", 1, world_id: world_id, rerank: false)

      assert_in_delta similarity, 1.0, 1.0e-9
    end
  end

  describe "a semantic fact with evidence episodes" do
    test "is found by their center, not by its text", %{world_id: world_id} do
      episodes = add_episodes(["weather in paris", "rain in paris"], world_id)
      center = VectorIndex.mean_vector(Enum.map(episodes, & &1.embedding))

      fact = SemanticFact.new("hello friend", center, Enum.map(episodes, & &1.id), [])
      {:ok, fact_id} = Store.add_semantic(fact, world_id: world_id)

      {:ok, stored} = Store.get_semantic(fact_id, world_id: world_id)
      assert stored.embedding == center
      refute stored.embedding == embed!(world_id, "hello friend")

      assert {:ok, [{%{id: ^fact_id}, by_episodes}]} =
               Store.query_semantic("weather rain paris", 1, world_id: world_id, rerank: false)

      assert {:ok, [{%{id: ^fact_id}, by_text}]} =
               Store.query_semantic("hello friend", 1, world_id: world_id, rerank: false)

      assert by_episodes > 0.9
      assert by_text == 0.0
    end
  end

  describe "a vocabulary rebuild" do
    test "re-embeds a fact with evidence by its center and one without by its text", %{world_id: world_id} do
      episodes = add_episodes(["weather in paris", "rain in paris"], world_id)
      evidence_ids = Enum.map(episodes, & &1.id)

      {:ok, centered_id} =
        Store.add_semantic(
          SemanticFact.new("weather in paris | rain in paris", VectorIndex.mean_vector(Enum.map(episodes, & &1.embedding)), evidence_ids, [],
            action: "weather"
          ),
          world_id: world_id
        )

      {:ok, textual_id} = Store.add_semantic(SemanticFact.new("hello friend", [], [], []), world_id: world_id)

      :ok = WorldEmbedder.load_model(world_id, @rebuilt_model)

      assert {:ok, %{episodes: 2, semantic_facts: 2}} = Store.vocabulary_rebuilt(world_id)

      rebuilt_episodes =
        Enum.map(evidence_ids, fn id ->
          {:ok, episode} = Store.get_episode(id, world_id: world_id)
          episode
        end)

      assert Enum.all?(rebuilt_episodes, &(length(&1.embedding) == length(@rebuilt_words)))

      {:ok, centered} = Store.get_semantic(centered_id, world_id: world_id)
      assert centered.embedding == VectorIndex.mean_vector(Enum.map(rebuilt_episodes, & &1.embedding))
      assert centered.action == "weather"

      {:ok, textual} = Store.get_semantic(textual_id, world_id: world_id)
      assert textual.embedding == embed!(world_id, "hello friend")
      assert length(textual.embedding) == length(@rebuilt_words)

      assert {:ok, [{%{id: ^textual_id}, similarity}]} =
               Store.query_semantic("hello friend", 1, world_id: world_id, rerank: false)

      assert_in_delta similarity, 1.0, 1.0e-9
    end

    test "stops at a fact that names evidence episodes the world does not hold, naming them", %{world_id: world_id} do
      [episode] = add_episodes(["weather in paris"], world_id)
      missing_id = Ecto.UUID.generate()

      fact = SemanticFact.new("weather in paris", episode.embedding, [episode.id, missing_id], [])
      {:ok, fact_id} = Store.add_semantic(fact, world_id: world_id)

      :ok = WorldEmbedder.load_model(world_id, @rebuilt_model)

      assert Store.vocabulary_rebuilt(world_id) == {:error, {:semantic_evidence_missing, fact_id, [missing_id]}}
    end
  end

  describe "evidence_episodes/2" do
    test "returns a fact's evidence episodes in the order it lists them", %{world_id: world_id} do
      [paris, rain] = add_episodes(["weather in paris", "rain in paris"], world_id)
      fact = SemanticFact.new("weather in paris | rain in paris", [], [rain.id, paris.id], [])

      assert {:ok, [%{id: first}, %{id: second}]} = Store.evidence_episodes(fact, world_id: world_id)
      assert [first, second] == [rain.id, paris.id]
    end

    test "names the evidence episodes the world does not hold", %{world_id: world_id} do
      [paris] = add_episodes(["weather in paris"], world_id)
      missing_id = Ecto.UUID.generate()
      fact = SemanticFact.new("weather in paris", [], [paris.id, missing_id], [])

      assert Store.evidence_episodes(fact, world_id: world_id) ==
               {:error, {:semantic_evidence_missing, fact.id, [missing_id]}}
    end

    test "raises without :world_id" do
      assert_raise ArgumentError, ~r/evidence_episodes requires a :world_id option/, fn ->
        Store.evidence_episodes(SemanticFact.new("a", [], [], []), [])
      end
    end
  end
end
