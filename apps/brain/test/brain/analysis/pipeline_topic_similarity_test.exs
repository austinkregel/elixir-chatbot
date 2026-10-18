defmodule Brain.Analysis.PipelineTopicSimilarityTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.Pipeline

  describe "topic_similarity/3" do
    setup do
      # Creating a world writes to Atlas (its graphs and its scheduling
      # driver), in this process.
      owner = Brain.Test.AtlasSandbox.checkout_allowed!()
      on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

      setup_world_sandbox()

      {:ok, world} = create_test_world("topic_similarity")

      :ok =
        Brain.Worlds.Embedder.load_model(world.id, %{
          vocabulary: %{"weather" => 0, "forecast" => 1, "music" => 2, "jazz" => 3},
          idf_weights: %{"weather" => 1.0, "forecast" => 1.0, "music" => 1.0, "jazz" => 1.0}
        })

      {:ok, world_id: world.id}
    end

    test "a related recent topic is similar to the domain", %{world_id: world_id} do
      similarity = Pipeline.topic_similarity("weather", ["jazz music", "weather forecast"], world_id)

      assert similarity > 0.0
      assert similarity <= 1.0
    end

    test "only unrelated recent topics are not similar to the domain", %{world_id: world_id} do
      assert Pipeline.topic_similarity("weather", ["jazz music"], world_id) == 0.0
    end

    test "no recent topics give no similarity", %{world_id: world_id} do
      assert Pipeline.topic_similarity("weather", [], world_id) == 0.0
    end
  end
end
