defmodule Brain.Lattice.FragmentVectorizerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Lattice.FragmentVectorizer

  # The world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  # Feature extraction reads the knowledge graph; without a checked-out
  # connection those lookups fail silently.
  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    :ok
  end

  @tag :integration
  test "vectorize_fragment_text returns vector of expected dimension" do
    Mix.Task.run("app.start")

    text = "Hello! What is the weather like in Seattle this week? It might rain on Tuesday."

    assert {:ok, fv} = FragmentVectorizer.vectorize_fragment_text(text, @world_id)
    assert length(fv) == ChunkFeatures.vector_dimension()
    assert length(fv) == FragmentVectorizer.vector_dimension()
    assert Enum.any?(fv, &(&1 != 0.0))
  end

  test "vectorize_fragment_text rejects empty string" do
    assert_raise ArgumentError, ~r/a fragment's text is blank/, fn ->
      FragmentVectorizer.vectorize_fragment_text("   ", @world_id)
    end
  end

  @tag :integration
  test "a fragment's vector is the one the pipeline classified the fragment from" do
    Mix.Task.run("app.start")

    text = "Hello! What is the weather like in Seattle this week?"
    analysis = Brain.Analysis.Pipeline.analyze_chunk(text, world_id: @world_id, side_effects: false)

    assert {:ok, fv} = FragmentVectorizer.vectorize_fragment_text(text, @world_id)
    assert fv == analysis.feature_vector
  end
end
