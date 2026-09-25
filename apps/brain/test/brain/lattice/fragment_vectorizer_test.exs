defmodule Brain.Lattice.FragmentVectorizerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Lattice.FragmentVectorizer

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

    assert {:ok, fv} = FragmentVectorizer.vectorize_fragment_text(text)
    assert length(fv) == ChunkFeatures.vector_dimension()
    assert length(fv) == FragmentVectorizer.vector_dimension()
    assert Enum.any?(fv, &(&1 != 0.0))
  end

  test "vectorize_fragment_text rejects empty string" do
    assert {:error, :empty_text} = FragmentVectorizer.vectorize_fragment_text("   ")
  end
end
