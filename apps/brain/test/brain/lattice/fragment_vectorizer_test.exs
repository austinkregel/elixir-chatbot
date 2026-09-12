defmodule Brain.Lattice.FragmentVectorizerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Lattice.FragmentVectorizer

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
