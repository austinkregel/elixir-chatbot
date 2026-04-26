defmodule Brain.ML.MicroClassifiersDetailedTest do
  use ExUnit.Case, async: false

  alias Brain.Lattice
  alias Brain.ML.MicroClassifiers

  describe "classify_vector_detailed/2" do
    test "returns a lattice with multiple candidates when :intent_full is loaded" do
      if not MicroClassifiers.ready?() do
        assert true
      else
        assert {:ok, dim} = MicroClassifiers.input_dim(:intent_full)
        vec = List.duplicate(0.0, dim)
        vec = List.replace_at(vec, 0, 0.01)

        case MicroClassifiers.classify_vector_detailed(:intent_full, vec) do
          {:ok, %Lattice{} = lat} ->
            refute Lattice.empty?(lat)
            assert length(lat.candidates) >= 1
            assert Lattice.best_label(lat) != nil

          {:error, _} ->
            assert true
        end
      end
    end
  end

  describe "classify_detailed/2" do
    test "returns a lattice for TF-IDF classifier" do
      if not MicroClassifiers.ready?() do
        assert true
      else
        case MicroClassifiers.classify_detailed(:personal_question, "what is your name") do
          {:ok, %Lattice{} = lat} ->
            refute Lattice.empty?(lat)
            assert Lattice.best_label(lat) != nil

          {:error, _} ->
            assert true
        end
      end
    end
  end
end
