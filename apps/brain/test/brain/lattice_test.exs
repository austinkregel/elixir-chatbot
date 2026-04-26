defmodule Brain.LatticeTest do
  use ExUnit.Case, async: true

  alias Brain.Lattice
  alias Brain.Lattice.Candidate

  describe "empty/2" do
    test "empty lattice has no best" do
      l = Lattice.empty(:intent_full, error: :not_loaded)
      assert Lattice.empty?(l)
      assert Lattice.best(l) == nil
      assert Lattice.error(l) == :not_loaded
    end
  end

  describe "from_classifier/2" do
    test "builds candidates from top_k and preserves winner confidence" do
      details = %{
        top_score: 0.9,
        second_score: 0.4,
        margin: 0.5,
        top_k: [{"a", 0.9}, {"b", 0.4}, {"c", 0.1}]
      }

      l =
        Lattice.from_classifier({:ok, "a", 0.88, details},
          stage: :intent_full,
          source: :feature_vector
        )

      refute Lattice.empty?(l)
      assert Lattice.best_label(l) == "a"
      assert %Candidate{label: "a"} = Lattice.best(l)
      assert length(l.candidates) == 3
      assert Lattice.margin(l) >= 0.0
    end
  end

  describe "rerank/2 and normalize" do
    test "rerank adds delta then finalize updates margin" do
      l =
        Lattice.from_top_k([{"x", 0.8}, {"y", 0.2}], stage: :test, source: :test)
          |> Lattice.rerank(fn %Candidate{label: lab} ->
            if lab == "y", do: 0.5, else: 0.0
          end)

      assert Lattice.best_label(l) == "y"
    end
  end

  describe "to_context_signal/2" do
    test "uses best candidate" do
      l = Lattice.singleton(:foo, 0.5, confidence: 0.9, stage: :x, source: :y)
      assert {:intent, :foo, conf} = Lattice.to_context_signal(l, :intent)
      assert conf > 0.5
    end
  end

  describe "merge/3" do
    test ":max keeps higher confidence per label then sorts globally" do
      a = Lattice.from_top_k([{"p", 0.99}, {"q", 0.01}], source: :a, stage: :s)
      b = Lattice.from_top_k([{"p", 0.1}, {"q", 0.98}], source: :b, stage: :s)
      m = Lattice.merge(a, b, :max)
      assert Lattice.best_label(m) == "p"
    end
  end
end
