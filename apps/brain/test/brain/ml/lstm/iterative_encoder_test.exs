defmodule Brain.ML.LSTM.IterativeEncoderTest do
  use ExUnit.Case, async: false

  alias Brain.ML.LSTM.IterativeEncoder

  describe "apply_gate/3" do
    @tag :known_answer
    test "gate=1 selects current, gate=0 selects previous" do
      current = Nx.tensor([[1.0, 2.0, 3.0]])
      previous = Nx.tensor([[4.0, 5.0, 6.0]])

      all_current = IterativeEncoder.apply_gate(current, previous, Nx.tensor([[1.0, 1.0, 1.0]]))
      all_previous = IterativeEncoder.apply_gate(current, previous, Nx.tensor([[0.0, 0.0, 0.0]]))

      assert_close(all_current, current)
      assert_close(all_previous, previous)
    end

    @tag :known_answer
    test "gate=0.5 gives average of current and previous" do
      current = Nx.tensor([[2.0, 4.0]])
      previous = Nx.tensor([[6.0, 8.0]])
      gate = Nx.tensor([[0.5, 0.5]])

      result = IterativeEncoder.apply_gate(current, previous, gate)
      expected = Nx.tensor([[4.0, 6.0]])

      assert_close(result, expected)
    end
  end

  describe "entropy/1" do
    @tag :known_answer
    test "uniform distribution has maximum entropy" do
      uniform = Nx.tensor([[0.25, 0.25, 0.25, 0.25]])
      peaked = Nx.tensor([[0.9, 0.05, 0.025, 0.025]])

      uniform_ent = IterativeEncoder.entropy(uniform) |> Nx.squeeze() |> Nx.to_number()
      peaked_ent = IterativeEncoder.entropy(peaked) |> Nx.squeeze() |> Nx.to_number()

      assert uniform_ent > peaked_ent,
        "Uniform entropy (#{uniform_ent}) should be > peaked (#{peaked_ent})"
    end

    @tag :property
    test "entropy is non-negative" do
      probs = Nx.tensor([[0.7, 0.2, 0.1]])
      ent = IterativeEncoder.entropy(probs) |> Nx.squeeze() |> Nx.to_number()
      assert ent >= 0.0
    end

    @tag :known_answer
    test "certain distribution has near-zero entropy" do
      certain = Nx.tensor([[0.99, 0.005, 0.005]])
      ent = IterativeEncoder.entropy(certain) |> Nx.squeeze() |> Nx.to_number()
      assert ent < 0.1
    end
  end

  describe "should_exit?/2" do
    test "exits when entropy is below threshold" do
      confident = Nx.tensor([[0.95, 0.03, 0.02]])
      result = IterativeEncoder.should_exit?(confident, 0.5) |> Nx.to_number()
      assert result == 1
    end

    test "does not exit when entropy is above threshold" do
      uncertain = Nx.tensor([[0.34, 0.33, 0.33]])
      result = IterativeEncoder.should_exit?(uncertain, 0.5) |> Nx.to_number()
      assert result == 0
    end
  end

  describe "iterate/6" do
    test "runs at least one pass" do
      encode_fn = fn input -> Nx.add(input, 0.1) end
      pool_fn = fn output -> Nx.mean(output, axes: [1]) end
      classify_fn = fn _pooled -> Nx.tensor([[0.9, 0.05, 0.05]]) end
      gate_fn = fn _pooled -> Nx.broadcast(0.5, {1, 3}) end

      input = Nx.tensor([[1.0, 2.0, 3.0]])

      {output, passes} = IterativeEncoder.iterate(
        input, encode_fn, classify_fn, pool_fn, gate_fn,
        max_passes: 3,
        entropy_threshold: 0.1
      )

      assert passes >= 1
      assert tuple_size(Nx.shape(output)) >= 2
    end

    test "early exits on confident predictions" do
      encode_fn = fn input -> input end
      pool_fn = fn output -> Nx.mean(output, axes: [1]) end
      classify_fn = fn _pooled -> Nx.tensor([[0.99, 0.005, 0.005]]) end
      gate_fn = fn _pooled -> Nx.broadcast(0.5, {1, 3}) end

      input = Nx.tensor([[1.0, 2.0, 3.0]])

      {_output, passes} = IterativeEncoder.iterate(
        input, encode_fn, classify_fn, pool_fn, gate_fn,
        max_passes: 5,
        entropy_threshold: 0.5
      )

      assert passes < 5, "Should exit early on confident predictions, used #{passes} passes"
    end
  end

  defp assert_close(actual, expected) do
    diff = Nx.subtract(actual, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff < 1.0e-5, "Tensors not close: max diff #{diff}"
  end
end
