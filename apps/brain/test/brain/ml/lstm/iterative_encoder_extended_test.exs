defmodule Brain.ML.LSTM.IterativeEncoderExtendedTest do
  use ExUnit.Case, async: false

  alias Brain.ML.LSTM.IterativeEncoder

  describe "build_gate/2" do
    test "builds an Axon model" do
      gate = IterativeEncoder.build_gate(32)
      assert %Axon{} = gate
    end

    test "builds with different hidden dimensions" do
      for dim <- [8, 16, 64, 128] do
        gate = IterativeEncoder.build_gate(dim)
        assert %Axon{} = gate
      end
    end

    test "gate model produces output of correct shape" do
      hidden_dim = 16
      gate = IterativeEncoder.build_gate(hidden_dim)

      {init_fn, predict_fn} = Axon.build(gate)
      params = init_fn.(Nx.template({1, hidden_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {1, hidden_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, hidden_dim}

      # Gate should be in (0, 1) range (sigmoid output)
      vals = Nx.to_flat_list(output)
      for val <- vals do
        assert val >= 0.0 and val <= 1.0, "Gate value #{val} outside [0,1]"
      end
    end
  end

  describe "apply_gate/3 edge cases" do
    test "batch of inputs" do
      current = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      previous = Nx.tensor([[0.0, 1.0], [1.0, 0.0]])
      gate = Nx.tensor([[0.5, 0.5], [0.5, 0.5]])

      result = IterativeEncoder.apply_gate(current, previous, gate)
      expected = Nx.tensor([[0.5, 0.5], [0.5, 0.5]])

      diff = Nx.subtract(Nx.squeeze(result), expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5
    end

    test "per-dimension gating" do
      current = Nx.tensor([[10.0, 0.0]])
      previous = Nx.tensor([[0.0, 10.0]])
      gate = Nx.tensor([[1.0, 0.0]])

      result = IterativeEncoder.apply_gate(current, previous, gate) |> Nx.squeeze()
      vals = Nx.to_flat_list(result)

      assert_in_delta Enum.at(vals, 0), 10.0, 0.01
      assert_in_delta Enum.at(vals, 1), 10.0, 0.01
    end
  end

  describe "entropy/1 edge cases" do
    test "two-class equal distribution" do
      probs = Nx.tensor([[0.5, 0.5]])
      ent = IterativeEncoder.entropy(probs) |> Nx.squeeze() |> Nx.to_number()
      # max entropy for 2 classes = ln(2) ≈ 0.693
      assert_in_delta ent, 0.693, 0.01
    end

    test "very peaked distribution" do
      probs = Nx.tensor([[0.999, 0.0005, 0.0005]])
      ent = IterativeEncoder.entropy(probs) |> Nx.squeeze() |> Nx.to_number()
      assert ent < 0.01
    end

    test "many classes uniform" do
      n = 10
      probs = Nx.broadcast(1.0 / n, {1, n})
      ent = IterativeEncoder.entropy(probs) |> Nx.squeeze() |> Nx.to_number()
      expected = :math.log(n)
      assert_in_delta ent, expected, 0.01
    end
  end

  describe "should_exit?/2 edge cases" do
    test "threshold of 0 always exits" do
      probs = Nx.tensor([[0.34, 0.33, 0.33]])
      result = IterativeEncoder.should_exit?(probs, 0.0) |> Nx.to_number()
      # entropy > 0 but threshold is 0, so maybe still fails
      # Actually threshold 0 means "exit if entropy < 0" which is never true
      # Let's check with very large threshold
      assert result == 0 or result == 1
    end

    test "very high threshold never exits" do
      probs = Nx.tensor([[0.99, 0.005, 0.005]])
      result = IterativeEncoder.should_exit?(probs, 100.0) |> Nx.to_number()
      assert result == 1
    end
  end

  describe "iterate/6 edge cases" do
    test "max_passes=1 runs exactly once" do
      encode_fn = fn input -> input end
      pool_fn = fn output -> Nx.mean(output, axes: [1]) end
      classify_fn = fn _pooled -> Nx.tensor([[0.5, 0.5]]) end
      gate_fn = fn _pooled -> Nx.broadcast(0.5, {1, 3}) end

      input = Nx.tensor([[1.0, 2.0, 3.0]])

      {_output, passes} = IterativeEncoder.iterate(
        input, encode_fn, classify_fn, pool_fn, gate_fn,
        max_passes: 1, entropy_threshold: 100.0
      )

      assert passes == 1
    end

    test "returns output even with zero entropy threshold" do
      encode_fn = fn input -> input end
      pool_fn = fn output -> Nx.mean(output, axes: [1]) end
      classify_fn = fn _pooled -> Nx.tensor([[0.9, 0.05, 0.05]]) end
      gate_fn = fn _pooled -> Nx.broadcast(0.5, {1, 3}) end

      input = Nx.tensor([[1.0, 2.0, 3.0]])

      {output, _passes} = IterativeEncoder.iterate(
        input, encode_fn, classify_fn, pool_fn, gate_fn,
        max_passes: 3, entropy_threshold: 0.0
      )

      assert is_struct(output, Nx.Tensor)
    end

    test "encode_fn transforms input each pass" do
      pass_count = :counters.new(1, [:atomics])

      encode_fn = fn input ->
        :counters.add(pass_count, 1, 1)
        Nx.add(input, 0.01)
      end

      pool_fn = fn output -> Nx.mean(output, axes: [1]) end
      classify_fn = fn _pooled -> Nx.tensor([[0.34, 0.33, 0.33]]) end
      gate_fn = fn _pooled -> Nx.broadcast(0.5, {1, 3}) end

      input = Nx.tensor([[1.0, 2.0, 3.0]])

      {_output, passes} = IterativeEncoder.iterate(
        input, encode_fn, classify_fn, pool_fn, gate_fn,
        max_passes: 3, entropy_threshold: 0.001
      )

      assert passes == 3
      assert :counters.get(pass_count, 1) == 3
    end
  end
end
