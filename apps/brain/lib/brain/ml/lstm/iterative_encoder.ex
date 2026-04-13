defmodule Brain.ML.LSTM.IterativeEncoder do
  @moduledoc """
  Multi-pass LSTM encoder with learned gating and entropy-based early exit.

  Inspired by Zhu et al. "Scaling Latent Reasoning via Looped Language Models" (2025).
  The concept of iterative refinement is adapted for BiLSTM encoders: run the
  shared encoder multiple times, feeding output back as input, allowing the
  model to "think longer" on harder inputs.

  After each pass, a learned gating function blends the current and previous
  representations:

      h_new = gate * h_pass_n + (1 - gate) * h_pass_n_minus_1

  An entropy-based early exit stops iteration when the task head's output
  entropy drops below a threshold (indicating confident predictions).
  """

  import Nx.Defn

  @default_max_passes 3
  @default_entropy_threshold 0.5

  @doc """
  Build an iterative encoder with a gating network.

  ## Parameters
    - `hidden_dim` - Hidden dimension of the LSTM encoder
    - `opts` - Options:
      - `:max_passes` - Maximum number of forward passes (default: #{@default_max_passes})
  """
  def build_gate(hidden_dim, _opts \\ []) do
    Axon.input("gate_input", shape: {nil, hidden_dim})
    |> Axon.dense(hidden_dim, activation: :sigmoid, name: "gate_dense")
  end

  @doc """
  Apply iterative encoding with gating.

  Takes the encoder output from one pass and blends it with the previous
  pass using a learned gate.

  ## Parameters
    - `current` - Current pass output tensor {batch, seq_len, hidden}
    - `previous` - Previous pass output tensor {batch, seq_len, hidden}
    - `gate_values` - Gate tensor {batch, hidden} in [0, 1]

  ## Returns
    Blended output tensor.
  """
  defn apply_gate(current, previous, gate_values) do
    gate_expanded = Nx.new_axis(gate_values, 1)

    Nx.add(
      Nx.multiply(gate_expanded, current),
      Nx.multiply(Nx.subtract(1.0, gate_expanded), previous)
    )
  end

  @doc """
  Compute the entropy of a probability distribution.

  Used to determine when to stop iterating (low entropy = confident prediction).

  ## Parameters
    - `probs` - Probability distribution tensor {batch, num_classes}

  ## Returns
    Entropy tensor {batch}.
  """
  defn entropy(probs) do
    eps = 1.0e-10
    log_probs = Nx.log(Nx.add(probs, eps))
    Nx.negate(Nx.sum(Nx.multiply(probs, log_probs), axes: [1]))
  end

  @doc """
  Check if the output is confident enough to stop iterating.

  Returns true if the maximum entropy across the batch is below the threshold.
  """
  defn should_exit?(probs, threshold) do
    ent = entropy(probs)
    max_entropy = Nx.reduce_max(ent)
    Nx.less(max_entropy, threshold)
  end

  @doc """
  Run iterative encoding with early exit.

  This is a pure function that performs multiple passes through the encoder,
  applying gating between passes and checking entropy for early exit.

  ## Parameters
    - `input` - Input tensor
    - `encode_fn` - Function that takes input and returns encoder output
    - `classify_fn` - Function that takes pooled output and returns class probs
    - `pool_fn` - Function that takes encoder output and returns pooled representation
    - `gate_fn` - Function that takes pooled output and returns gate values
    - `opts` - Options:
      - `:max_passes` - Maximum number of passes (default: #{@default_max_passes})
      - `:entropy_threshold` - Early exit threshold (default: #{@default_entropy_threshold})

  ## Returns
    `{final_output, num_passes}` where `final_output` is the refined encoder output.
  """
  def iterate(input, encode_fn, classify_fn, pool_fn, gate_fn, opts \\ []) do
    max_passes = Keyword.get(opts, :max_passes, @default_max_passes)
    threshold = Keyword.get(opts, :entropy_threshold, @default_entropy_threshold)

    initial_output = encode_fn.(input)

    {final_output, passes} = Enum.reduce_while(2..max_passes//1, {initial_output, 1}, fn pass, {prev_output, _} ->
      current_output = encode_fn.(prev_output)

      pooled = pool_fn.(current_output)
      gate_values = gate_fn.(pooled)
      blended = apply_gate(current_output, prev_output, gate_values)

      probs = classify_fn.(pooled)

      if should_exit_eager?(probs, threshold) do
        {:halt, {blended, pass}}
      else
        {:cont, {blended, pass}}
      end
    end)

    {final_output, passes}
  end

  defp should_exit_eager?(probs, threshold) do
    should_exit?(probs, threshold)
    |> Nx.to_number()
    |> Kernel.==(1)
  end
end
