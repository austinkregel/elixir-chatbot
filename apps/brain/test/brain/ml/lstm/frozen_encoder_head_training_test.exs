defmodule Brain.ML.LSTM.FrozenEncoderHeadTrainingTest do
  @moduledoc """
  Verifies that training a classification head with a frozen EXLA-backed encoder
  works without backend mismatches (FunctionClauseError in Nx.BinaryBackend.to_binary/1).

  Uses compiler: EXLA on Axon.predict when params come from Loop.run(compiler: EXLA).
  The crash (BinaryBackend.to_binary on EXLA tensor) may be environment-dependent.

  This mirrors the code path in train_head_with_frozen_encoder/7:
  - Train encoder+head with Loop.run(compiler: EXLA) -> EXLA-backed params
  - Axon.predict(encoder, exla_params, inputs) -> must use compiler: EXLA
  - Nx.mean, Nx.backend_transfer -> handoff to head training
  - Loop.run head with compiler: EXLA

  Uses minimal data to keep runtime under a few seconds. No tags - runs in every mix test.
  """
  use ExUnit.Case, async: true

  alias Axon.Loop
  alias Polaris.Optimizers

  @vocab_size 20
  @embedding_size 8
  @hidden_size 8
  @max_seq_length 5
  @batch_size 2
  @num_classes 3
  @learning_rate 0.001

  describe "frozen encoder + head training (EXLA-backed params)" do
    test "predict with EXLA-backed encoder params does not crash and head trains successfully" do
      config = %{
        vocab_size: @vocab_size,
        embedding_size: @embedding_size,
        hidden_size: @hidden_size,
        max_seq_length: @max_seq_length,
        batch_size: @batch_size,
        learning_rate: @learning_rate
      }

      # 1. Build encoder (same structure as UnifiedModel.build_encoder)
      encoder =
        Axon.input("input", shape: {nil, config.max_seq_length})
        |> Axon.embedding(config.vocab_size, config.embedding_size)
        |> Axon.lstm(config.hidden_size, name: "encoder_lstm")
        |> then(fn {seq, _state} -> seq end)

      # 2. Build combined encoder+head and train with EXLA (same as train_encoder_and_intent)
      combined_model =
        encoder
        |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)
        |> Axon.dense(64, activation: :relu, name: "intent_dense")
        |> Axon.dropout(rate: 0.1)
        |> Axon.dense(@num_classes, activation: :softmax, name: "intent_output")

      # Minimal fake data: 4 examples, 2 batches of 2
      batches =
        [
          {Nx.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], type: :s64),
           Nx.tensor([[1, 0, 0], [0, 1, 0]], type: :f32)},
          {Nx.tensor([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], type: :s64),
           Nx.tensor([[0, 0, 1], [1, 0, 0]], type: :f32)}
        ]
        |> Enum.map(fn {inputs, targets} -> {%{"input" => inputs}, targets} end)

      loop =
        combined_model
        |> Loop.trainer(:categorical_cross_entropy, Optimizers.adam(learning_rate: config.learning_rate))
        |> Loop.metric(:accuracy)

      trained_state = Loop.run(loop, batches, %{}, epochs: 1, compiler: EXLA, strict?: false)

      # trained_state is EXLA-backed (encoder params)
      encoder_params = trained_state

      # 3. Pre-compute encoder outputs with EXLA-backed params (the crash point without compiler: EXLA)
      fake_sentiment_data = [
        %{input: [0, 1, 2, 3, 4], sentiment: 0},
        %{input: [1, 2, 3, 4, 5], sentiment: 1},
        %{input: [2, 3, 4, 5, 6], sentiment: 2},
        %{input: [3, 4, 5, 6, 7], sentiment: 0}
      ]

      encoded_data =
        fake_sentiment_data
        |> Enum.chunk_every(config.batch_size)
        |> Enum.filter(fn batch -> length(batch) == config.batch_size end)
        |> Enum.map(fn batch ->
          inputs = batch |> Enum.map(& &1.input) |> Nx.tensor(type: :s64)
          labels = batch |> Enum.map(& &1.sentiment) |> Nx.tensor(type: :s64) |> Nx.new_axis(1)

          # Must use compiler: EXLA when params are EXLA-backed (from Loop.run with compiler: EXLA).
          # Without it: FunctionClauseError in Nx.BinaryBackend.to_binary/1 during LSTM recurrence.
          encoder_output =
            Axon.predict(encoder, encoder_params, %{"input" => inputs}, compiler: EXLA)

          pooled =
            encoder_output
            |> Nx.mean(axes: [1])
            |> Nx.backend_transfer(Nx.BinaryBackend)

          targets =
            Nx.equal(
              Nx.iota({config.batch_size, @num_classes}, axis: 1),
              labels
            )
            |> Nx.as_type(:f32)
            |> Nx.backend_transfer(Nx.BinaryBackend)

          {pooled, targets}
        end)

      # 4. Build standalone head and train (same as train_head_with_frozen_encoder)
      head_model =
        Axon.input("sentiment_input", shape: {nil, config.hidden_size})
        |> Axon.dense(64, activation: :relu, name: "sentiment_dense")
        |> Axon.dropout(rate: 0.1)
        |> Axon.dense(@num_classes, activation: :softmax, name: "sentiment_output")

      head_loop =
        head_model
        |> Loop.trainer(:categorical_cross_entropy, Optimizers.adam(learning_rate: config.learning_rate))
        |> Loop.metric(:accuracy)

      train_data =
        encoded_data
        |> Enum.map(fn {pooled, targets} -> {%{"sentiment_input" => pooled}, targets} end)

      head_state = Loop.run(head_loop, train_data, %{}, epochs: 1, compiler: EXLA, strict?: false)

      # 5. Assert we got trained params back
      assert head_state != nil
      param_data = if is_struct(head_state, Axon.ModelState), do: head_state.data, else: head_state
      assert is_map(param_data)
      assert map_size(param_data) > 0
    end
  end
end
