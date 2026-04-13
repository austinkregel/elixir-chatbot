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
  use ExUnit.Case, async: false

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
      head_hidden = min(256, max(64, @num_classes * 2))

      combined_model =
        encoder
        |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)
        |> Axon.dense(head_hidden, activation: :relu, name: "intent_dense")
        |> Axon.dropout(rate: 0.3)
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

      trained_state = Loop.run(loop, batches, Axon.ModelState.empty(), epochs: 1, compiler: EXLA)

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
      sentiment_hidden = min(256, @num_classes * 2)

      head_model =
        Axon.input("sentiment_input", shape: {nil, config.hidden_size})
        |> Axon.dense(sentiment_hidden, activation: :relu, name: "sentiment_dense")
        |> Axon.dropout(rate: 0.3)
        |> Axon.dense(@num_classes, activation: :softmax, name: "sentiment_output")

      head_loop =
        head_model
        |> Loop.trainer(:categorical_cross_entropy, Optimizers.adam(learning_rate: config.learning_rate))
        |> Loop.metric(:accuracy)

      train_data =
        encoded_data
        |> Enum.map(fn {pooled, targets} -> {%{"sentiment_input" => pooled}, targets} end)

      head_state = Loop.run(head_loop, train_data, Axon.ModelState.empty(), epochs: 1, compiler: EXLA)

      # 5. Assert we got trained params back
      assert head_state != nil
      param_data = if is_struct(head_state, Axon.ModelState), do: head_state.data, else: head_state
      assert is_map(param_data)
      assert map_size(param_data) > 0
    end

    test "encoder params are bitwise identical before and after head training" do
      encoder = build_test_encoder()
      {encoder_params, _} = train_combined_encoder(encoder)

      encoder_params_binary = params_to_binary_snapshot(encoder_params)

      _head_state = train_frozen_head(encoder, encoder_params, :sentiment)

      encoder_params_after = params_to_binary_snapshot(encoder_params)

      assert encoder_params_binary == encoder_params_after,
             "Encoder params were mutated during frozen head training"
    end

    test "multiple heads can be trained on the same frozen encoder" do
      encoder = build_test_encoder()
      {encoder_params, _} = train_combined_encoder(encoder)

      encoder_before = params_to_binary_snapshot(encoder_params)

      sentiment_state = train_frozen_head(encoder, encoder_params, :sentiment)
      speech_act_state = train_frozen_head(encoder, encoder_params, :speech_act)

      encoder_after = params_to_binary_snapshot(encoder_params)

      assert encoder_before == encoder_after,
             "Encoder was mutated after training two heads"

      assert sentiment_state != nil
      assert speech_act_state != nil

      sent_data = extract_param_data(sentiment_state)
      sa_data = extract_param_data(speech_act_state)

      assert map_size(sent_data) > 0
      assert map_size(sa_data) > 0
    end

    test "standalone head produces valid predictions via Axon.predict" do
      encoder = build_test_encoder()
      {encoder_params, _} = train_combined_encoder(encoder)

      head_state = train_frozen_head(encoder, encoder_params, :sentiment, epochs: 5)

      head_model =
        Axon.input("sentiment_input", shape: {nil, @hidden_size})
        |> Axon.dense(min(256, @num_classes * 2), activation: :relu, name: "sentiment_dense")
        |> Axon.dropout(rate: 0.3)
        |> Axon.dense(@num_classes, name: "sentiment_output")

      test_input = Nx.tensor([[1, 2, 3, 4, 5]], type: :s64)

      encoder_output =
        Axon.predict(encoder, encoder_params, %{"input" => test_input}, compiler: EXLA)

      pooled =
        encoder_output
        |> Nx.mean(axes: [1])
        |> Nx.backend_transfer(Nx.BinaryBackend)

      prediction =
        Axon.predict(head_model, head_state, %{"sentiment_input" => pooled}, compiler: EXLA)

      assert prediction.shape == {1, @num_classes}
      prediction_list = Nx.to_flat_list(prediction)
      assert length(prediction_list) == @num_classes
      assert Enum.all?(prediction_list, &is_float/1)
    end

    test "head params contain only head-specific keys, not encoder keys" do
      encoder = build_test_encoder()
      {encoder_params, _} = train_combined_encoder(encoder)

      head_state = train_frozen_head(encoder, encoder_params, :sentiment)

      head_data = extract_param_data(head_state)
      head_keys = Map.keys(head_data)

      assert Enum.any?(head_keys, fn k -> String.contains?(to_string(k), "sentiment") end),
             "Head params should contain sentiment-prefixed keys, got: #{inspect(head_keys)}"

      refute Enum.any?(head_keys, fn k ->
               to_string(k) |> String.contains?("encoder") or
                 to_string(k) |> String.contains?("embedding")
             end),
             "Head params should NOT contain encoder keys, got: #{inspect(head_keys)}"
    end
  end

  # --- Test Helpers ---

  defp build_test_encoder do
    Axon.input("input", shape: {nil, @max_seq_length})
    |> Axon.embedding(@vocab_size, @embedding_size)
    |> Axon.lstm(@hidden_size, name: "encoder_lstm")
    |> then(fn {seq, _state} -> seq end)
  end

  defp train_combined_encoder(encoder) do
    head_hidden = min(256, max(64, @num_classes * 2))

    combined =
      encoder
      |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)
      |> Axon.dense(head_hidden, activation: :relu, name: "intent_dense")
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(@num_classes, activation: :softmax, name: "intent_output")

    batches =
      [
        {Nx.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], type: :s64),
         Nx.tensor([[1, 0, 0], [0, 1, 0]], type: :f32)},
        {Nx.tensor([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], type: :s64),
         Nx.tensor([[0, 0, 1], [1, 0, 0]], type: :f32)}
      ]
      |> Enum.map(fn {inputs, targets} -> {%{"input" => inputs}, targets} end)

    loop =
      combined
      |> Loop.trainer(:categorical_cross_entropy, Optimizers.adam(learning_rate: @learning_rate))
      |> Loop.metric(:accuracy)

    state = Loop.run(loop, batches, Axon.ModelState.empty(), epochs: 2, compiler: EXLA)
    {state, combined}
  end

  defp train_frozen_head(encoder, encoder_params, head_name, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 1)
    name = to_string(head_name)
    head_hidden = min(256, max(64, @num_classes * 2))

    data = [
      %{input: [0, 1, 2, 3, 4], label: 0},
      %{input: [1, 2, 3, 4, 5], label: 1},
      %{input: [2, 3, 4, 5, 6], label: 2},
      %{input: [3, 4, 5, 6, 7], label: 0}
    ]

    train_data =
      data
      |> Enum.chunk_every(@batch_size)
      |> Enum.filter(fn batch -> length(batch) == @batch_size end)
      |> Enum.map(fn batch ->
        inputs = batch |> Enum.map(& &1.input) |> Nx.tensor(type: :s64)
        labels = batch |> Enum.map(& &1.label) |> Nx.tensor(type: :s64) |> Nx.new_axis(1)

        encoder_output =
          Axon.predict(encoder, encoder_params, %{"input" => inputs}, compiler: EXLA)

        pooled =
          encoder_output
          |> Nx.mean(axes: [1])
          |> Nx.backend_transfer(Nx.BinaryBackend)

        targets =
          Nx.equal(
            Nx.iota({@batch_size, @num_classes}, axis: 1),
            labels
          )
          |> Nx.as_type(:f32)
          |> Nx.backend_transfer(Nx.BinaryBackend)

        {%{"#{name}_input" => pooled}, targets}
      end)

    head_model =
      Axon.input("#{name}_input", shape: {nil, @hidden_size})
      |> Axon.dense(head_hidden, activation: :relu, name: "#{name}_dense")
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(@num_classes, activation: :softmax, name: "#{name}_output")

    loop =
      head_model
      |> Loop.trainer(:categorical_cross_entropy, Optimizers.adam(learning_rate: @learning_rate))
      |> Loop.metric(:accuracy)

    Loop.run(loop, train_data, Axon.ModelState.empty(), epochs: epochs, compiler: EXLA)
  end

  defp extract_param_data(state) do
    if is_struct(state, Axon.ModelState), do: state.data, else: state
  end

  defp params_to_binary_snapshot(params) do
    data = extract_param_data(params)

    data
    |> Enum.sort()
    |> Enum.map(fn {key, val} ->
      {key, tensor_to_binary(val)}
    end)
  end

  defp tensor_to_binary(%Nx.Tensor{} = t) do
    t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_binary()
  end

  defp tensor_to_binary(map) when is_map(map) do
    map |> Enum.sort() |> Enum.map(fn {k, v} -> {k, tensor_to_binary(v)} end)
  end

  defp tensor_to_binary(other), do: other
end
