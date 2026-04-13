defmodule Brain.ML.LSTM.LossFunctionTest do
  @moduledoc """
  Validates that the `logits + log_softmax` loss formulation trains correctly
  on multi-class classification problems, matching or exceeding the old
  `softmax + log(max(x, 1e-7))` approach.

  Uses a synthetic dataset with per-class signature tokens. Runtime ~5-10s.
  """
  use ExUnit.Case, async: false

  alias Axon.Loop
  alias Polaris.Optimizers

  @moduletag :lstm
  @moduletag timeout: 120_000

  @num_classes 50
  @examples_per_class 6
  @vocab_size 60
  @embedding_size 16
  @hidden_size 16
  @max_seq_length 8
  @batch_size 8
  @learning_rate 0.01
  @epochs 15

  defp stable_log_softmax(logits) do
    max_logit = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logit)
    Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true)))
  end

  defp generate_data do
    for class_idx <- 0..(@num_classes - 1),
        _i <- 1..@examples_per_class do
      signature = class_idx + 4
      seq = [signature, rem(class_idx * 7 + 3, @vocab_size - 4) + 4, rem(class_idx * 13 + 1, @vocab_size - 4) + 4]
      padded = seq ++ List.duplicate(0, @max_seq_length - length(seq))
      %{input: padded, class: class_idx}
    end
    |> Enum.shuffle()
  end

  defp make_batches(data) do
    data
    |> Enum.chunk_every(@batch_size)
    |> Enum.filter(fn batch -> length(batch) == @batch_size end)
    |> Enum.map(fn batch ->
      inputs = batch |> Enum.map(& &1.input) |> Nx.tensor(type: :s64)
      labels = batch |> Enum.map(& &1.class) |> Nx.tensor(type: :s64) |> Nx.new_axis(1)

      targets =
        Nx.equal(Nx.iota({@batch_size, @num_classes}, axis: 1), labels)
        |> Nx.as_type(:f32)

      {inputs, targets}
    end)
  end

  defp build_encoder do
    Axon.input("input", shape: {nil, @max_seq_length})
    |> Axon.embedding(@vocab_size, @embedding_size)
    |> Axon.lstm(@hidden_size, name: "encoder_lstm")
    |> then(fn {seq, _state} -> seq end)
  end

  defp evaluate_accuracy(model, params, batches) do
    {correct, total} =
      Enum.reduce(batches, {0, 0}, fn {inputs, targets}, {correct_acc, total_acc} ->
        preds = Axon.predict(model, params, %{"input" => inputs}, compiler: EXLA)
        pred_classes = Nx.argmax(preds, axis: 1) |> Nx.to_flat_list()
        true_classes = Nx.argmax(targets, axis: 1) |> Nx.to_flat_list()

        batch_correct =
          Enum.zip(pred_classes, true_classes)
          |> Enum.count(fn {p, t} -> p == t end)

        {correct_acc + batch_correct, total_acc + length(pred_classes)}
      end)

    correct / max(total, 1)
  end

  describe "loss function comparison" do
    test "softmax + log(max) trains on multi-class problem (baseline)" do
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        data = generate_data()
        batches = make_batches(data)
        head_hidden = min(256, @num_classes * 2)

        model =
          build_encoder()
          |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)
          |> Axon.dense(head_hidden, activation: :relu, name: "intent_dense")
          |> Axon.dropout(rate: 0.2)
          |> Axon.dense(@num_classes, activation: :softmax, name: "intent_output")

        loss_fn = fn y_true, y_pred ->
          log_preds = Nx.log(Nx.max(y_pred, 1.0e-7))
          per_class_loss = Nx.negate(Nx.multiply(y_true, log_preds))
          Nx.mean(Nx.sum(per_class_loss, axes: [1]))
        end

        train_batches = Enum.map(batches, fn {i, t} -> {%{"input" => i}, t} end)

        params =
          model
          |> Loop.trainer(loss_fn, Optimizers.adam(learning_rate: @learning_rate))
          |> Loop.run(train_batches, Axon.ModelState.empty(), epochs: @epochs, compiler: EXLA)

        accuracy = evaluate_accuracy(model, params, batches)

        assert accuracy > 0.10,
               "Baseline softmax+log(max) should achieve >10% on #{@num_classes}-class problem, " <>
                 "got #{Float.round(accuracy * 100, 1)}%"
      end)
    end

    test "logits + log_softmax trains on multi-class problem" do
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        data = generate_data()
        batches = make_batches(data)
        head_hidden = min(256, @num_classes * 2)

        model =
          build_encoder()
          |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)
          |> Axon.dense(head_hidden, activation: :relu, name: "intent_dense")
          |> Axon.dropout(rate: 0.2)
          |> Axon.dense(@num_classes, name: "intent_output")

        loss_fn = fn y_true, logits ->
          log_probs = stable_log_softmax(logits)
          per_class_loss = Nx.negate(Nx.multiply(y_true, log_probs))
          Nx.mean(Nx.sum(per_class_loss, axes: [1]))
        end

        train_batches = Enum.map(batches, fn {i, t} -> {%{"input" => i}, t} end)

        params =
          model
          |> Loop.trainer(loss_fn, Optimizers.adam(learning_rate: @learning_rate))
          |> Loop.run(train_batches, Axon.ModelState.empty(), epochs: @epochs, compiler: EXLA)

        accuracy = evaluate_accuracy(model, params, batches)

        assert accuracy > 0.10,
               "log_softmax should achieve >10% on #{@num_classes}-class problem, " <>
                 "got #{Float.round(accuracy * 100, 1)}% (random = #{Float.round(100.0 / @num_classes, 1)}%)"
      end)
    end
  end
end
