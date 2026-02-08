defmodule Brain.ML.LSTM.IntentHead do
  @moduledoc "Intent classification head for the multi-task LSTM model.\n\nTakes a sentence vector from the shared encoder and produces\nintent classification probabilities using a dense layer + softmax.\n\n## Architecture\n\n```\nsentence_vector [batch, hidden*2]\n      |\n      v\nDense [batch, num_intents]\n      |\n      v\nSoftmax [batch, num_intents]\n      |\n      v\nIntent probabilities\n```\n"

  alias Axon.ModelState
  require Logger
  import Nx.Defn

  @doc "Build the intent classification head.\n\n## Parameters\n- `input_size`: Size of input (encoder hidden_size * 2)\n- `num_intents`: Number of intent classes\n- `opts`: Additional options\n  - `:dropout` - Dropout rate before classification (default: 0.1)\n\n## Returns\nAxon model for intent classification\n"
  def build_model(input_size, num_intents, opts \\ []) do
    dropout_rate = Keyword.get(opts, :dropout, 0.1)
    input = Axon.input("sentence_vector", shape: {nil, input_size})

    input
    |> Axon.dropout(rate: dropout_rate, name: "intent_dropout")
    |> Axon.dense(num_intents, name: "intent_dense")
    |> Axon.softmax(name: "intent_softmax")
  end

  @doc "Initialize intent head parameters.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `input_size`: Size of input for template\n\n## Returns\nInitialized parameters map\n"
  def init_params(model, input_size) do
    {init_fn, _predict_fn} = Axon.build(model)

    template = %{"sentence_vector" => Nx.template({1, input_size}, :f32)}

    params = init_fn.(template, ModelState.empty())
    ModelState.new(params)
  end

  @doc "Classify intent from sentence vector.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `sentence_vector`: [batch, hidden*2] tensor from encoder\n- `params`: Model parameters\n\n## Returns\nIntent probabilities [batch, num_intents]\n"
  def forward(model, sentence_vector, params) do
    Axon.predict(model, params, %{"sentence_vector" => sentence_vector})
  end

  @doc "Classify a single input and return the predicted intent with confidence.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `sentence_vector`: [hidden*2] tensor (single sample, no batch dim)\n- `params`: Model parameters\n- `intent_labels`: List of intent label strings in order\n\n## Returns\n`{intent_label, confidence, all_scores}` where:\n- `intent_label`: String label of predicted intent\n- `confidence`: Float confidence score (0-1)\n- `all_scores`: Map of intent -> score for all intents\n"
  def classify(model, sentence_vector, params, intent_labels) do
    batched = Nx.new_axis(sentence_vector, 0)
    probs = forward(model, batched, params)
    probs = Nx.squeeze(probs, axes: [0])
    predicted_idx = Nx.argmax(probs) |> Nx.to_number()
    confidence = Nx.to_number(probs[predicted_idx])

    all_scores =
      intent_labels
      |> Enum.with_index()
      |> Enum.map(fn {label, idx} -> {label, Nx.to_number(probs[idx])} end)
      |> Enum.into(%{})

    intent_label = Enum.at(intent_labels, predicted_idx)

    {intent_label, confidence, all_scores}
  end

  @doc "Compute cross-entropy loss for intent classification.\n\n## Parameters\n- `predictions`: [batch, num_intents] predicted probabilities\n- `targets`: [batch] integer tensor of target intent indices\n\n## Returns\nScalar loss value\n"
  defn compute_loss(predictions, targets) do
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    num_classes = Nx.axis_size(predictions, 1)

    targets_onehot =
      Nx.equal(
        Nx.new_axis(targets, 1),
        Nx.iota({1, num_classes})
      )
      |> Nx.as_type(:f32)

    -Nx.sum(targets_onehot * Nx.log(predictions)) / Nx.axis_size(predictions, 0)
  end

  @doc "Get top-k predictions with their scores.\n\n## Parameters\n- `model`: Axon model\n- `sentence_vector`: [hidden*2] tensor\n- `params`: Model parameters\n- `intent_labels`: List of intent label strings\n- `k`: Number of top predictions to return\n\n## Returns\nList of `{intent_label, score}` tuples, sorted by score descending\n"
  def top_k(model, sentence_vector, params, intent_labels, k \\ 5) do
    batched = Nx.new_axis(sentence_vector, 0)
    probs = forward(model, batched, params) |> Nx.squeeze(axes: [0])

    intent_labels
    |> Enum.with_index()
    |> Enum.map(fn {label, idx} -> {label, Nx.to_number(probs[idx])} end)
    |> Enum.sort_by(fn {_label, score} -> -score end)
    |> Enum.take(k)
  end
end