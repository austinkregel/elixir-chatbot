defmodule Brain.ML.LSTM.IntentHead do
  @moduledoc """
  Intent classification head for the multi-task LSTM model.
  
  Takes a sentence vector from the shared encoder and produces
  intent classification probabilities using a dense layer + softmax.
  
  ## Architecture
  
  ```
  sentence_vector [batch, hidden*2]
        |
        v
  Dense [batch, num_intents]
        |
        v
  Softmax [batch, num_intents]
        |
        v
  Intent probabilities
  ```
  """
  
  require Logger
  import Nx.Defn

  @doc """
  Build the intent classification head.
  
  ## Parameters
  - `input_size`: Size of input (encoder hidden_size * 2)
  - `num_intents`: Number of intent classes
  - `opts`: Additional options
    - `:dropout` - Dropout rate before classification (default: 0.1)
  
  ## Returns
  Axon model for intent classification
  """
  def build_model(input_size, num_intents, opts \\ []) do
    dropout_rate = Keyword.get(opts, :dropout, 0.1)
    
    # Input: sentence vector from encoder [batch, input_size]
    input = Axon.input("sentence_vector", shape: {nil, input_size})
    
    input
    |> Axon.dropout(rate: dropout_rate, name: "intent_dropout")
    |> Axon.dense(num_intents, name: "intent_dense")
    |> Axon.softmax(name: "intent_softmax")
  end
  
  @doc """
  Initialize intent head parameters.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `input_size`: Size of input for template
  
  ## Returns
  Initialized parameters map
  """
  def init_params(model, input_size) do
    {init_fn, _predict_fn} = Axon.build(model)
    
    template = %{"sentence_vector" => Nx.template({1, input_size}, :f32)}
    
    params = init_fn.(template, Axon.ModelState.empty())
    Axon.ModelState.new(params)
  end
  
  @doc """
  Classify intent from sentence vector.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `sentence_vector`: [batch, hidden*2] tensor from encoder
  - `params`: Model parameters
  
  ## Returns
  Intent probabilities [batch, num_intents]
  """
  def forward(model, sentence_vector, params) do
    Axon.predict(model, params, %{"sentence_vector" => sentence_vector})
  end
  
  @doc """
  Classify a single input and return the predicted intent with confidence.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `sentence_vector`: [hidden*2] tensor (single sample, no batch dim)
  - `params`: Model parameters
  - `intent_labels`: List of intent label strings in order
  
  ## Returns
  `{intent_label, confidence, all_scores}` where:
  - `intent_label`: String label of predicted intent
  - `confidence`: Float confidence score (0-1)
  - `all_scores`: Map of intent -> score for all intents
  """
  def classify(model, sentence_vector, params, intent_labels) do
    # Add batch dimension
    batched = Nx.new_axis(sentence_vector, 0)
    
    # Get probabilities
    probs = forward(model, batched, params)
    
    # Remove batch dimension
    probs = Nx.squeeze(probs, axes: [0])
    
    # Get predicted class
    predicted_idx = Nx.argmax(probs) |> Nx.to_number()
    confidence = Nx.to_number(probs[predicted_idx])
    
    # Build all scores map
    all_scores = 
      intent_labels
      |> Enum.with_index()
      |> Enum.map(fn {label, idx} -> {label, Nx.to_number(probs[idx])} end)
      |> Enum.into(%{})
    
    intent_label = Enum.at(intent_labels, predicted_idx)
    
    {intent_label, confidence, all_scores}
  end
  
  @doc """
  Compute cross-entropy loss for intent classification.
  
  ## Parameters
  - `predictions`: [batch, num_intents] predicted probabilities
  - `targets`: [batch] integer tensor of target intent indices
  
  ## Returns
  Scalar loss value
  """
  defn compute_loss(predictions, targets) do
    # Add small epsilon to prevent log(0)
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    
    # One-hot encode targets
    num_classes = Nx.axis_size(predictions, 1)
    targets_onehot = Nx.equal(
      Nx.new_axis(targets, 1),
      Nx.iota({1, num_classes})
    ) |> Nx.as_type(:f32)
    
    # Cross-entropy: -sum(y * log(p))
    -Nx.sum(targets_onehot * Nx.log(predictions)) / Nx.axis_size(predictions, 0)
  end
  
  @doc """
  Get top-k predictions with their scores.
  
  ## Parameters
  - `model`: Axon model
  - `sentence_vector`: [hidden*2] tensor
  - `params`: Model parameters
  - `intent_labels`: List of intent label strings
  - `k`: Number of top predictions to return
  
  ## Returns
  List of `{intent_label, score}` tuples, sorted by score descending
  """
  def top_k(model, sentence_vector, params, intent_labels, k \\ 5) do
    # Add batch dimension
    batched = Nx.new_axis(sentence_vector, 0)
    probs = forward(model, batched, params) |> Nx.squeeze(axes: [0])
    
    # Convert to list and sort
    intent_labels
    |> Enum.with_index()
    |> Enum.map(fn {label, idx} -> {label, Nx.to_number(probs[idx])} end)
    |> Enum.sort_by(fn {_label, score} -> -score end)
    |> Enum.take(k)
  end
end
