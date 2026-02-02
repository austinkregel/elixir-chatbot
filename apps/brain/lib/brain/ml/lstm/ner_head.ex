defmodule Brain.ML.LSTM.NERHead do
  @moduledoc """
  Named Entity Recognition (NER) head for the multi-task LSTM model.
  
  Takes token-level outputs from the shared encoder and produces
  BIO tag probabilities for each token using a dense layer + softmax.
  
  ## BIO Tagging Scheme
  
  - B-TYPE: Beginning of an entity of TYPE
  - I-TYPE: Inside (continuation) of an entity of TYPE
  - O: Outside any entity
  
  For example:
  - Input: "what is the weather in London"
  - Output: ["O", "O", "O", "O", "O", "B-LOC"]
  
  ## Architecture
  
  ```
  token_outputs [batch, seq_len, hidden*2]
        |
        v
  Dense [batch, seq_len, num_bio_tags]
        |
        v
  Softmax [batch, seq_len, num_bio_tags]
        |
        v
  BIO tag probabilities per token
  ```
  """
  
  require Logger
  import Nx.Defn

  @doc """
  Build the NER head model.
  
  ## Parameters
  - `input_size`: Size of token representation (encoder hidden_size * 2)
  - `num_bio_tags`: Number of BIO tags (O + B-type + I-type for each entity type)
  - `opts`: Additional options
    - `:dropout` - Dropout rate (default: 0.1)
  
  ## Returns
  Axon model for sequence tagging
  """
  def build_model(input_size, num_bio_tags, opts \\ []) do
    dropout_rate = Keyword.get(opts, :dropout, 0.1)
    
    # Input: token outputs from encoder [batch, seq_len, input_size]
    input = Axon.input("token_outputs", shape: {nil, nil, input_size})
    
    input
    |> Axon.dropout(rate: dropout_rate, name: "ner_dropout")
    |> Axon.dense(num_bio_tags, name: "ner_dense")
    |> Axon.activation(:softmax, name: "ner_softmax")
  end
  
  @doc """
  Initialize NER head parameters.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `input_size`: Size of token representations
  - `seq_len`: Sequence length for template (default: 10)
  
  ## Returns
  Initialized parameters map
  """
  def init_params(model, input_size, seq_len \\ 10) do
    {init_fn, _predict_fn} = Axon.build(model)
    
    template = %{"token_outputs" => Nx.template({1, seq_len, input_size}, :f32)}
    
    params = init_fn.(template, Axon.ModelState.empty())
    Axon.ModelState.new(params)
  end
  
  @doc """
  Forward pass through NER head.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `token_outputs`: [batch, seq_len, hidden*2] tensor from encoder
  - `params`: Model parameters
  
  ## Returns
  BIO tag probabilities [batch, seq_len, num_bio_tags]
  """
  def forward(model, token_outputs, params) do
    Axon.predict(model, params, %{"token_outputs" => token_outputs})
  end
  
  @doc """
  Tag a sequence with BIO labels.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `token_outputs`: [seq_len, hidden*2] tensor (single sample)
  - `params`: Model parameters
  - `bio_labels`: List of BIO label strings in order
  
  ## Returns
  List of `{token_idx, bio_tag, confidence}` tuples
  """
  def tag_sequence(model, token_outputs, params, bio_labels) do
    # Add batch dimension
    batched = Nx.new_axis(token_outputs, 0)
    
    # Get predictions
    probs = forward(model, batched, params)
    probs = Nx.squeeze(probs, axes: [0])  # [seq_len, num_tags]
    
    seq_len = Nx.axis_size(probs, 0)
    
    # Get argmax for each position
    predicted_indices = Nx.argmax(probs, axis: 1) |> Nx.to_flat_list()
    
    # Get confidence for each position
    confidences = 
      for i <- 0..(seq_len - 1) do
        idx = Enum.at(predicted_indices, i)
        Nx.to_number(probs[i][idx])
      end
    
    # Build result
    Enum.zip([0..(seq_len - 1), predicted_indices, confidences])
    |> Enum.map(fn {token_idx, tag_idx, conf} ->
      tag = Enum.at(bio_labels, tag_idx, "O")
      {token_idx, tag, conf}
    end)
  end
  
  @doc """
  Extract entities from BIO tag sequence.
  
  ## Parameters
  - `tokens`: List of token strings
  - `bio_tags`: List of BIO tags (same length as tokens)
  
  ## Returns
  List of extracted entities:
  ```
  [
    %{text: "London", type: "location", start: 5, end: 5, tokens: ["London"]},
    %{text: "Taylor Swift", type: "artist", start: 2, end: 3, tokens: ["Taylor", "Swift"]}
  ]
  ```
  """
  def extract_entities_from_tags(tokens, bio_tags) do
    tokens
    |> Enum.zip(bio_tags)
    |> Enum.with_index()
    |> Enum.reduce([], fn {{token, tag}, idx}, entities ->
      cond do
        # Beginning of new entity
        String.starts_with?(tag, "B-") ->
          entity_type = String.replace_prefix(tag, "B-", "")
          new_entity = %{
            type: entity_type,
            start: idx,
            end: idx,
            tokens: [token]
          }
          [new_entity | entities]
        
        # Continuation of entity
        String.starts_with?(tag, "I-") and length(entities) > 0 ->
          [current | rest] = entities
          expected_type = String.replace_prefix(tag, "I-", "")
          
          if current.type == expected_type do
            # Extend current entity
            updated = %{current |
              end: idx,
              tokens: current.tokens ++ [token]
            }
            [updated | rest]
          else
            # Type mismatch - treat as new B- tag
            new_entity = %{
              type: expected_type,
              start: idx,
              end: idx,
              tokens: [token]
            }
            [new_entity | entities]
          end
        
        # Outside or I- without B- (treat as O)
        true ->
          entities
      end
    end)
    |> Enum.reverse()
    |> Enum.map(fn entity ->
      Map.put(entity, :text, Enum.join(entity.tokens, " "))
    end)
  end
  
  @doc """
  Tag a sequence and extract entities in one step.
  
  ## Parameters
  - `model`: Axon model
  - `tokens`: List of token strings
  - `token_outputs`: [seq_len, hidden*2] tensor
  - `params`: Model parameters
  - `bio_labels`: List of BIO label strings
  
  ## Returns
  List of extracted entities
  """
  def extract_entities(model, tokens, token_outputs, params, bio_labels) do
    tags_with_conf = tag_sequence(model, token_outputs, params, bio_labels)
    tags = Enum.map(tags_with_conf, fn {_idx, tag, _conf} -> tag end)
    
    # Truncate tags to match tokens length if needed
    tags = Enum.take(tags, length(tokens))
    
    extract_entities_from_tags(tokens, tags)
  end
  
  @doc """
  Compute cross-entropy loss for NER.
  
  ## Parameters
  - `predictions`: [batch, seq_len, num_tags] predicted probabilities
  - `targets`: [batch, seq_len] integer tensor of target BIO tag indices
  - `mask`: [batch, seq_len] mask tensor (1 for real tokens, 0 for padding)
  
  ## Returns
  Scalar loss value
  """
  defn compute_loss(predictions, targets, mask) do
    num_tags = Nx.axis_size(predictions, 2)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    
    # One-hot encode targets: [batch, seq_len] -> [batch, seq_len, num_tags]
    targets_expanded = Nx.new_axis(targets, 2)
    tag_indices = Nx.iota({1, 1, num_tags})
    targets_onehot = Nx.equal(targets_expanded, tag_indices) |> Nx.as_type(:f32)
    
    # Cross-entropy: -sum(y * log(p))
    token_losses = -Nx.sum(targets_onehot * Nx.log(predictions), axes: [2])
    
    # Apply mask and compute masked average
    masked_losses = token_losses * mask
    
    # Average over non-padding tokens (add epsilon to prevent div by zero)
    Nx.sum(masked_losses) / (Nx.sum(mask) + epsilon)
  end
  
  @doc """
  Compute cross-entropy loss for NER without masking.
  
  ## Parameters
  - `predictions`: [batch, seq_len, num_tags] predicted probabilities
  - `targets`: [batch, seq_len] integer tensor of target BIO tag indices
  
  ## Returns
  Scalar loss value
  """
  defn compute_loss_unmasked(predictions, targets) do
    num_tags = Nx.axis_size(predictions, 2)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    
    # One-hot encode targets: [batch, seq_len] -> [batch, seq_len, num_tags]
    targets_expanded = Nx.new_axis(targets, 2)
    tag_indices = Nx.iota({1, 1, num_tags})
    targets_onehot = Nx.equal(targets_expanded, tag_indices) |> Nx.as_type(:f32)
    
    # Cross-entropy: -sum(y * log(p))
    token_losses = -Nx.sum(targets_onehot * Nx.log(predictions), axes: [2])
    
    Nx.mean(token_losses)
  end
  
  @doc """
  Get entity type from BIO tag.
  
  ## Examples
  
      iex> NERHead.entity_type_from_bio("B-location")
      "location"
      iex> NERHead.entity_type_from_bio("I-artist")
      "artist"
      iex> NERHead.entity_type_from_bio("O")
      nil
  """
  def entity_type_from_bio(tag) do
    cond do
      String.starts_with?(tag, "B-") -> String.replace_prefix(tag, "B-", "")
      String.starts_with?(tag, "I-") -> String.replace_prefix(tag, "I-", "")
      true -> nil
    end
  end
  
  @doc """
  Get all unique entity types from BIO vocabulary.
  """
  def entity_types_from_bio_vocab(bio_labels) do
    bio_labels
    |> Enum.filter(&String.starts_with?(&1, "B-"))
    |> Enum.map(&String.replace_prefix(&1, "B-", ""))
    |> Enum.uniq()
    |> Enum.sort()
  end
end
