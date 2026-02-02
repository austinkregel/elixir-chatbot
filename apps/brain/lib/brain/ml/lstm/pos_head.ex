defmodule Brain.ML.LSTM.POSHead do
  @moduledoc """
  Part-of-Speech (POS) tagging head for the multi-task LSTM model.
  
  Takes token-level outputs from the shared encoder and produces
  POS tag probabilities for each token using a dense layer + softmax.
  
  ## POS Tags (Universal Dependencies)
  
  The model uses Universal POS tags:
  - NOUN: noun
  - VERB: verb
  - ADJ: adjective
  - ADV: adverb
  - PRON: pronoun
  - DET: determiner
  - ADP: adposition (preposition/postposition)
  - CONJ: conjunction
  - NUM: numeral
  - PART: particle
  - INTJ: interjection
  - PUNCT: punctuation
  - PROPN: proper noun
  - AUX: auxiliary verb
  - X: other
  
  ## Architecture
  
  ```
  token_outputs [batch, seq_len, hidden*2]
        |
        v
  Dense [batch, seq_len, num_pos_tags]
        |
        v
  Softmax [batch, seq_len, num_pos_tags]
        |
        v
  POS tag probabilities per token
  ```
  """
  
  require Logger
  import Nx.Defn

  # Standard Universal POS tags
  @pos_tags ~w(
    NOUN PROPN VERB AUX ADJ ADV PRON DET ADP
    CONJ PART NUM INTJ PUNCT SYM X
  )

  @doc """
  Get the standard POS tag list.
  """
  def pos_tags, do: @pos_tags

  @doc """
  Build the POS tagging head model.
  
  ## Parameters
  - `input_size`: Size of token representation (encoder hidden_size * 2)
  - `num_pos_tags`: Number of POS tags (default: 16)
  - `opts`: Additional options
    - `:dropout` - Dropout rate (default: 0.1)
  
  ## Returns
  Axon model for POS tagging
  """
  def build_model(input_size, num_pos_tags \\ 16, opts \\ []) do
    dropout_rate = Keyword.get(opts, :dropout, 0.1)
    
    # Input: token outputs from encoder [batch, seq_len, input_size]
    input = Axon.input("token_outputs", shape: {nil, nil, input_size})
    
    input
    |> Axon.dropout(rate: dropout_rate, name: "pos_dropout")
    |> Axon.dense(num_pos_tags, name: "pos_dense")
    |> Axon.activation(:softmax, name: "pos_softmax")
  end
  
  @doc """
  Initialize POS head parameters.
  
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
  Forward pass through POS head.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `token_outputs`: [batch, seq_len, hidden*2] tensor from encoder
  - `params`: Model parameters
  
  ## Returns
  POS tag probabilities [batch, seq_len, num_pos_tags]
  """
  def forward(model, token_outputs, params) do
    Axon.predict(model, params, %{"token_outputs" => token_outputs})
  end
  
  @doc """
  Tag a sequence with POS labels.
  
  ## Parameters
  - `model`: Axon model from `build_model/3`
  - `token_outputs`: [seq_len, hidden*2] tensor (single sample)
  - `params`: Model parameters
  - `pos_labels`: List of POS label strings in order
  
  ## Returns
  List of `{token_idx, pos_tag, confidence}` tuples
  """
  def tag_sequence(model, token_outputs, params, pos_labels) do
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
      tag = Enum.at(pos_labels, tag_idx, "X")
      {token_idx, tag, conf}
    end)
  end
  
  @doc """
  Tag tokens and return token-tag pairs.
  
  ## Parameters
  - `model`: Axon model
  - `tokens`: List of token strings
  - `token_outputs`: [seq_len, hidden*2] tensor
  - `params`: Model parameters
  - `pos_labels`: List of POS label strings
  
  ## Returns
  List of `{token, pos_tag}` tuples
  """
  def tag_tokens(model, tokens, token_outputs, params, pos_labels) do
    tags_with_conf = tag_sequence(model, token_outputs, params, pos_labels)
    
    tokens
    |> Enum.with_index()
    |> Enum.map(fn {token, idx} ->
      case Enum.find(tags_with_conf, fn {i, _, _} -> i == idx end) do
        {_, tag, _conf} -> {token, tag}
        nil -> {token, "X"}
      end
    end)
  end
  
  @doc """
  Compute cross-entropy loss for POS tagging.
  
  ## Parameters
  - `predictions`: [batch, seq_len, num_tags] predicted probabilities
  - `targets`: [batch, seq_len] integer tensor of target POS tag indices
  - `mask`: [batch, seq_len] mask tensor (1 for real tokens, 0 for padding)
  
  ## Returns
  Scalar loss value
  """
  defn compute_loss(predictions, targets, mask) do
    num_tags = Nx.axis_size(predictions, 2)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    
    # One-hot encode targets
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
  Compute cross-entropy loss for POS tagging without masking.
  
  ## Parameters
  - `predictions`: [batch, seq_len, num_tags] predicted probabilities
  - `targets`: [batch, seq_len] integer tensor of target POS tag indices
  
  ## Returns
  Scalar loss value
  """
  defn compute_loss_unmasked(predictions, targets) do
    num_tags = Nx.axis_size(predictions, 2)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    
    # One-hot encode targets
    targets_expanded = Nx.new_axis(targets, 2)
    tag_indices = Nx.iota({1, 1, num_tags})
    targets_onehot = Nx.equal(targets_expanded, tag_indices) |> Nx.as_type(:f32)
    
    # Cross-entropy: -sum(y * log(p))
    token_losses = -Nx.sum(targets_onehot * Nx.log(predictions), axes: [2])
    
    Nx.mean(token_losses)
  end
  
  @doc """
  Build POS vocabulary mappings.
  
  Returns `{pos_to_idx, idx_to_pos}` maps using standard Universal POS tags.
  """
  def build_pos_vocabulary do
    pos_to_idx =
      @pos_tags
      |> Enum.with_index()
      |> Enum.into(%{})
    
    idx_to_pos =
      pos_to_idx
      |> Enum.map(fn {k, v} -> {v, k} end)
      |> Enum.into(%{})
    
    {pos_to_idx, idx_to_pos}
  end
  
  @doc """
  Check if a POS tag indicates a content word (noun, verb, adj, adv).
  """
  def content_word?(pos_tag) do
    pos_tag in ["NOUN", "PROPN", "VERB", "ADJ", "ADV"]
  end
  
  @doc """
  Check if a POS tag indicates a function word (determiner, preposition, etc.).
  """
  def function_word?(pos_tag) do
    pos_tag in ["DET", "ADP", "CONJ", "PART", "AUX"]
  end
  
  @doc """
  Get the general category of a POS tag.
  """
  def pos_category(pos_tag) do
    cond do
      pos_tag in ["NOUN", "PROPN"] -> :noun
      pos_tag in ["VERB", "AUX"] -> :verb
      pos_tag in ["ADJ"] -> :adjective
      pos_tag in ["ADV"] -> :adverb
      pos_tag in ["PRON"] -> :pronoun
      pos_tag in ["DET", "ADP", "CONJ", "PART"] -> :function
      pos_tag in ["NUM"] -> :number
      pos_tag in ["INTJ"] -> :interjection
      pos_tag in ["PUNCT"] -> :punctuation
      true -> :other
    end
  end
end
