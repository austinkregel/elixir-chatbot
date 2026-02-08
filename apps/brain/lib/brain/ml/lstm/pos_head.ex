defmodule Brain.ML.LSTM.POSHead do
  @moduledoc "Part-of-Speech (POS) tagging head for the multi-task LSTM model.\n\nTakes token-level outputs from the shared encoder and produces\nPOS tag probabilities for each token using a dense layer + softmax.\n\n## POS Tags (Universal Dependencies)\n\nThe model uses Universal POS tags:\n- NOUN: noun\n- VERB: verb\n- ADJ: adjective\n- ADV: adverb\n- PRON: pronoun\n- DET: determiner\n- ADP: adposition (preposition/postposition)\n- CONJ: conjunction\n- NUM: numeral\n- PART: particle\n- INTJ: interjection\n- PUNCT: punctuation\n- PROPN: proper noun\n- AUX: auxiliary verb\n- X: other\n\n## Architecture\n\n```\ntoken_outputs [batch, seq_len, hidden*2]\n      |\n      v\nDense [batch, seq_len, num_pos_tags]\n      |\n      v\nSoftmax [batch, seq_len, num_pos_tags]\n      |\n      v\nPOS tag probabilities per token\n```\n"

  alias Axon.ModelState
  require Logger
  import Nx.Defn
  @pos_tags ~w(
    NOUN PROPN VERB AUX ADJ ADV PRON DET ADP
    CONJ PART NUM INTJ PUNCT SYM X
  )

  @doc "Get the standard POS tag list.\n"
  def pos_tags do
    @pos_tags
  end

  @doc "Build the POS tagging head model.\n\n## Parameters\n- `input_size`: Size of token representation (encoder hidden_size * 2)\n- `num_pos_tags`: Number of POS tags (default: 16)\n- `opts`: Additional options\n  - `:dropout` - Dropout rate (default: 0.1)\n\n## Returns\nAxon model for POS tagging\n"
  def build_model(input_size, num_pos_tags \\ 16, opts \\ []) do
    dropout_rate = Keyword.get(opts, :dropout, 0.1)
    input = Axon.input("token_outputs", shape: {nil, nil, input_size})

    input
    |> Axon.dropout(rate: dropout_rate, name: "pos_dropout")
    |> Axon.dense(num_pos_tags, name: "pos_dense")
    |> Axon.activation(:softmax, name: "pos_softmax")
  end

  @doc "Initialize POS head parameters.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `input_size`: Size of token representations\n- `seq_len`: Sequence length for template (default: 10)\n\n## Returns\nInitialized parameters map\n"
  def init_params(model, input_size, seq_len \\ 10) do
    {init_fn, _predict_fn} = Axon.build(model)

    template = %{"token_outputs" => Nx.template({1, seq_len, input_size}, :f32)}

    params = init_fn.(template, ModelState.empty())
    ModelState.new(params)
  end

  @doc "Forward pass through POS head.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `token_outputs`: [batch, seq_len, hidden*2] tensor from encoder\n- `params`: Model parameters\n\n## Returns\nPOS tag probabilities [batch, seq_len, num_pos_tags]\n"
  def forward(model, token_outputs, params) do
    Axon.predict(model, params, %{"token_outputs" => token_outputs})
  end

  @doc "Tag a sequence with POS labels.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `token_outputs`: [seq_len, hidden*2] tensor (single sample)\n- `params`: Model parameters\n- `pos_labels`: List of POS label strings in order\n\n## Returns\nList of `{token_idx, pos_tag, confidence}` tuples\n"
  def tag_sequence(model, token_outputs, params, pos_labels) do
    batched = Nx.new_axis(token_outputs, 0)
    probs = forward(model, batched, params)
    probs = Nx.squeeze(probs, axes: [0])

    seq_len = Nx.axis_size(probs, 0)
    predicted_indices = Nx.argmax(probs, axis: 1) |> Nx.to_flat_list()

    confidences =
      for i <- 0..(seq_len - 1) do
        idx = Enum.at(predicted_indices, i)
        Nx.to_number(probs[i][idx])
      end

    Enum.zip([0..(seq_len - 1), predicted_indices, confidences])
    |> Enum.map(fn {token_idx, tag_idx, conf} ->
      tag = Enum.at(pos_labels, tag_idx, "X")
      {token_idx, tag, conf}
    end)
  end

  @doc "Tag tokens and return token-tag pairs.\n\n## Parameters\n- `model`: Axon model\n- `tokens`: List of token strings\n- `token_outputs`: [seq_len, hidden*2] tensor\n- `params`: Model parameters\n- `pos_labels`: List of POS label strings\n\n## Returns\nList of `{token, pos_tag}` tuples\n"
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

  @doc "Compute cross-entropy loss for POS tagging.\n\n## Parameters\n- `predictions`: [batch, seq_len, num_tags] predicted probabilities\n- `targets`: [batch, seq_len] integer tensor of target POS tag indices\n- `mask`: [batch, seq_len] mask tensor (1 for real tokens, 0 for padding)\n\n## Returns\nScalar loss value\n"
  defn compute_loss(predictions, targets, mask) do
    num_tags = Nx.axis_size(predictions, 2)
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    targets_expanded = Nx.new_axis(targets, 2)
    tag_indices = Nx.iota({1, 1, num_tags})
    targets_onehot = Nx.equal(targets_expanded, tag_indices) |> Nx.as_type(:f32)
    token_losses = -Nx.sum(targets_onehot * Nx.log(predictions), axes: [2])
    masked_losses = token_losses * mask
    Nx.sum(masked_losses) / (Nx.sum(mask) + epsilon)
  end

  @doc "Compute cross-entropy loss for POS tagging without masking.\n\n## Parameters\n- `predictions`: [batch, seq_len, num_tags] predicted probabilities\n- `targets`: [batch, seq_len] integer tensor of target POS tag indices\n\n## Returns\nScalar loss value\n"
  defn compute_loss_unmasked(predictions, targets) do
    num_tags = Nx.axis_size(predictions, 2)
    epsilon = 1.0e-7
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)
    targets_expanded = Nx.new_axis(targets, 2)
    tag_indices = Nx.iota({1, 1, num_tags})
    targets_onehot = Nx.equal(targets_expanded, tag_indices) |> Nx.as_type(:f32)
    token_losses = -Nx.sum(targets_onehot * Nx.log(predictions), axes: [2])

    Nx.mean(token_losses)
  end

  @doc "Build POS vocabulary mappings.\n\nReturns `{pos_to_idx, idx_to_pos}` maps using standard Universal POS tags.\n"
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

  @doc "Check if a POS tag indicates a content word (noun, verb, adj, adv).\n"
  def content_word?(pos_tag) do
    pos_tag in ["NOUN", "PROPN", "VERB", "ADJ", "ADV"]
  end

  @doc "Check if a POS tag indicates a function word (determiner, preposition, etc.).\n"
  def function_word?(pos_tag) do
    pos_tag in ["DET", "ADP", "CONJ", "PART", "AUX"]
  end

  @doc "Get the general category of a POS tag.\n"
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