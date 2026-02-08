defmodule Brain.ML.LSTM.NERHead do
  @moduledoc "Named Entity Recognition (NER) head for the multi-task LSTM model.\n\nTakes token-level outputs from the shared encoder and produces\nBIO tag probabilities for each token using a dense layer + softmax.\n\n## BIO Tagging Scheme\n\n- B-TYPE: Beginning of an entity of TYPE\n- I-TYPE: Inside (continuation) of an entity of TYPE\n- O: Outside any entity\n\nFor example:\n- Input: \"what is the weather in London\"\n- Output: [\"O\", \"O\", \"O\", \"O\", \"O\", \"B-LOC\"]\n\n## Architecture\n\n```\ntoken_outputs [batch, seq_len, hidden*2]\n      |\n      v\nDense [batch, seq_len, num_bio_tags]\n      |\n      v\nSoftmax [batch, seq_len, num_bio_tags]\n      |\n      v\nBIO tag probabilities per token\n```\n"

  alias Axon.ModelState
  require Logger
  import Nx.Defn

  @doc "Build the NER head model.\n\n## Parameters\n- `input_size`: Size of token representation (encoder hidden_size * 2)\n- `num_bio_tags`: Number of BIO tags (O + B-type + I-type for each entity type)\n- `opts`: Additional options\n  - `:dropout` - Dropout rate (default: 0.1)\n\n## Returns\nAxon model for sequence tagging\n"
  def build_model(input_size, num_bio_tags, opts \\ []) do
    dropout_rate = Keyword.get(opts, :dropout, 0.1)
    input = Axon.input("token_outputs", shape: {nil, nil, input_size})

    input
    |> Axon.dropout(rate: dropout_rate, name: "ner_dropout")
    |> Axon.dense(num_bio_tags, name: "ner_dense")
    |> Axon.activation(:softmax, name: "ner_softmax")
  end

  @doc "Initialize NER head parameters.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `input_size`: Size of token representations\n- `seq_len`: Sequence length for template (default: 10)\n\n## Returns\nInitialized parameters map\n"
  def init_params(model, input_size, seq_len \\ 10) do
    {init_fn, _predict_fn} = Axon.build(model)

    template = %{"token_outputs" => Nx.template({1, seq_len, input_size}, :f32)}

    params = init_fn.(template, ModelState.empty())
    ModelState.new(params)
  end

  @doc "Forward pass through NER head.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `token_outputs`: [batch, seq_len, hidden*2] tensor from encoder\n- `params`: Model parameters\n\n## Returns\nBIO tag probabilities [batch, seq_len, num_bio_tags]\n"
  def forward(model, token_outputs, params) do
    Axon.predict(model, params, %{"token_outputs" => token_outputs})
  end

  @doc "Tag a sequence with BIO labels.\n\n## Parameters\n- `model`: Axon model from `build_model/3`\n- `token_outputs`: [seq_len, hidden*2] tensor (single sample)\n- `params`: Model parameters\n- `bio_labels`: List of BIO label strings in order\n\n## Returns\nList of `{token_idx, bio_tag, confidence}` tuples\n"
  def tag_sequence(model, token_outputs, params, bio_labels) do
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
      tag = Enum.at(bio_labels, tag_idx, "O")
      {token_idx, tag, conf}
    end)
  end

  @doc "Extract entities from BIO tag sequence.\n\n## Parameters\n- `tokens`: List of token strings\n- `bio_tags`: List of BIO tags (same length as tokens)\n\n## Returns\nList of extracted entities:\n```\n[\n  %{text: \"London\", type: \"location\", start: 5, end: 5, tokens: [\"London\"]},\n  %{text: \"Taylor Swift\", type: \"artist\", start: 2, end: 3, tokens: [\"Taylor\", \"Swift\"]}\n]\n```\n"
  def extract_entities_from_tags(tokens, bio_tags) do
    tokens
    |> Enum.zip(bio_tags)
    |> Enum.with_index()
    |> Enum.reduce([], fn {{token, tag}, idx}, entities ->
      cond do
        String.starts_with?(tag, "B-") ->
          entity_type = String.replace_prefix(tag, "B-", "")

          new_entity = %{
            type: entity_type,
            start: idx,
            end: idx,
            tokens: [token]
          }

          [new_entity | entities]

        String.starts_with?(tag, "I-") and entities != [] ->
          [current | rest] = entities
          expected_type = String.replace_prefix(tag, "I-", "")

          if current.type == expected_type do
            updated = %{current | end: idx, tokens: current.tokens ++ [token]}
            [updated | rest]
          else
            new_entity = %{
              type: expected_type,
              start: idx,
              end: idx,
              tokens: [token]
            }

            [new_entity | entities]
          end

        true ->
          entities
      end
    end)
    |> Enum.reverse()
    |> Enum.map(fn entity ->
      Map.put(entity, :text, Enum.join(entity.tokens, " "))
    end)
  end

  @doc "Tag a sequence and extract entities in one step.\n\n## Parameters\n- `model`: Axon model\n- `tokens`: List of token strings\n- `token_outputs`: [seq_len, hidden*2] tensor\n- `params`: Model parameters\n- `bio_labels`: List of BIO label strings\n\n## Returns\nList of extracted entities\n"
  def extract_entities(model, tokens, token_outputs, params, bio_labels) do
    tags_with_conf = tag_sequence(model, token_outputs, params, bio_labels)
    tags = Enum.map(tags_with_conf, fn {_idx, tag, _conf} -> tag end)
    tags = Enum.take(tags, length(tokens))

    extract_entities_from_tags(tokens, tags)
  end

  @doc "Compute cross-entropy loss for NER.\n\n## Parameters\n- `predictions`: [batch, seq_len, num_tags] predicted probabilities\n- `targets`: [batch, seq_len] integer tensor of target BIO tag indices\n- `mask`: [batch, seq_len] mask tensor (1 for real tokens, 0 for padding)\n\n## Returns\nScalar loss value\n"
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

  @doc "Compute cross-entropy loss for NER without masking.\n\n## Parameters\n- `predictions`: [batch, seq_len, num_tags] predicted probabilities\n- `targets`: [batch, seq_len] integer tensor of target BIO tag indices\n\n## Returns\nScalar loss value\n"
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

  @doc "Get entity type from BIO tag.\n\n## Examples\n\n    iex> NERHead.entity_type_from_bio(\"B-location\")\n    \"location\"\n    iex> NERHead.entity_type_from_bio(\"I-artist\")\n    \"artist\"\n    iex> NERHead.entity_type_from_bio(\"O\")\n    nil\n"
  def entity_type_from_bio(tag) do
    cond do
      String.starts_with?(tag, "B-") -> String.replace_prefix(tag, "B-", "")
      String.starts_with?(tag, "I-") -> String.replace_prefix(tag, "I-", "")
      true -> nil
    end
  end

  @doc "Get all unique entity types from BIO vocabulary.\n"
  def entity_types_from_bio_vocab(bio_labels) do
    bio_labels
    |> Enum.filter(&String.starts_with?(&1, "B-"))
    |> Enum.map(&String.replace_prefix(&1, "B-", ""))
    |> Enum.uniq()
    |> Enum.sort()
  end
end