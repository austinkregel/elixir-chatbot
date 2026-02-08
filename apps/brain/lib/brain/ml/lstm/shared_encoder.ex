defmodule Brain.ML.LSTM.SharedEncoder do
  @moduledoc "Shared Bidirectional LSTM encoder for multi-task NLP models.\n\nThis encoder produces both:\n- Token-level outputs: [batch, seq_len, hidden*2] for sequence tagging (NER, POS)\n- Sentence-level vector: [batch, hidden*2] for classification (intent)\n\nThe encoder weights are shared across all downstream tasks, allowing\nthe model to learn rich contextual representations that benefit\nintent classification, entity extraction, and POS tagging simultaneously.\n"

  alias Axon.ModelState
  require Logger

  @doc "Build shared encoder model using Axon.\n\n## Parameters\n- `vocab_size`: Size of vocabulary\n- `embedding_size`: Dimension of word embeddings (default: 128)\n- `hidden_size`: Dimension of LSTM hidden state (default: 128)\n- `opts`: Additional options\n  - `:num_layers` - Number of LSTM layers (default: 1)\n  - `:dropout` - Dropout rate (default: 0.1)\n\n## Returns\nA map containing:\n- `:token_outputs_model` - Axon model for token-level outputs [batch, seq_len, hidden*2]\n- `:sentence_vector_model` - Axon model for sentence-level vector [batch, hidden*2]\n- `:config` - Configuration used to build the model\n"
  def build_model(vocab_size, embedding_size \\ 128, hidden_size \\ 128, opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, 1)
    dropout = Keyword.get(opts, :dropout, 0.1)
    input = Axon.input("input", shape: {nil, nil})

    embedded =
      input
      |> Axon.embedding(vocab_size, embedding_size, name: "shared_embedding")

    forward_lstm =
      embedded
      |> build_lstm_stack(hidden_size, num_layers, dropout, "forward")

    backward_lstm =
      embedded
      |> Axon.nx(fn x -> Nx.reverse(x, axes: [1]) end, name: "reverse_input")
      |> build_lstm_stack(hidden_size, num_layers, dropout, "backward")
      |> Axon.nx(fn x -> Nx.reverse(x, axes: [1]) end, name: "reverse_output")

    token_outputs =
      Axon.concatenate([forward_lstm, backward_lstm], axis: 2, name: "token_outputs")

    forward_final =
      Axon.nx(
        forward_lstm,
        fn x ->
          seq_len = Nx.axis_size(x, 1)

          Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "forward_final"
      )

    backward_final =
      Axon.nx(
        backward_lstm,
        fn x ->
          Nx.slice_along_axis(x, 0, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "backward_final"
      )

    sentence_vector =
      Axon.concatenate([forward_final, backward_final], axis: 1, name: "sentence_vector")

    config = %{
      vocab_size: vocab_size,
      embedding_size: embedding_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      output_size: hidden_size * 2
    }

    %{
      token_outputs_model: token_outputs,
      sentence_vector_model: sentence_vector,
      config: config
    }
  end

  defp build_lstm_stack(input, hidden_size, num_layers, dropout, prefix) do
    Enum.reduce(1..num_layers, input, fn layer_idx, x ->
      layer_name = "#{prefix}_lstm_#{layer_idx}"
      {lstm_output, _hidden_states} = Axon.lstm(x, hidden_size, name: layer_name)

      if layer_idx < num_layers and dropout > 0.0 do
        Axon.dropout(lstm_output, rate: dropout, name: "#{layer_name}_dropout")
      else
        lstm_output
      end
    end)
  end

  @doc "Initialize encoder parameters for training.\n\n## Parameters\n- `encoder`: Encoder map from `build_model/4`\n- `opts`: Options passed to Axon.build\n\n## Returns\nInitialized parameters map\n"
  def init_params(encoder, opts \\ []) do
    {init_fn, _predict_fn} = Axon.build(encoder.token_outputs_model, opts)
    template = %{"input" => Nx.template({1, 10}, :s64)}

    params = init_fn.(template, ModelState.empty())
    ModelState.new(params)
  end

  @doc "Encode input sequences to get both token outputs and sentence vector.\n\n## Parameters\n- `encoder`: Encoder map from `build_model/4`\n- `input`: [batch, seq_len] tensor of token indices\n- `params`: Model parameters\n\n## Returns\n`{token_outputs, sentence_vector}` where:\n- `token_outputs`: [batch, seq_len, hidden*2] - For NER and POS tagging\n- `sentence_vector`: [batch, hidden*2] - For intent classification\n"
  def encode(encoder, input, params) do
    token_outputs = Axon.predict(encoder.token_outputs_model, params, %{"input" => input})
    sentence_vector = Axon.predict(encoder.sentence_vector_model, params, %{"input" => input})

    {token_outputs, sentence_vector}
  end

  @doc "Encode a single sequence (non-batched).\n\nConvenience function for inference.\n\n## Parameters\n- `encoder`: Encoder map from `build_model/4`\n- `input`: [seq_len] tensor of token indices\n- `params`: Model parameters\n\n## Returns\n`{token_outputs, sentence_vector}` where:\n- `token_outputs`: [seq_len, hidden*2]\n- `sentence_vector`: [hidden*2]\n"
  def encode_single(encoder, input, params) do
    batched = Nx.new_axis(input, 0)

    {token_outputs, sentence_vector} = encode(encoder, batched, params)
    {Nx.squeeze(token_outputs, axes: [0]), Nx.squeeze(sentence_vector, axes: [0])}
  end

  @doc "Get the output dimension of the encoder.\n\nThis is hidden_size * 2 due to bidirectional concatenation.\n"
  def output_size(%{config: config}) do
    config.output_size
  end

  def output_size(hidden_size) when is_integer(hidden_size) do
    hidden_size * 2
  end
end