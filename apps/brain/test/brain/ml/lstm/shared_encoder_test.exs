defmodule Brain.ML.LSTM.SharedEncoderTest do
  use ExUnit.Case, async: true

  alias Brain.ML.LSTM.SharedEncoder

  @moduletag :lstm

  describe "build_model/4" do
    test "builds encoder with correct configuration" do
      vocab_size = 100
      embedding_size = 64
      hidden_size = 32

      encoder = SharedEncoder.build_model(vocab_size, embedding_size, hidden_size)

      assert %{
        token_outputs_model: _,
        sentence_vector_model: _,
        config: config
      } = encoder

      assert config.vocab_size == vocab_size
      assert config.embedding_size == embedding_size
      assert config.hidden_size == hidden_size
      assert config.output_size == hidden_size * 2  # Bidirectional
    end

    test "applies custom options" do
      encoder = SharedEncoder.build_model(100, 64, 32,
        num_layers: 2,
        dropout: 0.2
      )

      assert encoder.config.num_layers == 2
      assert encoder.config.dropout == 0.2
    end
  end

  describe "init_params/2" do
    test "initializes parameters successfully" do
      encoder = SharedEncoder.build_model(100, 64, 32)
      params = SharedEncoder.init_params(encoder)

      # Params should be an Axon.ModelState or map
      assert params != nil
      # Should have parameters (check data key or directly)
      param_data = if is_struct(params, Axon.ModelState), do: params.data, else: params
      assert is_map(param_data)
      assert map_size(param_data) > 0
    end
  end

  describe "encode/3" do
    test "encodes input and returns token outputs and sentence vector" do
      vocab_size = 100
      embedding_size = 64
      hidden_size = 32
      batch_size = 2
      seq_len = 10

      encoder = SharedEncoder.build_model(vocab_size, embedding_size, hidden_size)
      params = SharedEncoder.init_params(encoder)

      # Create sample input
      input = Nx.tensor([
        Enum.to_list(1..seq_len),
        Enum.to_list(1..seq_len)
      ], type: :s64)

      {token_outputs, sentence_vector} = SharedEncoder.encode(encoder, input, params)

      # Token outputs: [batch, seq_len, hidden*2]
      assert Nx.shape(token_outputs) == {batch_size, seq_len, hidden_size * 2}

      # Sentence vector: [batch, hidden*2]
      assert Nx.shape(sentence_vector) == {batch_size, hidden_size * 2}
    end
  end

  describe "encode_single/3" do
    test "encodes single input without batch dimension" do
      vocab_size = 100
      embedding_size = 64
      hidden_size = 32
      seq_len = 10

      encoder = SharedEncoder.build_model(vocab_size, embedding_size, hidden_size)
      params = SharedEncoder.init_params(encoder)

      # Create single input
      input = Nx.tensor(Enum.to_list(1..seq_len), type: :s64)

      {token_outputs, sentence_vector} = SharedEncoder.encode_single(encoder, input, params)

      # Token outputs: [seq_len, hidden*2]
      assert Nx.shape(token_outputs) == {seq_len, hidden_size * 2}

      # Sentence vector: [hidden*2]
      assert Nx.shape(sentence_vector) == {hidden_size * 2}
    end
  end

  describe "output_size/1" do
    test "returns correct output size from encoder" do
      encoder = SharedEncoder.build_model(100, 64, 32)
      assert SharedEncoder.output_size(encoder) == 64  # 32 * 2
    end

    test "returns correct output size from integer" do
      assert SharedEncoder.output_size(32) == 64
    end
  end
end
