defmodule Brain.ML.Ouro.Spec do
  @moduledoc """
  Bumblebee model specification for the Ouro LoopLM architecture.

  Ouro is a decoder-only transformer with weight-tied recurrence:
  24 physical transformer layers are applied R times (default R=4)
  with shared weights, producing 96 virtual blocks total. Each
  recurrence step applies a shared RMSNorm (`model.norm`) after
  the full layer stack.

  Architecture per block uses sandwich normalization (4 RMSNorms):
    input
      → input_layernorm → attention → input_layernorm_2 → + residual
      → post_attention_layernorm → SwiGLU FFN → post_attention_layernorm_2 → + residual
    output

  ## Architectures

    * `:base` - plain Ouro without any head on top

    * `:for_causal_language_modeling` - Ouro with a language modeling
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

    * `"attention_mask"` - `{batch_size, sequence_length}`

    * `"position_ids"` - `{batch_size, sequence_length}`

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

    * `"cache"` - opaque cache for iterative decoding
  """

  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 49_152,
        doc: "the vocabulary size of the token embedding"
      ],
      max_positions: [
        default: 65_536,
        doc: "the maximum sequence length (for RoPE position embeddings)"
      ],
      hidden_size: [
        default: 2048,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 5632,
        doc: "the dimensionality of the SwiGLU FFN intermediate layer"
      ],
      attention_head_size: [
        default: 128,
        doc: "the size of key, value, and query projection per attention head"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of physical transformer blocks (applied max_recurrence times)"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads"
      ],
      num_key_value_heads: [
        default: 16,
        doc: "the number of key-value heads (for GQA; equal to num_attention_heads for MHA)"
      ],
      max_recurrence: [
        default: 4,
        doc: "the number of recurrence steps (weight-tied passes through the block stack)"
      ],
      activation: [
        default: :silu,
        doc: "the activation function used in the gated FFN"
      ],
      rotary_embedding_base: [
        default: 1_000_000.0,
        doc: "base for computing rotary embedding frequency"
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc: "the standard deviation of the normal initializer for kernel parameters"
      ],
      tie_word_embeddings: [
        default: false,
        doc: "whether to tie input and output embedding weights"
      ]
    ] ++
      Shared.common_options([:output_hidden_states, :output_attentions]) ++
      Shared.token_options(bos_token_id: 1, eos_token_id: 2, pad_token_id: 2)

  defstruct [architecture: :for_causal_language_modeling] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures, do: [:base, :for_causal_language_modeling]

  @impl true
  def config(spec, opts) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(_spec) do
    %{"input_ids" => Nx.template({1, 1}, :s64)}
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks * spec.max_recurrence
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  # --- Model architectures ---

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)
    Layers.output(outputs)
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)
    outputs = core(inputs, spec)
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      gate_values: outputs.gate_values,
      cache: outputs.cache
    })
  end

  # --- Axon graph construction ---

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(inputs["input_ids"], inputs["input_embeddings"], spec, name: "embedder")

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    %{
      hidden_state: decoder_outputs.hidden_state,
      hidden_states: decoder_outputs.hidden_states,
      attentions: decoder_outputs.attentions,
      gate_values: decoder_outputs.gate_values,
      cache: decoder_outputs.cache
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    Layers.default input_embeddings do
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
  end

  defp decoder(hidden_state, position_ids, attention_mask, cache, spec, opts) do
    name = opts[:name]

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      gate_values: Axon.container({}),
      cache: cache
    }

    block_opts = [
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      kernel_initializer: kernel_initializer(spec),
      layer_norm:
        &Layers.rms_norm(&1, name: &2, epsilon: spec.layer_norm_epsilon),
      ffn:
        &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
          name: &2,
          activation: spec.activation
        ),
      block_type: sandwich_block_type(spec),
      causal: true,
      rotary_embedding: [
        position_ids: position_ids,
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base
      ],
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      output_use_bias: false
    ]

    state =
      for recurrence_step <- 0..(spec.max_recurrence - 1), reduce: state do
        state ->
          state =
            for physical_idx <- 0..(spec.num_blocks - 1), reduce: state do
              state ->
                virtual_idx = recurrence_step * spec.num_blocks + physical_idx
                block_cache = Layers.Decoder.get_block_cache(state.cache, virtual_idx)

                {block_hidden_state, attention, _cross_attention, block_cache, _bias} =
                  Layers.Transformer.block(
                    state.hidden_state,
                    [
                      attention_mask: attention_mask,
                      block_cache: block_cache,
                      offset: offset,
                      name: join(join(name, "blocks"), physical_idx)
                    ] ++ block_opts
                  )

                updated_cache =
                  Layers.Decoder.put_block_cache(state.cache, virtual_idx, block_cache)

                %{state |
                  hidden_state: block_hidden_state,
                  hidden_states: Layers.append(state.hidden_states, block_hidden_state),
                  attentions: Layers.append(state.attentions, attention),
                  cache: updated_cache
                }
            end

          normed =
            Layers.rms_norm(state.hidden_state,
              name: join("recurrence_norm", recurrence_step),
              epsilon: spec.layer_norm_epsilon
            )

          gate =
            Axon.dense(normed, 1,
              name: join("early_exit_gate", recurrence_step),
              use_bias: true
            )

          %{
            state
            | hidden_state: normed,
              gate_values: Layers.append(state.gate_values, gate)
          }
      end

    cache = Layers.Decoder.update_cache_offset(state.cache, hidden_state)

    %{
      hidden_state: state.hidden_state,
      hidden_states: state.hidden_states,
      attentions: state.attentions,
      gate_values: state.gate_values,
      cache: cache
    }
  end

  defp sandwich_block_type(spec) do
    fn hidden_state, steps, block_name ->
      shortcut = hidden_state

      {hidden_state, attention_info} =
        hidden_state
        |> steps.self_attention_norm.()
        |> steps.self_attention.()

      hidden_state =
        Layers.rms_norm(hidden_state,
          name: join(block_name, "post_self_attention_norm"),
          epsilon: spec.layer_norm_epsilon
        )

      hidden_state = Axon.add(hidden_state, shortcut)

      {hidden_state, cross_attention_info} =
        steps.cross_attention_maybe.(hidden_state, fn h ->
          {h, {Layers.none(), Layers.none()}}
        end)

      shortcut = hidden_state

      hidden_state =
        hidden_state
        |> steps.output_norm.()
        |> steps.ffn.()

      hidden_state =
        Layers.rms_norm(hidden_state,
          name: join(block_name, "post_ffn_norm"),
          epsilon: spec.layer_norm_epsilon
        )

      hidden_state = Axon.add(hidden_state, shortcut)

      {hidden_state, attention_info, cross_attention_info}
    end
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    gate =
      Axon.dense(hidden_state, intermediate_size, name: join(name, "gate"), use_bias: false)

    intermediate =
      Axon.dense(hidden_state, intermediate_size, name: join(name, "intermediate"), use_bias: false)

    hidden_state = Axon.multiply(intermediate, Axon.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  # --- HuggingFace config.json loading ---

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          max_recurrence: {"total_ut_steps", number()},
          activation: {"hidden_act", activation()},
          rotary_embedding_base: {"rope_theta", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  # --- HuggingFace safetensors parameter mapping ---

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "embedder.token_embedding" => "model.embed_tokens",
        "decoder.blocks.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention_norm" => "model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.output_norm" => "model.layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.post_self_attention_norm" =>
          "model.layers.{n}.input_layernorm_2",
        "decoder.blocks.{n}.post_ffn_norm" =>
          "model.layers.{n}.post_attention_layernorm_2",
        "decoder.blocks.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "recurrence_norm.{s}" => "model.norm",
        "early_exit_gate.{s}" => "model.early_exit_gate",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head")
      }
    end
  end
end
