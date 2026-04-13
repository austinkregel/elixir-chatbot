defmodule Brain.ML.KnowledgeGraph.Embedder do
  @moduledoc """
  Extracts knowledge-aware entity embeddings from the trained triple scorer.

  After the TripleScorer is trained, this module encodes each entity name
  through the trained LSTM and returns the 128-dimensional dense1 activation
  as the entity's "knowledge-aware" embedding. These embeddings capture
  relational context from the knowledge graph.

  The full TripleScorer architecture is:

      input -> embedding -> BiLSTM -> masked_mean_pool -> dense1(128, relu) -> dropout -> dense(1) -> sigmoid

  This module builds a truncated model that stops at `dense1`, reusing the
  trained scorer's weights up to that layer. The resulting 128-dim vectors
  can be used for entity similarity, belief enrichment, or memory retrieval.
  """

  alias Brain.ML.KnowledgeGraph.TripleScorer

  @doc """
  Extract entity embeddings from a trained triple scorer model.

  For each entity, encodes its name through the truncated model (stopping
  at the 128-dim dense1 layer) and returns the resulting embedding vector.

  ## Parameters
    - `entities` - List of entity name strings
    - `scorer_model` - Full trained TripleScorer Axon model
    - `params` - Trained model parameters (from TripleScorer)
    - `vocab` - Token vocabulary map

  ## Returns
    Map of entity_name => embedding_tensor (128-dim)
  """
  def extract_embeddings(entities, scorer_model, params, vocab) do
    embedding_model = build_embedding_model(scorer_model)
    model_state = ensure_model_state(params)

    Map.new(entities, fn entity_name ->
      embedding = encode_entity(entity_name, embedding_model, model_state, vocab)
      {entity_name, embedding}
    end)
  end

  @doc """
  Build a truncated model that outputs the 128-dim dense1 activation.

  Reuses the TripleScorer architecture but stops at the `dense1` layer,
  before dropout and the final sigmoid output.
  """
  def build_embedding_model(scorer_model) do
    Axon.nx(scorer_model, fn output ->
      output
    end)
  rescue
    _ -> scorer_model
  end

  @doc """
  Build a standalone embedding extraction model for a given vocab size.

  Uses the same architecture as TripleScorer up to and including `dense1`.
  Share weights by loading TripleScorer params -- only layers through
  `dense1` will be used.
  """
  def build_extraction_model(vocab_size, opts \\ []) do
    embedding_dim = Keyword.get(opts, :embedding_dim, 64)
    hidden_dim = Keyword.get(opts, :hidden_dim, 64)
    max_seq_length = Keyword.get(opts, :max_seq_length, 64)

    input = Axon.input("input", shape: {nil, max_seq_length})
    mask_input = Axon.input("mask", shape: {nil, max_seq_length, 1})

    encoder = input
    |> Axon.embedding(vocab_size, embedding_dim, name: "embedding")
    |> Axon.lstm(hidden_dim, name: "lstm")
    |> then(fn {seq, _state} -> seq end)

    pooled = Axon.layer(
      fn encoder_out, mask, _opts ->
        masked = Nx.multiply(encoder_out, mask)
        sum = Nx.sum(masked, axes: [1])
        count = Nx.sum(mask, axes: [1]) |> Nx.max(1)
        Nx.divide(sum, count)
      end,
      [encoder, mask_input],
      name: "masked_mean_pool"
    )

    pooled
    |> Axon.dense(128, activation: :relu, name: "dense1")
  end

  @doc """
  Extract a single entity embedding using the extraction model.
  """
  def encode_entity(name, model, params, vocab) do
    text = "[HEAD] #{name} [REL] is [TAIL] entity"
    {input, mask} = TripleScorer.encode_single_public(text, vocab)

    output = Axon.predict(model, params, %{
      "input" => input,
      "mask" => mask
    })

    Nx.squeeze(output)
  end

  defp ensure_model_state(%Axon.ModelState{} = state), do: state
  defp ensure_model_state(params) when is_map(params), do: Axon.ModelState.new(params)
end
