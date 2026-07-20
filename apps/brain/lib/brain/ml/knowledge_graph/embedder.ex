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

  @doc """
  Encode an entity type concept using its parent type as tail.

  Uses the IS_A relationship: `[HEAD] song [REL] IS_A [TAIL] media`.
  For root types without a parent, falls back to WordNet hypernym,
  then to the type name itself as a last resort.

  These triples exist in the training set (after edge label standardization),
  so the resulting vectors are well-conditioned.
  """
  def encode_entity_type(type_name, parent_type, model, params, vocab) do
    tail = parent_type || type_name
    text = "[HEAD] #{type_name} [REL] #{Atlas.Graph.EdgeLabels.is_a()} [TAIL] #{tail}"
    {input, mask} = TripleScorer.encode_single_public(text, vocab)

    output = Axon.predict(model, params, %{
      "input" => input,
      "mask" => mask
    })

    Nx.squeeze(output)
  end

end
