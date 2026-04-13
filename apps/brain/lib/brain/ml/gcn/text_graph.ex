defmodule Brain.ML.GCN.TextGraph do
  @moduledoc """
  Builds a heterogeneous text graph for GCN-based text classification.

  The graph contains two types of nodes:
  - Document nodes (one per training example)
  - Word nodes (one per unique token in the vocabulary)

  Edges:
  - Word-word edges weighted by Pointwise Mutual Information (PMI > 0)
  - Word-document edges weighted by TF-IDF

  All text processing uses `Brain.ML.Tokenizer`.
  """

  alias Brain.ML.Tokenizer

  @default_vocab_size 2000
  @pmi_window_size 20

  @doc """
  Build a text graph from a training corpus.

  ## Parameters
    - `documents` - List of `{text, label}` tuples
    - `opts` - Options:
      - `:vocab_size` - Maximum vocabulary size (default: #{@default_vocab_size})
      - `:window_size` - PMI co-occurrence window (default: #{@pmi_window_size})

  ## Returns
    `{adjacency, features, labels, node_map}` where:
    - `adjacency` - Dense adjacency matrix {N, N} as Nx tensor
    - `features` - Node feature matrix {N, F} as Nx tensor
    - `labels` - Label indices for document nodes, -1 for word nodes {N}
    - `node_map` - %{node_index => {:doc, idx} | {:word, token}}
  """
  def build(documents, opts \\ []) do
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    window_size = Keyword.get(opts, :window_size, @pmi_window_size)

    tokenized_docs = Enum.map(documents, fn {text, _label} ->
      Tokenizer.tokenize_normalized(text)
    end)

    vocabulary = build_vocabulary(tokenized_docs, vocab_size)
    num_docs = length(documents)
    num_words = map_size(vocabulary)
    num_nodes = num_docs + num_words

    node_map = build_node_map(documents, vocabulary, num_docs)

    label_list = Enum.map(documents, fn {_text, label} -> label end)
    unique_labels = label_list |> Enum.uniq() |> Enum.sort()
    label_to_idx = unique_labels |> Enum.with_index() |> Map.new()

    idf_weights = compute_idf(vocabulary, tokenized_docs)
    pmi_matrix = compute_pmi(tokenized_docs, vocabulary, window_size)
    tfidf_matrix = compute_tfidf(tokenized_docs, vocabulary, idf_weights)

    adjacency = build_adjacency(num_nodes, num_docs, pmi_matrix, tfidf_matrix, vocabulary)
    features = build_features(num_nodes, num_docs, tokenized_docs, vocabulary, idf_weights)
    labels = build_labels(num_nodes, num_docs, label_list, label_to_idx)

    %{
      adjacency: adjacency,
      features: features,
      labels: labels,
      node_map: node_map,
      vocabulary: vocabulary,
      label_to_idx: label_to_idx,
      idx_to_label: Map.new(label_to_idx, fn {k, v} -> {v, k} end),
      num_docs: num_docs,
      num_words: num_words,
      num_classes: length(unique_labels)
    }
  end

  @doc """
  Build a text graph from Atlas adjacency data.

  Converts the output of `Atlas.Graph.to_adjacency/2` into the format
  expected by the GCN model.
  """
  def from_atlas(%{node_ids: node_ids, edges: edges, features: features_map}) do
    num_nodes = length(node_ids)
    id_to_idx = node_ids |> Enum.with_index() |> Map.new()

    edge_entries = Enum.flat_map(edges, fn {from, to} ->
      from_idx = Map.get(id_to_idx, from, nil)
      to_idx = Map.get(id_to_idx, to, nil)

      if from_idx && to_idx do
        [{[from_idx, to_idx], 1.0}, {[to_idx, from_idx], 1.0}]
      else
        []
      end
    end)

    adjacency = build_tensor_from_entries({num_nodes, num_nodes}, edge_entries)

    feature_dim = features_map
    |> Map.values()
    |> List.first(%{})
    |> map_size()

    features = Nx.broadcast(0.0, {num_nodes, max(feature_dim, 1)})

    node_map = Map.new(id_to_idx, fn {id, idx} -> {idx, {:atlas, id}} end)

    %{
      adjacency: adjacency,
      features: features,
      node_map: node_map,
      num_nodes: num_nodes
    }
  end

  @doc """
  Compute Pointwise Mutual Information between word pairs.

  PMI(w1, w2) = log(P(w1, w2) / (P(w1) * P(w2)))

  Only returns positive PMI values (PPMI).
  """
  def compute_pmi(tokenized_docs, vocabulary, window_size \\ @pmi_window_size) do
    num_windows = count_windows(tokenized_docs, window_size)

    cooccurrence = count_cooccurrences(tokenized_docs, vocabulary, window_size)
    word_freq = count_word_windows(tokenized_docs, vocabulary, window_size)

    Enum.reduce(cooccurrence, %{}, fn {{w1, w2}, co_count}, acc ->
      p_w1 = Map.get(word_freq, w1, 0) / max(num_windows, 1)
      p_w2 = Map.get(word_freq, w2, 0) / max(num_windows, 1)
      p_co = co_count / max(num_windows, 1)

      if p_w1 > 0 and p_w2 > 0 and p_co > 0 do
        pmi = :math.log(p_co / (p_w1 * p_w2))
        if pmi > 0, do: Map.put(acc, {w1, w2}, pmi), else: acc
      else
        acc
      end
    end)
  end

  defp build_vocabulary(tokenized_docs, max_size) do
    tokenized_docs
    |> List.flatten()
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_word, count} -> -count end)
    |> Enum.take(max_size)
    |> Enum.with_index()
    |> Map.new(fn {{word, _count}, idx} -> {word, idx} end)
  end

  defp build_node_map(documents, vocabulary, num_docs) do
    doc_map = documents
    |> Enum.with_index()
    |> Map.new(fn {{_text, label}, idx} -> {idx, {:doc, idx, label}} end)

    word_map = vocabulary
    |> Map.new(fn {word, word_idx} -> {num_docs + word_idx, {:word, word}} end)

    Map.merge(doc_map, word_map)
  end

  defp compute_idf(vocabulary, tokenized_docs) do
    num_docs = length(tokenized_docs)
    doc_sets = Enum.map(tokenized_docs, &MapSet.new/1)

    Map.new(vocabulary, fn {word, _idx} ->
      doc_freq = Enum.count(doc_sets, &MapSet.member?(&1, word))
      idf = :math.log((num_docs + 1) / (doc_freq + 1)) + 1.0
      {word, idf}
    end)
  end

  defp compute_tfidf(tokenized_docs, vocabulary, idf_weights) do
    Enum.with_index(tokenized_docs)
    |> Map.new(fn {tokens, doc_idx} ->
      freq = Enum.frequencies(tokens)
      max_freq = freq |> Map.values() |> Enum.max(fn -> 1 end)

      tfidf = Map.new(vocabulary, fn {word, _word_idx} ->
        tf = Map.get(freq, word, 0) / max(max_freq, 1)
        idf = Map.get(idf_weights, word, 0.0)
        {word, tf * idf}
      end)

      {doc_idx, tfidf}
    end)
  end

  defp count_windows(tokenized_docs, window_size) do
    Enum.reduce(tokenized_docs, 0, fn tokens, acc ->
      acc + max(length(tokens) - window_size + 1, 1)
    end)
  end

  defp count_cooccurrences(tokenized_docs, vocabulary, window_size) do
    Enum.reduce(tokenized_docs, %{}, fn tokens, acc ->
      vocab_tokens = Enum.filter(tokens, &Map.has_key?(vocabulary, &1))

      windows = if length(vocab_tokens) <= window_size do
        [vocab_tokens]
      else
        Enum.chunk_every(vocab_tokens, window_size, 1, :discard)
      end

      Enum.reduce(windows, acc, fn window, inner_acc ->
        pairs = for w1 <- window, w2 <- window, w1 < w2, do: {w1, w2}
        Enum.reduce(pairs, inner_acc, fn pair, a ->
          Map.update(a, pair, 1, &(&1 + 1))
        end)
      end)
    end)
  end

  defp count_word_windows(tokenized_docs, vocabulary, window_size) do
    Enum.reduce(tokenized_docs, %{}, fn tokens, acc ->
      vocab_tokens = Enum.filter(tokens, &Map.has_key?(vocabulary, &1))

      windows = if length(vocab_tokens) <= window_size do
        [vocab_tokens]
      else
        Enum.chunk_every(vocab_tokens, window_size, 1, :discard)
      end

      Enum.reduce(windows, acc, fn window, inner_acc ->
        Enum.reduce(Enum.uniq(window), inner_acc, fn word, a ->
          Map.update(a, word, 1, &(&1 + 1))
        end)
      end)
    end)
  end

  defp build_adjacency(num_nodes, num_docs, pmi_matrix, tfidf_matrix, vocabulary) do
    pmi_entries = Enum.flat_map(pmi_matrix, fn {{w1, w2}, pmi_val} ->
      w1_idx = num_docs + Map.fetch!(vocabulary, w1)
      w2_idx = num_docs + Map.fetch!(vocabulary, w2)
      [{[w1_idx, w2_idx], pmi_val}, {[w2_idx, w1_idx], pmi_val}]
    end)

    tfidf_entries = Enum.flat_map(tfidf_matrix, fn {doc_idx, word_weights} ->
      Enum.flat_map(word_weights, fn {word, weight} ->
        if weight > 0 do
          word_idx = num_docs + Map.fetch!(vocabulary, word)
          [{[doc_idx, word_idx], weight}, {[word_idx, doc_idx], weight}]
        else
          []
        end
      end)
    end)

    build_tensor_from_entries({num_nodes, num_nodes}, pmi_entries ++ tfidf_entries)
  end

  defp build_features(num_nodes, num_docs, tokenized_docs, vocabulary, idf_weights) do
    feature_dim = map_size(vocabulary)

    doc_entries = Enum.flat_map(Enum.with_index(tokenized_docs), fn {tokens, doc_idx} ->
      freq = Enum.frequencies(tokens)
      max_freq = freq |> Map.values() |> Enum.max(fn -> 1 end)

      Enum.flat_map(vocabulary, fn {word, word_vocab_idx} ->
        tf = Map.get(freq, word, 0) / max(max_freq, 1)
        idf = Map.get(idf_weights, word, 0.0)
        val = tf * idf

        if val > 0 do
          [{[doc_idx, word_vocab_idx], val}]
        else
          []
        end
      end)
    end)

    word_identity_entries = Enum.map(vocabulary, fn {_word, word_vocab_idx} ->
      {[num_docs + word_vocab_idx, word_vocab_idx], 1.0}
    end)

    build_tensor_from_entries({num_nodes, feature_dim}, doc_entries ++ word_identity_entries)
  end

  defp build_tensor_from_entries(shape, entries) when entries == [] do
    Nx.broadcast(0.0, shape)
  end

  defp build_tensor_from_entries(shape, entries) do
    {indices, values} = Enum.unzip(entries)
    base = Nx.broadcast(0.0, shape)
    Nx.indexed_put(base, Nx.tensor(indices, type: :s32), Nx.tensor(values))
  end

  defp build_labels(num_nodes, num_docs, label_list, label_to_idx) do
    doc_labels = Enum.map(label_list, &Map.fetch!(label_to_idx, &1))
    word_labels = List.duplicate(-1, num_nodes - num_docs)
    Nx.tensor(doc_labels ++ word_labels, type: :s32)
  end
end
