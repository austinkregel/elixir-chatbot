defmodule Brain.ML.GCN.Model do
  @moduledoc """
  Two-layer Graph Convolutional Network for semi-supervised text classification.

  Architecture:
      Input: A_hat (normalized adjacency), X (node features)
        -> GCN Layer 1 (hidden_dim) -> ReLU -> Dropout
        -> GCN Layer 2 (num_classes) -> Softmax
        -> Cross-entropy loss on labeled nodes only

  This module provides both the Axon model builder and a GenServer for
  inference. The GenServer follows the project's `ready?()` pattern with
  100ms timeout and visible failure surfacing when unavailable.
  """

  use GenServer
  require Logger
  import Nx.Defn

  alias Brain.ML.GCN.Layer

  @default_hidden_dim 200
  @default_dropout 0.5
  @default_epochs 200
  @default_learning_rate 0.01

  defstruct [:model, :params, :adjacency_norm, :text_graph, :config, :ready]

  # --- Public API ---

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Classify a text's intent using the trained GCN model.

  Returns `{:ok, {intent, confidence}}` or `{:error, reason}`.
  """
  def classify(text, name \\ __MODULE__) do
    try do
      GenServer.call(name, {:classify, text}, 5_000)
    catch
      :exit, _ -> {:error, :not_ready}
    end
  end

  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 30_000)
  end

  @doc """
  Build a two-layer GCN Axon model.

  ## Parameters
    - `num_features` - Input feature dimension
    - `num_classes` - Number of output classes
    - `opts` - Options:
      - `:hidden_dim` - Hidden layer dimension (default: #{@default_hidden_dim})
      - `:dropout` - Dropout rate (default: #{@default_dropout})
  """
  def build_model(num_features, num_classes, opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    feature_input = Axon.input("features", shape: {nil, num_features})
    adjacency_input = Axon.input("adjacency", shape: {nil, nil})

    feature_input
    |> Layer.gcn_layer(adjacency_input,
      out_features: hidden_dim,
      name: "gcn1",
      activation: :relu
    )
    |> Axon.dropout(rate: dropout, name: "dropout1")
    |> Layer.gcn_layer(adjacency_input,
      out_features: num_classes,
      name: "gcn2",
      activation: nil
    )
    |> Axon.softmax(name: "output")
  end

  @doc """
  Build a training-specific model that includes a mask input for masked loss.

  Returns `{model, mask_input}` where the mask_input is used by the loss function.
  """
  def build_training_model(num_features, num_classes, opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    feature_input = Axon.input("features", shape: {nil, num_features})
    adjacency_input = Axon.input("adjacency", shape: {nil, nil})

    logits = feature_input
    |> Layer.gcn_layer(adjacency_input,
      out_features: hidden_dim,
      name: "gcn1",
      activation: :relu
    )
    |> Axon.dropout(rate: dropout, name: "dropout1")
    |> Layer.gcn_layer(adjacency_input,
      out_features: num_classes,
      name: "gcn2",
      activation: nil
    )
    |> Axon.softmax(name: "output")

    logits
  end

  @doc """
  Train a GCN model on a text graph.

  The masked cross-entropy loss is implemented by building the mask into the
  target tensor: word nodes get a uniform distribution target so their loss
  contribution is zero, while labeled document nodes get one-hot targets.

  ## Parameters
    - `text_graph` - Result of `TextGraph.build/2`
    - `opts` - Training options:
      - `:epochs` - Number of training epochs (default: #{@default_epochs})
      - `:learning_rate` - Learning rate (default: #{@default_learning_rate})
      - `:hidden_dim` - Hidden dimension (default: #{@default_hidden_dim})
      - `:dropout` - Dropout rate (default: #{@default_dropout})
      - `:verbose` - Print per-epoch loss (default: false)
  """
  def train(text_graph, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, @default_epochs)
    learning_rate = Keyword.get(opts, :learning_rate, @default_learning_rate)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    verbose = Keyword.get(opts, :verbose, false)

    %{adjacency: adjacency, features: features, labels: labels,
      num_classes: num_classes} = text_graph

    num_features = Nx.axis_size(features, 1)

    n = Nx.axis_size(adjacency, 0)
    IO.puts("[GCN] normalize_adjacency (#{n}x#{n}), backend: #{inspect(adjacency.data.__struct__)}")
    IO.puts("[GCN] defn compiler: #{inspect(Nx.Defn.default_options()[:compiler])}")

    IO.puts("[GCN]   running normalize_adjacency via EXLA JIT...")
    t0 = System.monotonic_time(:millisecond)
    adjacency_norm = Nx.Defn.jit(&Layer.normalize_adjacency/1, compiler: EXLA).(adjacency)
    IO.puts("[GCN]   normalize_adjacency done in #{System.monotonic_time(:millisecond) - t0}ms, backend: #{inspect(adjacency_norm.data.__struct__)}")

    model = build_model(num_features, num_classes,
      hidden_dim: hidden_dim,
      dropout: dropout
    )

    train_mask = Nx.not_equal(labels, -1) |> Nx.as_type(:f32)
    clamped = Nx.max(labels, 0)
    one_hot_labels = Nx.equal(Nx.new_axis(clamped, 1), Nx.iota({1, num_classes}))
    |> Nx.as_type(:f32)

    masked_labels = Nx.multiply(one_hot_labels, Nx.new_axis(train_mask, 1))
    mask_count = train_mask |> Nx.sum() |> Nx.max(1)

    IO.puts("[GCN] initializing params (mode: :inference, no PRNG state)...")
    t0 = System.monotonic_time(:millisecond)
    {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
    template_input = %{"features" => features, "adjacency" => adjacency_norm}
    initial_model_state = init_fn.(template_input, Axon.ModelState.empty())

    initial_params = case initial_model_state do
      %Axon.ModelState{data: data} -> data
      params when is_map(params) -> params
    end
    IO.puts("[GCN] params initialized in #{System.monotonic_time(:millisecond) - t0}ms, keys: #{inspect(Map.keys(initial_params) |> Enum.sort())}")

    {optimizer_init, optimizer_update} = Polaris.Optimizers.adam(learning_rate: learning_rate)
    optimizer_state = optimizer_init.(initial_params)

    IO.puts("[GCN] starting training loop (#{epochs} epochs, pure defn forward pass)...")

    params = Enum.reduce(1..epochs, {initial_params, optimizer_state}, fn epoch, {params, opt_state} ->
      {new_params, new_opt_state, loss} =
        gcn_train_step(params, opt_state, features, adjacency_norm, masked_labels, mask_count,
          optimizer_update)

      if verbose and (epoch == 1 or rem(epoch, 10) == 0 or epoch == epochs) do
        loss_val = Nx.to_number(loss)
        Logger.info("GCN epoch #{epoch}/#{epochs}: loss=#{Float.round(loss_val, 6)}")
      end

      {new_params, new_opt_state}
    end)
    |> elem(0)

    params = transfer_to_binary_backend(params)

    config = %{
      num_features: num_features,
      num_classes: num_classes,
      hidden_dim: hidden_dim,
      dropout: dropout
    }

    {:ok, model, params, adjacency_norm, config}
  end

  defn gcn_forward(params, features, adjacency_norm) do
    h = features
    |> Nx.dot([1], params["gcn1_dense"]["kernel"], [0])
    |> Nx.add(params["gcn1_dense"]["bias"])
    |> then(&Nx.dot(adjacency_norm, [1], &1, [0]))
    |> Nx.max(0)

    h
    |> Nx.dot([1], params["gcn2_dense"]["kernel"], [0])
    |> Nx.add(params["gcn2_dense"]["bias"])
    |> then(&Nx.dot(adjacency_norm, [1], &1, [0]))
    |> Axon.Activations.softmax(axis: 1)
  end

  defnp gcn_train_step(params, opt_state, features, adjacency_norm,
                        targets, mask_count, optimizer_update) do
    {loss, grads} = Nx.Defn.value_and_grad(params, fn p ->
      preds = gcn_forward(p, features, adjacency_norm)
      eps = 1.0e-7
      log_preds = Nx.log(Nx.add(preds, eps))
      per_node = Nx.negate(Nx.sum(Nx.multiply(targets, log_preds), axes: [1]))
      Nx.divide(Nx.sum(per_node), mask_count)
    end)

    {updates, new_opt_state} = optimizer_update.(grads, opt_state, params)
    new_params = Polaris.Updates.apply_updates(params, updates)
    {new_params, new_opt_state, loss}
  end

  @doc """
  Save a trained model to disk.
  """
  def save_model(params, text_graph, config, path) do
    File.mkdir_p!(Path.dirname(path))

    data = %{
      params: params,
      text_graph_meta: %{
        vocabulary: text_graph.vocabulary,
        label_to_idx: text_graph.label_to_idx,
        idx_to_label: text_graph.idx_to_label,
        num_docs: text_graph.num_docs,
        num_words: text_graph.num_words,
        num_classes: text_graph.num_classes
      },
      adjacency: text_graph.adjacency,
      features: text_graph.features,
      labels: text_graph.labels,
      config: config
    }

    File.write!(path, :erlang.term_to_binary(data))
    :ok
  end

  @doc """
  Load a trained model from disk.
  """
  def load_model(path) do
    case File.read(path) do
      {:ok, binary} ->
        data = :erlang.binary_to_term(binary)
        {:ok, %{data | params: ensure_model_state(data.params)}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(opts) do
    world_id = Keyword.get(opts, :world_id, "default")
    state = %__MODULE__{ready: false, config: %{world_id: world_id}}

    case try_load(world_id) do
      {:ok, loaded_state} ->
        {:ok, %{loaded_state | ready: true}}

      {:error, _reason} ->
        {:ok, state}
    end
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call({:classify, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  def handle_call({:classify, text}, _from, state) do
    result = do_classify(text, state)
    {:reply, result, state}
  end

  def handle_call(:reload, _from, state) do
    world_id = get_in(state.config, [:world_id]) || "default"

    case try_load(world_id) do
      {:ok, loaded_state} ->
        {:reply, :ok, %{loaded_state | ready: true}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  # --- Private ---

  defp try_load(world_id) do
    path = model_path(world_id)
    Brain.ML.ModelStore.ensure_local("#{world_id}/gcn/model.term", path)

    case load_model(path) do
      {:ok, data} ->
        adjacency_norm = Layer.normalize_adjacency(data.adjacency)

        model = build_model(
          data.config.num_features,
          data.config.num_classes,
          hidden_dim: data.config.hidden_dim,
          dropout: data.config.dropout
        )

        state = %__MODULE__{
          model: model,
          params: ensure_model_state(data.params),
          adjacency_norm: adjacency_norm,
          text_graph: data,
          config: Map.put(data.config, :world_id, world_id),
          ready: true
        }

        {:ok, state}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp do_classify(text, state) do
    %{model: model, params: params, adjacency_norm: adjacency_norm,
      text_graph: %{text_graph_meta: meta, features: features}} = state

    tokens = Brain.ML.Tokenizer.tokenize_normalized(text)

    feature_vec = build_query_feature(tokens, meta.vocabulary, Nx.axis_size(features, 1))

    all_features = Nx.concatenate([features, Nx.new_axis(feature_vec, 0)], axis: 0)

    n = Nx.axis_size(adjacency_norm, 0)
    new_n = n + 1
    query_idx = new_n - 1

    doc_edges = build_query_doc_edges(tokens, meta, n)

    edge_entries =
      Enum.flat_map(doc_edges, fn {word_node_idx, weight} ->
        [{[query_idx, word_node_idx], weight}, {[word_node_idx, query_idx], weight}]
      end) ++ [{[query_idx, query_idx], 1.0}]

    {edge_indices, edge_values} = Enum.unzip(edge_entries)

    padded_adj = Nx.pad(adjacency_norm, 0.0, [{0, 1, 0}, {0, 1, 0}])
    padded_adj = Nx.indexed_put(
      padded_adj,
      Nx.tensor(edge_indices, type: :s32),
      Nx.tensor(edge_values)
    )

    renormed = Layer.normalize_adjacency(padded_adj)

    output = Axon.predict(model, params, %{
      "features" => all_features,
      "adjacency" => renormed
    })

    query_probs = output[query_idx]
    best_class = Nx.argmax(query_probs) |> Nx.to_number()
    confidence = query_probs[best_class] |> Nx.to_number()

    intent = Map.get(meta.idx_to_label, best_class, "unknown")
    {:ok, {intent, confidence}}
  end

  defp build_query_feature(tokens, vocabulary, feature_dim) do
    freq = Enum.frequencies(tokens)
    max_freq = freq |> Map.values() |> Enum.max(fn -> 1 end)

    entries = Enum.flat_map(vocabulary, fn {word, word_idx} ->
      tf = Map.get(freq, word, 0) / max(max_freq, 1)
      if tf > 0, do: [{[word_idx], tf}], else: []
    end)

    case entries do
      [] ->
        Nx.broadcast(0.0, {feature_dim})

      _ ->
        {indices, values} = Enum.unzip(entries)
        vec = Nx.broadcast(0.0, {feature_dim})
        Nx.indexed_put(vec, Nx.tensor(indices, type: :s32), Nx.tensor(values))
    end
  end

  defp build_query_doc_edges(tokens, meta, _adj_size) do
    vocab = meta.vocabulary
    num_docs = meta.num_docs

    Enum.flat_map(tokens, fn token ->
      case Map.get(vocab, token) do
        nil -> []
        word_idx -> [{num_docs + word_idx, 1.0}]
      end
    end)
    |> Enum.uniq_by(fn {idx, _} -> idx end)
  end


  defp ensure_model_state(%Axon.ModelState{} = state), do: state
  defp ensure_model_state(params) when is_map(params), do: Axon.ModelState.new(params)

  defp transfer_to_binary_backend(%Axon.ModelState{} = model_state) do
    data = model_state.data
    |> Map.new(fn {key, value} -> {key, transfer_value(value)} end)

    %{model_state | data: data}
  end

  defp transfer_to_binary_backend(params) when is_map(params) do
    Map.new(params, fn {key, value} ->
      {key, transfer_value(value)}
    end)
  end

  defp transfer_value(%Nx.Tensor{} = tensor) do
    Nx.backend_transfer(tensor, Nx.BinaryBackend)
  end

  defp transfer_value(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {k, transfer_value(v)} end)
  end

  defp transfer_value(other), do: other

  defp model_path(world_id) do
    base =
      case Application.get_env(:brain, :ml, [])[:models_path] do
        nil ->
          priv = :code.priv_dir(:brain) |> to_string()
          Path.join(priv, "ml_models")

        path ->
          path
      end

    Path.join([base, world_id, "gcn", "model.term"])
  end
end
