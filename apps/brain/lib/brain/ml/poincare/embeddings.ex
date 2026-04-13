defmodule Brain.ML.Poincare.Embeddings do
  @moduledoc """
  Poincare embedding trainer and lookup GenServer.

  Trains Poincare ball embeddings from hierarchical (parent-child) pairs
  using Riemannian SGD with negative sampling. Provides a lookup table
  mapping entity/type names to their learned hyperbolic embeddings.

  Follows the project's `ready?()` pattern with 100ms timeout and visible
  failure surfacing when unavailable.
  """

  use GenServer
  require Logger

  alias Brain.ML.Poincare.{Distance, Optimizer}

  @default_dim 5
  @default_epochs 50
  @default_learning_rate 0.01
  @default_num_negatives 10
  @default_batch_size 50

  defstruct [:embeddings, :entity_to_idx, :idx_to_entity, :dim, :ready]

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
  Look up the Poincare embedding for an entity/type name.

  Returns `{:ok, embedding_tensor}` or `{:error, :not_found}`.
  """
  def lookup(entity_name, name \\ __MODULE__) do
    try do
      GenServer.call(name, {:lookup, entity_name}, 5_000)
    catch
      :exit, _ -> {:error, :not_ready}
    end
  end

  @doc """
  Compute the Poincare distance between two entities.

  Returns `{:ok, distance}` or `{:error, reason}`.
  """
  def entity_distance(entity_a, entity_b, name \\ __MODULE__) do
    try do
      GenServer.call(name, {:distance, entity_a, entity_b}, 5_000)
    catch
      :exit, _ -> {:error, :not_ready}
    end
  end

  @doc """
  Get all entity embeddings as a map.
  """
  def all_embeddings(name \\ __MODULE__) do
    try do
      GenServer.call(name, :all_embeddings, 5_000)
    catch
      :exit, _ -> {:error, :not_ready}
    end
  end

  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 30_000)
  end

  @doc """
  Train Poincare embeddings from hierarchical pairs.

  ## Parameters
    - `pairs` - List of `{child_name, parent_name}` tuples
    - `opts` - Options:
      - `:dim` - Embedding dimension (default: #{@default_dim})
      - `:epochs` - Number of training epochs (default: #{@default_epochs})
      - `:learning_rate` - Learning rate (default: #{@default_learning_rate})
      - `:num_negatives` - Negatives per positive (default: #{@default_num_negatives})
      - `:batch_size` - Training batch size (default: #{@default_batch_size})
      - `:verbose` - Log epoch progress (default: false)

  ## Returns
    `{:ok, embeddings_tensor, entity_to_idx, idx_to_entity}`
  """
  def train(pairs, opts \\ []) do
    dim = Keyword.get(opts, :dim, @default_dim)
    epochs = Keyword.get(opts, :epochs, @default_epochs)
    lr = Keyword.get(opts, :learning_rate, @default_learning_rate)
    num_negatives = Keyword.get(opts, :num_negatives, @default_num_negatives)
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)
    verbose = Keyword.get(opts, :verbose, false)

    entities = pairs
    |> Enum.flat_map(fn {child, parent} -> [child, parent] end)
    |> Enum.uniq()
    |> Enum.sort()

    entity_to_idx = entities |> Enum.with_index() |> Map.new()
    idx_to_entity = entities |> Enum.with_index() |> Map.new(fn {e, i} -> {i, e} end)
    num_entities = length(entities)

    pair_indices = pairs
    |> Enum.map(fn {child, parent} ->
      {Map.fetch!(entity_to_idx, child), Map.fetch!(entity_to_idx, parent)}
    end)

    key = Nx.Random.key(42)
    {embeddings, _key} = Nx.Random.uniform(key, -0.001, 0.001, shape: {num_entities, dim})

    positive_set = MapSet.new(pair_indices)

    positive_counts = Enum.reduce(pair_indices, %{}, fn {c, _p}, acc ->
      Map.update(acc, c, 1, &(&1 + 1))
    end)

    embeddings = train_loop(embeddings, pair_indices, positive_set, positive_counts,
      num_entities, epochs, lr, num_negatives, batch_size, verbose)

    embeddings = Nx.backend_transfer(embeddings, Nx.BinaryBackend)

    {:ok, embeddings, entity_to_idx, idx_to_entity}
  end

  @doc """
  Save trained embeddings to disk.
  """
  def save(embeddings, entity_to_idx, idx_to_entity, dim, path) do
    File.mkdir_p!(Path.dirname(path))
    data = %{
      embeddings: embeddings,
      entity_to_idx: entity_to_idx,
      idx_to_entity: idx_to_entity,
      dim: dim
    }
    File.write!(path, :erlang.term_to_binary(data))
    :ok
  end

  @doc """
  Load trained embeddings from disk.
  """
  def load(path) do
    case File.read(path) do
      {:ok, binary} -> {:ok, :erlang.binary_to_term(binary)}
      {:error, reason} -> {:error, reason}
    end
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(opts) do
    world_id = Keyword.get(opts, :world_id, "default")
    state = %__MODULE__{ready: false, dim: @default_dim}

    case try_load(world_id) do
      {:ok, loaded_state} ->
        {:ok, loaded_state}

      {:error, _} ->
        {:ok, state}
    end
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call({:lookup, _entity_name}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  def handle_call({:lookup, entity_name}, _from, state) do
    case Map.get(state.entity_to_idx, entity_name) do
      nil -> {:reply, {:error, :not_found}, state}
      idx -> {:reply, {:ok, state.embeddings[idx]}, state}
    end
  end

  def handle_call({:distance, _entity_a, _entity_b}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  def handle_call({:distance, entity_a, entity_b}, _from, state) do
    with idx_a when not is_nil(idx_a) <- Map.get(state.entity_to_idx, entity_a),
         idx_b when not is_nil(idx_b) <- Map.get(state.entity_to_idx, entity_b) do
      emb_a = state.embeddings[idx_a]
      emb_b = state.embeddings[idx_b]
      dist = Distance.distance(emb_a, emb_b) |> Nx.to_number()
      {:reply, {:ok, dist}, state}
    else
      _ -> {:reply, {:error, :not_found}, state}
    end
  end

  def handle_call(:all_embeddings, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  def handle_call(:all_embeddings, _from, state) do
    result = Map.new(state.entity_to_idx, fn {name, idx} ->
      {name, state.embeddings[idx]}
    end)
    {:reply, {:ok, result}, state}
  end

  def handle_call(:reload, _from, state) do
    case try_load("default") do
      {:ok, loaded_state} ->
        {:reply, :ok, loaded_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  # --- Private ---

  defp try_load(world_id) do
    path = model_path(world_id)
    Brain.ML.ModelStore.ensure_local("#{world_id}/poincare/embeddings.term", path)

    case load(path) do
      {:ok, data} ->
        state = %__MODULE__{
          embeddings: data.embeddings,
          entity_to_idx: data.entity_to_idx,
          idx_to_entity: data.idx_to_entity,
          dim: data.dim,
          ready: true
        }
        {:ok, state}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp train_loop(embeddings, pair_indices, positive_set, positive_counts,
                   num_entities, epochs, lr, num_negatives, batch_size, verbose) do
    log_interval = max(div(epochs, 10), 1)

    Enum.reduce(1..epochs, embeddings, fn epoch, emb ->
      {updated, last_loss} =
        pair_indices
        |> Enum.shuffle()
        |> Enum.chunk_every(batch_size)
        |> Enum.reduce({emb, 0.0}, fn batch, {emb_acc, _prev_loss} ->
          pos_pairs = Nx.tensor(Enum.map(batch, fn {c, p} -> [c, p] end), type: :s32)

          neg_indices = generate_negatives(batch, positive_set, positive_counts, num_entities, num_negatives)
          neg_tensor = Nx.tensor(neg_indices, type: :s32)

          {step_updated, loss} = Optimizer.train_step(emb_acc, pos_pairs, neg_tensor, lr)
          {step_updated, Nx.to_number(loss)}
        end)

      if verbose and (epoch == 1 or rem(epoch, log_interval) == 0) do
        IO.puts(:stderr, "  Poincare epoch #{epoch}/#{epochs}, loss: #{Float.round(last_loss, 6)}")
      end

      updated
    end)
  end

  defp generate_negatives(batch, positive_set, positive_counts, num_entities, num_negatives) do
    Enum.map(batch, fn {child_idx, _parent_idx} ->
      generate_entity_negatives(child_idx, positive_set, positive_counts, num_entities, num_negatives)
    end)
  end

  defp generate_entity_negatives(entity_idx, positive_set, positive_counts, num_entities, count) do
    num_positives = Map.get(positive_counts, entity_idx, 0)
    available = max(num_entities - 1 - num_positives, 0)
    actual_count = min(count, available)

    sampled =
      if actual_count == 0 do
        []
      else
        Stream.repeatedly(fn -> :rand.uniform(num_entities) - 1 end)
        |> Stream.reject(fn neg ->
          neg == entity_idx or MapSet.member?(positive_set, {entity_idx, neg})
        end)
        |> Enum.take(actual_count)
      end

    pad_to_length(sampled, count, entity_idx)
  end

  defp pad_to_length(list, target, _fallback) when length(list) >= target, do: Enum.take(list, target)
  defp pad_to_length([], target, fallback), do: List.duplicate(fallback, target)
  defp pad_to_length(list, target, _fallback) do
    padding = List.duplicate(hd(list), target - length(list))
    list ++ padding
  end

  defp model_path(world_id) do
    priv = :code.priv_dir(:brain) |> to_string()
    Path.join([priv, "ml_models", world_id, "poincare", "embeddings.term"])
  end
end
