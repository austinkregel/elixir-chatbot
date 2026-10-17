defmodule Brain.ML.Poincare.Embeddings do
  @moduledoc """
  Poincare ball embeddings of the entity type and intent hierarchy: training,
  and each world's copy.

  `train/2` learns the embeddings from hierarchical (child, parent) pairs using
  Riemannian SGD with negative sampling; `mix train_poincare` writes them to
  `trained_path/0`.

  The embeddings place a world's types and intents, so they belong to a world.
  A world holds its own copy at `<world>/models/poincare_embeddings.term`
  (`path/1`), seeded by copying the trained embeddings:

  - a persistent world is seeded when `World.Manager` creates it, and an existing
    world, or every world after a retrain, with
    `mix world.seed_model poincare_embeddings`;
  - an ephemeral world has no directory, so it is seeded in memory when it is
    created (`seed_in_memory!/1`) and dropped with it (`drop/1`).

  `loaded!/1` reads a world's copy the first time the world asks for it and keeps
  it in `:persistent_term`. Every lookup names its world. A world with no copy
  raises naming the world: there is no fallback to the trained embeddings or to
  another world.
  """

  require Logger

  alias Brain.ML.Poincare.{Distance, Optimizer}
  alias Brain.ML.{TrainingSeed, WorldModelFile}

  @default_dim 5
  @default_epochs 50
  @default_learning_rate 0.01
  @default_num_negatives 10
  @default_batch_size 50

  @file_name "poincare_embeddings.term"
  @model_keys [:embeddings, :entity_to_idx, :idx_to_entity, :dim]

  # --- Lookups in a world's embeddings ---

  @doc """
  Look up the world's Poincare embedding for an entity/type name.

  Returns `{:ok, embedding_tensor}` or `{:error, :not_found}`. Raises when the
  world has no copy of the embeddings.
  """
  @spec lookup(String.t(), String.t()) :: {:ok, Nx.Tensor.t()} | {:error, :not_found}
  def lookup(world_id, entity_name) when is_binary(world_id) do
    model = loaded!(world_id)

    case Map.get(model.entity_to_idx, entity_name) do
      nil -> {:error, :not_found}
      idx -> {:ok, model.embeddings[idx]}
    end
  end

  @doc """
  Compute the Poincare distance between two entities in the world's embeddings.

  Returns `{:ok, distance}`, or `{:error, :not_found}` when either entity has no
  embedding. Raises when the world has no copy of the embeddings.
  """
  @spec entity_distance(String.t(), String.t(), String.t()) :: {:ok, float()} | {:error, :not_found}
  def entity_distance(world_id, entity_a, entity_b) when is_binary(world_id) do
    model = loaded!(world_id)

    with idx_a when not is_nil(idx_a) <- Map.get(model.entity_to_idx, entity_a),
         idx_b when not is_nil(idx_b) <- Map.get(model.entity_to_idx, entity_b) do
      dist = Distance.distance(model.embeddings[idx_a], model.embeddings[idx_b]) |> Nx.to_number()
      {:ok, dist}
    else
      _ -> {:error, :not_found}
    end
  end

  @doc """
  Every entity embedding in the world's copy, as `{:ok, %{name => tensor}}`.
  Raises when the world has no copy.
  """
  @spec all_embeddings(String.t()) :: {:ok, %{String.t() => Nx.Tensor.t()}}
  def all_embeddings(world_id) when is_binary(world_id) do
    model = loaded!(world_id)
    {:ok, Map.new(model.entity_to_idx, fn {name, idx} -> {name, model.embeddings[idx]} end)}
  end

  # --- A world's copy ---

  @doc "Where a world keeps its copy of the Poincare embeddings."
  @spec path(String.t()) :: Path.t()
  def path(world_id) when is_binary(world_id), do: WorldModelFile.path(world_id, @file_name)

  @doc "Where training writes the Poincare embeddings that worlds are seeded from."
  @spec trained_path() :: Path.t()
  def trained_path, do: WorldModelFile.trained_path(@file_name)

  @doc "The world's embeddings, read from its copy the first time they are asked for."
  @spec loaded!(String.t()) :: map()
  def loaded!(world_id) when is_binary(world_id) do
    case :persistent_term.get(key(world_id), nil) do
      nil ->
        model = read!(path(world_id), world_id)
        :persistent_term.put(key(world_id), model)
        model

      model ->
        model
    end
  end

  @doc """
  Copies the trained embeddings into the world's models directory, checks the
  copy byte for byte, and makes them the world's loaded embeddings. Replaces any
  copy the world held, which is how retrained embeddings reach a world.
  """
  @spec seed!(String.t()) :: :ok
  def seed!(world_id) when is_binary(world_id) do
    model = read!(trained_path(), :trained)
    :ok = WorldModelFile.copy!(world_id, @file_name)
    :persistent_term.put(key(world_id), model)
    :ok
  end

  @doc "Makes a copy of the trained embeddings the loaded embeddings of a world that has no directory."
  @spec seed_in_memory!(String.t()) :: :ok
  def seed_in_memory!(world_id) when is_binary(world_id) do
    :persistent_term.put(key(world_id), read!(trained_path(), :trained))
    :ok
  end

  @doc "Forgets the world's loaded embeddings, for a world that is destroyed."
  @spec drop(String.t()) :: :ok
  def drop(world_id) when is_binary(world_id) do
    :persistent_term.erase(key(world_id))
    :ok
  end

  @doc """
  Reads and checks a Poincare embeddings file. Raises when it is missing, or does
  not hold embeddings whose width is the stored `:dim`.
  """
  @spec read!(Path.t(), String.t() | :trained) :: map()
  def read!(file, owner) do
    Brain.Telemetry.measure_model_load("poincare_embeddings", fn -> read_checked!(file, owner) end)
  end

  defp read_checked!(file, owner) do
    unless File.regular?(file) do
      raise missing_message(file, owner)
    end

    case file |> File.read!() |> :erlang.binary_to_term() do
      %{} = model ->
        case Enum.reject(@model_keys, &Map.has_key?(model, &1)) do
          [] ->
            check_dim!(model, file)

          missing ->
            raise "Poincare.Embeddings: #{file} lacks #{inspect(missing)}; retrain with `mix train_poincare`"
        end

      other ->
        raise "Poincare.Embeddings: #{file} does not hold Poincare embeddings, got #{inspect(other, limit: 3)}"
    end
  end

  defp check_dim!(%{embeddings: %Nx.Tensor{} = embeddings, dim: dim} = model, file) do
    case Nx.shape(embeddings) do
      {_entities, ^dim} ->
        model

      shape ->
        raise "Poincare.Embeddings: #{file} stores dim #{inspect(dim)} but its embeddings have shape #{inspect(shape)}"
    end
  end

  defp check_dim!(%{embeddings: other}, file),
    do: raise("Poincare.Embeddings: #{file} holds embeddings that are not a tensor: #{inspect(other, limit: 3)}")

  defp missing_message(file, :trained),
    do: "Poincare.Embeddings: no trained Poincare embeddings at #{file}; train them with `mix train_poincare`"

  defp missing_message(file, world_id),
    do:
      "Poincare.Embeddings: world #{inspect(world_id)} has no Poincare embeddings at #{file}; " <>
        "seed it with `mix world.seed_model poincare_embeddings --world #{world_id}`"

  defp key(world_id), do: {__MODULE__, world_id}

  # --- Training ---

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
      - `:seed` - Seed for the initial embeddings, per-epoch shuffling and
        negative sampling (default: `Brain.ML.TrainingSeed.get!/0`)

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
    seed = Keyword.get_lazy(opts, :seed, &TrainingSeed.get!/0)

    entities =
      pairs
      |> Enum.flat_map(fn {child, parent} -> [child, parent] end)
      |> Enum.uniq()
      |> Enum.sort()

    entity_to_idx = entities |> Enum.with_index() |> Map.new()
    idx_to_entity = entities |> Enum.with_index() |> Map.new(fn {e, i} -> {i, e} end)
    num_entities = length(entities)

    pair_indices =
      pairs
      |> Enum.map(fn {child, parent} ->
        {Map.fetch!(entity_to_idx, child), Map.fetch!(entity_to_idx, parent)}
      end)

    key = Nx.Random.key(seed)
    {embeddings, _key} = Nx.Random.uniform(key, -0.001, 0.001, shape: {num_entities, dim})

    positive_set = MapSet.new(pair_indices)

    positive_counts =
      Enum.reduce(pair_indices, %{}, fn {c, _p}, acc ->
        Map.update(acc, c, 1, &(&1 + 1))
      end)

    embeddings =
      train_loop(
        embeddings,
        pair_indices,
        positive_set,
        positive_counts,
        num_entities,
        epochs,
        lr,
        num_negatives,
        batch_size,
        verbose,
        TrainingSeed.state(seed)
      )

    embeddings = Nx.backend_transfer(embeddings, Nx.BinaryBackend)

    {:ok, embeddings, entity_to_idx, idx_to_entity}
  end

  @doc """
  Save trained embeddings to disk.

  Pass `training_seed:` to record the seed the embeddings were trained with.
  """
  def save(embeddings, entity_to_idx, idx_to_entity, dim, path, opts \\ []) do
    File.mkdir_p!(Path.dirname(path))

    data =
      %{
        embeddings: embeddings,
        entity_to_idx: entity_to_idx,
        idx_to_entity: idx_to_entity,
        dim: dim
      }
      |> then(fn data ->
        case Keyword.fetch(opts, :training_seed) do
          {:ok, seed} -> Map.put(data, :training_seed, seed)
          :error -> data
        end
      end)

    File.write!(path, Brain.ML.ModelStore.serialize(data))
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

  # --- Private ---

  defp train_loop(
         embeddings,
         pair_indices,
         positive_set,
         positive_counts,
         num_entities,
         epochs,
         lr,
         num_negatives,
         batch_size,
         verbose,
         rand
       ) do
    log_interval = max(div(epochs, 10), 1)

    {embeddings, _rand} =
      Enum.reduce(1..epochs, {embeddings, rand}, fn epoch, {emb, rand} ->
        {shuffled, rand} = TrainingSeed.shuffle(pair_indices, rand)

        {updated, last_loss, rand} =
          shuffled
          |> Enum.chunk_every(batch_size)
          |> Enum.reduce({emb, 0.0, rand}, fn batch, {emb_acc, _prev_loss, rand} ->
            pos_pairs = Nx.tensor(Enum.map(batch, fn {c, p} -> [c, p] end), type: :s32)

            {neg_indices, rand} =
              generate_negatives(batch, positive_set, positive_counts, num_entities, num_negatives, rand)

            neg_tensor = Nx.tensor(neg_indices, type: :s32)

            {step_updated, loss} = Optimizer.train_step(emb_acc, pos_pairs, neg_tensor, lr)
            {step_updated, Nx.to_number(loss), rand}
          end)

        if verbose and (epoch == 1 or rem(epoch, log_interval) == 0) do
          IO.puts(
            :stderr,
            "  Poincare epoch #{epoch}/#{epochs}, loss: #{Float.round(last_loss, 6)}"
          )
        end

        {updated, rand}
      end)

    embeddings
  end

  defp generate_negatives(batch, positive_set, positive_counts, num_entities, num_negatives, rand) do
    Enum.map_reduce(batch, rand, fn {child_idx, _parent_idx}, rand ->
      generate_entity_negatives(
        child_idx,
        positive_set,
        positive_counts,
        num_entities,
        num_negatives,
        rand
      )
    end)
  end

  defp generate_entity_negatives(entity_idx, positive_set, positive_counts, num_entities, count, rand) do
    num_positives = Map.get(positive_counts, entity_idx, 0)
    available = max(num_entities - 1 - num_positives, 0)
    actual_count = min(count, available)

    {sampled, rand} =
      sample_negative_indices(entity_idx, positive_set, num_entities, actual_count, rand, [])

    {pad_to_length(sampled, count, entity_idx), rand}
  end

  # Draws entity indices that are neither the entity itself nor one of its
  # positives, until `count` are drawn.
  defp sample_negative_indices(_entity_idx, _set, _n, 0, rand, acc), do: {Enum.reverse(acc), rand}

  defp sample_negative_indices(entity_idx, set, num_entities, count, rand, acc) do
    {draw, rand} = :rand.uniform_s(num_entities, rand)
    neg = draw - 1

    if neg == entity_idx or MapSet.member?(set, {entity_idx, neg}) do
      sample_negative_indices(entity_idx, set, num_entities, count, rand, acc)
    else
      sample_negative_indices(entity_idx, set, num_entities, count - 1, rand, [neg | acc])
    end
  end

  defp pad_to_length(list, target, _fallback) when length(list) >= target,
    do: Enum.take(list, target)

  defp pad_to_length([], target, fallback), do: List.duplicate(fallback, target)

  defp pad_to_length(list, target, _fallback) do
    padding = List.duplicate(hd(list), target - length(list))
    list ++ padding
  end
end
