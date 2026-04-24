defmodule Brain.ML.WeightOptimizer do
  @moduledoc """
  Genetic algorithm optimizer for per-dimension feature weights.

  Evolves a weight vector that maximizes a composite fitness (50% raw
  accuracy + 50% balanced accuracy) when applied element-wise to feature
  vectors before centroid-based cosine classification. The composite
  prevents the GA from sacrificing majority class accuracy to boost
  minority class recall or vice versa.

  Fitness evaluation uses `defn` + the EXLA compiler so XLA can
  JIT-compile and fuse all tensor ops into a single optimized
  executable per unique input shape, eliminating per-op dispatch
  overhead.

  ## Algorithm

  - **Chromosome**: N floats (one weight per feature dimension), range [0.0, 3.0]
  - **Initial population**: Fisher-seeded + uniform + random individuals
  - **Fitness**: Composite (50% raw accuracy + 50% balanced accuracy) on stratified validation split
  - **Selection**: Tournament (configurable size)
  - **Crossover**: Uniform crossover
  - **Mutation**: Adaptive Gaussian — aggressive when stale, conservative when improving
  - **Termination**: Max generations or early-stop on plateau
  """

  import Nx.Defn
  require Logger

  @default_opts [
    population_size: 100,
    max_generations: 200,
    early_stop_generations: 15,
    tournament_size: 5,
    mutation_rate: 0.12,
    mutation_sigma: 0.25,
    weight_min: 0.0,
    weight_max: 3.0,
    validation_split: 0.2,
    seed: {42, 137, 256},
    verbose: true
  ]

  @type training_example :: {list(float()), String.t()}

  @type result :: %{
          weights: list(float()),
          fitness: float(),
          generation: non_neg_integer(),
          history: list({non_neg_integer(), float()})
        }

  @doc """
  Find optimal per-dimension weights for the given training data.

  Returns a map with `:weights`, `:fitness`, `:generation`, and `:history`.
  """
  @spec optimize([training_example()], keyword()) :: result()
  def optimize(training_data, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    {seed_a, seed_b, seed_c} = opts[:seed]
    :rand.seed(:exsplus, {seed_a, seed_b, seed_c})

    previous_backend = Nx.default_backend()

    if Code.ensure_loaded?(EXLA) do
      Nx.default_backend(EXLA.Backend)
    end

    try do
      do_optimize(training_data, opts)
    after
      Nx.default_backend(previous_backend)
    end
  end

  defp do_optimize(training_data, opts) do
    dim = training_data |> List.first() |> elem(0) |> length()
    {train_set, val_set} = stratified_split(training_data, opts[:validation_split])

    {train_vecs, train_labels} = unzip_data(train_set)
    {val_vecs, val_labels} = unzip_data(val_set)

    label_set = Enum.uniq(train_labels) |> Enum.sort()
    label_to_idx = label_set |> Enum.with_index() |> Map.new()
    n_classes = length(label_set)

    if opts[:verbose] do
      Logger.info("WeightOptimizer: #{dim} dims, #{length(train_set)} train, #{length(val_set)} val, #{n_classes} classes")
    end

    train_t = Nx.tensor(train_vecs, type: :f32)
    val_t = Nx.tensor(val_vecs, type: :f32)

    train_label_idx = Enum.map(train_labels, &Map.fetch!(label_to_idx, &1))
    val_label_idx = Enum.map(val_labels, fn l -> Map.get(label_to_idx, l, -1) end)
    val_label_t = Nx.tensor(val_label_idx, type: :s32)

    {mask_t, train_counts_t} = build_class_masks(train_label_idx, n_classes)
    val_counts_t = build_val_counts(val_label_idx, n_classes)

    fisher_weights = compute_fisher_weights(train_vecs, train_labels, dim)
    population = initialize_population(dim, fisher_weights, opts)

    if opts[:verbose] do
      Logger.info("WeightOptimizer: population=#{opts[:population_size]}, max_gen=#{opts[:max_generations]}, early_stop=#{opts[:early_stop_generations]}, backend=#{inspect(Nx.default_backend())}")
    end

    ctx = %{
      train_t: train_t,
      val_t: val_t,
      val_label_t: val_label_t,
      mask_t: mask_t,
      train_counts_t: train_counts_t,
      val_counts_t: val_counts_t,
      n_classes: n_classes,
      label_set: label_set,
      dim: dim
    }

    evolve(population, ctx, opts)
  end

  @doc """
  Compute Fisher discriminant weights for seeding the GA.
  """
  @spec compute_fisher_weights(list(list(float())), list(String.t()), non_neg_integer()) ::
          list(float())
  def compute_fisher_weights(vecs, labels, dim) do
    n_total = length(vecs)
    by_class = Enum.zip(vecs, labels) |> Enum.group_by(&elem(&1, 1), &elem(&1, 0))

    global_mean = compute_mean(vecs, dim)

    class_stats =
      Map.new(by_class, fn {label, class_vecs} ->
        n_k = length(class_vecs)
        class_mean = compute_mean(class_vecs, dim)

        within_var =
          Enum.reduce(class_vecs, List.duplicate(0.0, dim), fn vec, acc ->
            Enum.zip_with(vec, Enum.zip(class_mean, acc), fn x, {m, a} ->
              a + (x - m) * (x - m)
            end)
          end)
          |> Enum.map(&(&1 / max(n_k, 1)))

        {label, %{mean: class_mean, within_var: within_var, n: n_k}}
      end)

    between_var =
      Enum.reduce(class_stats, List.duplicate(0.0, dim), fn {_label, stats}, acc ->
        Enum.zip_with(stats.mean, Enum.zip(global_mean, acc), fn m_k, {g, a} ->
          a + stats.n * (m_k - g) * (m_k - g)
        end)
      end)
      |> Enum.map(&(&1 / n_total))

    within_var_pooled =
      Enum.reduce(class_stats, List.duplicate(0.0, dim), fn {_label, stats}, acc ->
        Enum.zip_with(stats.within_var, acc, fn wv, a ->
          a + stats.n * wv
        end)
      end)
      |> Enum.map(&(&1 / n_total))

    Enum.zip_with(between_var, within_var_pooled, fn b, w ->
      fisher = if w < 1.0e-12, do: 0.0, else: b / w
      min(:math.sqrt(fisher), 3.0)
    end)
  end

  # ── Batched Nx fitness ─────────────────────────────────────────────

  defp build_class_masks(train_label_idx, n_classes) do
    masks =
      Enum.map(0..(n_classes - 1), fn c ->
        Enum.map(train_label_idx, fn l -> if l == c, do: 1.0, else: 0.0 end)
      end)

    mask_t = Nx.tensor(masks, type: :f32)
    counts_t = Nx.sum(mask_t, axes: [1])
    {mask_t, counts_t}
  end

  defp build_val_counts(val_label_idx, n_classes) do
    counts =
      Enum.map(0..(n_classes - 1), fn c ->
        Enum.count(val_label_idx, &(&1 == c)) * 1.0
      end)

    Nx.tensor(counts, type: :f32)
  end

  # Evaluate all individuals in a single batched EXLA dispatch.
  # Pre-allocate the results tensor here (where pop_size is a plain
  # integer) so the defn never needs dynamic shapes.
  defp evaluate_population_batched(population, ctx) do
    %{
      train_t: train_t,
      val_t: val_t,
      val_label_t: val_label_t,
      mask_t: mask_t,
      train_counts_t: train_counts_t,
      val_counts_t: val_counts_t
    } = ctx

    pop_size = length(population)
    n_val = Nx.axis_size(val_t, 0)
    pop_t = Nx.tensor(population, type: :f32)
    results_t = Nx.broadcast(Nx.tensor(0.0, type: :f32), {pop_size, 3})
    n_val_t = Nx.tensor(n_val, type: :f32)
    pop_size_t = Nx.tensor(pop_size, type: :s32)

    result = Nx.Defn.jit(&evaluate_population_jit/10, compiler: EXLA).(
      pop_t, results_t, train_t, val_t, val_label_t, mask_t, train_counts_t, val_counts_t,
      n_val_t, pop_size_t
    )

    Nx.to_flat_list(result)
  end

  # Evaluates the entire population in one XLA dispatch.
  # results_t is pre-allocated as {pop_size, 3} by the caller so we
  # never construct shapes from tensor-valued dimensions inside defn.
  # n_val_t is a scalar tensor so we don't call Nx.axis_size inside defn.
  defn evaluate_population_jit(pop_t, results_t, train_t, val_t, val_label_t, mask_t, train_counts_t, val_counts_t, n_val_t, pop_size_t) do
    safe_train_counts = Nx.select(
      Nx.equal(train_counts_t, 0.0),
      Nx.tensor(1.0, type: :f32),
      train_counts_t
    )

    count_denom = Nx.reshape(safe_train_counts, {:auto, 1})

    # Every variable used inside the while body or condition must be
    # threaded through the accumulator tuple.
    {results, _i, _pop_size_t, _pop_t, _train_t, _val_t, _val_label_t,
     _mask_t, _count_denom, _val_counts_t, _train_counts_t, _n_val_t} =
      while {
        results_t,
        i = Nx.tensor(0, type: :s32),
        pop_size_t, pop_t, train_t, val_t, val_label_t, mask_t,
        count_denom, val_counts_t, train_counts_t, n_val_t
      }, Nx.less(i, pop_size_t) do
        w = pop_t[i]
        weighted_train = Nx.multiply(train_t, w)
        weighted_val = Nx.multiply(val_t, w)

        centroids = Nx.dot(mask_t, weighted_train) / count_denom

        predictions = cosine_classify(weighted_val, centroids)

        raw_correct = Nx.equal(predictions, val_label_t) |> Nx.sum()
        raw_acc = raw_correct / n_val_t

        bal_acc = balanced_accuracy(predictions, val_label_t, val_counts_t, train_counts_t)
        composite = 0.5 * raw_acc + 0.5 * bal_acc

        row = Nx.stack([composite, raw_acc, bal_acc])
        new_results = Nx.put_slice(results_t, [i, Nx.tensor(0, type: :s32)], Nx.reshape(row, {1, 3}))

        {new_results, i + 1, pop_size_t, pop_t, train_t, val_t, val_label_t,
         mask_t, count_denom, val_counts_t, train_counts_t, n_val_t}
      end

    results
  end

  defnp cosine_classify(vecs, centroids) do
    dot = Nx.dot(vecs, Nx.transpose(centroids))

    vec_norms = vecs |> Nx.pow(2) |> Nx.sum(axes: [1]) |> Nx.sqrt() |> Nx.reshape({:auto, 1})
    cen_norms = centroids |> Nx.pow(2) |> Nx.sum(axes: [1]) |> Nx.sqrt() |> Nx.reshape({1, :auto})

    norm_product = Nx.multiply(vec_norms, cen_norms)
    safe_norms = Nx.select(Nx.equal(norm_product, 0.0), Nx.tensor(1.0, type: :f32), norm_product)

    similarities = Nx.divide(dot, safe_norms)
    Nx.argmax(similarities, axis: 1)
  end

  # Uses val_counts_t for recall denominator (correct fix), and
  # train_counts_t to determine which classes are present in training.
  # n_classes is derived from the shape of val_counts_t (which is {n_classes}).
  defnp balanced_accuracy(predictions, val_label_t, val_counts_t, train_counts_t) do
    all_classes = Nx.iota(Nx.shape(val_counts_t), type: :s32)
    truth_matrix = Nx.equal(Nx.reshape(val_label_t, {:auto, 1}), Nx.reshape(all_classes, {1, :auto}))
    pred_matrix = Nx.equal(Nx.reshape(predictions, {:auto, 1}), Nx.reshape(all_classes, {1, :auto}))

    correct_per_class = Nx.logical_and(truth_matrix, pred_matrix) |> Nx.sum(axes: [0])

    # Use validation counts for recall denominator
    safe_val_counts = Nx.select(
      Nx.equal(val_counts_t, 0.0),
      Nx.tensor(1.0, type: :f32),
      val_counts_t
    )
    recall_per_class = Nx.as_type(correct_per_class, :f32) / safe_val_counts

    # Only count classes present in both train and val
    present_mask =
      Nx.logical_and(Nx.greater(train_counts_t, 0.0), Nx.greater(val_counts_t, 0.0))
      |> Nx.as_type(:f32)

    recall_sum = Nx.sum(recall_per_class * present_mask)
    classes_present = Nx.sum(present_mask)
    safe_classes = Nx.select(Nx.equal(classes_present, 0.0), Nx.tensor(1.0, type: :f32), classes_present)

    recall_sum / safe_classes
  end

  # ── Evolution loop ──────────────────────────────────────────────────

  defp evolve(population, ctx, opts) do
    max_gen = opts[:max_generations]
    early_stop = opts[:early_stop_generations]

    initial_state = %{
      population: population,
      best_weights: List.first(population),
      best_fitness: 0.0,
      best_gen: 0,
      stale_count: 0,
      history: []
    }

    result =
      Enum.reduce_while(0..(max_gen - 1), initial_state, fn gen, state ->
        all_metrics = evaluate_population_batched(state.population, ctx)

        # all_metrics is a flat list of [c0,r0,b0, c1,r1,b1, ...] — reshape
        pop_size = length(state.population)
        metrics = chunk_metrics(all_metrics, pop_size)

        fitnesses = Enum.map(metrics, fn {composite, _raw, _bal} -> composite end)

        gen_best_idx = fitnesses |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
        gen_best_fitness = Enum.at(fitnesses, gen_best_idx)
        gen_best_weights = Enum.at(state.population, gen_best_idx)
        {_best_composite, best_raw, best_bal} = Enum.at(metrics, gen_best_idx)

        gen_avg = Enum.sum(fitnesses) / max(length(fitnesses), 1)

        {new_best_weights, new_best_fitness, new_best_gen, new_stale} =
          if gen_best_fitness > state.best_fitness do
            {gen_best_weights, gen_best_fitness, gen, 0}
          else
            {state.best_weights, state.best_fitness, state.best_gen, state.stale_count + 1}
          end

        if opts[:verbose] and (rem(gen, 10) == 0 or gen_best_fitness > state.best_fitness) do
          {m_rate, m_sigma} = adaptive_mutation_params(new_stale, opts)
          Logger.info(
            "WeightOptimizer gen #{String.pad_leading(Integer.to_string(gen), 3)}: " <>
              "fitness=#{pct(new_best_fitness)} " <>
              "raw=#{pct(best_raw)} bal=#{pct(best_bal)} " <>
              "avg=#{pct(gen_avg)} stale=#{new_stale} " <>
              "mut=#{Float.round(m_rate, 3)}/#{Float.round(m_sigma, 3)}"
          )
        end

        new_history = state.history ++ [{gen, gen_best_fitness}]

        if new_stale >= early_stop do
          if opts[:verbose] do
            Logger.info("WeightOptimizer: early stop at gen #{gen} (#{early_stop} gens w/o improvement)")
          end

          {:halt,
           %{
             state
             | best_weights: new_best_weights,
               best_fitness: new_best_fitness,
               best_gen: new_best_gen,
               history: new_history
           }}
        else
          {mutation_rate, mutation_sigma} = adaptive_mutation_params(new_stale, opts)

          ranked = Enum.zip(state.population, fitnesses) |> Enum.sort_by(&elem(&1, 1), :desc)

          elite_count = max(div(opts[:population_size], 10), 2)
          elites = ranked |> Enum.take(elite_count) |> Enum.map(&elem(&1, 0))

          # Catastrophic restart: re-randomize bottom 20% when deeply stale
          base_pop =
            if new_stale >= 7 do
              restart_count = div(opts[:population_size], 5)
              keep = ranked |> Enum.take(opts[:population_size] - restart_count) |> Enum.map(&elem(&1, 0))
              randoms = random_individuals(restart_count, ctx.dim, opts[:weight_min], opts[:weight_max])
              keep ++ randoms
            else
              state.population
            end

          children_needed = opts[:population_size] - elite_count

          children =
            Enum.map(1..children_needed, fn _ ->
              parent_a = tournament_select(base_pop, fitnesses, opts[:tournament_size])
              parent_b = tournament_select(base_pop, fitnesses, opts[:tournament_size])
              child = uniform_crossover(parent_a, parent_b)
              mutate(child, mutation_rate, mutation_sigma, opts[:weight_min], opts[:weight_max])
            end)

          new_population = elites ++ children

          {:cont,
           %{
             population: new_population,
             best_weights: new_best_weights,
             best_fitness: new_best_fitness,
             best_gen: new_best_gen,
             stale_count: new_stale,
             history: new_history
           }}
        end
      end)

    if opts[:verbose] do
      Logger.info(
        "WeightOptimizer: DONE — best fitness=#{pct(result.best_fitness)} at gen #{result.best_gen}"
      )

      alive = Enum.count(result.best_weights, &(&1 > 0.01))
      suppressed = Enum.count(result.best_weights, &(&1 <= 0.01))
      Logger.info("WeightOptimizer: #{alive} dims alive (weight > 0.01), #{suppressed} suppressed")
    end

    %{
      weights: result.best_weights,
      fitness: result.best_fitness,
      generation: result.best_gen,
      history: result.history
    }
  end

  # Parse the flat list returned by evaluate_population_batched into
  # a list of {composite, raw, balanced} tuples.
  defp chunk_metrics(flat_list, pop_size) do
    flat_list
    |> Enum.chunk_every(3)
    |> Enum.take(pop_size)
    |> Enum.map(fn [c, r, b] -> {c, r, b} end)
  end

  # Adaptive mutation: starts with base rates, boosts when stale to
  # escape local optima, decays back when improving.
  defp adaptive_mutation_params(stale_count, opts) do
    base_rate = opts[:mutation_rate]
    base_sigma = opts[:mutation_sigma]

    cond do
      stale_count >= 7 ->
        {min(base_rate * 2.5, 0.40), min(base_sigma * 2.5, 0.75)}

      stale_count >= 3 ->
        {min(base_rate * 1.8, 0.30), min(base_sigma * 1.8, 0.50)}

      stale_count == 0 ->
        {base_rate * 0.8, base_sigma * 0.8}

      true ->
        {base_rate, base_sigma}
    end
  end

  defp pct(value), do: "#{Float.round(value * 100, 1)}%"

  # ── GA operators ────────────────────────────────────────────────────

  defp initialize_population(dim, fisher_weights, opts) do
    pop_size = opts[:population_size]
    w_min = opts[:weight_min]
    w_max = opts[:weight_max]

    fisher_individual = Enum.map(fisher_weights, &clamp(&1, w_min, w_max))
    uniform_individual = List.duplicate(1.0, dim)

    random_count = pop_size - 2

    random_individuals =
      Enum.map(1..random_count, fn _ ->
        Enum.map(1..dim, fn _ -> :rand.uniform() * (w_max - w_min) + w_min end)
      end)

    [fisher_individual, uniform_individual | random_individuals]
  end

  defp random_individuals(count, dim, w_min, w_max) do
    Enum.map(1..count, fn _ ->
      Enum.map(1..dim, fn _ -> :rand.uniform() * (w_max - w_min) + w_min end)
    end)
  end

  defp tournament_select(population, fitnesses, tournament_size) do
    pop_size = length(population)
    pop_vec = :array.from_list(population)
    fit_vec = :array.from_list(fitnesses)

    best_idx =
      Enum.max_by(
        Enum.map(1..tournament_size, fn _ -> :rand.uniform(pop_size) - 1 end),
        fn i -> :array.get(i, fit_vec) end
      )

    :array.get(best_idx, pop_vec)
  end

  defp uniform_crossover(parent_a, parent_b) do
    Enum.zip_with(parent_a, parent_b, fn a, b ->
      if :rand.uniform() < 0.5, do: a, else: b
    end)
  end

  defp mutate(chromosome, mutation_rate, sigma, w_min, w_max) do
    Enum.map(chromosome, fn gene ->
      if :rand.uniform() < mutation_rate do
        clamp(gene + :rand.normal() * sigma, w_min, w_max)
      else
        gene
      end
    end)
  end

  # ── Data utilities ──────────────────────────────────────────────────

  defp stratified_split(data, split_ratio) do
    by_class = Enum.group_by(data, &elem(&1, 1))

    {train_acc, val_acc} =
      Enum.reduce(by_class, {[], []}, fn {_label, examples}, {train, val} ->
        shuffled = Enum.shuffle(examples)
        split_point = max(round(length(shuffled) * (1 - split_ratio)), 1)
        {class_train, class_val} = Enum.split(shuffled, split_point)
        {train ++ class_train, val ++ class_val}
      end)

    {Enum.shuffle(train_acc), Enum.shuffle(val_acc)}
  end

  defp unzip_data(data) do
    vecs = Enum.map(data, &elem(&1, 0))
    labels = Enum.map(data, &elem(&1, 1))
    {vecs, labels}
  end

  defp compute_mean(vecs, dim) do
    n = length(vecs)

    Enum.reduce(vecs, List.duplicate(0.0, dim), fn vec, acc ->
      Enum.zip_with(vec, acc, &(&1 + &2))
    end)
    |> Enum.map(&(&1 / max(n, 1)))
  end

  defp clamp(value, min_val, max_val) do
    value |> max(min_val) |> min(max_val)
  end
end
