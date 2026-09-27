defmodule Brain.ML.WeightOptimizerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.WeightOptimizer

  # Tiny linearly-separable dataset across two classes — keeps the GA
  # cheap (under a second per generation even with EXLA boot cost).
  defp tiny_dataset do
    [
      {[1.0, 0.0, 0.0, 0.0], "a"},
      {[0.9, 0.1, 0.0, 0.0], "a"},
      {[1.0, 0.0, 0.1, 0.0], "a"},
      {[0.0, 0.0, 1.0, 0.0], "b"},
      {[0.0, 0.1, 0.9, 0.0], "b"},
      {[0.0, 0.0, 1.0, 0.1], "b"}
    ]
  end

  defp attach_capture do
    test_pid = self()
    handler_id = "weight-optimizer-test-#{System.unique_integer([:positive])}"

    # Module-function capture instead of an anonymous fn so :telemetry
    # doesn't print a "local function handler" warning for every test.
    # Destination pid travels via the `config` map.
    :telemetry.attach_many(
      handler_id,
      [
        [:brain, :weight_optimizer, :start],
        [:brain, :weight_optimizer, :generation],
        [:brain, :weight_optimizer, :stop],
        [:brain, :weight_optimizer, :exception]
      ],
      &__MODULE__.__handle_telemetry__/4,
      %{target: test_pid}
    )

    on_exit(fn -> :telemetry.detach(handler_id) end)
    :ok
  end

  @doc false
  def __handle_telemetry__(event, measurements, metadata, %{target: target})
      when is_pid(target) do
    send(target, {:telemetry, event, measurements, metadata})
    :ok
  end

  describe "optimize/2 telemetry" do
    test "emits :start with classifier and shape metadata" do
      attach_capture()

      run_id = "test-run-#{System.unique_integer([:positive])}"

      WeightOptimizer.optimize(tiny_dataset(),
        population_size: 6,
        max_generations: 2,
        early_stop_generations: 100,
        verbose: false,
        classifier: "test_classifier",
        run_id: run_id
      )

      assert_receive {:telemetry, [:brain, :weight_optimizer, :start], measurements,
                      %{
                        run_id: ^run_id,
                        classifier: "test_classifier",
                        dim: 4,
                        n_train: train_n,
                        n_val: val_n,
                        n_classes: 2
                      }},
                     30_000

      assert is_integer(measurements.system_time)
      assert train_n + val_n == 6
    end

    test "emits :generation events with full per-generation metrics" do
      attach_capture()

      run_id = "gen-run-#{System.unique_integer([:positive])}"

      WeightOptimizer.optimize(tiny_dataset(),
        population_size: 6,
        max_generations: 2,
        early_stop_generations: 100,
        verbose: false,
        classifier: "test_classifier",
        run_id: run_id
      )

      assert_receive {:telemetry, [:brain, :weight_optimizer, :generation], m,
                      %{run_id: ^run_id, classifier: "test_classifier", improved?: improved?}},
                     30_000

      assert is_boolean(improved?)
      assert is_integer(m.generation) and m.generation >= 0
      assert is_number(m.best_fitness)
      assert is_number(m.gen_best_fitness)
      assert is_number(m.raw_acc)
      assert is_number(m.balanced_acc)
      assert is_number(m.avg_fitness)
      assert is_integer(m.stale_count)
      assert is_number(m.mutation_rate)
      assert is_number(m.mutation_sigma)

      # The :stop event should always close the run regardless of how
      # many generations actually executed.
      assert_receive {:telemetry, [:brain, :weight_optimizer, :stop], _, %{run_id: ^run_id}},
                     30_000
    end

    test "emits :stop with duration and aggregate measurements" do
      attach_capture()

      run_id = "stop-run-#{System.unique_integer([:positive])}"

      result =
        WeightOptimizer.optimize(tiny_dataset(),
          population_size: 6,
          max_generations: 2,
          early_stop_generations: 100,
          verbose: false,
          classifier: "stop_classifier",
          run_id: run_id
        )

      assert_receive {:telemetry, [:brain, :weight_optimizer, :stop], measurements,
                      %{
                        run_id: ^run_id,
                        classifier: "stop_classifier",
                        status: status,
                        history: history,
                        weights: weights
                      }},
                     30_000

      assert status in [:complete, :early_stop]
      assert is_list(history) and history != []
      assert is_list(weights) and length(weights) == 4
      assert measurements.duration_ms >= 0
      assert measurements.total_dims == 4
      assert measurements.alive_dims >= 0 and measurements.alive_dims <= 4
      assert measurements.best_fitness >= 0.0 and measurements.best_fitness <= 1.0

      assert result.status == status
      assert result.run_id == run_id
      assert result.classifier == "stop_classifier"
      assert result.generations_run >= 1
    end
  end

  describe "optimize/2 result shape" do
    test "auto-generates run_id when not provided" do
      result =
        WeightOptimizer.optimize(tiny_dataset(),
          population_size: 4,
          max_generations: 1,
          early_stop_generations: 100,
          verbose: false,
          classifier: "auto_id"
        )

      assert is_binary(result.run_id)
      assert String.starts_with?(result.run_id, "ga-")
      assert result.classifier == "auto_id"
    end

    test "round-trips a caller-provided run_id" do
      result =
        WeightOptimizer.optimize(tiny_dataset(),
          population_size: 4,
          max_generations: 1,
          early_stop_generations: 100,
          verbose: false,
          run_id: "caller-supplied-id"
        )

      assert result.run_id == "caller-supplied-id"
      assert result.classifier == :unknown
    end
  end

  describe "determinism" do
    @quick [population_size: 12, max_generations: 6, early_stop_generations: 3, verbose: false]

    # Three overlapping classes built from a formula (no random draws). Unlike
    # tiny_dataset/0, no initial individual separates them perfectly, so the
    # search has to move and every random step affects the weights it returns.
    defp overlapping_dataset do
      for i <- 1..60 do
        label = Enum.at(["a", "b", "c"], rem(i, 3))
        c = rem(i, 3)
        {[:math.sin(i) + c * 0.3, :math.cos(i * 2), :math.sin(i * 3) - c * 0.2, c * 0.1 + :math.cos(i)], label}
      end
    end

    test "the same data and seed always produce the same weights" do
      first = WeightOptimizer.optimize(overlapping_dataset(), [seed: 5, run_id: "d1"] ++ @quick)
      second = WeightOptimizer.optimize(overlapping_dataset(), [seed: 5, run_id: "d2"] ++ @quick)

      assert first.weights == second.weights
      assert first.history == second.history
    end

    test "records the seed it ran with, defaulting to the configured one" do
      assert WeightOptimizer.optimize(tiny_dataset(), [seed: 5] ++ @quick).training_seed == 5

      assert WeightOptimizer.optimize(tiny_dataset(), @quick).training_seed ==
               Brain.ML.TrainingSeed.get!()
    end

    test "leaves the calling process's random state untouched" do
      :rand.seed(:exsss, {1, 2, 3})
      expected = :rand.uniform()

      :rand.seed(:exsss, {1, 2, 3})
      WeightOptimizer.optimize(tiny_dataset(), [seed: 5] ++ @quick)

      assert :rand.uniform() == expected
    end
  end
end
