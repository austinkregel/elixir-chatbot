defmodule Brain.ML.WeightOptimizer.TrackerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.WeightOptimizer.Tracker

  setup do
    Phoenix.PubSub.subscribe(Brain.PubSub, Tracker.topic())
    :ok
  end

  defp synthetic_run_id, do: "tracker-test-#{System.unique_integer([:positive])}"

  defp emit(:start, run_id) do
    :telemetry.execute(
      [:brain, :weight_optimizer, :start],
      %{system_time: System.system_time()},
      %{
        run_id: run_id,
        classifier: "test_clf",
        dim: 4,
        n_train: 8,
        n_val: 2,
        n_classes: 2,
        opts: %{population_size: 6, max_generations: 3}
      }
    )
  end

  defp emit(:generation, run_id, gen, fitness) do
    :telemetry.execute(
      [:brain, :weight_optimizer, :generation],
      %{
        generation: gen,
        best_fitness: fitness,
        gen_best_fitness: fitness,
        raw_acc: fitness,
        balanced_acc: fitness,
        avg_fitness: fitness * 0.8,
        stale_count: 0,
        mutation_rate: 0.1,
        mutation_sigma: 0.2
      },
      %{run_id: run_id, classifier: "test_clf", improved?: gen == 0}
    )
  end

  defp emit(:stop, run_id, status) do
    :telemetry.execute(
      [:brain, :weight_optimizer, :stop],
      %{
        duration_ms: 42,
        best_fitness: 0.85,
        best_generation: 1,
        generations_run: 2,
        alive_dims: 3,
        total_dims: 4
      },
      %{
        run_id: run_id,
        classifier: "test_clf",
        status: status,
        history: [{0, 0.5}, {1, 0.85}],
        weights: [1.0, 0.5, 0.5, 0.0]
      }
    )
  end

  describe "telemetry → state" do
    test "records an active run on :start and broadcasts :run_started" do
      run_id = synthetic_run_id()

      emit(:start, run_id)

      assert_receive {:run_started, run}, 1_000
      assert run.run_id == run_id
      assert run.classifier == "test_clf"
      assert run.status == :running
      assert run.dim == 4
      assert run.n_classes == 2

      active = Tracker.list_active()
      assert Enum.any?(active, &(&1.run_id == run_id))
    end

    test "applies generation snapshots and broadcasts {:generation, run_id, snapshot}" do
      run_id = synthetic_run_id()

      emit(:start, run_id)
      assert_receive {:run_started, _}, 1_000

      emit(:generation, run_id, 0, 0.4)
      emit(:generation, run_id, 1, 0.7)

      assert_receive {:generation, ^run_id, snap1}, 1_000
      assert snap1.generation == 0
      assert snap1.best_fitness == 0.4
      assert snap1.improved? == true

      assert_receive {:generation, ^run_id, snap2}, 1_000
      assert snap2.generation == 1
      assert snap2.best_fitness == 0.7

      run = Tracker.get_run(run_id)
      assert run.generation == 1
      assert run.best_fitness == 0.7
      assert length(run.history) == 2
    end

    test "moves run from active → recent on :stop and broadcasts :run_complete" do
      run_id = synthetic_run_id()

      emit(:start, run_id)
      assert_receive {:run_started, _}, 1_000
      emit(:generation, run_id, 0, 0.5)
      assert_receive {:generation, _, _}, 1_000
      emit(:stop, run_id, :complete)

      assert_receive {:run_complete, completed}, 1_000
      assert completed.run_id == run_id
      assert completed.status == :complete
      assert completed.best_fitness == 0.85
      assert completed.alive_dims == 3
      assert completed.total_dims == 4
      assert completed.duration_ms == 42

      refute Enum.any?(Tracker.list_active(), &(&1.run_id == run_id))
      assert Enum.any?(Tracker.list_recent(), &(&1.run_id == run_id))
    end

    test "captures :exception runs as :error" do
      run_id = synthetic_run_id()

      emit(:start, run_id)
      assert_receive {:run_started, _}, 1_000

      :telemetry.execute(
        [:brain, :weight_optimizer, :exception],
        %{duration_ms: 5},
        %{
          run_id: run_id,
          classifier: "test_clf",
          kind: :error,
          reason: %RuntimeError{message: "boom"},
          stacktrace: []
        }
      )

      assert_receive {:run_failed, failed}, 1_000
      assert failed.run_id == run_id
      assert failed.status == :error
      assert failed.error.kind == :error
    end
  end

  describe "start_run/2" do
    test "rejects an unknown classifier" do
      assert {:error, {:unknown_classifier, "nope"}} = Tracker.start_run("nope", [])
    end

    test "feature_vector_classifiers/0 lists the GA-eligible models" do
      classifiers = Tracker.feature_vector_classifiers()

      for expected <- ~w(intent_full intent_domain tense_class aspect_class urgency certainty_level) do
        assert expected in classifiers,
               "expected #{expected} in feature_vector_classifiers/0"
      end
    end
  end

  describe "cancel_run/1" do
    test "returns :not_found when the run isn't active" do
      assert {:error, :not_found} = Tracker.cancel_run("does-not-exist")
    end
  end
end
