defmodule Brain.ML.EvaluationStoreResultsTest do
  @moduledoc """
  `Brain.ML.EvaluationStore` reads evaluation results from Atlas only, and
  imports the results saved as JSON files that Atlas does not hold, once.
  """
  use ExUnit.Case, async: false

  alias Atlas.Schemas.EvaluationResult
  alias Brain.ML.EvaluationStore

  setup tags do
    {:ok, _apps} = Application.ensure_all_started(:atlas)
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    dir = Path.join(System.tmp_dir!(), "test_evaluation_results_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    {:ok, dir: dir}
  end

  test "results are read from Atlas, newest first, and a task Atlas holds none of has none" do
    assert EvaluationStore.latest("sentiment") == nil
    assert EvaluationStore.list_runs("sentiment") == []

    insert!(result("sentiment", 0.5), ~U[2026-01-01 00:00:00.000000Z])
    insert!(result("sentiment", 0.6), ~U[2026-02-01 00:00:00.000000Z])

    assert %{"accuracy" => 0.6, "timestamp" => "2026-02-01T00:00:00.000000Z"} = EvaluationStore.latest("sentiment")
    assert [%{"accuracy" => 0.6}, %{"accuracy" => 0.5}] = EvaluationStore.list_runs("sentiment")
  end

  test "the result files Atlas does not hold are imported at the time they record, once", %{dir: dir} do
    held = result("speech_act", 0.52)
    insert!(held, ~U[2026-04-26 06:18:59.000000Z])

    # The same run's file, which Atlas holds; a later run with the same scores,
    # which it does not; and a run it does not hold.
    write_file!(dir, held, "2026-04-26T06:18:58.807966Z")
    write_file!(dir, held, "2026-04-26T06:22:41.044960Z")
    write_file!(dir, result("speech_act", 0.53), "2026-04-30T06:33:12.263031Z")

    assert EvaluationStore.import_result_files!(dir) == %{files: 3, imported: 2, held: 1}

    assert [
             %{"accuracy" => 0.53, "timestamp" => "2026-04-30T06:33:12.263031Z"},
             %{"accuracy" => 0.52, "timestamp" => "2026-04-26T06:22:41.044960Z"},
             %{"accuracy" => 0.52, "timestamp" => "2026-04-26T06:18:59.000000Z"}
           ] = EvaluationStore.list_runs("speech_act")

    assert EvaluationStore.import_result_files!(dir) == %{files: 3, imported: 0, held: 3}
    assert length(EvaluationStore.list_runs("speech_act")) == 3
  end

  test "a file that cannot be read as a result raises, and nothing is imported", %{dir: dir} do
    write_file!(dir, result("ner", 0.3), "2026-09-30T15:55:15.045424Z")
    File.write!(Path.join(dir, "ner_broken.json"), ~s({"task": "ner"}))

    assert_raise RuntimeError, ~r/timestamp|holds no/, fn -> EvaluationStore.import_result_files!(dir) end
    assert EvaluationStore.list_runs("ner") == []
  end

  defp result(task, accuracy) do
    %{
      task: task,
      accuracy: accuracy,
      macro_f1: accuracy / 2,
      weighted_f1: accuracy / 3,
      total_examples: 100,
      per_class: %{"a" => %{"precision" => accuracy, "support" => 10}},
      confusion_matrix: %{"a" => %{"a" => 10}}
    }
  end

  defp insert!(result, inserted_at) do
    %EvaluationResult{}
    |> EvaluationResult.changeset(result)
    |> Ecto.Changeset.put_change(:inserted_at, inserted_at)
    |> Ecto.Changeset.put_change(:updated_at, inserted_at)
    |> Atlas.Repo.insert!()
  end

  defp write_file!(dir, result, timestamp) do
    content = result |> Map.put(:timestamp, timestamp) |> Map.put(:notes, "written by the test") |> Jason.encode!()
    File.write!(Path.join(dir, "#{result.task}_#{String.replace(timestamp, ":", "-")}.json"), content)
  end
end
