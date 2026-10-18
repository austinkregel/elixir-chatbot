defmodule Mix.Tasks.TrainModelsTest do
  use ExUnit.Case, async: false

  # Boot loads the span reliability model, which is trained from the entity
  # model, so the entity model has to be trainable before any boot artifact
  # exists. The test suite's own application is already booted, so the start
  # path runs in a fresh VM that starts nothing (`mix run --no-start`), with
  # the models directory pointed at an empty temporary one: if the start path
  # booted the brain application, its loaders would raise there.

  @tag timeout: 600_000
  test "entity training starts and trains with none of the boot artifacts present" do
    dir = Path.join(System.tmp_dir!(), "train_models_entity_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    assert File.ls!(dir) == []

    code = """
    dir = System.fetch_env!("ENTITY_TRAINING_MODELS_PATH")
    Application.put_env(:brain, :ml, Keyword.put(Application.get_env(:brain, :ml, []), :models_path, dir))
    Mix.Tasks.TrainModels.start_entity_training!()
    rows = "ner" |> Brain.ML.EvaluationStore.load_gold_standard(:train) |> Enum.take(25)
    %{entity_model_trained: true} =
      Brain.ML.Trainer.train_entity_model(%{}, models_path: Path.dirname(Brain.ML.EntityModel.trained_path()), rows: rows)

    IO.puts("brain_started=" <> inspect(List.keymember?(Application.started_applications(), :brain, 0)))
    """

    {output, status} =
      System.cmd("mix", ["run", "--no-start", "--no-compile", "--no-deps-check", "-e", code],
        cd: Brain.umbrella_root(),
        env: [{"MIX_ENV", "test"}, {"ENTITY_TRAINING_MODELS_PATH", dir}],
        stderr_to_stdout: true
      )

    assert status == 0, output
    assert output =~ "brain_started=false"
    assert File.ls!(dir) == ["entity_model.term"]

    model = dir |> Path.join("entity_model.term") |> Brain.ML.EntityModel.read!(:trained)
    assert is_float(model.transition_smoothing)
  end
end
