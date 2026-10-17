defmodule Mix.Tasks.Test.ModelsTest do
  use ExUnit.Case, async: false

  # The suite's own VM has already started EXLA, which hid that the task, run
  # on its own, never started what rebuilding the triple scorer reads when it
  # seeds the test world. A fresh VM that starts nothing runs it as
  # `MIX_ENV=test mix test.models` does. The task is idempotent: the models
  # it promotes are the ones the suite already runs on.
  @tag timeout: 600_000
  test "promotes and seeds the test world in a VM that starts nothing else" do
    {output, status} =
      System.cmd("mix", ["run", "--no-start", "--no-compile", "--no-deps-check", "-e", "Mix.Tasks.Test.Models.run([])"],
        cd: Brain.umbrella_root(),
        env: [{"MIX_ENV", "test"}],
        stderr_to_stdout: true
      )

    assert status == 0, output
    assert output =~ "seeded world default:"

    assert File.read!(Brain.ML.KnowledgeGraph.TripleScorer.path("default")) ==
             File.read!(Brain.ML.KnowledgeGraph.TripleScorer.trained_path())
  end
end
