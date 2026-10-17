defmodule Mix.Tasks.GenMicroDataTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.GenMicroData

  @strict [only: :string, stats: :boolean, world: :string]

  defp opts(args) do
    {opts, [], []} = OptionParser.parse(args, strict: @strict)
    opts
  end

  describe "trains_language_artifacts?/2" do
    # Without `--stats` the switch is absent from the parsed options; reading
    # it raised (`not nil`), so the task could not run in its writing mode.
    test "a run without --stats that generates a feature-vector classifier trains them" do
      assert GenMicroData.trains_language_artifacts?(["intent_full", "coarse_semantic_class"], opts(["--world", "w"]))
    end

    test "a --stats run trains neither" do
      refute GenMicroData.trains_language_artifacts?(["intent_full"], opts(["--world", "w", "--stats"]))
    end

    test "a run generating only a text classifier trains neither" do
      refute GenMicroData.trains_language_artifacts?(["coarse_semantic_class"], opts(["--world", "w", "--only", "coarse_semantic_class"]))
    end
  end
end
