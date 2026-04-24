defmodule Brain.ML.GoldStandardMigratorTest do
  use ExUnit.Case, async: false

  alias Brain.ML.GoldStandardMigrator

  describe "exclude_context_variants option" do
    test "preview with exclude_context_variants filters out context-bound usersays files" do
      {examples, _entities, source_files} =
        GoldStandardMigrator.preview(:all, exclude_context_variants: true)

      context_sources =
        Enum.filter(source_files, fn path ->
          basename = Path.basename(path)
          String.match?(basename, ~r/context_.*_usersays_en\.json$/)
        end)

      assert context_sources == [],
             "Expected no context-variant source files, got: #{inspect(context_sources)}"

      context_intents =
        examples
        |> Enum.map(& &1["intent"])
        |> Enum.filter(&String.contains?(&1, " - context_"))

      assert context_intents == [],
             "Expected no context-variant intent labels, got: #{inspect(Enum.uniq(context_intents))}"
    end

    test "preview without exclude_context_variants includes context-bound files" do
      {_examples, _entities, source_files} =
        GoldStandardMigrator.preview(:all, exclude_context_variants: false)

      context_sources =
        Enum.filter(source_files, fn path ->
          basename = Path.basename(path)
          String.match?(basename, ~r/context_.*_usersays_en\.json$/)
        end)

      assert length(context_sources) > 0,
             "Expected some context-variant source files to be included by default"
    end
  end
end
