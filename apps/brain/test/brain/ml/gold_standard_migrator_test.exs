defmodule Brain.ML.GoldStandardMigratorTest do
  use ExUnit.Case, async: false

  alias Brain.Corpus.Label
  alias Brain.ML.GoldStandardMigrator

  describe "contextual variants" do
    test "the export has no variant files, so no source or label names one" do
      {examples, _entities, source_files} = GoldStandardMigrator.preview(:all)

      variant_sources = Enum.filter(source_files, &String.contains?(Path.basename(&1), " - "))
      assert variant_sources == [], "contextual-variant source files remain: #{inspect(variant_sources)}"

      variant_labels =
        examples
        |> Enum.map(& &1["intent"])
        |> Enum.filter(&String.contains?(&1, " - "))
        |> Enum.uniq()

      assert variant_labels == [], "contextual-variant intent labels remain: #{inspect(variant_labels)}"
    end

    test "a variant name raises instead of being filed under its base" do
      assert_raise ArgumentError, ~r/contextual-variant name/, fn ->
        Label.canonical("smarthome.device.volume - context_device_volume - comment_points")
      end
    end
  end
end
