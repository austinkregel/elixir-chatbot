defmodule Brain.ML.TrainingData.CatalogTest do
  @moduledoc """
  Reading a page of a source under a filter, and editing or deleting the
  record a page row shows. Each row carries its position in the unfiltered
  file, because that position is what `delete_record/2` and
  `update_record/3` act on.

  The source is a gazetteer file in a temporary training-data directory, so
  nothing under the real `data/` tree is read or written.
  """
  use ExUnit.Case, async: false

  alias Brain.ML.TrainingData.Catalog

  @moduletag :tmp_dir

  @source :gaz_catalog_fixture_entries_en

  @records [
    %{"value" => "alpha", "synonyms" => []},
    %{"value" => "bravo", "synonyms" => []},
    %{"value" => "charlie-x", "synonyms" => []},
    %{"value" => "delta-x", "synonyms" => []},
    %{"value" => "echo", "synonyms" => []},
    %{"value" => "foxtrot-x", "synonyms" => []}
  ]

  setup %{tmp_dir: tmp_dir} do
    ml = Application.fetch_env!(:brain, :ml)
    Application.put_env(:brain, :ml, Keyword.put(ml, :training_data_path, tmp_dir))
    on_exit(fn -> Application.put_env(:brain, :ml, ml) end)

    path = Path.join([tmp_dir, "entities", "catalog_fixture_entries_en.json"])
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(@records))

    %{path: path}
  end

  defp file_records(path), do: path |> File.read!() |> Jason.decode!()

  describe "read_source_page/4" do
    test "with no filter, each row carries its file position and both counts are the file's" do
      assert {:ok, %{rows: rows, total: 6, matching: 6}} = Catalog.read_source_page(@source, 2, 2)

      assert rows == [{2, Enum.at(@records, 2)}, {3, Enum.at(@records, 3)}]
    end

    test "a filter keeps each row's position in the unfiltered file" do
      assert {:ok, %{rows: rows, total: 6, matching: 3}} =
               Catalog.read_source_page(@source, 0, 50, filter: "-x")

      assert rows == [
               {2, Enum.at(@records, 2)},
               {3, Enum.at(@records, 3)},
               {5, Enum.at(@records, 5)}
             ]
    end

    test "an offset pages through the matches, not the file" do
      assert {:ok, %{rows: rows, total: 6, matching: 3}} =
               Catalog.read_source_page(@source, 2, 2, filter: "-x")

      assert rows == [{5, Enum.at(@records, 5)}]
    end
  end

  describe "editing the first row of a filtered page" do
    test "delete_record/2 removes that row's record and nothing else", %{path: path} do
      {:ok, %{rows: [{index, record} | _]}} = Catalog.read_source_page(@source, 0, 50, filter: "-x")

      assert record == %{"value" => "charlie-x", "synonyms" => []}
      assert :ok = Catalog.delete_record(@source, index)

      assert file_records(path) == List.delete_at(@records, 2)
    end

    test "update_record/3 replaces that row's record and nothing else", %{path: path} do
      {:ok, %{rows: [{index, _record} | _]}} = Catalog.read_source_page(@source, 0, 50, filter: "-x")
      replacement = %{"value" => "charlie-y", "synonyms" => ["cy"]}

      assert :ok = Catalog.update_record(@source, index, replacement)

      assert file_records(path) == List.replace_at(@records, 2, replacement)
    end
  end
end
