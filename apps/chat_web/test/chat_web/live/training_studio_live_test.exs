defmodule ChatWeb.TrainingStudioLiveTest do
  @moduledoc """
  The Training Studio browse tab under a filter: Edit and Delete on a visible
  row change the record that row shows, the delete confirmation names that
  record's position in the file, and the page bar counts the whole file and
  the matches separately.

  The source is a gazetteer file in a temporary training-data directory, so
  nothing under the real `data/` tree is read or written.
  """
  use ChatWeb.ConnCase, async: false

  import Phoenix.LiveViewTest

  @moduletag :tmp_dir

  @source "gaz_studio_fixture_entries_en"

  @records [
    %{"value" => "alpha", "synonyms" => []},
    %{"value" => "bravo", "synonyms" => []},
    %{"value" => "charlie-x", "synonyms" => []},
    %{"value" => "delta-x", "synonyms" => []},
    %{"value" => "echo", "synonyms" => []}
  ]

  setup %{tmp_dir: tmp_dir} do
    ml = Application.fetch_env!(:brain, :ml)
    Application.put_env(:brain, :ml, Keyword.put(ml, :training_data_path, tmp_dir))
    on_exit(fn -> Application.put_env(:brain, :ml, ml) end)

    path = Path.join([tmp_dir, "entities", "studio_fixture_entries_en.json"])
    File.mkdir_p!(Path.dirname(path))

    %{path: path}
  end

  defp write_records(path, records), do: File.write!(path, Jason.encode!(records))
  defp file_records(path), do: path |> File.read!() |> Jason.decode!()

  defp browse(conn, filter) do
    live(conn, "/training-studio?" <> URI.encode_query(%{tab: "browse", source: @source, page: 1, filter: filter}))
  end

  # Opens the delete confirmation on the visible row at `row` (1-based),
  # confirms it, and returns the confirmation as it read before confirming.
  defp delete_visible_row(view, row) do
    view |> element("tbody tr:nth-child(#{row}) button[title='Delete record']") |> render_click()
    confirmation = view |> element("[role=dialog]") |> render()
    view |> element("[role=dialog] button[id$='-confirm']") |> render_click()
    confirmation
  end

  defp edit_visible_row(view, row, value, synonyms) do
    view |> element("tbody tr:nth-child(#{row}) button[title='Edit record']") |> render_click()

    view
    |> form("form[phx-submit=save_edit]", %{record: %{value: value, synonyms: synonyms}})
    |> render_submit()
  end

  describe "with a filter that hides the first records" do
    setup %{path: path} do
      write_records(path, @records)
      :ok
    end

    test "the filter shows only the matching records", %{conn: conn} do
      {:ok, _view, html} = browse(conn, "-x")

      assert html =~ "charlie-x"
      assert html =~ "delta-x"
      refute html =~ "alpha"
      refute html =~ "bravo"
    end

    test "Delete on the first visible row deletes that record and nothing else", %{conn: conn, path: path} do
      {:ok, view, _html} = browse(conn, "-x")

      confirmation = delete_visible_row(view, 1)

      assert file_records(path) == List.delete_at(@records, 2)
      assert confirmation =~ "record 3"
    end

    test "Delete on the second visible row deletes that record and nothing else", %{conn: conn, path: path} do
      {:ok, view, _html} = browse(conn, "-x")

      confirmation = delete_visible_row(view, 2)

      assert file_records(path) == List.delete_at(@records, 3)
      assert confirmation =~ "record 4"
    end

    test "Edit on the first visible row saves over that record and nothing else", %{conn: conn, path: path} do
      {:ok, view, _html} = browse(conn, "-x")

      edit_visible_row(view, 1, "charlie-y", "cy")

      assert file_records(path) ==
               List.replace_at(@records, 2, %{"value" => "charlie-y", "synonyms" => ["cy"]})
    end

    test "Edit on the second visible row saves over that record and nothing else", %{conn: conn, path: path} do
      {:ok, view, _html} = browse(conn, "-x")

      edit_visible_row(view, 2, "delta-y", "dy")

      assert file_records(path) ==
               List.replace_at(@records, 3, %{"value" => "delta-y", "synonyms" => ["dy"]})
    end
  end

  describe "the page bar under a filter" do
    test "counts every record in the file and, separately, the matches", %{conn: conn, path: path} do
      records =
        for n <- 1..120 do
          value = if rem(n, 2) == 0, do: "even-#{n}", else: "odd-#{n}"
          %{"value" => value, "synonyms" => []}
        end

      write_records(path, records)

      {:ok, _view, html} = browse(conn, "even")

      assert html =~ "Showing 1–50 of 120"
      assert html =~ "60 match the filter"
    end
  end
end
