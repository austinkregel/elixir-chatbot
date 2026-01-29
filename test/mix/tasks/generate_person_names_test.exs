defmodule Mix.Tasks.GeneratePersonNamesTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.GeneratePersonNames

  @moduletag :mix_task

  setup do
    # Create a unique temp directory for each test
    tmp_dir = Path.join(System.tmp_dir!(), "generate_person_names_test_#{:rand.uniform(100_000)}")
    File.mkdir_p!(tmp_dir)

    # Capture Mix.shell output
    Mix.shell(Mix.Shell.Process)

    on_exit(fn ->
      Mix.shell(Mix.Shell.IO)
      File.rm_rf!(tmp_dir)
    end)

    %{tmp_dir: tmp_dir}
  end

  describe "run/1 with --use-builtin" do
    test "generates person names from builtin list", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      GeneratePersonNames.run(["--use-builtin", "--output", output_path])

      # Verify file was created
      assert File.exists?(output_path)

      # Verify content
      {:ok, content} = File.read(output_path)
      {:ok, entries} = Jason.decode(content)

      assert is_list(entries)
      assert length(entries) > 100

      # Check structure of entries
      first_entry = hd(entries)
      assert Map.has_key?(first_entry, "value")
      assert Map.has_key?(first_entry, "synonyms")
    end

    test "filters names by minimum length", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      # Use higher min-length to reduce entries
      GeneratePersonNames.run(["--use-builtin", "--output", output_path, "--min-length", "5"])

      {:ok, content} = File.read(output_path)
      {:ok, entries} = Jason.decode(content)

      # All names should be at least 5 characters
      Enum.each(entries, fn entry ->
        assert String.length(entry["value"]) >= 5
      end)
    end

    test "includes common names like James and Mary", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      GeneratePersonNames.run(["--use-builtin", "--output", output_path])

      {:ok, content} = File.read(output_path)
      {:ok, entries} = Jason.decode(content)

      values = Enum.map(entries, & &1["value"])

      assert "James" in values
      assert "Mary" in values
      assert "Michael" in values
      assert "Elizabeth" in values
    end

    test "creates output directory if it does not exist", %{tmp_dir: tmp_dir} do
      output_path = Path.join([tmp_dir, "nested", "dir", "person_entries.json"])

      GeneratePersonNames.run(["--use-builtin", "--output", output_path])

      assert File.exists?(output_path)
    end

    test "shows success message with count", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      GeneratePersonNames.run(["--use-builtin", "--output", output_path])

      output = collect_shell_output()

      assert output =~ "Successfully generated"
      assert output =~ "person name entries"
    end
  end

  describe "option parsing" do
    test "accepts all supported options", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      # This should not raise
      GeneratePersonNames.run([
        "--use-builtin",
        "--output",
        output_path,
        "--min-count",
        "500",
        "--min-length",
        "4",
        "--years",
        "30"
      ])

      assert File.exists?(output_path)
    end
  end

  describe "name filtering" do
    test "excludes month names from stoplist", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      GeneratePersonNames.run(["--use-builtin", "--output", output_path])

      {:ok, content} = File.read(output_path)
      {:ok, entries} = Jason.decode(content)

      values = Enum.map(entries, &String.downcase(&1["value"]))

      # These should be excluded (months)
      refute "january" in values
      refute "february" in values
      refute "march" in values
    end

    test "excludes day names from stoplist", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      GeneratePersonNames.run(["--use-builtin", "--output", output_path])

      {:ok, content} = File.read(output_path)
      {:ok, entries} = Jason.decode(content)

      values = Enum.map(entries, &String.downcase(&1["value"]))

      # These should be excluded (days)
      refute "monday" in values
      refute "tuesday" in values
      refute "sunday" in values
    end

    test "entries are sorted alphabetically", %{tmp_dir: tmp_dir} do
      output_path = Path.join(tmp_dir, "person_entries.json")

      GeneratePersonNames.run(["--use-builtin", "--output", output_path])

      {:ok, content} = File.read(output_path)
      {:ok, entries} = Jason.decode(content)

      values = Enum.map(entries, & &1["value"])
      sorted_values = Enum.sort_by(values, &String.downcase/1)

      assert values == sorted_values
    end
  end

  # Helper to collect all shell output
  defp collect_shell_output do
    collect_shell_output([])
  end

  defp collect_shell_output(acc) do
    receive do
      {:mix_shell, :info, [msg]} ->
        collect_shell_output([msg | acc])

      {:mix_shell, :error, [msg]} ->
        collect_shell_output([msg | acc])
    after
      100 ->
        acc |> Enum.reverse() |> Enum.join("\n")
    end
  end
end
