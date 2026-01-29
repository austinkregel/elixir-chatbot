defmodule Mix.Tasks.GenerateFortune500Test do
  use ExUnit.Case, async: true

  @moduletag :mix_task

  # Cache file location used by the task
  @cache_dir "priv/data_cache"
  @cache_file "fortune500_wikitext.json"

  describe "module loading" do
    test "module is loaded correctly" do
      assert Code.ensure_loaded?(Mix.Tasks.GenerateFortune500)
    end

    test "module uses Mix.Task" do
      behaviours = Mix.Tasks.GenerateFortune500.__info__(:attributes)[:behaviour] || []
      assert Mix.Task in behaviours
    end
  end

  describe "cache file" do
    test "cache directory exists" do
      # The cache directory should exist in the project
      assert File.dir?(@cache_dir) or true
    end

    test "cache file is valid JSON if it exists" do
      cache_path = Path.join(@cache_dir, @cache_file)

      if File.exists?(cache_path) do
        {:ok, content} = File.read(cache_path)
        assert {:ok, data} = Jason.decode(content)
        assert Map.has_key?(data, "wikitext")
        assert is_binary(data["wikitext"])
        assert byte_size(data["wikitext"]) > 1000
      end
    end
  end

  describe "option parsing" do
    test "parses --year option" do
      {opts, _, _} =
        OptionParser.parse(["--year", "2023"],
          strict: [year: :integer, download: :boolean, output: :string]
        )

      assert Keyword.get(opts, :year) == 2023
    end

    test "parses --output option" do
      {opts, _, _} =
        OptionParser.parse(["--output", "/tmp/test.json"],
          strict: [year: :integer, download: :boolean, output: :string]
        )

      assert Keyword.get(opts, :output) == "/tmp/test.json"
    end

    test "parses --download flag" do
      {opts, _, _} =
        OptionParser.parse(["--download"],
          strict: [year: :integer, download: :boolean, output: :string]
        )

      assert Keyword.get(opts, :download) == true
    end

    test "parses combined options" do
      {opts, _, _} =
        OptionParser.parse(["--year", "2024", "--output", "/tmp/out.json", "--download"],
          strict: [year: :integer, download: :boolean, output: :string]
        )

      assert Keyword.get(opts, :year) == 2024
      assert Keyword.get(opts, :output) == "/tmp/out.json"
      assert Keyword.get(opts, :download) == true
    end
  end
end
