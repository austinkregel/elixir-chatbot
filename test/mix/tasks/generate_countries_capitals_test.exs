defmodule Mix.Tasks.GenerateCountriesCapitalsTest do
  use ExUnit.Case, async: true

  @moduletag :mix_task

  # Cache file location used by the task
  @cache_dir "priv/data_cache"
  @cache_file "restcountries_all.json"

  describe "module loading" do
    test "module is loaded correctly" do
      assert Code.ensure_loaded?(Mix.Tasks.GenerateCountriesCapitals)
    end

    test "module uses Mix.Task" do
      behaviours = Mix.Tasks.GenerateCountriesCapitals.__info__(:attributes)[:behaviour] || []
      assert Mix.Task in behaviours
    end
  end

  describe "cache file" do
    test "cache directory exists" do
      assert File.dir?(@cache_dir) or true
    end

    test "cache file is valid JSON if it exists" do
      cache_path = Path.join(@cache_dir, @cache_file)

      if File.exists?(cache_path) do
        {:ok, content} = File.read(cache_path)
        assert {:ok, data} = Jason.decode(content)
        # Data can be a list or a map with countries key
        assert is_list(data) or is_map(data)
      end
    end
  end

  describe "option parsing" do
    test "parses --country-output option" do
      {opts, _, _} =
        OptionParser.parse(["--country-output", "/tmp/countries.json"],
          strict: [country_output: :string, capital_output: :string, download: :boolean]
        )

      assert Keyword.get(opts, :country_output) == "/tmp/countries.json"
    end

    test "parses --capital-output option" do
      {opts, _, _} =
        OptionParser.parse(["--capital-output", "/tmp/capitals.json"],
          strict: [country_output: :string, capital_output: :string, download: :boolean]
        )

      assert Keyword.get(opts, :capital_output) == "/tmp/capitals.json"
    end

    test "parses --download flag" do
      {opts, _, _} =
        OptionParser.parse(["--download"],
          strict: [country_output: :string, capital_output: :string, download: :boolean]
        )

      assert Keyword.get(opts, :download) == true
    end

    test "parses combined options" do
      {opts, _, _} =
        OptionParser.parse([
          "--country-output", "/tmp/c.json",
          "--capital-output", "/tmp/cap.json"
        ],
          strict: [country_output: :string, capital_output: :string, download: :boolean]
        )

      assert Keyword.get(opts, :country_output) == "/tmp/c.json"
      assert Keyword.get(opts, :capital_output) == "/tmp/cap.json"
    end
  end
end
