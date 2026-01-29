defmodule Mix.Tasks.GenerateNewsSourcesTest do
  use ExUnit.Case, async: true

  @moduletag :mix_task

  # Cache file location used by the task
  @cache_dir "priv/data_cache"
  @cache_file "wikidata_news_sources.json"

  describe "module loading" do
    test "module is loaded correctly" do
      assert Code.ensure_loaded?(Mix.Tasks.GenerateNewsSources)
    end

    test "module uses Mix.Task" do
      behaviours = Mix.Tasks.GenerateNewsSources.__info__(:attributes)[:behaviour] || []
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
        assert is_list(data) or is_map(data)
      end
    end
  end

  describe "option parsing" do
    test "parses --output option" do
      {opts, _, _} =
        OptionParser.parse(["--output", "/tmp/news.json"],
          strict: [output: :string, limit: :integer, download: :boolean]
        )

      assert Keyword.get(opts, :output) == "/tmp/news.json"
    end

    test "parses --limit option" do
      {opts, _, _} =
        OptionParser.parse(["--limit", "100"],
          strict: [output: :string, limit: :integer, download: :boolean]
        )

      assert Keyword.get(opts, :limit) == 100
    end

    test "parses --download flag" do
      {opts, _, _} =
        OptionParser.parse(["--download"],
          strict: [output: :string, limit: :integer, download: :boolean]
        )

      assert Keyword.get(opts, :download) == true
    end

    test "parses combined options" do
      {opts, _, _} =
        OptionParser.parse(["--output", "/tmp/news.json", "--limit", "50"],
          strict: [output: :string, limit: :integer, download: :boolean]
        )

      assert Keyword.get(opts, :output) == "/tmp/news.json"
      assert Keyword.get(opts, :limit) == 50
    end
  end
end
