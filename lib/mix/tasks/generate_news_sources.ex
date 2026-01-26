defmodule Mix.Tasks.GenerateNewsSources do
  @moduledoc """
  Generate a bounded news-source entity list from Wikidata (CC0).

  This task is designed to produce a *reasonable* list (hundreds to low thousands)
  rather than an exhaustive corpus, keeping gazetteer size and false-positives manageable.

  ## Usage

      mix generate_news_sources [options]

  ## Options

    --limit N             Max number of sources to include (default: 1000)
    --download            Force re-download (ignore cached response)
    --output PATH         Output JSON path (default: data/entities/news-source_entries_en.json)

  ## Data Source

  Wikidata SPARQL endpoint (CC0): `https://query.wikidata.org/sparql`
  """

  use Mix.Task
  require Logger

  @shortdoc "Generate news-source entities from Wikidata"

  @default_limit 1000
  @default_output "data/entities/news-source_entries_en.json"
  @cache_dir "priv/data_cache"
  @cache_file "wikidata_news_sources.json"

  @sparql_endpoint "https://query.wikidata.org/sparql"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          limit: :integer,
          download: :boolean,
          output: :string
        ]
      )

    limit = Keyword.get(opts, :limit, @default_limit)
    force_download = Keyword.get(opts, :download, false)
    output_path = Keyword.get(opts, :output, @default_output)

    Mix.shell().info("Generating news-source entities from Wikidata...")
    Mix.shell().info("  Limit: #{limit}")
    Mix.shell().info("  Output: #{output_path}")
    Mix.shell().info("")

    File.mkdir_p!(@cache_dir)
    cache_path = Path.join(@cache_dir, @cache_file)

    rows =
      case load_rows(cache_path, force_download, limit) do
        {:ok, rows} -> rows
        {:error, reason} -> fatal("Failed to load Wikidata results: #{inspect(reason)}")
      end

    entries =
      rows
      |> Enum.map(&row_to_entry/1)
      |> Enum.reject(&is_nil/1)
      |> dedupe_entries()
      |> Enum.sort_by(fn %{"value" => v} -> String.downcase(v) end)

    write_json(entries, output_path)

    Mix.shell().info("Successfully generated #{length(entries)} news-source entries")
    Mix.shell().info("Next steps:")
    Mix.shell().info("  1. Run `mix train_models --gazetteer-only`")
    Mix.shell().info("  2. Restart the app")
  end

  defp load_rows(cache_path, force_download, limit) do
    if not force_download and File.exists?(cache_path) do
      with {:ok, content} <- File.read(cache_path),
           {:ok, %{"bindings" => bindings, "limit" => cached_limit}} <- Jason.decode(content),
           true <- is_list(bindings),
           true <- cached_limit == limit do
        Mix.shell().info("Using cached Wikidata response: #{cache_path}")
        {:ok, bindings}
      else
        _ -> download_rows(cache_path, limit)
      end
    else
      download_rows(cache_path, limit)
    end
  end

  defp download_rows(cache_path, limit) do
    Mix.shell().info("Querying Wikidata SPARQL endpoint...")

    query = sparql_query(limit)

    headers = [
      {"accept", "application/sparql-results+json"},
      {"user-agent", "chat-bot (Mix task GenerateNewsSources)"}
    ]

    case Req.get(@sparql_endpoint,
           params: %{"format" => "json", "query" => query},
           headers: headers,
           receive_timeout: 60_000,
           connect_options: [timeout: 30_000]
         ) do
      {:ok, %{status: 200, body: body}} when is_map(body) ->
        bindings = get_in(body, ["results", "bindings"]) || []

        if is_list(bindings) do
          File.write!(
            cache_path,
            Jason.encode!(%{"limit" => limit, "bindings" => bindings}, pretty: true)
          )

          {:ok, bindings}
        else
          {:error, :unexpected_response}
        end

      {:ok, %{status: status}} ->
        {:error, {:http_status, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp sparql_query(limit) do
    # Pull a bounded list of common sources (newspapers, online newspapers, news agencies),
    # ordered by sitelinks as a rough popularity signal.
    #
    # Q11032  = newspaper
    # Q1153191 = online newspaper
    # Q102156 = news agency
    """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?item ?itemLabel (GROUP_CONCAT(DISTINCT ?altLabel; separator="|") AS ?altLabels) ?sitelinks WHERE {
      VALUES ?class { wd:Q11032 wd:Q1153191 wd:Q102156 } .
      ?item wdt:P31/wdt:P279* ?class .
      ?item wikibase:sitelinks ?sitelinks .
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
        ?item rdfs:label ?itemLabel .
      }
      OPTIONAL {
        ?item skos:altLabel ?altLabel .
        FILTER(LANG(?altLabel) = "en") .
      }
    }
    GROUP BY ?item ?itemLabel ?sitelinks
    ORDER BY DESC(?sitelinks)
    LIMIT #{limit}
    """
  end

  defp row_to_entry(binding) when is_map(binding) do
    label = get_in(binding, ["itemLabel", "value"]) |> safe_trim()
    alt_labels = get_in(binding, ["altLabels", "value"]) |> safe_trim()

    if label == "" do
      nil
    else
      synonyms =
        alt_labels
        |> split_alt_labels()
        |> Enum.concat(derived_news_source_synonyms(label))
        |> Enum.map(&safe_trim/1)
        |> Enum.reject(&(&1 == "" or &1 == label))
        |> Enum.uniq()

      %{"value" => label, "synonyms" => Enum.uniq([label | synonyms])}
    end
  end

  defp row_to_entry(_), do: nil

  defp split_alt_labels(""), do: []

  defp split_alt_labels(alt_labels) do
    alt_labels
    |> String.split("|", trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
  end

  defp derived_news_source_synonyms(label) do
    # Conservative: remove a leading "The " (common in English outlet names).
    case String.split(label, " ", parts: 2) do
      ["The", rest] -> [rest]
      ["the", rest] -> [rest]
      _ -> []
    end
  end

  defp dedupe_entries(entries) do
    entries
    |> Enum.reduce(%{}, fn %{"value" => v, "synonyms" => syns}, acc ->
      key = String.downcase(v)

      Map.update(acc, key, %{"value" => v, "synonyms" => syns}, fn existing ->
        %{
          "value" => existing["value"],
          "synonyms" => Enum.uniq(existing["synonyms"] ++ syns)
        }
      end)
    end)
    |> Map.values()
  end

  defp write_json(data, output_path) do
    output_path |> Path.dirname() |> File.mkdir_p!()
    File.write!(output_path, Jason.encode!(data, pretty: true))
  end

  defp safe_trim(nil), do: ""
  defp safe_trim(v) when is_binary(v), do: String.trim(v)
  defp safe_trim(_), do: ""

  defp fatal(message) do
    Mix.shell().error(message)
    System.halt(1)
  end
end
