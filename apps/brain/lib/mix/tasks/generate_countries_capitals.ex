defmodule Mix.Tasks.GenerateCountriesCapitals do
  @moduledoc """
  Generate country and capital entity lists from REST Countries (MPL-2.0).

  This produces small, high-signal entities (~250 countries; ~200-250 capitals),
  suitable for always-on entity extraction.

  ## Usage

      mix generate_countries_capitals [options]

  ## Options

    --download                 Force re-download of the REST Countries response
    --country-output PATH      Output JSON path (default: data/entities/country_entries_en.json)
    --capital-output PATH      Output JSON path (default: data/entities/capital_entries_en.json)

  ## Data Source

  REST Countries v3.1 (MPL-2.0): `https://restcountries.com/`
  """

  use Mix.Task
  require Logger

  @shortdoc "Generate country and capital entities"

  @default_country_output "data/entities/country_entries_en.json"
  @default_capital_output "data/entities/capital_entries_en.json"
  @cache_dir "priv/data_cache"
  @cache_file "restcountries_all.json"

  @endpoint "https://restcountries.com/v3.1/all"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          download: :boolean,
          country_output: :string,
          capital_output: :string
        ]
      )

    force_download = Keyword.get(opts, :download, false)
    country_output = Keyword.get(opts, :country_output, @default_country_output)
    capital_output = Keyword.get(opts, :capital_output, @default_capital_output)

    Mix.shell().info("Generating country + capital entities...")
    Mix.shell().info("  Country output: #{country_output}")
    Mix.shell().info("  Capital output: #{capital_output}")
    Mix.shell().info("")

    File.mkdir_p!(@cache_dir)
    cache_path = Path.join(@cache_dir, @cache_file)

    countries =
      case load_countries(cache_path, force_download) do
        {:ok, list} -> list
        {:error, reason} -> fatal("Failed to load REST Countries data: #{inspect(reason)}")
      end

    {country_entries, capital_entries} = build_entries(countries)

    write_json(country_entries, country_output)
    write_json(capital_entries, capital_output)

    Mix.shell().info("Successfully generated:")
    Mix.shell().info("  - #{length(country_entries)} countries")
    Mix.shell().info("  - #{length(capital_entries)} capitals")
    Mix.shell().info("")
    Mix.shell().info("Next steps:")
    Mix.shell().info("  1. Run `mix train_models --gazetteer-only`")
    Mix.shell().info("  2. Restart the app")
  end

  defp load_countries(cache_path, force_download) do
    if not force_download and File.exists?(cache_path) do
      with {:ok, content} <- File.read(cache_path),
           {:ok, %{"countries" => list}} <- Jason.decode(content),
           true <- is_list(list) do
        Mix.shell().info("Using cached REST Countries response: #{cache_path}")
        {:ok, list}
      else
        _ -> download_countries(cache_path)
      end
    else
      download_countries(cache_path)
    end
  end

  defp download_countries(cache_path) do
    Mix.shell().info("Fetching REST Countries dataset...")

    params = %{
      "fields" => "name,capital,altSpellings,cca2,cca3"
    }

    case Req.get(@endpoint,
           params: params,
           receive_timeout: 60_000,
           connect_options: [timeout: 30_000]
         ) do
      {:ok, %{status: 200, body: body}} when is_list(body) ->
        File.write!(cache_path, Jason.encode!(%{"countries" => body}, pretty: true))
        {:ok, body}

      {:ok, %{status: status}} ->
        {:error, {:http_status, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_entries(countries) when is_list(countries) do
    {countries_map, capitals_map} =
      Enum.reduce(countries, {%{}, %{}}, fn country, {country_acc, capital_acc} ->
        {country_entry, capital_entries} = country_to_entries(country)

        country_acc =
          case country_entry do
            nil ->
              country_acc

            %{"value" => v, "synonyms" => syns} ->
              key = String.downcase(v)

              Map.update(country_acc, key, %{"value" => v, "synonyms" => syns}, fn existing ->
                %{
                  "value" => existing["value"],
                  "synonyms" => Enum.uniq(existing["synonyms"] ++ syns)
                }
              end)
          end

        capital_acc =
          Enum.reduce(capital_entries, capital_acc, fn %{"value" => cv, "synonyms" => csyns},
                                                       acc ->
            key = String.downcase(cv)

            Map.update(acc, key, %{"value" => cv, "synonyms" => csyns}, fn existing ->
              %{
                "value" => existing["value"],
                "synonyms" => Enum.uniq(existing["synonyms"] ++ csyns)
              }
            end)
          end)

        {country_acc, capital_acc}
      end)

    country_entries =
      countries_map
      |> Map.values()
      |> Enum.map(fn %{"value" => v, "synonyms" => syns} ->
        %{"value" => v, "synonyms" => Enum.uniq([v | syns])}
      end)
      |> Enum.sort_by(fn %{"value" => v} -> String.downcase(v) end)

    capital_entries =
      capitals_map
      |> Map.values()
      |> Enum.map(fn %{"value" => v, "synonyms" => syns} ->
        %{"value" => v, "synonyms" => Enum.uniq([v | syns])}
      end)
      |> Enum.sort_by(fn %{"value" => v} -> String.downcase(v) end)

    {country_entries, capital_entries}
  end

  defp country_to_entries(country) when is_map(country) do
    common = get_in(country, ["name", "common"]) |> safe_trim()
    official = get_in(country, ["name", "official"]) |> safe_trim()
    alt_spellings = Map.get(country, "altSpellings", []) |> List.wrap()
    cca3 = Map.get(country, "cca3") |> safe_trim()
    capital_list = Map.get(country, "capital", []) |> List.wrap()

    country_entry =
      if common == "" do
        nil
      else
        synonyms =
          [official]
          |> Enum.concat(alt_spellings)
          |> Enum.concat([cca3])
          |> Enum.map(&safe_trim/1)
          |> Enum.reject(&(&1 == "" or &1 == common))
          |> Enum.uniq()

        %{"value" => common, "synonyms" => synonyms}
      end

    capital_entries =
      capital_list
      |> Enum.map(&safe_trim/1)
      |> Enum.reject(&(&1 == ""))
      |> Enum.map(fn cap ->
        %{"value" => cap, "synonyms" => derived_capital_synonyms(cap)}
      end)

    {country_entry, capital_entries}
  end

  defp country_to_entries(_), do: {nil, []}

  defp derived_capital_synonyms(capital) do
    # Conservative: normalize "City" punctuation variants a bit.
    #
    # Example: "Washington, D.C." -> also "Washington DC"
    no_periods = String.replace(capital, ".", "")
    no_commas = String.replace(no_periods, ",", "")
    collapsed = String.replace(no_commas, ~r/\s+/, " ") |> String.trim()

    [collapsed]
    |> Enum.reject(&(&1 == "" or &1 == capital))
    |> Enum.uniq()
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
