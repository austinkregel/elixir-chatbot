defmodule Mix.Tasks.GenerateFortune500 do
  @moduledoc """
  Generate a bounded company entity list from Wikipedia's Fortune 500 page.

  This is intentionally *not* an exhaustive company dataset; it is capped to the
  Fortune 500 list for a given year to keep gazetteer size and false-positives manageable.

  ## Usage

      mix generate_fortune500 [options]

  ## Options

    --year YEAR           Fortune 500 list year to extract (default: 2024)
    --download            Force re-download of the Wikipedia wikitext
    --output PATH         Output JSON path (default: data/entities/company_entries_en.json)

  ## Data Source

  Wikipedia page: `Fortune 500` via MediaWiki API (CC BY-SA).
  """

  use Mix.Task
  require Logger

  @shortdoc "Generate company entities from Fortune 500 (Wikipedia)"

  @default_year 2024
  @default_output "data/entities/company_entries_en.json"
  @cache_dir "priv/data_cache"
  @cache_file "fortune500_wikitext.json"

  @wiki_api "https://en.wikipedia.org/w/api.php"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          year: :integer,
          download: :boolean,
          output: :string
        ]
      )

    year = Keyword.get(opts, :year, @default_year)
    force_download = Keyword.get(opts, :download, false)
    output_path = Keyword.get(opts, :output, @default_output)

    Mix.shell().info("Generating Fortune 500 company entities...")
    Mix.shell().info("  Year: #{year}")
    Mix.shell().info("  Output: #{output_path}")
    Mix.shell().info("")

    File.mkdir_p!(@cache_dir)
    cache_path = Path.join(@cache_dir, @cache_file)

    wikitext =
      case load_wikitext(cache_path, force_download) do
        {:ok, text} -> text
        {:error, reason} -> fatal("Failed to load Wikipedia wikitext: #{inspect(reason)}")
      end

    companies =
      case extract_companies_from_wikitext(wikitext, year) do
        {:ok, list} -> list
        {:error, reason} -> fatal("Failed to parse Fortune 500 list for #{year}: #{reason}")
      end

    entries =
      companies
      |> Enum.map(fn %{value: value, synonyms: synonyms} ->
        %{"value" => value, "synonyms" => Enum.uniq([value | synonyms])}
      end)
      |> Enum.sort_by(fn %{"value" => v} -> String.downcase(v) end)

    write_json(entries, output_path)

    Mix.shell().info("Successfully generated #{length(entries)} company entries")
    Mix.shell().info("Next steps:")
    Mix.shell().info("  1. Run `mix train_models --gazetteer-only`")
    Mix.shell().info("  2. Restart the app")
  end

  defp load_wikitext(cache_path, force_download) do
    if not force_download and File.exists?(cache_path) do
      with {:ok, content} <- File.read(cache_path),
           {:ok, %{"wikitext" => wikitext}} <- Jason.decode(content),
           true <- is_binary(wikitext) do
        Mix.shell().info("Using cached Wikipedia wikitext: #{cache_path}")
        {:ok, wikitext}
      else
        _ -> download_wikitext(cache_path)
      end
    else
      download_wikitext(cache_path)
    end
  end

  defp download_wikitext(cache_path) do
    Mix.shell().info("Downloading Wikipedia wikitext (Fortune 500)...")

    params = %{
      "action" => "parse",
      "page" => "Fortune_500",
      "prop" => "wikitext",
      "format" => "json",
      "formatversion" => "2"
    }

    case Req.get(@wiki_api,
           params: params,
           receive_timeout: 60_000,
           connect_options: [timeout: 30_000]
         ) do
      {:ok, %{status: 200, body: body}} when is_map(body) ->
        wikitext = get_in(body, ["parse", "wikitext"])

        if is_binary(wikitext) and byte_size(wikitext) > 0 do
          File.write!(cache_path, Jason.encode!(%{"wikitext" => wikitext}, pretty: true))
          {:ok, wikitext}
        else
          {:error, :missing_wikitext}
        end

      {:ok, %{status: status}} ->
        {:error, {:http_status, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp extract_companies_from_wikitext(wikitext, year) do
    with {:ok, section_text} <- extract_year_section(wikitext, year),
         {:ok, table_text} <- extract_first_wikitable(section_text),
         {:ok, headers, rows} <- parse_wikitable(table_text),
         {:ok, company_idx} <- find_company_column(headers) do
      companies =
        rows
        |> Enum.map(&Enum.at(&1, company_idx))
        |> Enum.reject(&is_nil/1)
        |> Enum.map(&String.trim/1)
        |> Enum.reject(&(&1 == ""))
        |> Enum.map(&company_from_cell/1)
        |> Enum.reject(&is_nil/1)
        |> dedupe_company_entries()

      {:ok, companies}
    end
  end

  defp extract_year_section(wikitext, year) do
    # Prefer explicit section "Fortune 500 list of YEAR"
    year_heading = ~r/^==\s*Fortune\s+500\s+list\s+of\s+#{year}\s*==\s*$/m

    case Regex.split(year_heading, wikitext, include_captures: false, parts: 2) do
      [_before, after_heading] ->
        # Cut off at next top-level section if present
        next_heading = ~r/^==[^=].*==\s*$/m

        section =
          case Regex.split(next_heading, after_heading, include_captures: false, parts: 2) do
            [current | _] -> current
            _ -> after_heading
          end

        {:ok, section}

      _ ->
        # Fallback: try to find the first section that looks like a Fortune 500 list and matches year
        fallback = ~r/^==\s*Fortune\s+500\s+list\s+of\s+(\d{4})\s*==\s*$/m

        case Regex.scan(fallback, wikitext) do
          [] ->
            {:error, "could not find a 'Fortune 500 list of YEAR' section in the page"}

          matches ->
            years =
              matches
              |> Enum.map(fn [_, y] -> String.to_integer(y) end)
              |> Enum.uniq()

            {:error, "available years in page: #{Enum.sort(years) |> Enum.join(", ")}"}
        end
    end
  end

  defp extract_first_wikitable(text) do
    # Find the first wikitable in the provided text.
    case Regex.run(~r/\{\|\s.*?\n\|\}\s*/s, text) do
      [table] -> {:ok, table}
      _ -> {:error, "could not find a wikitable in the section"}
    end
  end

  defp parse_wikitable(table_text) do
    # Minimal wikitext table parsing for the Fortune 500 section.
    # We only need: headers, then rows as list of cell strings.
    #
    # Table structure:
    #   {| ... header lines ...
    #   ! Rank !! Company !! ...
    #   |-
    #   | 1 || [[Walmart]] || ...
    #   |-
    #   ...
    #   |}
    table_body =
      table_text
      |> String.replace_prefix("{|", "")
      |> String.replace_suffix("|}", "")

    # Split into chunks by row delimiter; the first chunk includes header and maybe intro lines.
    chunks = String.split(table_body, "\n|-\n", trim: true)

    case chunks do
      [header_chunk | row_chunks] ->
        headers = parse_header_cells(header_chunk)

        rows =
          row_chunks
          |> Enum.map(&parse_row_cells/1)
          |> Enum.reject(&Enum.empty?/1)

        if headers == [] or rows == [] do
          {:error, "table did not contain recognizable headers/rows"}
        else
          {:ok, headers, rows}
        end

      _ ->
        {:error, "unexpected table structure"}
    end
  end

  defp parse_header_cells(header_chunk) do
    header_chunk
    |> String.split("\n")
    |> Enum.filter(&String.starts_with?(String.trim(&1), "!"))
    |> Enum.flat_map(fn line ->
      line
      |> String.trim()
      |> String.trim_leading("!")
      |> String.split("!!")
      |> Enum.map(&String.trim/1)
    end)
    |> Enum.reject(&(&1 == ""))
  end

  defp parse_row_cells(row_chunk) do
    row_chunk
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.filter(&String.starts_with?(&1, "|"))
    |> Enum.reject(&String.starts_with?(&1, "|}"))
    |> Enum.flat_map(fn line ->
      line
      |> String.trim_leading("|")
      |> String.split("||")
      |> Enum.map(&strip_cell_attributes/1)
      |> Enum.map(&String.trim/1)
    end)
    |> Enum.reject(&(&1 == ""))
  end

  defp strip_cell_attributes(cell) do
    trimmed = String.trim(cell)

    if String.starts_with?(trimmed, "style=") or String.starts_with?(trimmed, "scope=") or
         String.starts_with?(trimmed, "rowspan=") or String.starts_with?(trimmed, "colspan=") or
         String.starts_with?(trimmed, "align=") do
      # In tables, attributes are before the first "|" that separates attrs from content.
      case String.split(trimmed, "|", parts: 2) do
        [_attrs, content] -> String.trim(content)
        _ -> trimmed
      end
    else
      trimmed
    end
  end

  defp find_company_column(headers) do
    normalized =
      headers
      |> Enum.map(fn h -> h |> String.downcase() |> String.trim() end)

    idx =
      Enum.find_index(normalized, fn h ->
        h == "company" or h == "name" or String.contains?(h, "company") or
          String.contains?(h, "name")
      end)

    if is_integer(idx) do
      {:ok, idx}
    else
      {:error, "could not find a 'Company'/'Name' column in headers: #{Enum.join(headers, ", ")}"}
    end
  end

  defp company_from_cell(cell) do
    # Extract a reasonable plain-text label, plus some safe synonyms.
    #
    # Examples:
    #   [[Walmart]] -> value "Walmart"
    #   [[Alphabet Inc.|Alphabet]] -> value "Alphabet", synonym "Alphabet Inc."
    text = cell |> strip_refs_templates() |> String.trim()

    case Regex.scan(~r/\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]/, text) do
      [] ->
        value = strip_wiki_markup(text)
        value = normalize_company_value(value)
        build_company_entry(value, [])

      links ->
        {value, synonyms} =
          links
          |> Enum.reduce({nil, []}, fn [_, target, display], {val, syns} ->
            display = (display || target) |> strip_wiki_markup() |> normalize_company_value()
            target = target |> strip_wiki_markup() |> normalize_company_value()

            val = val || display
            syns = [target | syns]
            {val, syns}
          end)

        value = normalize_company_value(value || "")
        build_company_entry(value, synonyms)
    end
  end

  defp build_company_entry(value, synonyms) do
    value = String.trim(value)

    if value == "" do
      nil
    else
      synonyms =
        synonyms
        |> Enum.map(&String.trim/1)
        |> Enum.reject(&(&1 == "" or &1 == value))
        |> Enum.concat(derived_company_synonyms(value))
        |> Enum.uniq()

      %{value: value, synonyms: synonyms}
    end
  end

  defp derived_company_synonyms(value) do
    # Keep these conservative to avoid false positives.
    base = value |> String.replace(~r/\s+/, " ") |> String.trim()

    suffixes = [
      ~r/,\s*inc\.?$/i,
      ~r/\s+inc\.?$/i,
      ~r/,\s*corp\.?$/i,
      ~r/\s+corp\.?$/i,
      ~r/\s+corporation$/i,
      ~r/\s+co\.?$/i,
      ~r/\s+company$/i,
      ~r/\s+holdings$/i,
      ~r/\s+group$/i,
      ~r/\s+ltd\.?$/i,
      ~r/\s+limited$/i,
      ~r/\s+plc$/i
    ]

    without_suffix =
      Enum.reduce(suffixes, base, fn rx, acc -> Regex.replace(rx, acc, "") end)
      |> String.trim()

    without_the =
      case String.split(base, " ", parts: 2) do
        ["The", rest] -> rest
        ["the", rest] -> rest
        _ -> base
      end
      |> String.trim()

    [without_suffix, without_the]
    |> Enum.uniq()
    |> Enum.reject(&(&1 == "" or &1 == base))
  end

  defp strip_refs_templates(text) do
    text
    |> Regex.replace(~r/<ref[^>]*>.*?<\/ref>/s, "")
    |> Regex.replace(~r/<ref[^\/]*\/>/, "")
    |> Regex.replace(~r/\{\{.*?\}\}/s, "")
    |> Regex.replace(~r/<!--.*?-->/s, "")
  end

  defp strip_wiki_markup(text) do
    text
    |> Regex.replace(~r/\[\[([^\]\|]+)\|([^\]]+)\]\]/, "\\2")
    |> Regex.replace(~r/\[\[([^\]]+)\]\]/, "\\1")
    |> String.replace("'''", "")
    |> String.replace("''", "")
    |> String.replace("&amp;", "&")
    |> String.replace("&nbsp;", " ")
    |> String.replace("&quot;", "\"")
    |> String.replace("&#39;", "'")
    |> String.trim()
  end

  defp normalize_company_value(text) do
    text
    |> String.replace("\u00A0", " ")
    |> String.replace(~r/\s+/, " ")
    |> String.trim()
  end

  defp dedupe_company_entries(entries) do
    entries
    |> Enum.reduce(%{}, fn %{value: v, synonyms: syns}, acc ->
      key = String.downcase(v)

      Map.update(acc, key, %{value: v, synonyms: syns}, fn existing ->
        %{existing | synonyms: Enum.uniq(existing.synonyms ++ syns)}
      end)
    end)
    |> Map.values()
  end

  defp write_json(data, output_path) do
    output_path |> Path.dirname() |> File.mkdir_p!()
    File.write!(output_path, Jason.encode!(data, pretty: true))
  end

  defp fatal(message) do
    Mix.shell().error(message)
    System.halt(1)
  end
end
