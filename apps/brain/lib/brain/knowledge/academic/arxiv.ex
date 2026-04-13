defmodule Brain.Knowledge.Academic.Arxiv do
  @moduledoc """
  Client for the arXiv API.

  arXiv is a free distribution service and open-access archive for
  scholarly articles in physics, mathematics, computer science, and more.

  API Documentation: https://info.arxiv.org/help/api/index.html

  Rate Limits:
  - 3 requests per second
  - Bulk requests should include 3 second delay

  ## Example

      {:ok, papers} = Arxiv.search("attention mechanism", limit: 10)
      {:ok, papers} = Arxiv.search("cat:cs.LG", limit: 5)  # By category
  """

  require Logger

  alias Brain.Knowledge.Academic.Paper

  @base_url "http://export.arxiv.org/api/query"

  # Configurable HTTP client (allows mocking in tests)
  @http_client Application.compile_env(:brain, :http_client, Req)

  # Rate limiting between requests (milliseconds) - arXiv recommends 3s for bulk
  @rate_limit_ms 1000

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Searches for papers matching the given query.

  ## Query Syntax
  - Simple search: "machine learning"
  - By field: "ti:attention" (title), "au:hinton" (author), "abs:transformer" (abstract)
  - By category: "cat:cs.LG" (machine learning), "cat:cs.CL" (computation and language)
  - Combined: "ti:attention AND cat:cs.LG"

  ## Options
    - :limit - Maximum number of results (default: 10, max: 100)
    - :offset - Pagination offset (default: 0)
    - :sort_by - Sort field: "relevance", "lastUpdatedDate", "submittedDate" (default: "relevance")
    - :sort_order - "ascending" or "descending" (default: "descending")

  ## Returns
    - {:ok, [%Paper{}]} on success
    - {:error, reason} on failure
  """
  @spec search(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def search(query, opts \\ []) when is_binary(query) do
    limit = Keyword.get(opts, :limit, 10) |> min(100)
    offset = Keyword.get(opts, :offset, 0)
    sort_by = Keyword.get(opts, :sort_by, "relevance")
    sort_order = Keyword.get(opts, :sort_order, "descending")

    # Build query parameters
    params = %{
      "search_query" => build_search_query(query),
      "start" => offset,
      "max_results" => limit,
      "sortBy" => sort_by,
      "sortOrder" => sort_order
    }

    case do_request(@base_url, params) do
      {:ok, xml_body} ->
        papers = parse_atom_feed(xml_body)
        Logger.debug("arXiv search completed", query: query, results: length(papers))
        {:ok, papers}

      {:error, reason} = error ->
        Logger.error("arXiv search failed", query: query, error: inspect(reason))
        error
    end
  end

  @doc """
  Retrieves a single paper by its arXiv ID.

  ## Examples
      get_paper("2301.12345")
      get_paper("cs.LG/0601001")
  """
  @spec get_paper(String.t()) :: {:ok, Paper.t()} | {:error, term()}
  def get_paper(arxiv_id) when is_binary(arxiv_id) do
    # Normalize ID format
    normalized_id = normalize_arxiv_id(arxiv_id)

    params = %{
      "id_list" => normalized_id,
      "max_results" => 1
    }

    case do_request(@base_url, params) do
      {:ok, xml_body} ->
        case parse_atom_feed(xml_body) do
          [paper | _] -> {:ok, paper}
          [] -> {:error, :not_found}
        end

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Returns the PDF URL for an arXiv paper.
  """
  @spec get_pdf_url(String.t()) :: String.t()
  def get_pdf_url(arxiv_id) do
    normalized = normalize_arxiv_id(arxiv_id)
    "https://arxiv.org/pdf/#{normalized}.pdf"
  end

  @doc """
  Returns the abstract page URL for an arXiv paper.
  """
  @spec get_abs_url(String.t()) :: String.t()
  def get_abs_url(arxiv_id) do
    normalized = normalize_arxiv_id(arxiv_id)
    "https://arxiv.org/abs/#{normalized}"
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp do_request(url, params) do
    # Apply rate limiting
    Process.sleep(@rate_limit_ms)

    case @http_client.get(url, params: params, receive_timeout: 30_000) do
      {:ok, %{status: status, body: body}} when status in 200..299 ->
        {:ok, body}

      {:ok, %{status: status, body: body}} ->
        Logger.warning("arXiv API error", status: status, body: inspect(body))
        {:error, {:http_error, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_search_query(query) do
    # If query already contains arXiv field prefixes, use as-is
    if String.contains?(query, ":") do
      query
    else
      # Search in all fields by default
      "all:#{query}"
    end
  end

  defp normalize_arxiv_id(id) do
    id
    |> String.trim()
    |> String.replace_leading("arxiv:", "")
    |> String.replace_leading("arXiv:", "")
    |> String.replace(~r/^https?:\/\/arxiv\.org\/(abs|pdf)\//, "")
    |> String.replace_trailing(".pdf", "")
  end

  # ============================================================================
  # Atom XML Parsing
  # ============================================================================

  defp parse_atom_feed(xml_body) when is_binary(xml_body) do
    try do
      # Parse XML using Erlang's xmerl
      {doc, _} =
        xml_body
        |> String.to_charlist()
        |> :xmerl_scan.string(quiet: true)

      # Extract entries
      entries = :xmerl_xpath.string(~c"//entry", doc)

      Enum.map(entries, &parse_entry/1)
      |> Enum.reject(&is_nil/1)
    rescue
      e ->
        Logger.error("Failed to parse arXiv XML", error: Exception.message(e))
        []
    end
  end

  defp parse_atom_feed(xml_body) when is_list(xml_body) do
    # Handle charlist input
    parse_atom_feed(List.to_string(xml_body))
  end

  defp parse_atom_feed(_other) do
    # Unknown format - return empty
    []
  end

  defp parse_entry(entry) do
    id = get_text(entry, ~c"./id/text()")
    arxiv_id = extract_arxiv_id(id)

    title =
      get_text(entry, ~c"./title/text()")
      |> clean_text()

    abstract =
      get_text(entry, ~c"./summary/text()")
      |> clean_text()

    # Get authors
    author_nodes = :xmerl_xpath.string(~c"./author", entry)
    authors = Enum.map(author_nodes, &parse_author/1)

    # Get published/updated dates
    published = get_text(entry, ~c"./published/text()")
    year = extract_year(published)

    # Get categories
    category_nodes = :xmerl_xpath.string(~c"./category/@term", entry)
    categories = Enum.map(category_nodes, &extract_attr_value/1)

    # Get primary category for venue
    primary_category = List.first(categories)

    # Build PDF URL
    pdf_url = get_pdf_url(arxiv_id)

    if arxiv_id && title do
      Paper.new(
        id: arxiv_id,
        title: title,
        abstract: abstract,
        authors: authors,
        venue: primary_category,
        year: year,
        citation_count: 0,
        url: get_abs_url(arxiv_id),
        pdf_url: pdf_url,
        external_ids: %{"ArXiv" => arxiv_id},
        fields_of_study: categories,
        source: :arxiv
      )
    else
      nil
    end
  end

  defp parse_author(author_node) do
    name = get_text(author_node, ~c"./name/text()")
    %{id: nil, name: name}
  end

  defp get_text(node, xpath) when is_list(xpath) do
    # xpath is already a charlist
    case :xmerl_xpath.string(xpath, node) do
      [text_node | _] -> extract_text_value(text_node)
      [] -> nil
    end
  end

  defp get_text(node, xpath) when is_binary(xpath) do
    # Convert string to charlist for xmerl
    case :xmerl_xpath.string(String.to_charlist(xpath), node) do
      [text_node | _] -> extract_text_value(text_node)
      [] -> nil
    end
  end

  defp extract_text_value({:xmlText, _, _, _, value, _}) do
    List.to_string(value)
  end

  defp extract_text_value(_), do: nil

  defp extract_attr_value({:xmlAttribute, _, _, _, _, _, _, _, value, _}) do
    List.to_string(value)
  end

  defp extract_attr_value(_), do: nil

  defp extract_arxiv_id(nil), do: nil

  defp extract_arxiv_id(url) when is_binary(url) do
    # Extract ID from URL like "http://arxiv.org/abs/2301.12345v1"
    case Regex.run(~r/arxiv\.org\/abs\/(.+)$/, url) do
      [_, id] ->
        # Remove version suffix if present
        String.replace(id, ~r/v\d+$/, "")

      nil ->
        nil
    end
  end

  defp extract_year(nil), do: nil

  defp extract_year(date_string) when is_binary(date_string) do
    case Regex.run(~r/^(\d{4})/, date_string) do
      [_, year] -> String.to_integer(year)
      nil -> nil
    end
  end

  defp clean_text(nil), do: nil

  defp clean_text(text) when is_binary(text) do
    text
    |> String.replace(~r/\s+/, " ")
    |> String.trim()
  end
end
