defmodule Brain.Knowledge.Academic.SemanticScholar do
  @moduledoc """
  Client for the Semantic Scholar Academic Graph API.

  Semantic Scholar is an AI-powered research tool that provides access to
  a large corpus of academic papers with structured metadata.

  API Documentation: https://api.semanticscholar.org/api-docs/

  Rate Limits:
  - Free tier: 5,000 requests/day
  - 100 requests per 5 minutes without API key

  ## Example

      {:ok, papers} = SemanticScholar.search("attention mechanism transformers", limit: 10)
      {:ok, paper} = SemanticScholar.get_paper("arxiv:1706.03762")
  """

  require Logger

  alias Brain.Knowledge.Academic.Paper

  @base_url "https://api.semanticscholar.org/graph/v1"

  # Fields to request from the API
  @search_fields "paperId,title,abstract,authors,venue,year,citationCount,url,externalIds,fieldsOfStudy"
  @paper_fields "paperId,title,abstract,authors,venue,year,citationCount,url,externalIds,fieldsOfStudy,references,citations"

  # Configurable HTTP client (allows mocking in tests)
  @http_client Application.compile_env(:brain, :http_client, Req)

  # Rate limiting between requests (milliseconds)
  @rate_limit_ms 500

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Searches for papers matching the given query.

  ## Options
    - :limit - Maximum number of results (default: 10, max: 100)
    - :offset - Pagination offset (default: 0)
    - :year - Filter by publication year or range (e.g., "2020-2023")
    - :fields_of_study - Filter by field (e.g., "Computer Science")

  ## Returns
    - {:ok, [%Paper{}]} on success
    - {:error, reason} on failure
  """
  @spec search(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def search(query, opts \\ []) when is_binary(query) do
    limit = Keyword.get(opts, :limit, 10) |> min(100)
    offset = Keyword.get(opts, :offset, 0)

    params = %{
      "query" => query,
      "limit" => limit,
      "offset" => offset,
      "fields" => @search_fields
    }

    # Add optional filters
    params =
      params
      |> maybe_add_param("year", Keyword.get(opts, :year))
      |> maybe_add_param("fieldsOfStudy", Keyword.get(opts, :fields_of_study))

    url = "#{@base_url}/paper/search"

    case do_request(url, params) do
      {:ok, %{"data" => papers}} when is_list(papers) ->
        parsed = Enum.map(papers, &parse_paper/1)
        Logger.debug("Semantic Scholar search completed", query: query, results: length(parsed))
        {:ok, parsed}

      {:ok, %{"data" => nil}} ->
        {:ok, []}

      {:ok, response} ->
        Logger.warning("Unexpected response format", response: inspect(response))
        {:ok, []}

      {:error, reason} = error ->
        Logger.error("Semantic Scholar search failed", query: query, error: inspect(reason))
        error
    end
  end

  @doc """
  Retrieves a single paper by its ID.

  The paper_id can be:
  - Semantic Scholar ID: "649def34f8be52c8b66281af98ae884c09aef38b"
  - arXiv ID: "arxiv:1706.03762"
  - DOI: "10.1145/3295222"
  - PubMed ID: "pubmed:12345678"

  ## Returns
    - {:ok, %Paper{}} on success
    - {:error, :not_found} if paper doesn't exist
    - {:error, reason} on other failures
  """
  @spec get_paper(String.t()) :: {:ok, Paper.t()} | {:error, term()}
  def get_paper(paper_id) when is_binary(paper_id) do
    # URL encode the paper ID for special characters in DOIs
    encoded_id = URI.encode(paper_id, &URI.char_unreserved?/1)
    url = "#{@base_url}/paper/#{encoded_id}"
    params = %{"fields" => @paper_fields}

    case do_request(url, params) do
      {:ok, %{"paperId" => _} = paper} ->
        {:ok, parse_paper(paper)}

      {:ok, %{"error" => "Paper not found"}} ->
        {:error, :not_found}

      {:error, {:http_error, 404}} ->
        {:error, :not_found}

      {:error, reason} = error ->
        Logger.error("Failed to fetch paper", paper_id: paper_id, error: inspect(reason))
        error
    end
  end

  @doc """
  Retrieves papers that cite the given paper.

  ## Options
    - :limit - Maximum number of results (default: 10)
    - :offset - Pagination offset (default: 0)
  """
  @spec get_citations(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def get_citations(paper_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10) |> min(100)
    offset = Keyword.get(opts, :offset, 0)

    encoded_id = URI.encode(paper_id, &URI.char_unreserved?/1)
    url = "#{@base_url}/paper/#{encoded_id}/citations"
    params = %{"fields" => @search_fields, "limit" => limit, "offset" => offset}

    case do_request(url, params) do
      {:ok, %{"data" => citations}} when is_list(citations) ->
        papers =
          citations
          |> Enum.map(&Map.get(&1, "citingPaper", %{}))
          |> Enum.reject(&(&1 == %{}))
          |> Enum.map(&parse_paper/1)

        {:ok, papers}

      {:ok, _} ->
        {:ok, []}

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Retrieves papers that the given paper references.

  ## Options
    - :limit - Maximum number of results (default: 10)
    - :offset - Pagination offset (default: 0)
  """
  @spec get_references(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def get_references(paper_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10) |> min(100)
    offset = Keyword.get(opts, :offset, 0)

    encoded_id = URI.encode(paper_id, &URI.char_unreserved?/1)
    url = "#{@base_url}/paper/#{encoded_id}/references"
    params = %{"fields" => @search_fields, "limit" => limit, "offset" => offset}

    case do_request(url, params) do
      {:ok, %{"data" => references}} when is_list(references) ->
        papers =
          references
          |> Enum.map(&Map.get(&1, "citedPaper", %{}))
          |> Enum.reject(&(&1 == %{}))
          |> Enum.map(&parse_paper/1)

        {:ok, papers}

      {:ok, _} ->
        {:ok, []}

      {:error, _} = error ->
        error
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp do_request(url, params) do
    # Apply rate limiting
    Process.sleep(@rate_limit_ms)

    headers = build_headers()

    case @http_client.get(url, params: params, headers: headers, receive_timeout: 15_000) do
      {:ok, %{status: status, body: body}} when status in 200..299 ->
        {:ok, body}

      {:ok, %{status: 404}} ->
        {:error, {:http_error, 404}}

      {:ok, %{status: 429}} ->
        Logger.warning("Semantic Scholar rate limit hit")
        {:error, :rate_limited}

      {:ok, %{status: status, body: body}} ->
        Logger.warning("Semantic Scholar API error", status: status, body: inspect(body))
        {:error, {:http_error, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_headers do
    # Add API key if configured
    case Application.get_env(:brain, :semantic_scholar_api_key) do
      nil -> []
      key -> [{"x-api-key", key}]
    end
  end

  defp maybe_add_param(params, _key, nil), do: params
  defp maybe_add_param(params, key, value), do: Map.put(params, key, value)

  defp parse_paper(data) when is_map(data) do
    Paper.new(
      id: Map.get(data, "paperId"),
      title: Map.get(data, "title"),
      abstract: Map.get(data, "abstract"),
      authors: parse_authors(Map.get(data, "authors", [])),
      venue: Map.get(data, "venue"),
      year: Map.get(data, "year"),
      citation_count: Map.get(data, "citationCount", 0),
      url: Map.get(data, "url"),
      external_ids: Map.get(data, "externalIds", %{}),
      fields_of_study: Map.get(data, "fieldsOfStudy", []),
      source: :semantic_scholar
    )
  end

  defp parse_authors(authors) when is_list(authors) do
    Enum.map(authors, fn author ->
      %{
        id: Map.get(author, "authorId"),
        name: Map.get(author, "name")
      }
    end)
  end

  defp parse_authors(_), do: []
end
