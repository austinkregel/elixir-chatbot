defmodule Brain.Knowledge.Academic.OpenAlex do
  @moduledoc """
  Client for the OpenAlex API.

  OpenAlex is a free and open catalog of the global research system,
  indexing over 250 million scholarly works.

  API Documentation: https://docs.openalex.org/

  Rate Limits:
  - Completely open, no API key required
  - 100,000 requests per day
  - 10 requests per second

  ## Example

      {:ok, papers} = OpenAlex.search("transformer attention", limit: 10)
      {:ok, papers} = OpenAlex.search_cs("deep learning", limit: 5)  # CS only
  """

  require Logger

  alias Brain.Knowledge.Academic.Paper

  @base_url "https://api.openalex.org"

  # Configurable HTTP client (allows mocking in tests)
  @http_client Application.compile_env(:brain, :http_client, Req)

  # Rate limiting between requests (milliseconds)
  @rate_limit_ms 200

  # OpenAlex concept ID for Computer Science
  @cs_concept_id "C41008148"

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Searches for works (papers) matching the given query.

  ## Options
    - :limit - Maximum number of results (default: 10, max: 200)
    - :page - Page number for pagination (default: 1)
    - :filter - Additional filter string (OpenAlex filter syntax)
    - :sort - Sort field (default: "cited_by_count:desc")
    - :from_year - Filter papers from this year onwards
    - :to_year - Filter papers up to this year

  ## Returns
    - {:ok, [%Paper{}]} on success
    - {:error, reason} on failure
  """
  @spec search(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def search(query, opts \\ []) when is_binary(query) do
    limit = Keyword.get(opts, :limit, 10) |> min(200)
    page = Keyword.get(opts, :page, 1)
    sort = Keyword.get(opts, :sort, "cited_by_count:desc")

    # Build base parameters
    params = %{
      "search" => query,
      "per_page" => limit,
      "page" => page,
      "sort" => sort
    }

    # Add optional filters
    filter_parts = build_filters(opts)

    params =
      if filter_parts != [] do
        Map.put(params, "filter", Enum.join(filter_parts, ","))
      else
        params
      end

    url = "#{@base_url}/works"

    case do_request(url, params) do
      {:ok, %{"results" => works}} when is_list(works) ->
        papers = Enum.map(works, &parse_work/1) |> Enum.reject(&is_nil/1)
        Logger.debug("OpenAlex search completed", query: query, results: length(papers))
        {:ok, papers}

      {:ok, %{"results" => nil}} ->
        {:ok, []}

      {:ok, response} ->
        Logger.warning("Unexpected response format", response: inspect(response))
        {:ok, []}

      {:error, reason} = error ->
        Logger.error("OpenAlex search failed", query: query, error: inspect(reason))
        error
    end
  end

  @doc """
  Searches for Computer Science papers only.

  Applies a filter for the Computer Science concept.
  """
  @spec search_cs(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def search_cs(query, opts \\ []) do
    # Add CS concept filter
    existing_filter = Keyword.get(opts, :filter, "")

    cs_filter =
      if existing_filter == "" do
        "concepts.id:#{@cs_concept_id}"
      else
        "#{existing_filter},concepts.id:#{@cs_concept_id}"
      end

    opts = Keyword.put(opts, :filter, cs_filter)
    search(query, opts)
  end

  @doc """
  Retrieves a single work by its OpenAlex ID or DOI.

  ## Examples
      get_work("W2741809807")  # OpenAlex ID
      get_work("https://doi.org/10.1038/s41586-021-03819-2")  # DOI
  """
  @spec get_work(String.t()) :: {:ok, Paper.t()} | {:error, term()}
  def get_work(work_id) when is_binary(work_id) do
    url = "#{@base_url}/works/#{URI.encode(work_id)}"

    case do_request(url, %{}) do
      {:ok, %{"id" => _} = work} ->
        case parse_work(work) do
          nil -> {:error, :parse_failed}
          paper -> {:ok, paper}
        end

      {:ok, %{"error" => _}} ->
        {:error, :not_found}

      {:error, {:http_error, 404}} ->
        {:error, :not_found}

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Gets papers that cite the given work.

  ## Options
    - :limit - Maximum number of results (default: 10)
    - :page - Page number for pagination (default: 1)
  """
  @spec get_citations(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def get_citations(work_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10) |> min(200)
    page = Keyword.get(opts, :page, 1)

    params = %{
      "filter" => "cites:#{work_id}",
      "per_page" => limit,
      "page" => page,
      "sort" => "cited_by_count:desc"
    }

    url = "#{@base_url}/works"

    case do_request(url, params) do
      {:ok, %{"results" => works}} when is_list(works) ->
        papers = Enum.map(works, &parse_work/1) |> Enum.reject(&is_nil/1)
        {:ok, papers}

      {:ok, _} ->
        {:ok, []}

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Gets papers that the given work references.

  ## Options
    - :limit - Maximum number of results (default: 10)
  """
  @spec get_references(String.t(), keyword()) :: {:ok, [Paper.t()]} | {:error, term()}
  def get_references(work_id, opts \\ []) do
    # First get the work to find its references
    case get_work(work_id) do
      {:ok, _paper} ->
        # OpenAlex doesn't have a direct references endpoint,
        # but we can filter by referenced_works
        limit = Keyword.get(opts, :limit, 10) |> min(200)

        params = %{
          "filter" => "cited_by:#{work_id}",
          "per_page" => limit,
          "sort" => "cited_by_count:desc"
        }

        url = "#{@base_url}/works"

        case do_request(url, params) do
          {:ok, %{"results" => works}} when is_list(works) ->
            papers = Enum.map(works, &parse_work/1) |> Enum.reject(&is_nil/1)
            {:ok, papers}

          {:ok, _} ->
            {:ok, []}

          {:error, _} = error ->
            error
        end

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

    # Add polite pool email if configured
    params =
      case Application.get_env(:brain, :openalex_email) do
        nil -> params
        email -> Map.put(params, "mailto", email)
      end

    case @http_client.get(url, params: params, receive_timeout: 15_000) do
      {:ok, %{status: status, body: body}} when status in 200..299 ->
        {:ok, body}

      {:ok, %{status: 404}} ->
        {:error, {:http_error, 404}}

      {:ok, %{status: 429}} ->
        Logger.warning("OpenAlex rate limit hit")
        {:error, :rate_limited}

      {:ok, %{status: status, body: body}} ->
        Logger.warning("OpenAlex API error", status: status, body: inspect(body))
        {:error, {:http_error, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_filters(opts) do
    filters = []

    # Type filter (default to journal articles)
    filters =
      case Keyword.get(opts, :type) do
        nil -> filters
        type -> ["type:#{type}" | filters]
      end

    # Year range filter
    filters =
      case Keyword.get(opts, :from_year) do
        nil -> filters
        year -> ["from_publication_date:#{year}-01-01" | filters]
      end

    filters =
      case Keyword.get(opts, :to_year) do
        nil -> filters
        year -> ["to_publication_date:#{year}-12-31" | filters]
      end

    # Custom filter
    case Keyword.get(opts, :filter) do
      nil -> filters
      "" -> filters
      custom -> [custom | filters]
    end
  end

  defp parse_work(work) when is_map(work) do
    # Extract OpenAlex ID
    openalex_id = Map.get(work, "id", "") |> extract_openalex_id()

    title = Map.get(work, "title")

    # Abstract is stored inverted in OpenAlex
    abstract = parse_abstract(Map.get(work, "abstract_inverted_index"))

    # Parse authorships
    authorships = Map.get(work, "authorships", [])
    authors = Enum.map(authorships, &parse_authorship/1)

    # Get venue from primary_location
    venue = get_venue(work)

    # Publication year
    year = Map.get(work, "publication_year")

    # Citation count
    citation_count = Map.get(work, "cited_by_count", 0)

    # URLs
    doi = Map.get(work, "doi")

    url =
      cond do
        doi -> doi
        Map.get(work, "id") -> Map.get(work, "id")
        true -> nil
      end

    # External IDs
    ids = Map.get(work, "ids", %{})

    external_ids = %{
      "openalex" => openalex_id,
      "DOI" => doi |> maybe_extract_doi(),
      "pmid" => Map.get(ids, "pmid")
    }

    # Concepts as fields of study
    concepts = Map.get(work, "concepts", [])

    fields_of_study =
      concepts
      |> Enum.filter(fn c -> Map.get(c, "score", 0) > 0.3 end)
      |> Enum.map(fn c -> Map.get(c, "display_name") end)
      |> Enum.reject(&is_nil/1)

    # Get PDF URL if available
    pdf_url = get_pdf_url(work)

    if openalex_id && title do
      Paper.new(
        id: openalex_id,
        title: title,
        abstract: abstract,
        authors: authors,
        venue: venue,
        year: year,
        citation_count: citation_count,
        url: url,
        pdf_url: pdf_url,
        external_ids: external_ids,
        fields_of_study: fields_of_study,
        source: :openalex
      )
    else
      nil
    end
  end

  defp parse_work(_), do: nil

  defp extract_openalex_id(id) when is_binary(id) do
    # Extract ID from URL like "https://openalex.org/W2741809807"
    case Regex.run(~r/W\d+$/, id) do
      [match] -> match
      nil -> id
    end
  end

  defp extract_openalex_id(_), do: nil

  defp parse_abstract(nil), do: nil

  defp parse_abstract(inverted_index) when is_map(inverted_index) do
    # OpenAlex stores abstracts as inverted index: {"word": [positions]}
    # We need to reconstruct the text
    if map_size(inverted_index) == 0 do
      nil
    else
      # Find max position to determine array size
      max_pos =
        inverted_index
        |> Map.values()
        |> List.flatten()
        |> Enum.max(fn -> 0 end)

      # Create array and fill in words at their positions
      words = :array.new(max_pos + 1, default: "")

      words =
        Enum.reduce(inverted_index, words, fn {word, positions}, acc ->
          Enum.reduce(positions, acc, fn pos, inner_acc ->
            :array.set(pos, word, inner_acc)
          end)
        end)

      # Convert to list and join
      words
      |> :array.to_list()
      |> Enum.join(" ")
      |> String.trim()
    end
  rescue
    _ -> nil
  end

  defp parse_authorship(authorship) do
    author = Map.get(authorship, "author", %{})

    %{
      id: Map.get(author, "id") |> extract_openalex_id(),
      name: Map.get(author, "display_name")
    }
  end

  defp get_venue(work) do
    # Try primary_location first
    location = Map.get(work, "primary_location", %{}) || %{}
    source = Map.get(location, "source", %{}) || %{}

    cond do
      Map.get(source, "display_name") -> Map.get(source, "display_name")
      Map.get(work, "host_venue", %{}) |> Map.get("display_name") -> Map.get(work, "host_venue") |> Map.get("display_name")
      true -> nil
    end
  end

  defp get_pdf_url(work) do
    # Check primary_location for PDF
    location = Map.get(work, "primary_location", %{}) || %{}
    pdf = Map.get(location, "pdf_url")

    if pdf do
      pdf
    else
      # Check best_oa_location
      oa_location = Map.get(work, "best_oa_location", %{}) || %{}
      Map.get(oa_location, "pdf_url")
    end
  end

  defp maybe_extract_doi(nil), do: nil

  defp maybe_extract_doi(doi) when is_binary(doi) do
    # Extract DOI from URL if needed
    case Regex.run(~r/10\.\d{4,}\/[^\s]+/, doi) do
      [match] -> match
      nil -> nil
    end
  end
end
