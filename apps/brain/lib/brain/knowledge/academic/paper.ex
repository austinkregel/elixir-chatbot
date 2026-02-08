defmodule Brain.Knowledge.Academic.Paper do
  @moduledoc "Unified paper representation for academic sources.\n\nThis struct normalizes paper data from different academic APIs\n(Semantic Scholar, arXiv, OpenAlex) into a common format.\n"

  alias Brain.Knowledge.Types.{Finding, SourceInfo}

  @type author :: %{
          id: String.t() | nil,
          name: String.t()
        }

  @type source :: :semantic_scholar | :arxiv | :openalex

  @type t :: %__MODULE__{
          id: String.t(),
          title: String.t(),
          abstract: String.t() | nil,
          authors: [author()],
          venue: String.t() | nil,
          year: non_neg_integer() | nil,
          citation_count: non_neg_integer(),
          url: String.t() | nil,
          pdf_url: String.t() | nil,
          external_ids: map(),
          fields_of_study: [String.t()],
          source: source(),
          fetched_at: DateTime.t()
        }

  @enforce_keys [:id, :title, :source]
  defstruct [
    :id,
    :title,
    :abstract,
    :venue,
    :year,
    :url,
    :pdf_url,
    :fetched_at,
    authors: [],
    citation_count: 0,
    external_ids: %{},
    fields_of_study: [],
    source: :semantic_scholar
  ]

  @doc "Creates a new Paper struct.\n"
  def new(opts) when is_list(opts) do
    %__MODULE__{
      id: Keyword.fetch!(opts, :id),
      title: Keyword.fetch!(opts, :title),
      abstract: Keyword.get(opts, :abstract),
      authors: Keyword.get(opts, :authors, []),
      venue: Keyword.get(opts, :venue),
      year: Keyword.get(opts, :year),
      citation_count: Keyword.get(opts, :citation_count, 0),
      url: Keyword.get(opts, :url),
      pdf_url: Keyword.get(opts, :pdf_url),
      external_ids: Keyword.get(opts, :external_ids, %{}),
      fields_of_study: Keyword.get(opts, :fields_of_study, []),
      source: Keyword.get(opts, :source, :semantic_scholar),
      fetched_at: DateTime.utc_now()
    }
  end

  @doc "Converts a Paper to a Finding struct for the knowledge system.\n\nUses the paper's abstract as the primary claim, with full metadata\npreserved in the source info.\n"
  @spec to_finding(t()) :: Finding.t() | nil
  def to_finding(%__MODULE__{abstract: nil}) do
    nil
  end

  def to_finding(%__MODULE__{abstract: abstract}) when byte_size(abstract) < 50 do
    nil
  end

  def to_finding(%__MODULE__{} = paper) do
    source = to_source_info(paper)
    entity = paper.title || "unknown"

    Finding.new(
      paper.abstract,
      entity,
      source,
      entity_type: "academic_paper",
      raw_context: build_context(paper),
      confidence: citation_to_confidence(paper.citation_count)
    )
  end

  @doc "Converts a Paper to a SourceInfo struct.\n"
  @spec to_source_info(t()) :: SourceInfo.t()
  def to_source_info(%__MODULE__{} = paper) do
    url = paper.url || build_url(paper)

    SourceInfo.new(url,
      title: paper.title,
      fetched_at: paper.fetched_at,
      reliability_score: source_reliability(paper.source),
      trust_tier: :verified
    )
  end

  @doc "Gets the arXiv ID from external IDs if available.\n"
  @spec arxiv_id(t()) :: String.t() | nil
  def arxiv_id(%__MODULE__{external_ids: ids}) do
    Map.get(ids, "ArXiv") || Map.get(ids, "arxiv")
  end

  @doc "Gets the DOI from external IDs if available.\n"
  @spec doi(t()) :: String.t() | nil
  def doi(%__MODULE__{external_ids: ids}) do
    Map.get(ids, "DOI") || Map.get(ids, "doi")
  end

  @doc "Gets the PDF URL for the paper, preferring arXiv.\n"
  @spec pdf_url(t()) :: String.t() | nil
  def pdf_url(%__MODULE__{pdf_url: url}) when is_binary(url) do
    url
  end

  def pdf_url(%__MODULE__{} = paper) do
    case arxiv_id(paper) do
      nil -> nil
      id -> "https://arxiv.org/pdf/#{id}.pdf"
    end
  end

  @doc "Returns author names as a formatted string.\n"
  @spec author_string(t()) :: String.t()
  def author_string(%__MODULE__{authors: []}) do
    "Unknown authors"
  end

  def author_string(%__MODULE__{authors: authors}) do
    names = Enum.map(authors, fn a -> a.name || "Unknown" end)

    case names do
      [single] -> single
      [first, second] -> "#{first} and #{second}"
      [first | rest] -> "#{first} et al. (#{length(rest) + 1} authors)"
    end
  end

  defp build_context(%__MODULE__{} = paper) do
    parts = [
      "Title: #{paper.title}",
      "Authors: #{author_string(paper)}",
      if(paper.venue) do
        "Venue: #{paper.venue}"
      else
        nil
      end,
      if(paper.year) do
        "Year: #{paper.year}"
      else
        nil
      end,
      "Citations: #{paper.citation_count}"
    ]

    parts
    |> Enum.reject(&is_nil/1)
    |> Enum.join("\n")
  end

  defp build_url(%__MODULE__{} = paper) do
    cond do
      paper.url -> paper.url
      arxiv_id(paper) -> "https://arxiv.org/abs/#{arxiv_id(paper)}"
      doi(paper) -> "https://doi.org/#{doi(paper)}"
      paper.id -> "https://www.semanticscholar.org/paper/#{paper.id}"
      true -> "https://semanticscholar.org"
    end
  end

  defp source_reliability(:semantic_scholar) do
    0.95
  end

  defp source_reliability(:arxiv) do
    0.95
  end

  defp source_reliability(:openalex) do
    0.9
  end

  defp source_reliability(_) do
    0.8
  end

  defp citation_to_confidence(count) when count > 500 do
    0.95
  end

  defp citation_to_confidence(count) when count > 100 do
    0.85
  end

  defp citation_to_confidence(count) when count > 10 do
    0.7
  end

  defp citation_to_confidence(_) do
    0.5
  end
end