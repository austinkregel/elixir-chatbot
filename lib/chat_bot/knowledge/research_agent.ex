defmodule ChatBot.Knowledge.ResearchAgent do
  @moduledoc """
  Stateless worker module for fetching and analyzing web content.

  Research agents:
  - Accept a research goal (topic/questions)
  - Fetch content from web sources
  - Extract factual claims using the Analysis Pipeline
  - Return structured findings with source metadata

  Agents are designed to be run as supervised Tasks through the
  Learning Center's AgentSupervisor.

  ## Example

      goal = ResearchGoal.new("France", questions: ["What is the capital?"])
      {:ok, findings} = ResearchAgent.research(goal)
  """

  require Logger

  alias ChatBot.Analysis.Pipeline
  alias ChatBot.Knowledge.{HtmlProcessor, SourceReliability}
  alias ChatBot.Knowledge.Types.{Finding, SourceInfo, ResearchGoal}
  alias ChatBot.Telemetry

  # Configurable HTTP client (allows mocking in tests)
  @http_client Application.compile_env(:chat_bot, :http_client, Req)

  # Rate limiting: minimum delay between requests to same domain
  @rate_limit_ms 1000

  # Agent for tracking per-domain request times
  @rate_limiter_agent ChatBot.Knowledge.RateLimiter

  @type fetch_result :: {:ok, [Finding.t()]} | {:error, term()}

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Fetches and analyzes content for a research goal.

  Returns extracted findings with source metadata.

  ## Options
    - :sources - List of source types to use (default: [:web])
    - :max_pages - Maximum pages to fetch per source (default: 5)
    - :timeout - Request timeout in ms (default: 10000)
    - :mock - If true, uses mock data for testing
  """
  @spec research(ResearchGoal.t(), keyword()) :: fetch_result()
  def research(%ResearchGoal{} = goal, opts \\ []) do
    Telemetry.span(:knowledge_research, %{topic: goal.topic, questions: length(goal.questions)}, fn ->
      do_research(goal, opts)
    end)
  end

  defp do_research(%ResearchGoal{} = goal, opts) do
    sources = Keyword.get(opts, :sources, [:web])
    max_pages = Keyword.get(opts, :max_pages, 5)
    mock? = Keyword.get(opts, :mock, false)

    Logger.info("Starting research",
      topic: goal.topic,
      questions: length(goal.questions),
      sources: sources
    )

    try do
      # Generate search queries from goal
      queries = expand_goal_to_queries(goal)

      # Include goal in opts for task source
      opts_with_goal = Keyword.put(opts, :goal, goal)

      # Fetch from each source type
      raw_results =
        if mock? do
          generate_mock_results(goal, max_pages)
        else
          sources
          |> Enum.flat_map(&fetch_from_source(&1, queries, max_pages, opts_with_goal))
        end

      # Extract claims using Pipeline
      findings =
        raw_results
        |> Enum.flat_map(&extract_findings/1)
        |> Enum.map(&enrich_with_source_reliability/1)

      Logger.info("Research completed",
        topic: goal.topic,
        findings: length(findings)
      )

      {:ok, findings}
    rescue
      e ->
        Logger.error("Research failed",
          topic: goal.topic,
          error: Exception.message(e)
        )

        {:error, {:research_failed, Exception.message(e)}}
    end
  end

  @doc """
  Fetches content from a single URL.

  Used for direct URL fetching when the URL is already known.
  Respects rate limiting per domain.
  """
  @spec fetch_url(String.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def fetch_url(url, opts \\ []) when is_binary(url) do
    timeout = Keyword.get(opts, :timeout, 10_000)
    domain = SourceInfo.extract_domain(url)

    # Check if domain is blocked
    if SourceReliability.ready?() and SourceReliability.blocked?(domain) do
      {:error, :blocked_domain}
    else
      # Apply rate limiting
      wait_for_rate_limit(domain)

      # Fetch content
      case do_fetch(url, timeout) do
        {:ok, %{status: status, body: body}} when status in 200..299 ->
          record_request(domain)

          {:ok,
           %{
             url: url,
             content: body,
             source: SourceInfo.new(url, fetched_at: DateTime.utc_now())
           }}

        {:ok, %{status: status}} ->
          {:error, {:http_error, status}}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  # ============================================================================
  # Private Functions - Query Expansion
  # ============================================================================

  defp expand_goal_to_queries(%ResearchGoal{topic: topic, questions: questions}) do
    base_queries = [topic]

    question_queries =
      questions
      |> Enum.map(&question_to_query/1)

    (base_queries ++ question_queries)
    |> Enum.uniq()
    |> Enum.take(10)
  end

  defp question_to_query(question) when is_binary(question) do
    # Convert question to search query by removing question words
    question
    |> String.replace(~r/^(what|where|when|who|how|why|is|are|was|were|does|do|did)\s+/i, "")
    |> String.replace("?", "")
    |> String.trim()
  end

  # ============================================================================
  # Private Functions - Fetching
  # ============================================================================

  defp fetch_from_source(:web, queries, max_pages, opts) do
    # For web sources, we construct search-like URLs or use known sources
    # In a production system, this would integrate with a search API

    queries
    |> Enum.take(max_pages)
    |> Enum.flat_map(fn query ->
      # Generate potential source URLs based on query
      urls = generate_source_urls(query)

      urls
      |> Enum.take(3)
      |> Enum.map(fn url ->
        case fetch_url(url, opts) do
          {:ok, result} -> result
          {:error, _} -> nil
        end
      end)
      |> Enum.reject(&is_nil/1)
    end)
  end

  defp fetch_from_source(:mock, _queries, max_pages, _opts) do
    # Generate mock results for testing
    1..max_pages
    |> Enum.map(fn i ->
      %{
        url: "https://mock-source-#{i}.com/article",
        content: "This is mock content #{i} with factual claims.",
        source: SourceInfo.new("https://mock-source-#{i}.com/article")
      }
    end)
  end

  defp fetch_from_source(:task, _queries, max_pages, opts) do
    # Fetch from domain-specific NLP tasks
    # This provides high-quality, curated training data
    goal = Keyword.get(opts, :goal)

    if goal do
      alias ChatBot.Knowledge.TaskSource

      case TaskSource.fetch_for_goal(goal, max_tasks: max_pages, max_instances: 20) do
        {:ok, findings} ->
          # Convert findings to the expected raw result format
          # Note: Finding struct uses entity_type for task metadata, source.title for task_id
          Enum.map(findings, fn finding ->
            task_id = finding.source.title || "unknown"

            %{
              url: "task://#{task_id}",
              content: "#{finding.raw_context}\n\nAnswer: #{finding.claim}",
              source: finding.source,
              finding: finding
            }
          end)

        {:error, _reason} ->
          []
      end
    else
      []
    end
  end

  defp fetch_from_source(_source, _queries, _max_pages, _opts) do
    # Unknown source type
    []
  end

  defp generate_source_urls(query) do
    # Generate URLs for known reliable sources
    # In production, this would use a search API
    encoded_query = URI.encode(query)

    [
      "https://en.wikipedia.org/wiki/#{String.replace(query, " ", "_")}",
      "https://www.britannica.com/search?query=#{encoded_query}"
    ]
  end

  defp generate_mock_results(%ResearchGoal{topic: topic}, max_pages) do
    # Generate mock data for testing
    1..min(max_pages, 3)
    |> Enum.map(fn i ->
      %{
        url: "https://reliable-source-#{i}.com/#{topic}",
        content: """
        #{topic} is an important subject. Here are some facts about #{topic}.
        The main characteristic of #{topic} is well-documented.
        According to research, #{topic} has significant properties.
        """,
        source:
          SourceInfo.new("https://reliable-source-#{i}.com/#{topic}",
            reliability_score: 0.8,
            trust_tier: :verified
          )
      }
    end)
  end

  defp do_fetch(url, timeout) do
    # Use the configured HTTP client
    try do
      case @http_client.get(url, receive_timeout: timeout) do
        {:ok, response} -> {:ok, response}
        {:error, reason} -> {:error, reason}
      end
    rescue
      e -> {:error, {:fetch_error, Exception.message(e)}}
    end
  end

  # ============================================================================
  # Private Functions - Rate Limiting
  # ============================================================================

  defp wait_for_rate_limit(domain) do
    # Get or create rate limiter agent
    ensure_rate_limiter_started()

    case Agent.get(@rate_limiter_agent, &Map.get(&1, domain)) do
      nil ->
        :ok

      last_request_time ->
        elapsed = System.monotonic_time(:millisecond) - last_request_time
        remaining = @rate_limit_ms - elapsed

        if remaining > 0 do
          Process.sleep(remaining)
        end
    end
  end

  defp record_request(domain) do
    ensure_rate_limiter_started()
    Agent.update(@rate_limiter_agent, &Map.put(&1, domain, System.monotonic_time(:millisecond)))
  end

  defp ensure_rate_limiter_started do
    case Process.whereis(@rate_limiter_agent) do
      nil ->
        {:ok, _} = Agent.start_link(fn -> %{} end, name: @rate_limiter_agent)

      _pid ->
        :ok
    end
  end

  # ============================================================================
  # Private Functions - Extraction
  # ============================================================================

  # Handle pre-extracted findings from TaskSource
  defp extract_findings(%{finding: finding}) when is_map(finding) do
    [finding]
  end

  defp extract_findings(%{content: content, source: source}) when is_binary(content) do
    # Clean HTML content if present
    clean_content = clean_content(content)

    # Skip if no meaningful content after cleaning
    if String.length(clean_content) < 50 do
      Logger.debug("Content too short after cleaning", url: source.url)
      []
    else
      # Run through Pipeline for sentence segmentation and analysis
      case Pipeline.process(clean_content, skip_entity_extraction: false) do
        %{analyses: analyses} ->
          analyses
          |> Enum.filter(&is_factual_claim?/1)
          |> Enum.map(&build_finding(&1, source, clean_content))
          |> Enum.reject(&is_nil/1)

        _ ->
          []
      end
    end
  rescue
    e ->
      Logger.warning("Failed to extract findings",
        error: Exception.message(e),
        url: source.url
      )

      []
  end

  defp extract_findings(_), do: []

  defp clean_content(content) when is_binary(content) do
    if HtmlProcessor.is_html?(content) do
      case HtmlProcessor.extract_article_text(content) do
        {:ok, text} ->
          Logger.debug("Cleaned HTML content",
            original_length: String.length(content),
            clean_length: String.length(text)
          )

          text

        {:error, :no_content} ->
          # Fallback to basic HTML-to-text
          case HtmlProcessor.html_to_text(content, min_length: 20) do
            {:ok, text} -> text
            {:error, _} -> content
          end
      end
    else
      # Not HTML, return as-is
      content
    end
  end

  defp clean_content(content), do: to_string(content)

  defp is_factual_claim?(analysis) do
    # Consider assertive speech acts as potential factual claims
    # Exclude questions, commands, and expressives
    case analysis do
      %{speech_act: %{category: :assertive}} -> true
      %{speech_act: %{category: :commissive}} -> false
      %{speech_act: %{category: :directive}} -> false
      %{speech_act: %{category: :expressive}} -> false
      _ -> false
    end
  end

  defp build_finding(analysis, source, raw_content) do
    # Extract entities from the analysis
    entities = Map.get(analysis, :entities, [])

    # Find the primary entity (first one with high confidence)
    primary_entity =
      entities
      |> Enum.max_by(fn e -> Map.get(e, :confidence, 0) end, fn -> nil end)

    # Get the claim text
    claim = Map.get(analysis, :text, "")

    if primary_entity && String.length(claim) > 10 do
      entity_value = Map.get(primary_entity, :value) || Map.get(primary_entity, "value")
      entity_type = Map.get(primary_entity, :entity_type) || Map.get(primary_entity, "type")

      Finding.new(claim, entity_value || "unknown",
        source,
        entity_type: entity_type,
        raw_context: extract_context(raw_content, claim),
        confidence: Map.get(analysis, :confidence, 0.5)
      )
    else
      nil
    end
  end

  defp extract_context(content, claim) when is_binary(content) and is_binary(claim) do
    # Extract surrounding context for the claim
    case :binary.match(content, claim) do
      {start, len} ->
        context_start = max(0, start - 100)
        context_end = min(String.length(content), start + len + 100)

        String.slice(content, context_start, context_end - context_start)

      :nomatch ->
        String.slice(claim, 0, 200)
    end
  end

  defp extract_context(_, claim), do: claim

  defp enrich_with_source_reliability(%Finding{} = finding) do
    if SourceReliability.ready?() do
      case SourceReliability.lookup(finding.source.url) do
        {:ok, enriched_source} ->
          %{finding | source: enriched_source}

        _ ->
          finding
      end
    else
      finding
    end
  end
end
