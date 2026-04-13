defmodule Brain.Knowledge.ResearchAgent do
  @moduledoc "Stateless worker module for fetching and analyzing web content.\n\nResearch agents:\n- Accept a research goal (topic/questions)\n- Fetch content from web sources\n- Extract factual claims using the Analysis Pipeline\n- Return structured findings with source metadata\n\nAgents are designed to be run as supervised Tasks through the\nLearning Center's AgentSupervisor.\n\n## Example\n\n    goal = ResearchGoal.new(\"France\", questions: [\"What is the capital?\"])\n    {:ok, findings} = ResearchAgent.research(goal)\n"

  # Tasks.Source is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, Tasks.Source}

  alias Brain.Knowledge.Academic
  alias Brain.Knowledge.Types
  alias Brain.Knowledge
  require Logger

  alias Brain.Analysis.Pipeline
  alias Brain.Analysis.ComprehensionAssessor
  alias Knowledge.{HtmlProcessor, SourceReliability}
  alias Types.{Finding, SourceInfo, ResearchGoal}
  alias Brain.Telemetry
  @http_client Application.compile_env(:brain, :http_client, Req)
  @rate_limit_ms 1000
  @rate_limiter_agent Brain.Knowledge.RateLimiter

  @type fetch_result :: {:ok, [Finding.t()]} | {:error, term()}

  @doc "Fetches and analyzes content for a research goal.\n\nReturns extracted findings with source metadata.\n\n## Options\n  - :sources - List of source types to use (default: [:web])\n  - :max_pages - Maximum pages to fetch per source (default: 5)\n  - :timeout - Request timeout in ms (default: 10_000)\n  - :mock - If true, uses mock data for testing\n"
  @spec research(ResearchGoal.t(), keyword()) :: fetch_result()
  def research(%ResearchGoal{} = goal, opts \\ []) do
    Telemetry.span(
      :knowledge_research,
      %{topic: goal.topic, questions: length(goal.questions)},
      fn ->
        do_research(goal, opts)
      end
    )
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
      queries = expand_goal_to_queries(goal)
      opts_with_goal = Keyword.put(opts, :goal, goal)

      raw_results =
        if mock? do
          generate_mock_results(goal, max_pages)
        else
          sources
          |> Enum.flat_map(&fetch_from_source(&1, queries, max_pages, opts_with_goal))
        end

      findings =
        raw_results
        |> Enum.flat_map(&extract_findings/1)
        |> Enum.map(&enrich_with_source_reliability/1)

      Logger.info("Research completed", topic: goal.topic, findings: length(findings))

      {:ok, findings}
    rescue
      e ->
        Logger.error("Research failed", topic: goal.topic, error: Exception.message(e))

        {:error, {:research_failed, Exception.message(e)}}
    end
  end

  @doc "Fetches content from a single URL.\n\nUsed for direct URL fetching when the URL is already known.\nRespects rate limiting per domain.\n"
  @spec fetch_url(String.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def fetch_url(url, opts \\ []) when is_binary(url) do
    timeout = Keyword.get(opts, :timeout, 10_000)
    domain = SourceInfo.extract_domain(url)

    if SourceReliability.ready?() and SourceReliability.blocked?(domain) do
      {:error, :blocked_domain}
    else
      wait_for_rate_limit(domain)

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

  defp expand_goal_to_queries(%ResearchGoal{topic: topic, questions: questions}) do
    base_queries = [topic]

    question_queries =
      questions
      |> Enum.map(&question_to_query/1)

    (base_queries ++ question_queries)
    |> Enum.uniq()
    |> Enum.take(10)
  end

  @question_words MapSet.new(~w(what where when who how why is are was were does do did))

  defp question_to_query(question) when is_binary(question) do
    tokens = Brain.ML.Tokenizer.tokenize_normalized(question)

    tokens
    |> Enum.drop_while(fn token -> MapSet.member?(@question_words, token) end)
    |> Enum.join(" ")
    |> String.trim()
  end

  defp fetch_from_source(:web, queries, max_pages, opts) do
    queries
    |> Enum.take(max_pages)
    |> Enum.flat_map(fn query ->
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
    goal = Keyword.get(opts, :goal)

    if goal do
      alias Tasks.Source, as: TaskSource

      case TaskSource.fetch_for_goal(goal, max_tasks: max_pages, max_instances: 20) do
        {:ok, findings} ->
          Enum.map(findings, fn finding ->
            task_id = finding.source.title || "unknown"

            %{
              url: "task://#{task_id}",
              content: "#{finding.raw_context}

Answer: #{finding.claim}",
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

  defp fetch_from_source(:academic, queries, max_pages, _opts) do
    alias Academic.{SemanticScholar, Arxiv, OpenAlex, PaperModelBuilder}

    queries
    |> Enum.take(max_pages)
    |> Enum.flat_map(fn query ->
      tasks = [
        Task.async(fn -> SemanticScholar.search(query, limit: 5) end),
        Task.async(fn -> OpenAlex.search_cs(query, limit: 5) end),
        Task.async(fn -> Arxiv.search(query, limit: 3) end)
      ]

      papers =
        tasks
        |> Task.await_many(20_000)
        |> Enum.flat_map(fn
          {:ok, result} -> result
          {:error, _} -> []
        end)

      Logger.debug("Fetched academic papers", query: query, paper_count: length(papers))

      {:ok, _node_ids} = PaperModelBuilder.ingest_papers(papers)
      Logger.debug("Ingested papers into epistemic model", count: length(papers))

      Enum.map(papers, &paper_to_raw_result/1)
    end)
  end

  defp fetch_from_source(_source, _queries, _max_pages, _opts) do
    []
  end

  defp paper_to_raw_result(paper) do
    alias Brain.Knowledge.Academic.Paper

    source = Paper.to_source_info(paper)

    content =
      [
        "Title: #{paper.title}",
        if(paper.abstract) do
          "
Abstract: #{paper.abstract}"
        else
          ""
        end,
        "
Authors: #{Paper.author_string(paper)}",
        if(paper.venue) do
          "
Venue: #{paper.venue}"
        else
          ""
        end,
        if(paper.year) do
          "
Year: #{paper.year}"
        else
          ""
        end,
        "
Citations: #{paper.citation_count}"
      ]
      |> Enum.reject(&(&1 == ""))
      |> Enum.join("")

    %{
      url: paper.url || "academic://#{paper.source}/#{paper.id}",
      content: content,
      source: source,
      paper: paper
    }
  end

  defp generate_source_urls(query) do
    encoded_query = URI.encode(query)

    [
      "https://en.wikipedia.org/wiki/#{String.replace(query, " ", "_")}",
      "https://www.britannica.com/search?query=#{encoded_query}"
    ]
  end

  defp generate_mock_results(%ResearchGoal{topic: topic}, max_pages) do
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
    try do
      case @http_client.get(url, receive_timeout: timeout) do
        {:ok, response} -> {:ok, response}
        {:error, reason} -> {:error, reason}
      end
    rescue
      e -> {:error, {:fetch_error, Exception.message(e)}}
    end
  end

  defp wait_for_rate_limit(domain) do
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

  defp extract_findings(%{finding: finding}) when is_map(finding) do
    [finding]
  end

  defp extract_findings(%{paper: paper, source: _source}) when is_struct(paper) do
    alias Brain.Knowledge.Academic.Paper

    case Paper.to_finding(paper) do
      nil -> []
      finding -> [finding]
    end
  end

  defp extract_findings(%{content: content, source: source}) when is_binary(content) do
    clean_content = clean_content(content)

    if String.length(clean_content) < 50 do
      Logger.debug("Content too short after cleaning", url: source.url)
      []
    else
      case Pipeline.process(clean_content, skip_entity_extraction: false) do
        %{analyses: analyses} ->
          # Comprehension gate: assess whether we understand this text
          {profile, analyses} = assess_comprehension(analyses)

          if profile.learnable do
            confidence_multiplier =
              if profile.verdict == :partial, do: profile.composite_score, else: 1.0

            analyses
            |> Enum.filter(&is_factual_claim?/1)
            |> Enum.map(
              &build_finding(&1, source, clean_content,
                comprehension_profile_id: profile.id,
                confidence_multiplier: confidence_multiplier
              )
            )
            |> Enum.reject(&is_nil/1)
          else
            gap_descriptions =
              profile.gaps |> Enum.map(& &1.description) |> Enum.join("; ")

            Logger.debug(
              "Comprehension gate blocked content: verdict=#{profile.verdict}, gaps=[#{gap_descriptions}]",
              url: source.url
            )

            []
          end

        _ ->
          []
      end
    end
  rescue
    e ->
      Logger.warning("Failed to extract findings", error: Exception.message(e), url: source.url)

      []
  end

  defp extract_findings(_) do
    []
  end

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
          case HtmlProcessor.html_to_text(content, min_length: 20) do
            {:ok, text} -> text
            {:error, _} -> content
          end
      end
    else
      content
    end
  end

  defp clean_content(content) do
    to_string(content)
  end

  defp is_factual_claim?(analysis) do
    case analysis do
      %{speech_act: %{category: :assertive}} -> true
      %{speech_act: %{category: :commissive}} -> false
      %{speech_act: %{category: :directive}} -> false
      %{speech_act: %{category: :expressive}} -> false
      _ -> false
    end
  end

  defp build_finding(analysis, source, raw_content, opts) do
    entities = Map.get(analysis, :entities, [])

    primary_entity =
      entities
      |> Enum.max_by(fn e -> Map.get(e, :confidence, 0) end, fn -> nil end)

    claim = Map.get(analysis, :text, "")

    if primary_entity && String.length(claim) > 10 do
      entity_value = Map.get(primary_entity, :value) || Map.get(primary_entity, "value")
      entity_type = Map.get(primary_entity, :entity_type) || Map.get(primary_entity, "type")

      base_confidence = Map.get(analysis, :confidence, 0.5)
      multiplier = Keyword.get(opts, :confidence_multiplier, 1.0)
      profile_id = Keyword.get(opts, :comprehension_profile_id)

      Finding.new(claim, entity_value || "unknown", source,
        entity_type: entity_type,
        raw_context: extract_context(raw_content, claim),
        confidence: base_confidence * multiplier,
        comprehension_profile_id: profile_id
      )
    else
      nil
    end
  end

  defp assess_comprehension(analyses) do
    if ComprehensionAssessor.ready?() do
      profile = ComprehensionAssessor.assess(analyses)
      {profile, analyses}
    else
      # Fallback: pass everything through when assessor is unavailable
      fallback_profile = %Brain.Analysis.ComprehensionAssessor.ComprehensionProfile{
        id: "fallback",
        composite_score: 1.0,
        verdict: :comprehended,
        learnable: true,
        gaps: [],
        dimensions: %{}
      }

      {fallback_profile, analyses}
    end
  end

  defp extract_context(content, claim) when is_binary(content) and is_binary(claim) do
    case :binary.match(content, claim) do
      {start, len} ->
        context_start = max(0, start - 100)
        context_end = min(String.length(content), start + len + 100)

        String.slice(content, context_start, context_end - context_start)

      :nomatch ->
        String.slice(claim, 0, 200)
    end
  end

  defp extract_context(_, claim) do
    claim
  end

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
