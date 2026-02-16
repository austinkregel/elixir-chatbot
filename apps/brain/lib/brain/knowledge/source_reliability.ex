defmodule Brain.Knowledge.SourceReliability do
  @moduledoc "GenServer managing source reputation and reliability scores.\n\nTracks per-domain reliability scores, bias ratings, and trust tiers.\nBootstraps from MediaBiasFactCheck/AllSides data and learns from\nadmin feedback over time.\n\n## Features\n\n- Bootstrap from curated source reliability data\n- Per-domain reliability scoring (0.0-1.0)\n- Bias rating (left/center/right spectrum)\n- Trust tiers (verified, neutral, untrusted, blocked)\n- Admin feedback learning (approvals/rejections adjust scores)\n- Persistence of learned adjustments\n\n## Example\n\n    {:ok, profile} = SourceReliability.lookup(\"wikipedia.org\")\n    # => %SourceProfile{domain: \"wikipedia.org\", trust_tier: :verified, ...}\n\n    SourceReliability.record_feedback(\"sketchy-news.com\", :rejected)\n    # Lowers reliability score for that domain\n"

  alias Brain.Knowledge.Types
  use GenServer
  require Logger

  alias Types.{SourceInfo, SourceProfile}

  defp bootstrap_file do
    Brain.priv_path("knowledge/source_reliability.json")
  end


  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Looks up reliability information for a URL or domain.\n\nReturns {:ok, SourceInfo} with reliability data, or a default profile\nfor unknown domains.\n"
  @spec lookup(String.t()) :: {:ok, SourceInfo.t()}
  def lookup(url_or_domain) when is_binary(url_or_domain) do
    GenServer.call(__MODULE__, {:lookup, url_or_domain})
  end

  @doc "Gets the full profile for a domain including historical data.\n"
  @spec get_profile(String.t()) :: {:ok, SourceProfile.t()} | {:error, :not_found}
  def get_profile(domain) when is_binary(domain) do
    GenServer.call(__MODULE__, {:get_profile, domain})
  end

  @doc "Records admin feedback (approval or rejection) for a domain.\n\nThis adjusts the domain's reliability score over time.\n\n## Options\n  - :candidate_id - ID of the candidate that was reviewed\n  - :notes - Optional notes from the reviewer\n"
  @spec record_feedback(String.t(), :approved | :rejected, keyword()) :: :ok
  def record_feedback(domain, decision, opts \\ [])
      when is_binary(domain) and decision in [:approved, :rejected] do
    GenServer.cast(__MODULE__, {:record_feedback, domain, decision, opts})
  end

  @doc "Gets statistics about the source reliability index.\n"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc "Gets statistics about the source reliability index.\nDeprecated: Use `stats/0` instead.\n"
  @spec get_stats() :: map()
  def get_stats do
    stats()
  end

  @doc "Reloads the bootstrap data from disk.\n"
  @spec reload_bootstrap() :: :ok
  def reload_bootstrap do
    GenServer.call(__MODULE__, :reload_bootstrap)
  end

  @doc "Persists learned adjustments to disk.\n"
  @spec persist() :: :ok | {:error, term()}
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc "Checks if a domain is blocked.\n"
  @spec blocked?(String.t()) :: boolean()
  def blocked?(domain) when is_binary(domain) do
    GenServer.call(__MODULE__, {:blocked?, domain})
  end

  @doc "Checks if the service is ready.\n"
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @impl true
  def init(_opts) do
    state = %{
      sources: %{},
      blocked_domains: MapSet.new(),
      last_updated: nil
    }

    state = load_bootstrap_data(state)
    state = load_from_atlas(state)

    Logger.info("SourceReliability initialized",
      sources: map_size(state.sources),
      blocked: MapSet.size(state.blocked_domains)
    )

    {:ok, state}
  end

  @impl true
  def handle_call({:lookup, url_or_domain}, _from, state) do
    domain = normalize_domain(url_or_domain)
    profile = get_or_create_profile(state.sources, domain)

    source_info = %SourceInfo{
      url: url_or_domain,
      domain: domain,
      reliability_score: SourceProfile.calculate_reliability(profile),
      bias_rating: profile.bias_rating,
      trust_tier: get_effective_trust_tier(profile, state.blocked_domains)
    }

    {:reply, {:ok, source_info}, state}
  end

  @impl true
  def handle_call({:get_profile, domain}, _from, state) do
    normalized = normalize_domain(domain)

    case Map.get(state.sources, normalized) do
      nil -> {:reply, {:error, :not_found}, state}
      profile -> {:reply, {:ok, profile}, state}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      total_sources: map_size(state.sources),
      blocked_domains: MapSet.size(state.blocked_domains),
      by_trust_tier: count_by_trust_tier(state.sources),
      by_bias_rating: count_by_bias_rating(state.sources),
      last_updated: state.last_updated
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:reload_bootstrap, _from, state) do
    new_state = load_bootstrap_data(state)
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:blocked?, domain}, _from, state) do
    normalized = normalize_domain(domain)
    blocked = MapSet.member?(state.blocked_domains, normalized)
    {:reply, blocked, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_cast({:record_feedback, domain, decision, opts}, state) do
    normalized = normalize_domain(domain)
    profile = get_or_create_profile(state.sources, normalized)
    updated_profile = SourceProfile.record_decision(profile, decision, opts)

    new_sources = Map.put(state.sources, normalized, updated_profile)
    new_state = %{state | sources: new_sources, last_updated: DateTime.utc_now()}

    Brain.AtlasIntegration.persist_source_reliability(normalized, updated_profile)

    Logger.debug("Recorded feedback for source",
      domain: normalized,
      decision: decision,
      new_reliability: SourceProfile.calculate_reliability(updated_profile)
    )

    {:noreply, new_state}
  end

  defp normalize_domain(url_or_domain) do
    SourceInfo.extract_domain(url_or_domain)
  end

  defp get_or_create_profile(sources, domain) do
    case Map.get(sources, domain) do
      nil ->
        SourceProfile.new(domain)

      profile ->
        profile
    end
  end

  defp get_effective_trust_tier(profile, blocked_domains) do
    if MapSet.member?(blocked_domains, profile.domain) do
      :blocked
    else
      profile.trust_tier
    end
  end

  defp load_bootstrap_data(state) do
    bootstrap_path = bootstrap_file()

    if File.exists?(bootstrap_path) do
      case File.read(bootstrap_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} ->
              sources = parse_bootstrap_sources(data)
              blocked = parse_blocked_domains(data)

              Logger.info("Loaded bootstrap data",
                sources: map_size(sources),
                blocked: MapSet.size(blocked)
              )

              %{state | sources: Map.merge(state.sources, sources), blocked_domains: blocked}

            {:error, reason} ->
              Logger.warning("Failed to parse bootstrap file", reason: inspect(reason))
              state
          end

        {:error, reason} ->
          Logger.warning("Failed to read bootstrap file", reason: inspect(reason))
          state
      end
    else
      Logger.info("No bootstrap file found, starting with empty source index")
      state
    end
  end

  defp parse_bootstrap_sources(data) do
    data
    |> Map.get("sources", %{})
    |> Enum.map(fn {domain, info} ->
      profile =
        SourceProfile.new(domain,
          factual_accuracy: Map.get(info, "factual_accuracy", 0.5),
          bias_rating: parse_bias_rating(Map.get(info, "bias_rating", "unknown")),
          trust_tier: parse_trust_tier(Map.get(info, "trust_tier", "neutral")),
          notes: Map.get(info, "notes")
        )

      {String.downcase(domain), profile}
    end)
    |> Map.new()
  end

  defp parse_blocked_domains(data) do
    data
    |> Map.get("blocked_domains", [])
    |> Enum.map(&String.downcase/1)
    |> MapSet.new()
  end

  defp parse_bias_rating(rating) when is_binary(rating) do
    case String.downcase(rating) do
      "left" -> :left
      "center_left" -> :center_left
      "center-left" -> :center_left
      "center" -> :center
      "center_right" -> :center_right
      "center-right" -> :center_right
      "right" -> :right
      _ -> :unknown
    end
  end

  defp parse_bias_rating(_) do
    :unknown
  end

  defp parse_trust_tier(tier) when is_binary(tier) do
    case String.downcase(tier) do
      "verified" -> :verified
      "neutral" -> :neutral
      "untrusted" -> :untrusted
      "blocked" -> :blocked
      _ -> :neutral
    end
  end

  defp parse_trust_tier(_) do
    :neutral
  end

  defp load_from_atlas(state) do
    case Brain.AtlasIntegration.load_source_reliability() do
      {:ok, atlas_sources} when atlas_sources != %{} ->
        merged_sources = Map.merge(state.sources, atlas_sources)
        Logger.info("Loaded learned source data from Atlas", sources: map_size(atlas_sources))
        %{state | sources: merged_sources}

      _ ->
        Logger.debug("No learned source data in Atlas")
        state
    end
  rescue
    e ->
      Logger.warning("Failed to load source reliability from Atlas: #{inspect(e)}")
      state
  end

  defp count_by_trust_tier(sources) do
    sources
    |> Map.values()
    |> Enum.group_by(& &1.trust_tier)
    |> Enum.map(fn {tier, profiles} -> {tier, length(profiles)} end)
    |> Map.new()
  end

  defp count_by_bias_rating(sources) do
    sources
    |> Map.values()
    |> Enum.group_by(& &1.bias_rating)
    |> Enum.map(fn {rating, profiles} -> {rating, length(profiles)} end)
    |> Map.new()
  end
end
