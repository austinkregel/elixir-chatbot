defmodule Brain.Epistemic.SourceAuthority do
  @moduledoc """
  GenServer managing source authority profiles and credibility tracking.

  Source authority describes WHO provided a belief and how much trust
  that carries. Each authority type (e.g., :mentor, :academic_expert,
  :stranger) has a trust profile that defines initial confidence,
  decay rate multiplier, and JTMS node behaviour.

  Trust profiles are bootstrapped from a JSON file and credibility
  is tracked over time based on how beliefs from each authority
  fare (confirmed vs contradicted).

  ## Example

      SourceAuthority.effective_confidence(:mentor)
      # => 0.85 (with fresh credibility)

      SourceAuthority.record_outcome(:mentor, :contradicted)
      # Lowers mentor credibility for future beliefs

      SourceAuthority.list_profiles()
      # => [%{key: :mentor, profile: %{...}, credibility: 0.95, ...}, ...]
  """

  use GenServer
  require Logger

  defp bootstrap_file do
    Brain.priv_path("knowledge/source_authority_profiles.json")
  end

  defp default_persistence_file do
    Brain.priv_path("data/source_authority_learned.term")
  end

  defp persistence_file do
    Application.get_env(:brain, :source_authority_path, default_persistence_file())
  end

  # ────────────────────────────────────────────────────────────────
  # Public API
  # ────────────────────────────────────────────────────────────────

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Returns the trust profile map for the given authority key."
  @spec get_profile(atom()) :: map() | nil
  def get_profile(authority_key) when is_atom(authority_key) do
    GenServer.call(__MODULE__, {:get_profile, authority_key})
  end

  @doc """
  Returns the effective confidence for a new belief from this authority.

  effective_confidence = profile.initial_confidence * current_credibility
  """
  @spec effective_confidence(atom()) :: float()
  def effective_confidence(authority_key) when is_atom(authority_key) do
    GenServer.call(__MODULE__, {:effective_confidence, authority_key})
  end

  @doc """
  Returns the effective decay rate for beliefs from this authority.

  effective_decay_rate = base_rate * profile.decay_rate_multiplier
  """
  @spec effective_decay_rate(atom(), float()) :: float()
  def effective_decay_rate(authority_key, base_rate)
      when is_atom(authority_key) and is_float(base_rate) do
    GenServer.call(__MODULE__, {:effective_decay_rate, authority_key, base_rate})
  end

  @doc """
  Records an outcome for an authority (confirmed or contradicted).

  This adjusts the running credibility score for future beliefs.
  """
  @spec record_outcome(atom(), :confirmed | :contradicted | :added) :: :ok
  def record_outcome(authority_key, outcome)
      when is_atom(authority_key) and outcome in [:confirmed, :contradicted, :added] do
    GenServer.cast(__MODULE__, {:record_outcome, authority_key, outcome})
  end

  @doc "Returns the current credibility score for an authority."
  @spec get_credibility(atom()) :: float()
  def get_credibility(authority_key) when is_atom(authority_key) do
    GenServer.call(__MODULE__, {:get_credibility, authority_key})
  end

  @doc "Returns all profiles with their current credibility data, for UI display."
  @spec list_profiles() :: [map()]
  def list_profiles do
    GenServer.call(__MODULE__, :list_profiles)
  end

  @doc "Persists learned credibility data to disk."
  @spec persist() :: :ok | {:error, term()}
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc "Clears all learned credibility data (useful for testing)."
  @spec clear() :: :ok
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  @doc "Checks if the service is ready."
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # ────────────────────────────────────────────────────────────────
  # GenServer callbacks
  # ────────────────────────────────────────────────────────────────

  @impl true
  def init(_opts) do
    state = %{
      profiles: %{},
      tracking: %{},
      last_updated: nil
    }

    state = load_bootstrap_data(state)
    state = load_learned_data(state)

    Logger.info("SourceAuthority initialized",
      profiles: map_size(state.profiles)
    )

    {:ok, state}
  end

  @impl true
  def handle_call({:get_profile, key}, _from, state) do
    {:reply, Map.get(state.profiles, key), state}
  end

  @impl true
  def handle_call({:effective_confidence, key}, _from, state) do
    case Map.get(state.profiles, key) do
      nil ->
        {:reply, 0.5, state}

      profile ->
        credibility = get_tracking_credibility(state.tracking, key, profile)
        {:reply, profile.initial_confidence * credibility, state}
    end
  end

  @impl true
  def handle_call({:effective_decay_rate, key, base_rate}, _from, state) do
    case Map.get(state.profiles, key) do
      nil ->
        {:reply, base_rate, state}

      profile ->
        {:reply, base_rate * profile.decay_rate_multiplier, state}
    end
  end

  @impl true
  def handle_call({:get_credibility, key}, _from, state) do
    case Map.get(state.profiles, key) do
      nil ->
        {:reply, 1.0, state}

      profile ->
        credibility = get_tracking_credibility(state.tracking, key, profile)
        {:reply, credibility, state}
    end
  end

  @impl true
  def handle_call(:list_profiles, _from, state) do
    profiles =
      state.profiles
      |> Enum.map(fn {key, profile} ->
        tracking = Map.get(state.tracking, key, %{})
        credibility = get_tracking_credibility(state.tracking, key, profile)

        %{
          key: key,
          profile: profile,
          credibility: credibility,
          confirmed_count: Map.get(tracking, :confirmed_count, 0),
          contradicted_count: Map.get(tracking, :contradicted_count, 0),
          total_added: Map.get(tracking, :total_added, 0),
          last_updated: Map.get(tracking, :last_updated)
        }
      end)
      |> Enum.sort_by(fn p -> -p.profile.initial_confidence end)

    {:reply, profiles, state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    result = persist_learned_data(state)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    new_state = %{state | tracking: %{}, last_updated: nil}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_cast({:record_outcome, key, outcome}, state) do
    tracking = Map.get(state.tracking, key, default_tracking())

    tracking =
      case outcome do
        :confirmed ->
          %{tracking | confirmed_count: tracking.confirmed_count + 1, last_updated: DateTime.utc_now()}

        :contradicted ->
          %{tracking | contradicted_count: tracking.contradicted_count + 1, last_updated: DateTime.utc_now()}

        :added ->
          %{tracking | total_added: tracking.total_added + 1, last_updated: DateTime.utc_now()}
      end

    new_tracking = Map.put(state.tracking, key, tracking)
    new_state = %{state | tracking: new_tracking, last_updated: DateTime.utc_now()}

    # Auto-persist after significant changes
    if rem(tracking.confirmed_count + tracking.contradicted_count, 5) == 0 do
      persist_learned_data(new_state)
    end

    {:noreply, new_state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # ────────────────────────────────────────────────────────────────
  # Private helpers
  # ────────────────────────────────────────────────────────────────

  defp default_tracking do
    %{
      confirmed_count: 0,
      contradicted_count: 0,
      total_added: 0,
      last_updated: nil
    }
  end

  defp get_tracking_credibility(tracking, key, profile) do
    case Map.get(tracking, key) do
      nil ->
        1.0

      t ->
        calculate_credibility(
          t.confirmed_count,
          t.contradicted_count,
          profile.credibility_floor
        )
    end
  end

  defp calculate_credibility(confirmed, contradicted, floor) do
    total = confirmed + contradicted

    if total == 0 do
      1.0
    else
      max(confirmed / total, floor)
    end
  end

  defp load_bootstrap_data(state) do
    bootstrap_path = bootstrap_file()

    if File.exists?(bootstrap_path) do
      case File.read(bootstrap_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} ->
              profiles = parse_profiles(data)

              Logger.info("Loaded source authority profiles",
                profiles: map_size(profiles)
              )

              %{state | profiles: profiles}

            {:error, reason} ->
              Logger.warning("Failed to parse source authority profiles",
                reason: inspect(reason)
              )

              state
          end

        {:error, reason} ->
          Logger.warning("Failed to read source authority profiles",
            reason: inspect(reason)
          )

          state
      end
    else
      Logger.info("No source authority profiles file found, starting empty")
      state
    end
  end

  defp parse_profiles(data) do
    data
    |> Map.get("profiles", %{})
    |> Enum.map(fn {key, info} ->
      atom_key =
        try do
          String.to_existing_atom(key)
        rescue
          _ -> String.to_atom(key)
        end

      profile = %{
        label: Map.get(info, "label", key),
        description: Map.get(info, "description", ""),
        initial_confidence: Map.get(info, "initial_confidence", 0.5),
        decay_rate_multiplier: Map.get(info, "decay_rate_multiplier", 1.0),
        jtms_node_type: parse_jtms_node_type(Map.get(info, "jtms_node_type", "assumption")),
        credibility_floor: Map.get(info, "credibility_floor", 0.0),
        category: Map.get(info, "category", "unknown")
      }

      {atom_key, profile}
    end)
    |> Map.new()
  end

  defp parse_jtms_node_type("premise"), do: :premise
  defp parse_jtms_node_type("assumption"), do: :assumption
  defp parse_jtms_node_type(_), do: :assumption

  defp load_learned_data(state) do
    path = persistence_file()
    full_path = Path.join(File.cwd!(), path)

    if File.exists?(full_path) do
      case File.read(full_path) do
        {:ok, binary} ->
          try do
            data = :erlang.binary_to_term(binary)
            tracking = Map.get(data, :tracking, %{})

            Logger.info("Loaded learned authority data",
              authorities: map_size(tracking)
            )

            %{state | tracking: tracking, last_updated: Map.get(data, :last_updated)}
          rescue
            e ->
              Logger.warning("Failed to parse learned authority data",
                error: inspect(e)
              )

              state
          end

        {:error, reason} ->
          Logger.warning("Failed to read learned authority data",
            reason: inspect(reason)
          )

          state
      end
    else
      state
    end
  end

  defp persist_learned_data(state) do
    path = persistence_file()
    full_path = Path.join(File.cwd!(), path)

    # Only persist authorities that have data
    learned_tracking =
      state.tracking
      |> Enum.filter(fn {_key, t} ->
        t.confirmed_count > 0 or t.contradicted_count > 0 or t.total_added > 0
      end)
      |> Map.new()

    data = %{
      tracking: learned_tracking,
      last_updated: state.last_updated,
      version: 1
    }

    full_path |> Path.dirname() |> File.mkdir_p!()

    case File.write(full_path, :erlang.term_to_binary(data)) do
      :ok ->
        Logger.debug("Persisted source authority data",
          authorities: map_size(learned_tracking)
        )

        :ok

      {:error, reason} ->
        Logger.error("Failed to persist source authority data",
          reason: inspect(reason)
        )

        {:error, reason}
    end
  end
end
