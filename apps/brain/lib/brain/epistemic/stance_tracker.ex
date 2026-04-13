defmodule Brain.Epistemic.StanceTracker do
  @moduledoc """
  Tracks stance drift in conversations for bidirectional opinion dynamics.

  Based on Jiang et al. "Beyond One-Way Influence" (CHI 2026): LLM responses
  shift significantly toward user opinions over multi-turn conversations, while
  user opinions shift only slightly.

  This module:
  1. Tracks the system's stance on topics across conversation turns
  2. Detects when the system's stance is drifting toward the user's position
  3. Flags excessive drift for review via the "persuasion resistance" parameter
  4. Exposes metrics for dashboard monitoring

  Stances are stored in the existing Belief system using the `metadata` field
  with `%{stance: true, topic: topic, position: position}`.
  """

  use GenServer
  require Logger

  alias Brain.Epistemic.BeliefStore

  @default_drift_threshold 0.3
  @max_drift_per_conversation 0.5

  defstruct [
    :conversations,
    :drift_threshold,
    :max_drift,
    :stats
  ]

  # --- Public API ---

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Record a stance observation for a topic in a conversation.

  ## Parameters
    - `conversation_id` - Current conversation identifier
    - `topic` - The topic being discussed
    - `position` - A float in [-1.0, 1.0] representing the stance
      (-1 = strongly against, 0 = neutral, 1 = strongly for)
    - `source` - `:system` or `:user`
  """
  def record_stance(conversation_id, topic, position, source, name \\ __MODULE__) do
    GenServer.cast(name, {:record_stance, conversation_id, topic, position, source})
  end

  @doc """
  Check if the system's stance on a topic has drifted beyond the threshold.

  Returns `{:ok, drift_info}` or `{:ok, :no_drift}`.
  """
  def check_drift(conversation_id, topic, name \\ __MODULE__) do
    try do
      GenServer.call(name, {:check_drift, conversation_id, topic}, 5_000)
    catch
      :exit, _ -> {:error, :not_ready}
    end
  end

  @doc """
  Get all stance drift metrics for monitoring.
  """
  def stats(name \\ __MODULE__) do
    try do
      GenServer.call(name, :stats, 5_000)
    catch
      :exit, _ -> {:error, :not_ready}
    end
  end

  @doc """
  Get the drift history for a specific conversation.
  """
  def conversation_stances(conversation_id, name \\ __MODULE__) do
    try do
      GenServer.call(name, {:conversation_stances, conversation_id}, 5_000)
    catch
      :exit, _ -> {:error, :not_ready}
    end
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(opts) do
    state = %__MODULE__{
      conversations: %{},
      drift_threshold: Keyword.get(opts, :drift_threshold, @default_drift_threshold),
      max_drift: Keyword.get(opts, :max_drift, @max_drift_per_conversation),
      stats: %{
        total_observations: 0,
        drift_warnings: 0,
        conversations_tracked: 0
      }
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  def handle_call({:check_drift, conversation_id, topic}, _from, state) do
    case get_in(state.conversations, [conversation_id, topic]) do
      nil ->
        {:reply, {:ok, :no_drift}, state}

      observations ->
        system_obs = Enum.filter(observations, &(&1.source == :system))

        if length(system_obs) < 2 do
          {:reply, {:ok, :no_drift}, state}
        else
          initial = hd(system_obs).position
          current = List.last(system_obs).position
          drift = current - initial

          user_obs = Enum.filter(observations, &(&1.source == :user))
          user_direction = if user_obs != [] do
            avg_user = Enum.sum(Enum.map(user_obs, & &1.position)) / length(user_obs)
            if avg_user > initial, do: :toward_user, else: :away_from_user
          else
            :unknown
          end

          drift_info = %{
            initial_position: initial,
            current_position: current,
            absolute_drift: abs(drift),
            direction: user_direction,
            exceeds_threshold: abs(drift) > state.drift_threshold,
            num_observations: length(system_obs)
          }

          {:reply, {:ok, drift_info}, state}
        end
    end
  end

  def handle_call(:stats, _from, state) do
    {:reply, {:ok, state.stats}, state}
  end

  def handle_call({:conversation_stances, conversation_id}, _from, state) do
    stances = Map.get(state.conversations, conversation_id, %{})
    {:reply, {:ok, stances}, state}
  end

  @impl true
  def handle_cast({:record_stance, conversation_id, topic, position, source}, state) do
    clamped_position = clamp(position, -1.0, 1.0)

    observation = %{
      position: clamped_position,
      source: source,
      timestamp: DateTime.utc_now()
    }

    conversations = state.conversations
    |> Map.update(conversation_id, %{topic => [observation]}, fn conv ->
      Map.update(conv, topic, [observation], &(&1 ++ [observation]))
    end)

    persist_stance_to_belief_store(conversation_id, topic, clamped_position, source)

    new_stats = %{state.stats |
      total_observations: state.stats.total_observations + 1,
      conversations_tracked: map_size(conversations)
    }

    new_stats = maybe_warn_drift(conversations, conversation_id, topic, state, new_stats)

    {:noreply, %{state | conversations: conversations, stats: new_stats}}
  end

  # --- Private ---

  defp maybe_warn_drift(conversations, conversation_id, topic, state, stats) do
    case get_in(conversations, [conversation_id, topic]) do
      nil -> stats
      observations ->
        system_obs = Enum.filter(observations, &(&1.source == :system))

        if length(system_obs) >= 2 do
          initial = hd(system_obs).position
          current = List.last(system_obs).position
          drift = abs(current - initial)

          if drift > state.drift_threshold do
            Logger.warning(
              "Stance drift detected on topic '#{topic}' in conversation #{conversation_id}: " <>
              "#{Float.round(drift, 3)} exceeds threshold #{state.drift_threshold}"
            )
            :telemetry.execute([:brain, :epistemic, :stance_drift], %{drift: drift}, %{
              conversation_id: conversation_id,
              topic: topic
            })
            %{stats | drift_warnings: stats.drift_warnings + 1}
          else
            stats
          end
        else
          stats
        end
    end
  end

  defp persist_stance_to_belief_store(conversation_id, topic, position, source) do
    if BeliefStore.ready?() do
      subject = if source == :system, do: :self, else: :user

      BeliefStore.add_belief(
        subject,
        :stance_on,
        topic,
        source: :inferred,
        confidence: 0.7,
        metadata: %{
          stance: true,
          topic: topic,
          position: position,
          conversation_id: conversation_id,
          stance_source: source
        }
      )
    end
  rescue
    _ -> :ok
  end

  defp clamp(value, min_val, max_val) do
    value
    |> max(min_val)
    |> min(max_val)
  end
end
