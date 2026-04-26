defmodule Brain.Knowledge.LearningTriggers do
  @moduledoc """
  GenServer that monitors conversation signals and auto-triggers
  LearningCenter research sessions.

  Triggers:
  - **Novelty trigger:** When 3+ novel inputs on the same domain arrive within 24h
  - **Uncertainty trigger:** When a concluded investigation has low lattice margin
    or high entropy (subscribes to `learning:investigation`)

  Subscribes to PubSub `learning:novel_input` and `learning:investigation`.
  Rate-limited with per-domain and global daily caps.
  """

  use GenServer
  require Logger

  alias Brain.Lattice
  alias Brain.Knowledge.LearningCenter
  alias Brain.Knowledge.Types.ResearchGoal

  @max_sessions_per_day_per_domain 2
  @max_sessions_per_day_total 5
  @novelty_cluster_threshold 3
  @novelty_window_ms 24 * 60 * 60 * 1000
  @cleanup_interval_ms 60 * 60 * 1000
  @uncertainty_margin_threshold 0.12
  @uncertainty_entropy_threshold 0.92

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Returns current trigger stats."
  def stats(name \\ __MODULE__) do
    GenServer.call(name, :stats)
  end

  @doc "Checks if the GenServer is ready."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @impl true
  def init(_opts) do
    if Process.whereis(Brain.PubSub) do
      Phoenix.PubSub.subscribe(Brain.PubSub, "learning:novel_input")
      Phoenix.PubSub.subscribe(Brain.PubSub, "learning:investigation")
    end

    Process.send_after(self(), :cleanup, @cleanup_interval_ms)

    Logger.info("LearningTriggers started")

    {:ok,
     %{
       novel_inputs: %{},
       sessions_per_domain: %{},
       sessions_total_today: 0,
       last_reset_date: Date.utc_today(),
       triggered_sessions: []
     }}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      novel_input_domains: Map.keys(state.novel_inputs),
      novel_input_counts:
        Map.new(state.novel_inputs, fn {domain, inputs} -> {domain, length(inputs)} end),
      sessions_per_domain: state.sessions_per_domain,
      sessions_total_today: state.sessions_total_today,
      max_sessions_per_day_per_domain: @max_sessions_per_day_per_domain,
      max_sessions_per_day_total: @max_sessions_per_day_total,
      triggered_sessions: length(state.triggered_sessions)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_info({:novel_input, text, score, domain, entities}, state) do
    accumulate_input(state, text, score, domain, entities)
  end

  @impl true
  def handle_info({:novel_input, text, score, domain}, state) do
    accumulate_input(state, text, score, domain, [])
  end

  @impl true
  def handle_info(
        {:investigation_concluded, session_id, _inv_id, %Lattice{} = lat, dominant_gap,
         domain},
        state
      ) do
    state = maybe_reset_daily_counter(state)
    dkey = domain_key(domain)

    if session_cap_ok?(state, dkey) and uncertain?(lat) do
      topic = "Follow-up: resolve hypothesis uncertainty (#{dkey})"

      goal =
        ResearchGoal.new(topic,
          questions: ["What additional evidence would reduce uncertainty?"],
          source_gaps: [dominant_gap],
          priority: :high
        )

      case LearningCenter.spawn_followup_goal(session_id, goal) do
        :ok ->
          Logger.info("LearningTriggers: uncertainty follow-up goal queued",
            session_id: session_id,
            margin: Lattice.margin(lat),
            entropy: Lattice.entropy(lat)
          )

          if Process.whereis(Brain.PubSub) do
            Phoenix.PubSub.broadcast(
              Brain.PubSub,
              "learning:uncertainty",
              {:uncertainty_followup, session_id, dominant_gap}
            )
          end

          {:noreply, consume_session_slot(state, dkey)}

        _ ->
          {:noreply, state}
      end
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info(:cleanup, state) do
    state = cleanup_old_inputs(state)
    Process.send_after(self(), :cleanup, @cleanup_interval_ms)
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  defp uncertain?(%Lattice{} = lat) do
    Lattice.margin(lat) < @uncertainty_margin_threshold or
      Lattice.entropy(lat) > @uncertainty_entropy_threshold
  end

  defp accumulate_input(state, text, score, domain, entities) do
    state = maybe_reset_daily_counter(state)

    entry = %{
      text: text,
      score: score,
      entities: entities || [],
      timestamp: System.system_time(:millisecond)
    }

    domain_inputs = Map.get(state.novel_inputs, domain, [])
    updated_inputs = [entry | domain_inputs]
    new_novel_inputs = Map.put(state.novel_inputs, domain, updated_inputs)

    state = %{state | novel_inputs: new_novel_inputs}
    state = maybe_trigger_session(state, domain)

    {:noreply, state}
  end

  defp maybe_trigger_session(state, domain) do
    dkey = domain_key(domain)

    if not session_cap_ok?(state, dkey) do
      state
    else
      domain_inputs = Map.get(state.novel_inputs, domain, [])
      now = System.system_time(:millisecond)
      cutoff = now - @novelty_window_ms

      recent = Enum.filter(domain_inputs, fn input -> input.timestamp >= cutoff end)

      if length(recent) >= @novelty_cluster_threshold do
        topic = build_topic_from_inputs(domain, recent)

        case trigger_learning_session(topic) do
          :ok ->
            Logger.info("LearningTriggers: auto-triggered session for domain #{inspect(domain)}",
              topic: topic,
              novel_input_count: length(recent)
            )

            new_novel_inputs = Map.delete(state.novel_inputs, domain)

            %{
              state
              | novel_inputs: new_novel_inputs,
                triggered_sessions: [
                  %{domain: domain, topic: topic, timestamp: now}
                  | state.triggered_sessions
                ]
            }
            |> consume_session_slot(dkey)

          :error ->
            state
        end
      else
        state
      end
    end
  end

  defp session_cap_ok?(state, domain_key) do
    total = Map.get(state, :sessions_total_today, 0)
    domain_count = Map.get(state.sessions_per_domain, domain_key, 0)
    total < @max_sessions_per_day_total and domain_count < @max_sessions_per_day_per_domain
  end

  defp consume_session_slot(state, domain_key) do
    total = Map.get(state, :sessions_total_today, 0)

    %{
      state
      | sessions_total_today: total + 1,
        sessions_per_domain:
          Map.update(state.sessions_per_domain, domain_key, 1, &(&1 + 1))
    }
  end

  defp trigger_learning_session(topic) do
    if Code.ensure_loaded?(LearningCenter) and
         function_exported?(LearningCenter, :start_session, 2) do
      case LearningCenter.start_session(topic, source: :auto_trigger) do
        {:ok, _} -> :ok
        _ -> :error
      end
    else
      :error
    end
  rescue
    _ -> :error
  end

  defp build_topic_from_inputs(domain, inputs) do
    entity_names = extract_entity_concepts(inputs)
    domain_str = to_string(domain)

    if entity_names != [] do
      Enum.join(entity_names, ", ")
    else
      best = Enum.max_by(inputs, & &1.score)
      extract_content_words(best.text, domain_str)
    end
  end

  defp extract_entity_concepts(inputs) do
    inputs
    |> Enum.flat_map(fn input -> Map.get(input, :entities, []) end)
    |> Enum.filter(fn entity ->
      confidence = Map.get(entity, :confidence, 0)
      confidence > 0.3
    end)
    |> Enum.map(fn entity ->
      Map.get(entity, :value) || Map.get(entity, "value") || ""
    end)
    |> Enum.reject(&(&1 == ""))
    |> Enum.uniq()
    |> Enum.take(3)
  end

  defp extract_content_words(text, domain_str) do
    alias Brain.ML.{Tokenizer, POSTagger}

    tokens = Tokenizer.tokenize_words(text)

    case POSTagger.load_model() do
      {:ok, model} ->
        tags = POSTagger.predict_tags(tokens, model)

        nouns =
          Enum.zip(tokens, tags)
          |> Enum.filter(fn {_token, tag} -> tag in ["NOUN", "PROPN"] end)
          |> Enum.map(fn {token, _tag} -> token end)

        if nouns != [] do
          Enum.join(nouns, " ")
        else
          "#{domain_str}: #{text}"
        end

      {:error, _} ->
        "#{domain_str}: #{text}"
    end
  end

  defp cleanup_old_inputs(state) do
    now = System.system_time(:millisecond)
    cutoff = now - @novelty_window_ms

    new_novel_inputs =
      state.novel_inputs
      |> Map.new(fn {domain, inputs} ->
        {domain, Enum.filter(inputs, fn input -> input.timestamp >= cutoff end)}
      end)
      |> Enum.reject(fn {_domain, inputs} -> inputs == [] end)
      |> Map.new()

    %{state | novel_inputs: new_novel_inputs}
  end

  defp maybe_reset_daily_counter(state) do
    today = Date.utc_today()

    if Date.compare(today, state.last_reset_date) != :eq do
      %{
        state
        | sessions_per_domain: %{},
          sessions_total_today: 0,
          last_reset_date: today
      }
    else
      state
    end
  end

  defp domain_key(domain) when is_binary(domain), do: domain
  defp domain_key(domain) when is_atom(domain), do: to_string(domain)
  defp domain_key(_), do: "unknown"
end
