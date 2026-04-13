defmodule Brain.Epistemic.BeliefStore do
  @moduledoc "GenServer managing beliefs with persistence.\n\nThe BeliefStore is the primary storage layer for the epistemic system.\nIt manages beliefs, tracks their relationships via the JTMS, and provides\nquery capabilities for the rest of the system.\n\nFeatures:\n- CRUD operations for beliefs\n- Querying by subject, predicate, user\n- Confidence updates with confirmation tracking\n- Persistence to disk\n- Integration with JTMS for justification tracking\n"

  alias Brain.Epistemic.Types
  use GenServer

  alias Types.{Belief, Config}
  alias Brain.Telemetry

  require Logger


  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Adds a new belief to the store.\n\nReturns {:ok, belief_id} on success.\n"
  def add_belief(%Belief{} = belief) do
    Telemetry.span(:belief_operation, %{operation: :add, subject: belief.subject}, fn ->
      if Config.enabled?() do
        GenServer.call(__MODULE__, {:add_belief, belief}, 5_000)
      else
        {:ok, belief.id}
      end
    end)
  end

  @doc "Creates and adds a belief from parameters.\n"
  def add_belief(subject, predicate, object, opts \\ []) do
    belief = Belief.new(subject, predicate, object, opts)
    add_belief(belief)
  end

  @doc "Retracts a belief by ID.\n\nThis marks the belief as retracted but keeps it for history.\nRelated justifications are updated.\n"
  def retract_belief(belief_id) do
    GenServer.call(__MODULE__, {:retract_belief, belief_id}, 5_000)
  end

  @doc "Gets a belief by ID.\n"
  def get_belief(belief_id) do
    GenServer.call(__MODULE__, {:get_belief, belief_id}, 5_000)
  end

  @doc "Queries beliefs by subject and/or predicate.\n\nOptions:\n- :subject - Filter by subject (:user, :world, :self, or string)\n- :predicate - Filter by predicate\n- :user_id - Filter by user_id\n- :min_confidence - Minimum confidence threshold\n- :source - Filter by source (:explicit, :inferred, etc.)\n"
  def query_beliefs(opts \\ []) do
    Telemetry.span(:belief_operation, %{operation: :query, filters: opts}, fn ->
      GenServer.call(__MODULE__, {:query_beliefs, opts}, 5_000)
    end)
  end

  @doc "Gets all beliefs about a specific user.\n"
  def get_beliefs_for_user(user_id) do
    query_beliefs(user_id: user_id)
  end

  @doc "Extract beliefs from an extracted event.\n\nCreates beliefs based on the event structure:\n- Actor performing an action suggests the actor wants/needs/likes the object\n- Imperative actions suggest user desires\n\n## Parameters\n  - event: An Event struct from EventExtractor\n  - user_id: The user ID to associate beliefs with (optional)\n\n## Examples\n\n    event = %Event{\n      action: %{lemma: \"want\", tense: :present},\n      actor: %{text: \"I\", type: \"pronoun\"},\n      object: %{text: \"coffee\", type: \"noun\"},\n      confidence: 0.85\n    }\n\n    extract_beliefs_from_event(event, \"user_123\")\n    # Creates belief: User wants coffee (confidence: 0.85)\n"
  def extract_beliefs_from_event(event, user_id \\ nil) do
    confidence = Map.get(event, :confidence, 0.0)

    if confidence >= 0.6 do
      do_extract_beliefs_from_event(event, user_id)
    else
      {:ok, []}
    end
  end

  defp do_extract_beliefs_from_event(event, user_id) do
    beliefs_created = []
    action = Map.get(event, :action, %{})
    actor = Map.get(event, :actor)
    object = Map.get(event, :object)
    confidence = Map.get(event, :confidence, 0.5)

    action_lemma = Map.get(action, :lemma, Map.get(action, :verb))
    tense = Map.get(action, :tense, :present)

    # Build source info with actor for attribution
    source_info = %{type: :event, actor: actor, tense: tense}

    beliefs_created =
      cond do
        action_lemma in ["want", "need", "desire", "wish", "like", "love", "prefer"] and
            object != nil ->
          object_text = Map.get(object, :text, "something")

          predicate =
            case action_lemma do
              "want" -> :wants
              "need" -> :needs
              "desire" -> :desires
              "wish" -> :wishes_for
              "like" -> :likes
              "love" -> :loves
              "prefer" -> :prefers
              _ -> :wants
            end

          case add_belief(:user, predicate, object_text,
                 source: source_info,
                 confidence: confidence * 0.9,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: predicate, object: object_text, confidence: confidence * 0.9})
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        action_lemma in ["ask", "wonder", "question", "inquire"] and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(:user, :interested_in, object_text,
                 source: :inferred,
                 confidence: confidence * 0.7,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: :interested_in, object: object_text, confidence: confidence * 0.7})
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        tense == :imperative and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(:user, :requests, "#{action_lemma} #{object_text}",
                 source: :inferred,
                 confidence: confidence * 0.8,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: :requests, object: "#{action_lemma} #{object_text}", confidence: confidence * 0.8})
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        action_lemma in ["know", "believe", "think", "understand"] and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(:user, :believes, object_text,
                 source: :inferred,
                 confidence: confidence * 0.8,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: :believes, object: object_text, confidence: confidence * 0.8})
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        true ->
          beliefs_created
      end

    {:ok, beliefs_created}
  end

  @doc "Extract beliefs from multiple events.\n\nProcesses a list of events and extracts beliefs from each.\n"
  def extract_beliefs_from_events(events, user_id \\ nil) when is_list(events) do
    results =
      Enum.map(events, fn event ->
        extract_beliefs_from_event(event, user_id)
      end)

    belief_ids =
      results
      |> Enum.flat_map(fn
        {:ok, ids} -> ids
        _ -> []
      end)

    {:ok, belief_ids}
  end

  @doc "Updates the confidence of a belief.\n\nOptions:\n- :confirm - If true, also updates last_confirmed timestamp\n"
  def update_confidence(belief_id, new_confidence, opts \\ []) do
    GenServer.call(__MODULE__, {:update_confidence, belief_id, new_confidence, opts}, 5_000)
  end

  @doc "Confirms a belief (updates last_confirmed and optionally boosts confidence).\n"
  def confirm_belief(belief_id, confidence_boost \\ 0.1) do
    GenServer.call(__MODULE__, {:confirm_belief, belief_id, confidence_boost}, 5_000)
  end

  @doc """
  Adds a belief with source authority context.

  The confidence and decay behaviour are determined by the authority
  profile loaded from SourceAuthority. This replaces the old
  `add_admin_belief/4` function.

  ## Parameters

    - subject: :user | :world | :self | String.t()
    - predicate: atom or string identifying the property
    - object: the value of the belief
    - authority_key: atom like :mentor, :academic_expert, :stranger
    - opts: additional keyword options (merged; authority defaults take precedence)
  """
  def add_belief_with_authority(subject, predicate, object, authority_key, opts \\ []) do
    alias Brain.Epistemic.SourceAuthority

    effective_conf =
      if SourceAuthority.ready?() do
        try do
          SourceAuthority.effective_confidence(authority_key)
        catch
          :exit, _ -> 0.5
        end
      else
        0.5
      end

    merged_opts =
      Keyword.merge(
        [
          source: :explicit,
          confidence: effective_conf,
          source_authority: authority_key,
          provenance: ["authority:#{authority_key}"]
        ],
        opts
      )

    result = add_belief(subject, predicate, object, merged_opts)

    if match?({:ok, _}, result) and SourceAuthority.ready?() do
      try do
        SourceAuthority.record_outcome(authority_key, :added)
      catch
        :exit, _ -> :ok
      end
    end

    result
  end

  @doc "Links a belief to a JTMS node.\n"
  def link_to_node(belief_id, node_id) do
    GenServer.call(__MODULE__, {:link_to_node, belief_id, node_id}, 5_000)
  end

  @doc "Gets store statistics.\n"
  def stats do
    GenServer.call(__MODULE__, :stats, 5_000)
  end

  @doc "Persists the store to disk.\n"
  def persist do
    GenServer.call(__MODULE__, :persist, 30_000)
  end

  @doc "Clears all beliefs (useful for testing).\n"
  def clear do
    GenServer.call(__MODULE__, :clear, 30_000)
  end

  @doc "Checks if the store is ready.\n"
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
      beliefs: %{},
      by_user: %{},
      by_subject: %{},
      by_predicate: %{},
      retracted: MapSet.new()
    }

    state = load_from_atlas(state)

    # Schedule confidence decay tick
    decay_interval = Config.get().decay_interval_ms
    if decay_interval > 0, do: Process.send_after(self(), :decay_tick, decay_interval)

    Logger.info("BeliefStore initialized", belief_count: map_size(state.beliefs))

    {:ok, state}
  end

  @impl true
  def handle_call({:add_belief, belief}, _from, state) do
    new_beliefs = Map.put(state.beliefs, belief.id, belief)
    new_by_user = add_to_index(state.by_user, belief.user_id, belief.id)
    new_by_subject = add_to_index(state.by_subject, belief.subject, belief.id)
    new_by_predicate = add_to_index(state.by_predicate, belief.predicate, belief.id)

    # Create JTMS node for justification tracking
    belief = maybe_create_jtms_node(belief)
    new_beliefs = Map.put(new_beliefs, belief.id, belief)

    new_state = %{
      state
      | beliefs: new_beliefs,
        by_user: new_by_user,
        by_subject: new_by_subject,
        by_predicate: new_by_predicate
    }

    Logger.debug("Belief added",
      id: belief.id,
      subject: belief.subject,
      predicate: belief.predicate
    )

    # Write-through to Atlas
    Brain.AtlasIntegration.persist_belief(belief)

    {:reply, {:ok, belief.id}, new_state}
  end

  @impl true
  def handle_call({:retract_belief, belief_id}, _from, state) do
    case Map.get(state.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        new_retracted = MapSet.put(state.retracted, belief_id)
        new_state = %{state | retracted: new_retracted}

        # Track credibility for the source authority (best-effort)
        if belief.source_authority do
          alias Brain.Epistemic.SourceAuthority

          if SourceAuthority.ready?() do
            try do
              SourceAuthority.record_outcome(belief.source_authority, :contradicted)
            catch
              :exit, _ -> :ok
            end
          end
        end

        Logger.debug("Belief retracted", id: belief_id)
        Brain.AtlasIntegration.retract_belief_in_atlas(belief_id)

        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:get_belief, belief_id}, _from, state) do
    case Map.get(state.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        if MapSet.member?(state.retracted, belief_id) do
          {:reply, {:error, :retracted}, state}
        else
          {:reply, {:ok, belief}, state}
        end
    end
  end

  @impl true
  def handle_call({:query_beliefs, opts}, _from, state) do
    beliefs =
      state.beliefs
      |> Map.values()
      |> Enum.reject(fn b -> MapSet.member?(state.retracted, b.id) end)
      |> apply_filters(opts)

    {:reply, {:ok, beliefs}, state}
  end

  @impl true
  def handle_call({:update_confidence, belief_id, new_confidence, opts}, _from, state) do
    case Map.get(state.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        confirm? = Keyword.get(opts, :confirm, false)
        updated = Belief.update_confidence(belief, new_confidence, confirm?)
        new_beliefs = Map.put(state.beliefs, belief_id, updated)
        new_state = %{state | beliefs: new_beliefs}

        Brain.AtlasIntegration.update_belief_confidence(
          belief_id,
          new_confidence,
          if(confirm?, do: updated.last_confirmed)
        )

        {:reply, {:ok, updated}, new_state}
    end
  end

  @impl true
  def handle_call({:confirm_belief, belief_id, confidence_boost}, _from, state) do
    case Map.get(state.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        new_confidence = min(belief.confidence + confidence_boost, 1.0)

        updated =
          belief
          |> Belief.update_confidence(new_confidence, true)

        new_beliefs = Map.put(state.beliefs, belief_id, updated)
        new_state = %{state | beliefs: new_beliefs}

        # Track credibility for the source authority (best-effort)
        if belief.source_authority do
          alias Brain.Epistemic.SourceAuthority

          if SourceAuthority.ready?() do
            try do
              SourceAuthority.record_outcome(belief.source_authority, :confirmed)
            catch
              :exit, _ -> :ok
            end
          end
        end

        {:reply, {:ok, updated}, new_state}
    end
  end

  @impl true
  def handle_call({:link_to_node, belief_id, node_id}, _from, state) do
    case Map.get(state.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        updated = %{belief | node_id: node_id}
        new_beliefs = Map.put(state.beliefs, belief_id, updated)
        new_state = %{state | beliefs: new_beliefs}

        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    active_count =
      state.beliefs
      |> Map.keys()
      |> Enum.reject(&MapSet.member?(state.retracted, &1))
      |> length()

    stats = %{
      total_beliefs: map_size(state.beliefs),
      active_beliefs: active_count,
      retracted_beliefs: MapSet.size(state.retracted),
      unique_users: map_size(state.by_user),
      unique_subjects: map_size(state.by_subject),
      unique_predicates: map_size(state.by_predicate)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    new_state = %{
      state
      | beliefs: %{},
        by_user: %{},
        by_subject: %{},
        by_predicate: %{},
        retracted: MapSet.new()
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_info(:decay_tick, state) do
    config = Config.get()
    new_state = apply_confidence_decay(state, config)

    # Re-schedule
    if config.decay_interval_ms > 0 do
      Process.send_after(self(), :decay_tick, config.decay_interval_ms)
    end

    {:noreply, new_state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  defp apply_confidence_decay(state, config) do
    now = DateTime.utc_now()
    min_age_seconds = div(config.decay_min_age_ms, 1000)

    {updated_beliefs, retracted_ids} =
      Enum.reduce(state.beliefs, {%{}, []}, fn {id, belief}, {acc, retracted} ->
        if belief.source in config.decay_exempt_sources or MapSet.member?(state.retracted, id) do
          {Map.put(acc, id, belief), retracted}
        else
          # Only decay if last_confirmed is old enough
          age_seconds =
            case Map.get(belief, :last_confirmed) do
              nil -> min_age_seconds + 1
              ts -> DateTime.diff(now, ts, :second)
            end

          if age_seconds >= min_age_seconds do
            # Authority-aware decay rate (falls back to base rate if SourceAuthority unavailable)
            rate =
              if belief.source_authority do
                alias Brain.Epistemic.SourceAuthority

                if SourceAuthority.ready?() do
                  try do
                    SourceAuthority.effective_decay_rate(
                      belief.source_authority,
                      config.decay_rate
                    )
                  catch
                    :exit, _ -> config.decay_rate
                  end
                else
                  config.decay_rate
                end
              else
                config.decay_rate
              end

            new_confidence = belief.confidence * (1.0 - rate)

            if new_confidence < 0.1 do
              # Auto-retract
              Logger.debug("Belief auto-retracted due to decay",
                id: id,
                subject: belief.subject,
                predicate: belief.predicate
              )

              {Map.put(acc, id, %{belief | confidence: 0.0}), [id | retracted]}
            else
              {Map.put(acc, id, %{belief | confidence: new_confidence}), retracted}
            end
          else
            {Map.put(acc, id, belief), retracted}
          end
        end
      end)

    new_retracted = Enum.reduce(retracted_ids, state.retracted, &MapSet.put(&2, &1))

    %{state | beliefs: updated_beliefs, retracted: new_retracted}
  end

  defp maybe_create_jtms_node(%Belief{node_id: node_id} = belief)
       when not is_nil(node_id) do
    # Already has a JTMS node
    belief
  end

  defp maybe_create_jtms_node(%Belief{source: source} = belief) do
    alias Brain.Epistemic.JTMS

    datum = "belief:#{belief.subject}:#{belief.predicate}:#{belief.object}"

    result =
      case source do
        :explicit ->
          JTMS.create_premise(datum)

        _ ->
          # :inferred, :learned, :consolidated — retractable assumptions
          JTMS.create_assumption(datum, true)
      end

    case result do
      {:ok, node_id} ->
        %{belief | node_id: node_id}

      _ ->
        belief
    end
  rescue
    _ -> belief
  end

  defp add_to_index(index, nil, _id) do
    index
  end

  defp add_to_index(index, key, id) do
    current = Map.get(index, key, MapSet.new())
    Map.put(index, key, MapSet.put(current, id))
  end

  defp apply_filters(beliefs, opts) do
    beliefs
    |> filter_by(:subject, Keyword.get(opts, :subject))
    |> filter_by(:predicate, Keyword.get(opts, :predicate))
    |> filter_by(:user_id, Keyword.get(opts, :user_id))
    |> filter_by(:source, Keyword.get(opts, :source))
    |> filter_by_min_confidence(Keyword.get(opts, :min_confidence))
  end

  defp filter_by(beliefs, _field, nil) do
    beliefs
  end

  defp filter_by(beliefs, field, value) do
    Enum.filter(beliefs, fn b -> Map.get(b, field) == value end)
  end

  defp filter_by_min_confidence(beliefs, nil) do
    beliefs
  end

  defp filter_by_min_confidence(beliefs, min_conf) do
    Enum.filter(beliefs, fn b -> b.confidence >= min_conf end)
  end

  defp load_from_atlas(state) do
    case Brain.AtlasIntegration.load_beliefs() do
      {:ok, beliefs} when beliefs != [] ->
        Logger.info("Loading BeliefStore from Atlas", count: length(beliefs))

        Enum.reduce(beliefs, state, fn belief, acc ->
          new_beliefs = Map.put(acc.beliefs, belief.id, belief)
          new_by_user = add_to_index(acc.by_user, belief.user_id, belief.id)
          new_by_subject = add_to_index(acc.by_subject, belief.subject, belief.id)
          new_by_predicate = add_to_index(acc.by_predicate, belief.predicate, belief.id)

          %{
            acc
            | beliefs: new_beliefs,
              by_user: new_by_user,
              by_subject: new_by_subject,
              by_predicate: new_by_predicate
          }
        end)

      _ ->
        Logger.debug("No beliefs in Atlas, starting with empty BeliefStore")
        state
    end
  rescue
    e ->
      Logger.warning("Failed to load BeliefStore from Atlas: #{inspect(e)}")
      state
  end
end
