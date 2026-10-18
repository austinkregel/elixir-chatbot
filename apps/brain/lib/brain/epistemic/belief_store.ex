defmodule Brain.Epistemic.BeliefStore do
  @moduledoc """
  GenServer managing beliefs with persistence.

  The BeliefStore is the primary storage layer for the epistemic system.
  It manages beliefs, tracks their relationships via the JTMS, and provides
  query capabilities for the rest of the system.

  Beliefs are held per world. Every function that reads or changes beliefs
  takes the world first; a belief added in one world is never found, counted,
  confirmed or retracted from another. A missing or malformed world raises
  `ArgumentError` in the caller's process. A belief's JTMS node lives in the
  same world's JTMS network (`Brain.Epistemic.JTMS`), and its Atlas row
  records the world.

  Features:
  - CRUD operations for beliefs
  - Querying by subject, predicate, user
  - Confidence updates with confirmation tracking
  - Persistence to Atlas
  - Integration with JTMS for justification tracking
  """

  alias Brain.Epistemic.Types
  use GenServer

  alias Types.{Belief, Config}
  alias Brain.Telemetry

  require Logger

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Adds a new belief to the world's store.

  Returns {:ok, belief_id} on success.
  """
  def add_belief(world_id, %Belief{} = belief) do
    world_id = world_id!(world_id, "add_belief")

    Telemetry.span(:belief_operation, %{operation: :add, subject: belief.subject}, fn ->
      if Config.enabled?() do
        GenServer.call(__MODULE__, {:add_belief, world_id, belief}, 5_000)
      else
        {:ok, belief.id}
      end
    end)
  end

  @doc "Creates and adds a belief from parameters, in the world.\n"
  def add_belief(world_id, subject, predicate, object, opts \\ []) do
    world_id = world_id!(world_id, "add_belief")
    belief = Belief.new(subject, predicate, object, opts)
    add_belief(world_id, belief)
  end

  @doc """
  Retracts a belief held in the world by ID.

  This marks the belief as retracted but keeps it for history.
  Related justifications are updated.
  """
  def retract_belief(world_id, belief_id) do
    world_id = world_id!(world_id, "retract_belief")
    GenServer.call(__MODULE__, {:retract_belief, world_id, belief_id}, 5_000)
  end

  @doc "Gets a belief held in the world by ID.\n"
  def get_belief(world_id, belief_id) do
    world_id = world_id!(world_id, "get_belief")
    GenServer.call(__MODULE__, {:get_belief, world_id, belief_id}, 5_000)
  end

  @doc """
  Queries the beliefs held in the world by subject and/or predicate.

  Options:
  - :subject - Filter by subject (:user, :world, :self, or string)
  - :predicate - Filter by predicate
  - :user_id - Filter by user_id
  - :min_confidence - Minimum confidence threshold
  - :source - Filter by source (:explicit, :inferred, etc.)
  """
  def query_beliefs(world_id, opts \\ []) do
    world_id = world_id!(world_id, "query_beliefs")

    Telemetry.span(:belief_operation, %{operation: :query, filters: opts}, fn ->
      GenServer.call(__MODULE__, {:query_beliefs, world_id, opts}, 5_000)
    end)
  end

  @doc "Gets all beliefs held in the world about a specific user.\n"
  def get_beliefs_for_user(world_id, user_id) do
    query_beliefs(world_id, user_id: user_id)
  end

  @doc """
  Extract beliefs from an extracted event, into the world.

  Creates beliefs based on the event structure:
  - Actor performing an action suggests the actor wants/needs/likes the object
  - Imperative actions suggest user desires

  ## Parameters
    - world_id: The world the beliefs are held in
    - event: An Event struct from EventExtractor
    - user_id: The user ID to associate beliefs with (optional)

  ## Examples

      event = %Event{
        action: %{lemma: "want", tense: :present},
        actor: %{text: "I", type: "pronoun"},
        object: %{text: "coffee", type: "noun"},
        confidence: 0.85
      }

      extract_beliefs_from_event("default", event, "user_123")
      # Creates belief: User wants coffee (confidence: 0.85)
  """
  def extract_beliefs_from_event(world_id, event, user_id \\ nil) do
    world_id = world_id!(world_id, "extract_beliefs_from_event")
    confidence = Map.get(event, :confidence, 0.0)

    if confidence >= 0.6 do
      do_extract_beliefs_from_event(world_id, event, user_id)
    else
      {:ok, []}
    end
  end

  defp do_extract_beliefs_from_event(world_id, event, user_id) do
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

          case add_belief(world_id, :user, predicate, object_text,
                 source: source_info,
                 confidence: confidence * 0.9,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: predicate, object: object_text, confidence: confidence * 0.9}, world_id)
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        action_lemma in ["ask", "wonder", "question", "inquire"] and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(world_id, :user, :interested_in, object_text,
                 source: :inferred,
                 confidence: confidence * 0.7,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: :interested_in, object: object_text, confidence: confidence * 0.7}, world_id)
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        tense == :imperative and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(world_id, :user, :requests, "#{action_lemma} #{object_text}",
                 source: :inferred,
                 confidence: confidence * 0.8,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: :requests, object: "#{action_lemma} #{object_text}", confidence: confidence * 0.8}, world_id)
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        action_lemma in ["know", "believe", "think", "understand"] and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(world_id, :user, :believes, object_text,
                 source: :inferred,
                 confidence: confidence * 0.8,
                 user_id: user_id
               ) do
            {:ok, belief_id} ->
              Brain.Graph.Writer.write_belief(%{id: belief_id, subject: :user, predicate: :believes, object: object_text, confidence: confidence * 0.8}, world_id)
              [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        true ->
          beliefs_created
      end

    {:ok, beliefs_created}
  end

  @doc """
  Extract beliefs from multiple events, into the world.

  Processes a list of events and extracts beliefs from each.
  """
  def extract_beliefs_from_events(world_id, events, user_id \\ nil) when is_list(events) do
    world_id = world_id!(world_id, "extract_beliefs_from_events")

    results =
      Enum.map(events, fn event ->
        extract_beliefs_from_event(world_id, event, user_id)
      end)

    belief_ids =
      results
      |> Enum.flat_map(fn
        {:ok, ids} -> ids
        _ -> []
      end)

    {:ok, belief_ids}
  end

  @doc """
  Updates the confidence of a belief held in the world.

  Options:
  - :confirm - If true, also updates last_confirmed timestamp
  """
  def update_confidence(world_id, belief_id, new_confidence, opts \\ []) do
    world_id = world_id!(world_id, "update_confidence")
    GenServer.call(__MODULE__, {:update_confidence, world_id, belief_id, new_confidence, opts}, 5_000)
  end

  @doc "Confirms a belief held in the world (updates last_confirmed and optionally boosts confidence).\n"
  def confirm_belief(world_id, belief_id, confidence_boost \\ 0.1) do
    world_id = world_id!(world_id, "confirm_belief")
    GenServer.call(__MODULE__, {:confirm_belief, world_id, belief_id, confidence_boost}, 5_000)
  end

  @doc """
  Adds a belief with source authority context, in the world.

  The confidence and decay behaviour are determined by the authority
  profile loaded from SourceAuthority. This replaces the old
  `add_admin_belief/4` function.

  ## Parameters

    - world_id: the world the belief is held in
    - subject: :user | :world | :self | String.t()
    - predicate: atom or string identifying the property
    - object: the value of the belief
    - authority_key: atom like :mentor, :academic_expert, :stranger
    - opts: additional keyword options (merged; authority defaults take precedence)
  """
  def add_belief_with_authority(world_id, subject, predicate, object, authority_key, opts \\ []) do
    alias Brain.Epistemic.SourceAuthority

    world_id = world_id!(world_id, "add_belief_with_authority")

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

    result = add_belief(world_id, subject, predicate, object, merged_opts)

    if match?({:ok, _}, result) and SourceAuthority.ready?() do
      try do
        SourceAuthority.record_outcome(authority_key, :added)
      catch
        :exit, _ -> :ok
      end
    end

    result
  end

  @doc "Links a belief held in the world to a JTMS node.\n"
  def link_to_node(world_id, belief_id, node_id) do
    world_id = world_id!(world_id, "link_to_node")
    GenServer.call(__MODULE__, {:link_to_node, world_id, belief_id, node_id}, 5_000)
  end

  @doc "Gets the statistics of the beliefs held in the world.\n"
  def stats(world_id) do
    world_id = world_id!(world_id, "stats")
    GenServer.call(__MODULE__, {:stats, world_id}, 5_000)
  end

  @doc "Persists the store to disk.\n"
  def persist do
    GenServer.call(__MODULE__, :persist, 30_000)
  end

  @doc "Clears the beliefs held in the world (useful for testing). Other worlds keep theirs.\n"
  def clear(world_id) do
    world_id = world_id!(world_id, "clear")
    GenServer.call(__MODULE__, {:clear, world_id}, 30_000)
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

  defp world_id!(world_id, _function) when is_binary(world_id) and world_id != "", do: world_id

  defp world_id!(other, function) do
    raise ArgumentError,
          "BeliefStore.#{function} needs the world the beliefs are held in, got: #{inspect(other)}"
  end

  defp empty_world do
    %{
      beliefs: %{},
      by_user: %{},
      by_subject: %{},
      by_predicate: %{},
      retracted: MapSet.new()
    }
  end

  defp world(state, world_id), do: Map.get(state.worlds, world_id, empty_world())

  defp put_world(state, world_id, world), do: %{state | worlds: Map.put(state.worlds, world_id, world)}

  @impl true
  def init(_opts) do
    state = load_from_atlas(%{worlds: %{}})

    # Schedule confidence decay tick
    decay_interval = Config.get().decay_interval_ms
    if decay_interval > 0, do: Process.send_after(self(), :decay_tick, decay_interval)

    Logger.info("BeliefStore initialized",
      belief_count: state.worlds |> Map.values() |> Enum.map(&map_size(&1.beliefs)) |> Enum.sum(),
      world_count: map_size(state.worlds)
    )

    {:ok, state}
  end

  @impl true
  def handle_call({:add_belief, world_id, belief}, _from, state) do
    world = world(state, world_id)

    # Create the JTMS node in the world's network for justification tracking
    belief = maybe_create_jtms_node(world_id, belief)

    world = index_belief(world, belief)

    Logger.debug("Belief added",
      id: belief.id,
      world_id: world_id,
      subject: belief.subject,
      predicate: belief.predicate
    )

    # Write-through to Atlas
    Brain.AtlasIntegration.persist_belief(world_id, belief)

    {:reply, {:ok, belief.id}, put_world(state, world_id, world)}
  end

  @impl true
  def handle_call({:retract_belief, world_id, belief_id}, _from, state) do
    world = world(state, world_id)

    case Map.get(world.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        world = %{world | retracted: MapSet.put(world.retracted, belief_id)}

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

        Logger.debug("Belief retracted", id: belief_id, world_id: world_id)
        Brain.AtlasIntegration.retract_belief_in_atlas(world_id, belief_id)

        {:reply, :ok, put_world(state, world_id, world)}
    end
  end

  @impl true
  def handle_call({:get_belief, world_id, belief_id}, _from, state) do
    world = world(state, world_id)

    case Map.get(world.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        if MapSet.member?(world.retracted, belief_id) do
          {:reply, {:error, :retracted}, state}
        else
          {:reply, {:ok, belief}, state}
        end
    end
  end

  @impl true
  def handle_call({:query_beliefs, world_id, opts}, _from, state) do
    world = world(state, world_id)

    beliefs =
      world.beliefs
      |> Map.values()
      |> Enum.reject(fn b -> MapSet.member?(world.retracted, b.id) end)
      |> apply_filters(opts)

    {:reply, {:ok, beliefs}, state}
  end

  @impl true
  def handle_call({:update_confidence, world_id, belief_id, new_confidence, opts}, _from, state) do
    world = world(state, world_id)

    case Map.get(world.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        confirm? = Keyword.get(opts, :confirm, false)
        updated = Belief.update_confidence(belief, new_confidence, confirm?)
        world = %{world | beliefs: Map.put(world.beliefs, belief_id, updated)}

        Brain.AtlasIntegration.update_belief_confidence(
          world_id,
          belief_id,
          new_confidence,
          if(confirm?, do: updated.last_confirmed)
        )

        {:reply, {:ok, updated}, put_world(state, world_id, world)}
    end
  end

  @impl true
  def handle_call({:confirm_belief, world_id, belief_id, confidence_boost}, _from, state) do
    world = world(state, world_id)

    case Map.get(world.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        new_confidence = min(belief.confidence + confidence_boost, 1.0)

        updated =
          belief
          |> Belief.update_confidence(new_confidence, true)

        world = %{world | beliefs: Map.put(world.beliefs, belief_id, updated)}

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

        {:reply, {:ok, updated}, put_world(state, world_id, world)}
    end
  end

  @impl true
  def handle_call({:link_to_node, world_id, belief_id, node_id}, _from, state) do
    world = world(state, world_id)

    case Map.get(world.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      belief ->
        updated = %{belief | node_id: node_id}
        world = %{world | beliefs: Map.put(world.beliefs, belief_id, updated)}

        {:reply, :ok, put_world(state, world_id, world)}
    end
  end

  @impl true
  def handle_call({:stats, world_id}, _from, state) do
    world = world(state, world_id)

    active_count =
      world.beliefs
      |> Map.keys()
      |> Enum.reject(&MapSet.member?(world.retracted, &1))
      |> length()

    stats = %{
      total_beliefs: map_size(world.beliefs),
      active_beliefs: active_count,
      retracted_beliefs: MapSet.size(world.retracted),
      unique_users: map_size(world.by_user),
      unique_subjects: map_size(world.by_subject),
      unique_predicates: map_size(world.by_predicate)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:clear, world_id}, _from, state) do
    {:reply, :ok, %{state | worlds: Map.delete(state.worlds, world_id)}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_info(:decay_tick, state) do
    config = Config.get()

    new_state = %{
      state
      | worlds: Map.new(state.worlds, fn {world_id, world} -> {world_id, apply_confidence_decay(world, config)} end)
    }

    # Re-schedule
    if config.decay_interval_ms > 0 do
      Process.send_after(self(), :decay_tick, config.decay_interval_ms)
    end

    {:noreply, new_state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  defp index_belief(world, belief) do
    %{
      world
      | beliefs: Map.put(world.beliefs, belief.id, belief),
        by_user: add_to_index(world.by_user, belief.user_id, belief.id),
        by_subject: add_to_index(world.by_subject, belief.subject, belief.id),
        by_predicate: add_to_index(world.by_predicate, belief.predicate, belief.id)
    }
  end

  defp apply_confidence_decay(world, config) do
    now = DateTime.utc_now()
    min_age_seconds = div(config.decay_min_age_ms, 1000)

    {updated_beliefs, retracted_ids} =
      Enum.reduce(world.beliefs, {%{}, []}, fn {id, belief}, {acc, retracted} ->
        if belief.source in config.decay_exempt_sources or MapSet.member?(world.retracted, id) do
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

    new_retracted = Enum.reduce(retracted_ids, world.retracted, &MapSet.put(&2, &1))

    %{world | beliefs: updated_beliefs, retracted: new_retracted}
  end

  defp maybe_create_jtms_node(_world_id, %Belief{node_id: node_id} = belief)
       when not is_nil(node_id) do
    # Already has a JTMS node
    belief
  end

  defp maybe_create_jtms_node(world_id, %Belief{source: source} = belief) do
    alias Brain.Epistemic.JTMS

    datum = "belief:#{belief.subject}:#{belief.predicate}:#{belief.object}"

    result =
      case source do
        :explicit ->
          JTMS.create_premise(world_id, datum)

        _ ->
          # :inferred, :learned, :consolidated — retractable assumptions
          JTMS.create_assumption(world_id, datum, true)
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

        Enum.reduce(beliefs, state, fn {world_id, belief}, acc ->
          put_world(acc, world_id, index_belief(world(acc, world_id), belief))
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
