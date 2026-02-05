defmodule Brain.Epistemic.BeliefStore do
  @moduledoc """
  GenServer managing beliefs with persistence.

  The BeliefStore is the primary storage layer for the epistemic system.
  It manages beliefs, tracks their relationships via the JTMS, and provides
  query capabilities for the rest of the system.

  Features:
  - CRUD operations for beliefs
  - Querying by subject, predicate, user
  - Confidence updates with confirmation tracking
  - Persistence to disk
  - Integration with JTMS for justification tracking
  """

  use GenServer

  alias Brain.Epistemic.Types.{Belief, Config}
  alias Brain.Telemetry

  require Logger

  # Default persistence path resolved at runtime
  defp default_persistence_path, do: Brain.priv_path("data/belief_store.term")

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Adds a new belief to the store.

  Returns {:ok, belief_id} on success.
  """
  def add_belief(%Belief{} = belief) do
    Telemetry.span(:belief_operation, %{operation: :add, subject: belief.subject}, fn ->
      if Config.enabled?() do
        GenServer.call(__MODULE__, {:add_belief, belief})
      else
        {:ok, belief.id}
      end
    end)
  end

  @doc """
  Creates and adds a belief from parameters.
  """
  def add_belief(subject, predicate, object, opts \\ []) do
    belief = Belief.new(subject, predicate, object, opts)
    add_belief(belief)
  end

  @doc """
  Retracts a belief by ID.

  This marks the belief as retracted but keeps it for history.
  Related justifications are updated.
  """
  def retract_belief(belief_id) do
    GenServer.call(__MODULE__, {:retract_belief, belief_id})
  end

  @doc """
  Gets a belief by ID.
  """
  def get_belief(belief_id) do
    GenServer.call(__MODULE__, {:get_belief, belief_id})
  end

  @doc """
  Queries beliefs by subject and/or predicate.

  Options:
  - :subject - Filter by subject (:user, :world, :self, or string)
  - :predicate - Filter by predicate
  - :user_id - Filter by user_id
  - :min_confidence - Minimum confidence threshold
  - :source - Filter by source (:explicit, :inferred, etc.)
  """
  def query_beliefs(opts \\ []) do
    Telemetry.span(:belief_operation, %{operation: :query, filters: opts}, fn ->
      GenServer.call(__MODULE__, {:query_beliefs, opts})
    end)
  end

  @doc """
  Gets all beliefs about a specific user.
  """
  def get_beliefs_for_user(user_id) do
    query_beliefs(user_id: user_id)
  end

  # ============================================================================
  # Event-Based Belief Extraction
  # ============================================================================

  @doc """
  Extract beliefs from an extracted event.

  Creates beliefs based on the event structure:
  - Actor performing an action suggests the actor wants/needs/likes the object
  - Imperative actions suggest user desires

  ## Parameters
    - event: An Event struct from EventExtractor
    - user_id: The user ID to associate beliefs with (optional)

  ## Examples

      event = %Event{
        action: %{lemma: "want", tense: :present},
        actor: %{text: "I", type: "pronoun"},
        object: %{text: "coffee", type: "noun"},
        confidence: 0.85
      }

      extract_beliefs_from_event(event, "user_123")
      # Creates belief: User wants coffee (confidence: 0.85)
  """
  def extract_beliefs_from_event(event, user_id \\ nil) do
    # Only extract beliefs from high-confidence events
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

    # Extract belief based on action type
    beliefs_created =
      cond do
        # User wants/needs/desires something
        action_lemma in ["want", "need", "desire", "wish", "like", "love", "prefer"] and object != nil ->
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
                 source: :inferred,
                 confidence: confidence * 0.9,
                 user_id: user_id
               ) do
            {:ok, belief_id} -> [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        # User asks about something (implies interest)
        action_lemma in ["ask", "wonder", "question", "inquire"] and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(:user, :interested_in, object_text,
                 source: :inferred,
                 confidence: confidence * 0.7,
                 user_id: user_id
               ) do
            {:ok, belief_id} -> [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        # Imperative commands suggest user wants bot to do something
        tense == :imperative and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(:user, :requests, "#{action_lemma} #{object_text}",
                 source: :inferred,
                 confidence: confidence * 0.8,
                 user_id: user_id
               ) do
            {:ok, belief_id} -> [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        # User knows/believes something
        action_lemma in ["know", "believe", "think", "understand"] and object != nil ->
          object_text = Map.get(object, :text, "something")

          case add_belief(:user, :believes, object_text,
                 source: :inferred,
                 confidence: confidence * 0.8,
                 user_id: user_id
               ) do
            {:ok, belief_id} -> [belief_id | beliefs_created]
            _ -> beliefs_created
          end

        true ->
          beliefs_created
      end

    {:ok, beliefs_created}
  end

  @doc """
  Extract beliefs from multiple events.

  Processes a list of events and extracts beliefs from each.
  """
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

  @doc """
  Updates the confidence of a belief.

  Options:
  - :confirm - If true, also updates last_confirmed timestamp
  """
  def update_confidence(belief_id, new_confidence, opts \\ []) do
    GenServer.call(__MODULE__, {:update_confidence, belief_id, new_confidence, opts})
  end

  @doc """
  Confirms a belief (updates last_confirmed and optionally boosts confidence).
  """
  def confirm_belief(belief_id, confidence_boost \\ 0.1) do
    GenServer.call(__MODULE__, {:confirm_belief, belief_id, confidence_boost})
  end

  @doc """
  Links a belief to a JTMS node.
  """
  def link_to_node(belief_id, node_id) do
    GenServer.call(__MODULE__, {:link_to_node, belief_id, node_id})
  end

  @doc """
  Gets store statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Persists the store to disk.
  """
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc """
  Clears all beliefs (useful for testing).
  """
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  @doc """
  Checks if the store is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    persistence_path = Keyword.get(opts, :persistence_path, default_persistence_path())

    state = %{
      beliefs: %{},
      by_user: %{},
      by_subject: %{},
      by_predicate: %{},
      retracted: MapSet.new(),
      persistence_path: persistence_path
    }

    # Try to load from disk
    state = maybe_load_from_disk(state)

    Logger.info("BeliefStore initialized", belief_count: map_size(state.beliefs))

    {:ok, state}
  end

  @impl true
  def handle_call({:add_belief, belief}, _from, state) do
    # Store the belief
    new_beliefs = Map.put(state.beliefs, belief.id, belief)

    # Update indices
    new_by_user = add_to_index(state.by_user, belief.user_id, belief.id)
    new_by_subject = add_to_index(state.by_subject, belief.subject, belief.id)
    new_by_predicate = add_to_index(state.by_predicate, belief.predicate, belief.id)

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

    {:reply, {:ok, belief.id}, new_state}
  end

  @impl true
  def handle_call({:retract_belief, belief_id}, _from, state) do
    case Map.get(state.beliefs, belief_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      _belief ->
        new_retracted = MapSet.put(state.retracted, belief_id)
        new_state = %{state | retracted: new_retracted}

        Logger.debug("Belief retracted", id: belief_id)

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
    # Start with all beliefs, then filter
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
    result = persist_to_disk(state)
    {:reply, result, state}
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

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp add_to_index(index, nil, _id), do: index

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

  defp filter_by(beliefs, _field, nil), do: beliefs

  defp filter_by(beliefs, field, value) do
    Enum.filter(beliefs, fn b -> Map.get(b, field) == value end)
  end

  defp filter_by_min_confidence(beliefs, nil), do: beliefs

  defp filter_by_min_confidence(beliefs, min_conf) do
    Enum.filter(beliefs, fn b -> b.confidence >= min_conf end)
  end

  defp maybe_load_from_disk(state) do
    path = state.persistence_path

    if File.exists?(path) do
      case File.read(path) do
        {:ok, binary} ->
          try do
            data = :erlang.binary_to_term(binary)

            Logger.info("Loaded BeliefStore from disk",
              beliefs: map_size(data.beliefs || %{})
            )

            %{
              state
              | beliefs: Map.get(data, :beliefs, %{}),
                by_user: Map.get(data, :by_user, %{}),
                by_subject: Map.get(data, :by_subject, %{}),
                by_predicate: Map.get(data, :by_predicate, %{}),
                retracted: Map.get(data, :retracted, MapSet.new())
            }
          rescue
            e ->
              Logger.warning("Failed to load BeliefStore: #{inspect(e)}")
              state
          end

        {:error, reason} ->
          Logger.warning("Could not read BeliefStore file: #{inspect(reason)}")
          state
      end
    else
      state
    end
  end

  defp persist_to_disk(state) do
    path = state.persistence_path

    # Ensure directory exists
    path |> Path.dirname() |> File.mkdir_p!()

    data = %{
      beliefs: state.beliefs,
      by_user: state.by_user,
      by_subject: state.by_subject,
      by_predicate: state.by_predicate,
      retracted: state.retracted
    }

    binary = :erlang.term_to_binary(data)

    case File.write(path, binary) do
      :ok ->
        Logger.info("BeliefStore persisted to disk", beliefs: map_size(state.beliefs))
        :ok

      {:error, reason} ->
        Logger.error("Failed to persist BeliefStore: #{inspect(reason)}")
        {:error, reason}
    end
  end
end
