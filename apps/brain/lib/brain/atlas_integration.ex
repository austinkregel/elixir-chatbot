defmodule Brain.AtlasIntegration do
  @moduledoc """
  Integration layer between Brain GenServers and Atlas persistence.

  Provides async write-through to Atlas for all Brain GenServers,
  with graceful fallback when Atlas is unavailable. Atlas is the
  source of truth; in-memory state provides fast access.

  All write operations are async (fire-and-forget via Task.Supervisor)
  to avoid blocking GenServer operations on database calls.
  """

  require Logger

  @doc "Returns true if Atlas.Repo is available for operations."
  def available? do
    Code.ensure_loaded?(Atlas.Repo) and
      is_pid(Process.whereis(Atlas.Repo))
  rescue
    _ -> false
  catch
    _, _ -> false
  end

  @doc "Execute an Atlas operation asynchronously via the AtlasTaskSupervisor."
  def async(fun) when is_function(fun, 0) do
    if available?() do
      Task.Supervisor.start_child(Brain.AtlasTaskSupervisor, fn ->
        try do
          fun.()
        rescue
          e ->
            Logger.debug("Atlas async operation failed: #{Exception.message(e)}")
        catch
          kind, reason ->
            Logger.debug("Atlas async operation failed: #{inspect(kind)} #{inspect(reason)}")
        end
      end)
    end

    :ok
  end

  @doc "Wait for all in-flight Atlas async tasks to complete."
  def drain(timeout \\ 5_000) do
    if Process.whereis(Brain.AtlasTaskSupervisor) do
      deadline = System.monotonic_time(:millisecond) + timeout
      do_drain(deadline)
    end

    :ok
  end

  defp do_drain(deadline) do
    case Task.Supervisor.children(Brain.AtlasTaskSupervisor) do
      [] ->
        :ok

      _children ->
        if System.monotonic_time(:millisecond) < deadline do
          Process.sleep(10)
          do_drain(deadline)
        else
          :ok
        end
    end
  end

  @doc "Execute an Atlas operation synchronously. Returns {:ok, result} or {:error, reason}."
  def sync(fun) when is_function(fun, 0) do
    if available?() do
      try do
        {:ok, fun.()}
      rescue
        e -> {:error, Exception.message(e)}
      catch
        kind, reason -> {:error, {kind, reason}}
      end
    else
      {:error, :atlas_unavailable}
    end
  end

  # ============================================================================
  # BeliefStore Integration
  # ============================================================================

  @doc "Load all beliefs from Atlas. Returns list of Brain.Epistemic.Types.Belief structs."
  def load_beliefs do
    with {:ok, result} <- sync(fn -> Atlas.Repo.all(Atlas.Schemas.Belief) end) do
      beliefs =
        Enum.map(result, fn row ->
          %Brain.Epistemic.Types.Belief{
            id: row.id,
            subject: parse_atom_or_string(row.subject),
            predicate: safe_to_atom(row.predicate),
            object: row.object,
            confidence: row.confidence || 0.5,
            source: safe_to_atom(row.source),
            source_authority: safe_to_atom(row.source_authority),
            user_id: row.user_id,
            node_id: row.node_id,
            created_at: row.inserted_at,
            last_confirmed: row.last_confirmed,
            provenance: Map.get(row.provenance || %{}, "list", [])
          }
        end)

      {:ok, beliefs}
    end
  end

  @doc "Persist a belief to Atlas asynchronously."
  def persist_belief(%Brain.Epistemic.Types.Belief{} = belief) do
    async(fn ->
      attrs = %{
        id: belief.id,
        subject: to_string(belief.subject),
        predicate: to_string(belief.predicate),
        object: to_string(belief.object),
        confidence: belief.confidence,
        source: to_string(belief.source),
        source_authority: if(belief.source_authority, do: to_string(belief.source_authority)),
        user_id: belief.user_id,
        node_id: belief.node_id,
        retracted: false,
        last_confirmed: belief.last_confirmed,
        provenance: %{"list" => belief.provenance || []},
        metadata: %{}
      }

      case Atlas.Repo.get(Atlas.Schemas.Belief, belief.id) do
        nil ->
          %Atlas.Schemas.Belief{}
          |> Atlas.Schemas.Belief.changeset(attrs)
          |> Atlas.Repo.insert()

        existing ->
          existing
          |> Atlas.Schemas.Belief.changeset(attrs)
          |> Atlas.Repo.update()
      end
    end)
  end

  @doc "Update belief confidence in Atlas."
  def update_belief_confidence(belief_id, confidence, last_confirmed) do
    async(fn ->
      case Atlas.Repo.get(Atlas.Schemas.Belief, belief_id) do
        nil -> :ok

        belief ->
          attrs = %{confidence: confidence}
          attrs = if last_confirmed, do: Map.put(attrs, :last_confirmed, last_confirmed), else: attrs

          belief
          |> Atlas.Schemas.Belief.changeset(attrs)
          |> Atlas.Repo.update()
      end
    end)
  end

  @doc "Mark a belief as retracted in Atlas."
  def retract_belief_in_atlas(belief_id) do
    async(fn ->
      case Atlas.Repo.get(Atlas.Schemas.Belief, belief_id) do
        nil -> :ok

        belief ->
          belief
          |> Atlas.Schemas.Belief.changeset(%{retracted: true})
          |> Atlas.Repo.update()
      end
    end)
  end

  # ============================================================================
  # Memory.Store Integration
  # ============================================================================

  @doc "Load all episodes from Atlas for a world. Returns map of id -> Episode."
  def load_episodes(world_id) do
    import Ecto.Query

    with {:ok, rows} <-
           sync(fn ->
             Atlas.Schemas.Episode
             |> where([e], e.world_id == ^world_id)
             |> Atlas.Repo.all()
           end) do
      episodes =
        rows
        |> Enum.map(fn row ->
          ep = %Brain.Memory.Types.Episode{
            id: row.id,
            state: row.state,
            action: row.action,
            outcome: row.outcome,
            tags: row.tags || [],
            embedding: row.embedding || [],
            timestamp: DateTime.to_unix(row.inserted_at, :millisecond),
            semantic_id: row.semantic_id
          }

          {row.id, ep}
        end)
        |> Map.new()

      {:ok, episodes}
    end
  end

  @doc "Load all semantic facts from Atlas for a world."
  def load_semantics(world_id) do
    import Ecto.Query

    with {:ok, rows} <-
           sync(fn ->
             Atlas.Schemas.SemanticFact
             |> where([s], s.world_id == ^world_id)
             |> Atlas.Repo.all()
           end) do
      semantics =
        rows
        |> Enum.map(fn row ->
          sf = %Brain.Memory.Types.SemanticFact{
            id: row.id,
            representation: row.content,
            embedding: row.embedding || [],
            evidence_ids: row.source_episodes || [],
            timestamp: DateTime.to_unix(row.inserted_at, :millisecond)
          }

          {row.id, sf}
        end)
        |> Map.new()

      {:ok, semantics}
    end
  end

  @doc "Persist an episode to Atlas."
  def persist_episode(%Brain.Memory.Types.Episode{} = episode, world_id) do
    async(fn ->
      attrs = %{
        id: episode.id,
        world_id: world_id,
        state: episode.state,
        action: episode.action,
        outcome: episode.outcome,
        tags: episode.tags || [],
        embedding: episode.embedding || [],
        semantic_id: episode.semantic_id
      }

      case Atlas.Repo.get(Atlas.Schemas.Episode, episode.id) do
        nil ->
          %Atlas.Schemas.Episode{}
          |> Atlas.Schemas.Episode.changeset(attrs)
          |> Atlas.Repo.insert()

        existing ->
          existing
          |> Atlas.Schemas.Episode.changeset(attrs)
          |> Atlas.Repo.update()
      end
    end)
  end

  @doc "Persist a semantic fact to Atlas."
  def persist_semantic(%Brain.Memory.Types.SemanticFact{} = semantic, world_id) do
    async(fn ->
      attrs = %{
        id: semantic.id,
        world_id: world_id,
        content: semantic.representation,
        category: "consolidated",
        confidence: 0.5,
        embedding: semantic.embedding || [],
        source_episodes: semantic.evidence_ids || []
      }

      case Atlas.Repo.get(Atlas.Schemas.SemanticFact, semantic.id) do
        nil ->
          %Atlas.Schemas.SemanticFact{}
          |> Atlas.Schemas.SemanticFact.changeset(attrs)
          |> Atlas.Repo.insert()

        existing ->
          existing
          |> Atlas.Schemas.SemanticFact.changeset(attrs)
          |> Atlas.Repo.update()
      end
    end)
  end

  @doc "Update episode's semantic_id link in Atlas."
  def link_episode_semantic(episode_id, semantic_id) do
    sync(fn ->
      case Atlas.Repo.get(Atlas.Schemas.Episode, episode_id) do
        nil -> :ok

        ep ->
          ep
          |> Atlas.Schemas.Episode.changeset(%{semantic_id: semantic_id})
          |> Atlas.Repo.update()
      end
    end)
  end

  @doc """
  Persist an episode to Atlas synchronously.

  Returns `{:ok, episode_id}` or `{:error, reason}`.
  Used by Memory.Store when Atlas is the primary store.
  """
  def persist_episode_sync(%Brain.Memory.Types.Episode{} = episode, world_id) do
    case Ecto.UUID.cast(episode.id) do
      {:ok, _uuid} ->
        sync(fn ->
          attrs = %{
            id: episode.id,
            world_id: world_id,
            state: episode.state,
            action: episode.action,
            outcome: episode.outcome,
            tags: episode.tags || [],
            embedding: episode.embedding || [],
            semantic_id: episode.semantic_id
          }

          case Atlas.Repo.get(Atlas.Schemas.Episode, episode.id) do
            nil ->
              %Atlas.Schemas.Episode{id: episode.id}
              |> Atlas.Schemas.Episode.changeset(attrs)
              |> Atlas.Repo.insert()

            existing ->
              existing
              |> Atlas.Schemas.Episode.changeset(attrs)
              |> Atlas.Repo.update()
          end
        end)
        |> case do
          {:ok, {:ok, _row}} -> {:ok, episode.id}
          {:ok, {:error, changeset}} -> {:error, changeset}
          {:error, reason} -> {:error, reason}
        end

      :error ->
        {:error, :invalid_id_format}
    end
  end

  @doc """
  Persist a semantic fact to Atlas synchronously.

  Returns `{:ok, semantic_id}` or `{:error, reason}`.
  """
  def persist_semantic_sync(%Brain.Memory.Types.SemanticFact{} = semantic, world_id) do
    case Ecto.UUID.cast(semantic.id) do
      {:ok, _uuid} ->
        sync(fn ->
          attrs = %{
            id: semantic.id,
            world_id: world_id,
            content: semantic.representation,
            category: "consolidated",
            confidence: 0.5,
            embedding: semantic.embedding || [],
            source_episodes: semantic.evidence_ids || [],
            tags: semantic.tags || []
          }

          case Atlas.Repo.get(Atlas.Schemas.SemanticFact, semantic.id) do
            nil ->
              %Atlas.Schemas.SemanticFact{id: semantic.id}
              |> Atlas.Schemas.SemanticFact.changeset(attrs)
              |> Atlas.Repo.insert()

            existing ->
              existing
              |> Atlas.Schemas.SemanticFact.changeset(attrs)
              |> Atlas.Repo.update()
          end
        end)
        |> case do
          {:ok, {:ok, _row}} -> {:ok, semantic.id}
          {:ok, {:error, changeset}} -> {:error, changeset}
          {:error, reason} -> {:error, reason}
        end

      :error ->
        {:error, :invalid_id_format}
    end
  end

  @doc """
  Get a single episode from Atlas by ID and world_id.
  """
  def get_episode(id, world_id) do
    import Ecto.Query

    case Ecto.UUID.cast(id) do
      {:ok, uuid} ->
        sync(fn ->
          Atlas.Schemas.Episode
          |> where([e], e.id == ^uuid and e.world_id == ^world_id)
          |> Atlas.Repo.one()
        end)
        |> case do
          {:ok, nil} -> {:error, :not_found}
          {:ok, row} -> {:ok, row_to_episode(row)}
          {:error, reason} -> {:error, reason}
        end

      :error ->
        {:error, :not_found}
    end
  end

  @doc """
  Query episodes by tags from Atlas.

  Returns episodes that have at least one matching tag, ordered by
  insertion time (most recent first).
  """
  def query_episodes_by_tags(world_id, tags, limit \\ 10) do
    import Ecto.Query

    sync(fn ->
      Atlas.Schemas.Episode
      |> where([e], e.world_id == ^world_id)
      |> where([e], fragment("? && ?", e.tags, ^tags))
      |> order_by([e], desc: e.inserted_at)
      |> limit(^limit)
      |> Atlas.Repo.all()
    end)
    |> case do
      {:ok, rows} ->
        {:ok, Enum.map(rows, &row_to_episode/1)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Load all episodes for a world, returned as a list (not a map).
  """
  def list_episodes(world_id) do
    import Ecto.Query

    sync(fn ->
      Atlas.Schemas.Episode
      |> where([e], e.world_id == ^world_id)
      |> order_by([e], desc: e.inserted_at)
      |> Atlas.Repo.all()
    end)
    |> case do
      {:ok, rows} -> {:ok, Enum.map(rows, &row_to_episode/1)}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Load all semantic facts for a world, returned as a list.
  """
  def list_semantics(world_id) do
    import Ecto.Query

    sync(fn ->
      Atlas.Schemas.SemanticFact
      |> where([s], s.world_id == ^world_id)
      |> order_by([s], desc: s.inserted_at)
      |> Atlas.Repo.all()
    end)
    |> case do
      {:ok, rows} -> {:ok, Enum.map(rows, &row_to_semantic/1)}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Get a single semantic fact from Atlas by ID and world_id.
  """
  def get_semantic(id, world_id) do
    import Ecto.Query

    case Ecto.UUID.cast(id) do
      {:ok, uuid} ->
        sync(fn ->
          Atlas.Schemas.SemanticFact
          |> where([s], s.id == ^uuid and s.world_id == ^world_id)
          |> Atlas.Repo.one()
        end)
        |> case do
          {:ok, nil} -> {:error, :not_found}
          {:ok, row} -> {:ok, row_to_semantic(row)}
          {:error, reason} -> {:error, reason}
        end

      :error ->
        {:error, :not_found}
    end
  end

  @doc """
  List all distinct world_ids that have episodes or semantics in Atlas.
  """
  def list_memory_worlds do
    import Ecto.Query

    sync(fn ->
      episode_worlds =
        Atlas.Schemas.Episode
        |> select([e], e.world_id)
        |> distinct(true)
        |> Atlas.Repo.all()

      semantic_worlds =
        Atlas.Schemas.SemanticFact
        |> select([s], s.world_id)
        |> distinct(true)
        |> Atlas.Repo.all()

      Enum.uniq(episode_worlds ++ semantic_worlds)
    end)
    |> case do
      {:ok, worlds} -> {:ok, worlds}
      {:error, _} -> {:ok, []}
    end
  end

  defp row_to_episode(row) do
    %Brain.Memory.Types.Episode{
      id: row.id,
      state: row.state,
      action: row.action,
      outcome: row.outcome,
      tags: row.tags || [],
      embedding: row.embedding || [],
      timestamp: DateTime.to_unix(row.inserted_at, :millisecond),
      semantic_id: row.semantic_id
    }
  end

  defp row_to_semantic(row) do
    %Brain.Memory.Types.SemanticFact{
      id: row.id,
      representation: row.content,
      embedding: row.embedding || [],
      evidence_ids: row.source_episodes || [],
      tags: row.tags || [],
      timestamp: DateTime.to_unix(row.inserted_at, :millisecond)
    }
  end

  # ============================================================================
  # ReviewQueue Integration
  @doc "Delete all episodes and semantic facts from Atlas for a world, or all worlds if nil."
  def clear_memory(world_id \\ nil) do
    import Ecto.Query

    sync(fn ->
      if world_id do
        Atlas.Schemas.Episode |> where([e], e.world_id == ^world_id) |> Atlas.Repo.delete_all()
        Atlas.Schemas.SemanticFact |> where([s], s.world_id == ^world_id) |> Atlas.Repo.delete_all()
      else
        Atlas.Repo.delete_all(Atlas.Schemas.Episode)
        Atlas.Repo.delete_all(Atlas.Schemas.SemanticFact)
      end

      :ok
    end)
  end

  # ============================================================================

  @doc "Load all review candidates from Atlas."
  def load_review_candidates do
    with {:ok, rows} <- sync(fn -> Atlas.Repo.all(Atlas.Schemas.ReviewCandidate) end) do
      candidates =
        Enum.map(rows, fn row ->
          source = %Brain.Knowledge.Types.SourceInfo{
            url: "",
            domain: "atlas"
          }

          finding = %Brain.Knowledge.Types.Finding{
            id: row.id,
            claim: Map.get(row.finding || %{}, "claim", ""),
            entity: Map.get(row.finding || %{}, "entity", ""),
            entity_type: Map.get(row.finding || %{}, "entity_type"),
            source: source,
            raw_context: Map.get(row.finding || %{}, "raw_context", ""),
            confidence: Map.get(row.finding || %{}, "confidence", 0.5)
          }

          candidate = %Brain.Knowledge.Types.ReviewCandidate{
            id: row.id,
            finding: finding,
            aggregate_confidence: row.aggregate_confidence || 0.5,
            status: safe_to_atom(row.status),
            reviewed_at: row.reviewed_at,
            reviewer_notes: row.reviewer_notes
          }

          {row.id, candidate}
        end)

      {:ok, candidates}
    end
  end

  @doc "Persist a review candidate to Atlas."
  def persist_review_candidate(%Brain.Knowledge.Types.ReviewCandidate{} = candidate) do
    async(fn ->
      finding_map = %{
        "claim" => candidate.finding.claim,
        "entity" => candidate.finding.entity,
        "entity_type" => candidate.finding.entity_type,
        "raw_context" => candidate.finding.raw_context,
        "confidence" => candidate.finding.confidence
      }

      corroborating =
        Enum.map(candidate.corroborating_sources, fn s ->
          %{"url" => s.url, "domain" => s.domain}
        end)

      attrs = %{
        id: candidate.id,
        status: to_string(candidate.status),
        finding: finding_map,
        aggregate_confidence: candidate.aggregate_confidence,
        corroborating_sources: corroborating,
        reviewer_notes: candidate.reviewer_notes,
        reviewed_at: candidate.reviewed_at
      }

      %Atlas.Schemas.ReviewCandidate{}
      |> Atlas.Schemas.ReviewCandidate.changeset(attrs)
      |> Atlas.Repo.insert(
        on_conflict: {:replace, [:status, :finding, :aggregate_confidence, :corroborating_sources, :reviewer_notes, :reviewed_at, :updated_at]},
        conflict_target: :id
      )
    end)
  end

  # ============================================================================
  # SourceReliability Integration
  # ============================================================================

  @doc "Load source reliability data from Atlas."
  def load_source_reliability do
    with {:ok, rows} <- sync(fn -> Atlas.Repo.all(Atlas.Schemas.SourceReliability) end) do
      sources =
        Enum.map(rows, fn row ->
          profile = %Brain.Knowledge.Types.SourceProfile{
            domain: row.domain,
            factual_accuracy: row.reliability_score || 0.5,
            bias_rating: safe_to_atom(row.bias_rating),
            trust_tier: safe_to_atom(row.trust_tier),
            admin_decisions: row.admin_decisions || [],
            last_updated: row.updated_at
          }

          {row.domain, profile}
        end)
        |> Map.new()

      {:ok, sources}
    end
  end

  @doc "Persist source reliability data to Atlas."
  def persist_source_reliability(domain, %Brain.Knowledge.Types.SourceProfile{} = profile) do
    async(fn ->
      attrs = %{
        domain: domain,
        reliability_score: profile.factual_accuracy,
        bias_rating: to_string(profile.bias_rating),
        trust_tier: to_string(profile.trust_tier),
        confirmed_count: count_decisions(profile.admin_decisions, :approved),
        rejected_count: count_decisions(profile.admin_decisions, :rejected),
        admin_decisions: Enum.take(profile.admin_decisions, 100)
      }

      %Atlas.Schemas.SourceReliability{}
      |> Atlas.Schemas.SourceReliability.changeset(attrs)
      |> Atlas.Repo.insert(
        on_conflict: {:replace, [:reliability_score, :bias_rating, :trust_tier, :confirmed_count, :rejected_count, :admin_decisions, :updated_at]},
        conflict_target: :domain
      )
    end)
  end

  # ============================================================================
  # FactDatabase Integration
  # ============================================================================

  @doc "Load learned facts from Atlas."
  def load_learned_facts do
    with {:ok, rows} <- sync(fn -> Atlas.Repo.all(Atlas.Schemas.LearnedFact) end) do
      facts =
        Enum.map(rows, fn row ->
          %Brain.FactDatabase.Fact{
            id: row.id,
            entity: row.entity,
            entity_type: row.entity_type,
            fact: row.fact,
            category: row.category,
            verification_source: row.verification_source,
            confidence: row.confidence || 0.8,
            learned_at: row.inserted_at
          }
        end)

      {:ok, facts}
    end
  end

  @doc "Persist a learned fact to Atlas."
  def persist_learned_fact(%Brain.FactDatabase.Fact{} = fact) do
    async(fn ->
      attrs = %{
        id: fact.id,
        entity: fact.entity,
        entity_type: fact.entity_type,
        fact: fact.fact,
        category: fact.category,
        confidence: fact.confidence,
        verification_source: fact.verification_source
      }

      case Atlas.Repo.get(Atlas.Schemas.LearnedFact, fact.id) do
        nil ->
          %Atlas.Schemas.LearnedFact{}
          |> Atlas.Schemas.LearnedFact.changeset(attrs)
          |> Atlas.Repo.insert()

        existing ->
          existing
          |> Atlas.Schemas.LearnedFact.changeset(attrs)
          |> Atlas.Repo.update()
      end
    end)
  end

  # ============================================================================
  # SourceAuthority Integration
  # ============================================================================

  @doc "Load source authority tracking data from Atlas."
  def load_source_authority do
    with {:ok, rows} <- sync(fn -> Atlas.Repo.all(Atlas.Schemas.SourceAuthority) end) do
      tracking =
        rows
        |> Enum.map(fn row ->
          key = safe_to_atom(row.authority_key)

          data = %{
            confirmed_count: row.confirmed_count || 0,
            contradicted_count: row.contradicted_count || 0,
            total_added: row.total_added || 0,
            credibility: row.credibility,
            last_updated: row.last_updated
          }

          {key, data}
        end)
        |> Map.new()

      {:ok, tracking}
    end
  end

  @doc "Persist source authority tracking to Atlas."
  def persist_source_authority(authority_key, tracking_data) do
    async(fn ->
      key_string = to_string(authority_key)

      attrs = %{
        authority_key: key_string,
        confirmed_count: Map.get(tracking_data, :confirmed_count, 0),
        contradicted_count: Map.get(tracking_data, :contradicted_count, 0),
        total_added: Map.get(tracking_data, :total_added, 0),
        credibility: Map.get(tracking_data, :credibility),
        last_updated: Map.get(tracking_data, :last_updated, DateTime.utc_now())
      }

      %Atlas.Schemas.SourceAuthority{}
      |> Atlas.Schemas.SourceAuthority.changeset(attrs)
      |> Atlas.Repo.insert(
        on_conflict: {:replace, [:confirmed_count, :contradicted_count, :total_added, :credibility, :last_updated, :updated_at]},
        conflict_target: :authority_key
      )
    end)
  end

  # ============================================================================
  # UserModel Integration
  # ============================================================================

  @doc "Load user models from Atlas."
  def load_user_models do
    with {:ok, rows} <- sync(fn -> Atlas.Repo.all(Atlas.Schemas.UserModel) end) do
      models =
        rows
        |> Enum.map(fn row ->
          model = %Brain.Epistemic.Types.UserModel{
            user_id: row.user_id,
            facts: atomize_keys(row.facts || %{}),
            interaction_patterns: row.interaction_patterns || %{},
            epistemic_bounds: atomize_keys(row.epistemic_bounds || %{}),
            provenance_map: atomize_keys(row.provenance_map || %{}),
            disclosure_history: row.disclosure_history || [],
            created_at: row.inserted_at,
            updated_at: row.updated_at
          }

          {row.user_id, model}
        end)
        |> Map.new()

      {:ok, models}
    end
  end

  @doc "Persist a user model to Atlas."
  def persist_user_model(%Brain.Epistemic.Types.UserModel{} = model) do
    async(fn ->
      attrs = %{
        user_id: model.user_id,
        facts: stringify_keys(model.facts || %{}),
        interaction_patterns: model.interaction_patterns || %{},
        epistemic_bounds: stringify_keys(model.epistemic_bounds || %{}),
        provenance_map: stringify_keys(model.provenance_map || %{}),
        disclosure_history: model.disclosure_history || []
      }

      %Atlas.Schemas.UserModel{}
      |> Atlas.Schemas.UserModel.changeset(attrs)
      |> Atlas.Repo.insert(
        on_conflict: {:replace, [:facts, :interaction_patterns, :epistemic_bounds, :provenance_map, :disclosure_history, :updated_at]},
        conflict_target: :user_id
      )
    end)
  end

  # ============================================================================
  # Graph Helpers
  # ============================================================================

  @doc """
  Find an existing graph node by label and name property.

  Returns `{:ok, vertex}` if found, `:not_found` otherwise.
  """
  def find_node(graph, label, name) when is_binary(graph) and is_binary(label) do
    escaped = String.replace(to_string(name), "'", "\\'")
    query = "MATCH (n:#{label}) WHERE n.name = '#{escaped}' RETURN n"

    case Atlas.Graph.cypher(graph, query) do
      {:ok, [[%Atlas.Graph.Types.Vertex{} = v] | _]} ->
        {:ok, v}

      {:ok, []} ->
        find_node_via_synonyms(graph, label, name)

      {:error, _} ->
        :not_found
    end
  rescue
    _ -> :not_found
  end

  defp find_node_via_synonyms(graph, label, name) do
    if Process.whereis(Brain.ML.Lexicon) do
      synonyms = Brain.ML.Lexicon.synonyms(to_string(name))

      Enum.find_value(Enum.take(synonyms, 5), :not_found, fn syn ->
        escaped = String.replace(syn, "'", "\\'")
        query = "MATCH (n:#{label}) WHERE n.name = '#{escaped}' RETURN n"

        case Atlas.Graph.cypher(graph, query) do
          {:ok, [[%Atlas.Graph.Types.Vertex{} = v] | _]} -> {:ok, v}
          _ -> nil
        end
      end)
    else
      :not_found
    end
  rescue
    _ -> :not_found
  end

  @doc """
  Find or create a graph node. Returns `{:ok, vertex}` always.

  If a node with matching label and name property exists, returns it.
  Otherwise creates a new node with the given properties.
  """
  def ensure_node(graph, label, properties) when is_map(properties) do
    name = Map.get(properties, :name) || Map.get(properties, "name")

    case find_node(graph, label, name) do
      {:ok, vertex} ->
        {:ok, vertex}

      :not_found ->
        enriched = enrich_with_lexicon(properties, name)
        Atlas.Graph.add_node(graph, label, enriched)
    end
  end

  defp enrich_with_lexicon(properties, name) when is_binary(name) do
    if Process.whereis(Brain.ML.Lexicon) and Brain.ML.Lexicon.known_word?(name) do
      definition =
        case Brain.ML.Lexicon.definition(name) do
          {:ok, defn} -> defn
          _ -> nil
        end

      synonyms = Brain.ML.Lexicon.synonyms(name) |> Enum.take(5)
      hypernyms = Brain.ML.Lexicon.hypernyms(name) |> Enum.take(3)

      properties
      |> maybe_put("wordnet_definition", definition)
      |> maybe_put("wordnet_synonyms", if(synonyms != [], do: Enum.join(synonyms, ", ")))
      |> maybe_put("wordnet_hypernyms", if(hypernyms != [], do: Enum.join(hypernyms, ", ")))
    else
      properties
    end
  end

  defp enrich_with_lexicon(properties, _), do: properties

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  @doc """
  Find or create an edge between two nodes.

  Checks if an edge with the given relationship type already exists
  between the two nodes. If not, creates it with the given properties.
  """
  def find_or_create_edge(graph, from_id, to_id, rel_type, properties \\ %{}) do
    query = "MATCH (a)-[r:#{rel_type}]->(b) WHERE id(a) = #{from_id} AND id(b) = #{to_id} RETURN r"

    case Atlas.Graph.cypher(graph, query) do
      {:ok, [[%Atlas.Graph.Types.Edge{} = e] | _]} ->
        {:ok, e}

      _ ->
        Atlas.Graph.add_edge(graph, from_id, to_id, rel_type, properties)
    end
  rescue
    _ -> Atlas.Graph.add_edge(graph, from_id, to_id, rel_type, properties)
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp safe_to_atom(nil), do: nil
  defp safe_to_atom(val) when is_atom(val), do: val

  defp safe_to_atom(val) when is_binary(val) do
    String.to_existing_atom(val)
  rescue
    ArgumentError -> String.to_atom(val)
  end

  defp parse_atom_or_string(val) when is_binary(val) do
    case val do
      "user" -> :user
      "world" -> :world
      "self" -> :self
      other -> other
    end
  end

  defp parse_atom_or_string(val), do: val

  defp count_decisions(decisions, type) when is_list(decisions) do
    Enum.count(decisions, fn d ->
      Map.get(d, :decision) == type or Map.get(d, "decision") == to_string(type)
    end)
  end

  defp count_decisions(_, _), do: 0

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_binary(k) -> {safe_to_atom(k), v}
      {k, v} -> {k, v}
    end)
  end

  defp atomize_keys(other), do: other

  defp stringify_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_atom(k) -> {Atom.to_string(k), v}
      {k, v} -> {to_string(k), v}
    end)
  end

  defp stringify_keys(other), do: other

  # ============================================================================
  # IntentReviewQueue Integration
  # ============================================================================

  @doc "Load all intent review candidates from Atlas."
  def load_intent_review_candidates do
    with {:ok, rows} <- sync(fn -> Atlas.Repo.all(Atlas.Schemas.IntentReviewCandidate) end) do
      candidates =
        Enum.map(rows, fn row ->
          candidate = %Brain.Analysis.Types.IntentReviewCandidate{
            id: row.id,
            text: row.text,
            timestamp: row.inserted_at,
            conversation_id: row.conversation_id,
            world_id: row.world_id,
            predicted_intent: row.predicted_intent,
            best_score: row.best_score || 0.0,
            second_score: row.second_score || 0.0,
            margin: row.margin || 0.0,
            top_k: (row.top_k || []) |> Enum.map(fn m -> {Map.get(m, "intent", ""), Map.get(m, "score", 0.0)} end),
            extracted_entities: row.extracted_entities || [],
            slot_fill_summary: row.slot_fill_summary || %{},
            annotation: atomize_annotation(row.annotation || %{}),
            status: safe_to_atom(row.status),
            reviewed_at: row.reviewed_at,
            reviewer_notes: row.reviewer_notes,
            promotion_action: safe_to_atom_or_nil(row.promotion_action),
            promoted_to_intent: row.promoted_to_intent
          }

          {row.id, candidate}
        end)

      {:ok, candidates}
    end
  end

  @doc "Persist an intent review candidate to Atlas."
  def persist_intent_review_candidate(%Brain.Analysis.Types.IntentReviewCandidate{} = candidate) do
    async(fn ->
      top_k_maps = Enum.map(candidate.top_k || [], fn
        {intent, score} -> %{"intent" => intent, "score" => score}
        other -> other
      end)

      annotation_map = stringify_annotation(candidate.annotation)

      attrs = %{
        id: candidate.id,
        text: candidate.text,
        status: to_string(candidate.status),
        predicted_intent: candidate.predicted_intent,
        best_score: candidate.best_score,
        second_score: candidate.second_score,
        margin: candidate.margin,
        top_k: top_k_maps,
        extracted_entities: candidate.extracted_entities || [],
        slot_fill_summary: candidate.slot_fill_summary || %{},
        annotation: annotation_map,
        conversation_id: candidate.conversation_id,
        world_id: candidate.world_id,
        reviewer_notes: candidate.reviewer_notes,
        reviewed_at: candidate.reviewed_at,
        promotion_action: if(candidate.promotion_action, do: to_string(candidate.promotion_action)),
        promoted_to_intent: candidate.promoted_to_intent
      }

      %Atlas.Schemas.IntentReviewCandidate{}
      |> Atlas.Schemas.IntentReviewCandidate.changeset(attrs)
      |> Atlas.Repo.insert(
        on_conflict: {:replace, [:status, :predicted_intent, :best_score, :second_score, :margin, :top_k, :extracted_entities, :slot_fill_summary, :annotation, :reviewer_notes, :reviewed_at, :promotion_action, :promoted_to_intent, :updated_at]},
        conflict_target: :id
      )
    end)
  end

  defp safe_to_atom_or_nil(nil), do: nil
  defp safe_to_atom_or_nil(""), do: nil
  defp safe_to_atom_or_nil(val) when is_atom(val), do: val

  defp safe_to_atom_or_nil(val) when is_binary(val) do
    String.to_existing_atom(val)
  rescue
    ArgumentError -> nil
  end

  defp atomize_annotation(annotation) when is_map(annotation) do
    tags = Map.get(annotation, "tags", Map.get(annotation, :tags, []))
    tags = Enum.map(tags, fn t -> if is_binary(t), do: String.to_atom(t), else: t end)

    %{
      tags: tags,
      notes: Map.get(annotation, "notes", Map.get(annotation, :notes)),
      domain_guess: Map.get(annotation, "domain_guess", Map.get(annotation, :domain_guess)),
      spans: Map.get(annotation, "spans", Map.get(annotation, :spans, []))
    }
  end

  defp atomize_annotation(_), do: %{tags: [], notes: nil, domain_guess: nil, spans: []}

  defp stringify_annotation(annotation) when is_map(annotation) do
    tags = Map.get(annotation, :tags, Map.get(annotation, "tags", []))
    tags = Enum.map(tags, &to_string/1)

    %{
      "tags" => tags,
      "notes" => Map.get(annotation, :notes, Map.get(annotation, "notes")),
      "domain_guess" => Map.get(annotation, :domain_guess, Map.get(annotation, "domain_guess")),
      "spans" => Map.get(annotation, :spans, Map.get(annotation, "spans", []))
    }
  end

  defp stringify_annotation(_), do: %{"tags" => [], "notes" => nil, "domain_guess" => nil, "spans" => []}
end
