defmodule Atlas.Importer do
  @moduledoc """
  Imports file-based data into PostgreSQL via Atlas schemas.

  Reads from .term, .json, and .enc files and bulk-inserts into the
  corresponding Atlas tables. Idempotent: uses on_conflict: :nothing
  where unique constraints exist; skips stores with existing data
  unless :force is set.
  """

  alias Atlas.Repo
  alias Atlas.Schemas.{
    Belief,
    Credential,
    Episode,
    LearnedFact,
    ReviewCandidate,
    SemanticFact,
    SourceReliability,
    SourceAuthority,
    UserModel,
    KnowledgeEntry,
    PersonaMemory
  }

  require Logger

  @store_names ~w(beliefs credentials episodes semantic_facts review_candidates learned_facts source_reliability source_authority user_models knowledge persona_memories)

  @doc "Import all stores. Opts: :only (list of store names), :dry_run (bool), :force (bool), :quiet (bool)."
  def import_all(opts \\ []) do
    only = Keyword.get(opts, :only)
    dry_run = Keyword.get(opts, :dry_run, false)
    force = Keyword.get(opts, :force, false)
    quiet = Keyword.get(opts, :quiet, false)

    stores =
      if only do
        only_list =
          if is_binary(only),
            do: String.split(only, ",") |> Enum.map(&String.trim/1),
            else: only

        Enum.filter(@store_names, &(&1 in only_list))
      else
        @store_names
      end

    unless quiet do
      Logger.info("Atlas.Importer starting", stores: stores, dry_run: dry_run, force: force)
    end

    results =
      Enum.map(stores, fn store ->
        result = import_store(store, dry_run: dry_run, force: force, quiet: quiet)
        {store, result}
      end)

    summary = Enum.into(results, %{})
    unless quiet, do: Logger.info("Atlas.Importer complete", summary: summary)
    {:ok, summary}
  end

  defp import_store("beliefs", opts), do: import_beliefs(opts)
  defp import_store("credentials", opts), do: import_credentials(opts)
  defp import_store("episodes", opts), do: import_episodes(opts)
  defp import_store("semantic_facts", opts), do: import_semantic_facts(opts)
  defp import_store("review_candidates", opts), do: import_review_candidates(opts)
  defp import_store("learned_facts", opts), do: import_learned_facts(opts)
  defp import_store("source_reliability", opts), do: import_source_reliability(opts)
  defp import_store("source_authority", opts), do: import_source_authority(opts)
  defp import_store("user_models", opts), do: import_user_models(opts)
  defp import_store("knowledge", opts), do: import_knowledge(opts)
  defp import_store("persona_memories", opts), do: import_persona_memories(opts)
  defp import_store(unknown, _opts), do: {:error, "Unknown store: #{unknown}"}

  # ============================================================================
  # Beliefs (belief_store.term)
  #
  # File shape (persisted by Brain.Epistemic.BeliefStore):
  #   %{
  #     beliefs: %{id => %Brain.Epistemic.Types.Belief{...}},
  #     by_user: %{...},
  #     by_subject: %{...},
  #     by_predicate: %{...},
  #     retracted: MapSet.t()
  #   }
  # ============================================================================
  defp import_beliefs(opts) do
    with_term_file(belief_store_path(), Belief, opts, fn data ->
      beliefs = data |> g(:beliefs, %{}) |> Map.values()
      retracted = g(data, :retracted, MapSet.new())

      rows =
        beliefs
        |> Enum.reject(fn b -> MapSet.member?(retracted, g(b, :id)) end)
        |> Enum.map(fn b ->
          %{
            subject: to_string(g(b, :subject)),
            predicate: to_string(g(b, :predicate)),
            object: to_string(g(b, :object)),
            confidence: g(b, :confidence) || 0.5,
            source: to_string(g(b, :source)),
            source_authority: safe_to_string(g(b, :source_authority)),
            user_id: safe_to_string(g(b, :user_id)),
            node_id: safe_to_string(g(b, :node_id)),
            retracted: false,
            last_confirmed: g(b, :last_confirmed),
            provenance: ensure_map(g(b, :provenance)),
            metadata: ensure_map(g(b, :metadata))
          }
        end)

      insert_rows(Belief, rows, opts, "beliefs")
    end)
  end

  defp belief_store_path do
    Application.get_env(:brain, :belief_store_path) ||
      Path.join(brain_priv(), "data/belief_store.term")
  end

  # ============================================================================
  # Credentials (credentials.enc)
  # ============================================================================
  defp import_credentials(opts) do
    path = credential_path()

    cond do
      !File.exists?(path) ->
        skip("file not found: #{path}", opts)

      !opts[:force] and Repo.aggregate(Credential, :count) > 0 ->
        skip("table has data", opts)

      true ->
        case File.read(path) do
          {:ok, encrypted} ->
            key = credential_encryption_key()

            case Plug.Crypto.decrypt(key, "credential_vault", encrypted, max_age: :infinity) do
              {:ok, serialized} ->
                entries = binary_to_term_safe(serialized)

                rows =
                  Enum.map(entries, fn {{world, service, k}, encrypted_value} ->
                    %{
                      world: to_string(world),
                      service: to_string(service),
                      key: to_string(k),
                      encrypted_value: encrypted_value
                    }
                  end)

                insert_rows(Credential, rows, opts, "credentials")

              {:error, reason} ->
                {:error, "decrypt failed: #{inspect(reason)}"}
            end

          {:error, reason} ->
            {:error, "could not read #{path}: #{inspect(reason)}"}
        end
    end
  end

  defp credential_encryption_key do
    case Application.get_env(:brain, :credential_encryption_key) do
      nil ->
        endpoint_config = get_chat_web_endpoint_config()

        case endpoint_config do
          nil ->
            :crypto.strong_rand_bytes(32)

          config ->
            secret = Keyword.get(config, :secret_key_base, "")

            if byte_size(secret) >= 32,
              do: :crypto.hash(:sha256, secret),
              else: :crypto.strong_rand_bytes(32)
        end

      key when is_binary(key) and byte_size(key) >= 32 ->
        :crypto.hash(:sha256, key)

      key when is_binary(key) ->
        :crypto.hash(:sha256, key <> String.duplicate("0", 32))
    end
  rescue
    _ -> :crypto.strong_rand_bytes(32)
  end

  defp get_chat_web_endpoint_config do
    if Code.ensure_loaded?(ChatWeb.Endpoint) do
      Application.get_env(:chat_web, ChatWeb.Endpoint)
    else
      nil
    end
  rescue
    _ -> nil
  end

  defp credential_path do
    base = Application.get_env(:brain, :secrets_path) || Path.join(brain_priv(), "secrets")
    Path.join(base, "credentials.enc")
  end

  # ============================================================================
  # Episodes (memory_store.term)
  #
  # File shape (persisted by Brain.Memory.Store):
  #   %{
  #     episodes: %{world_id => %{id => %Brain.Memory.Types.Episode{...}}},
  #     semantics: %{world_id => %{id => %Brain.Memory.Types.SemanticFact{...}}}
  #   }
  # Episode struct has: id, timestamp, state, action, outcome, tags, embedding, semantic_id
  # ============================================================================
  defp import_episodes(opts) do
    with_term_file(memory_store_path(), Episode, opts, fn data ->
      episodes_map = normalize_world_map(g(data, :episodes, %{}))

      rows =
        Enum.flat_map(episodes_map, fn {world_id, world_episodes} ->
          Enum.map(world_episodes, fn {_id, ep} ->
            %{
              world_id: world_id,
              state: to_string(g(ep, :state) || ""),
              action: to_string(g(ep, :action) || ""),
              outcome: to_string(g(ep, :outcome) || ""),
              tags: g(ep, :tags) || [],
              embedding: g(ep, :embedding) || [],
              semantic_id: normalize_uuid(g(ep, :semantic_id))
            }
          end)
        end)

      insert_rows(Episode, rows, opts, "episodes")
    end)
  end

  # ============================================================================
  # Semantic Facts (memory_store.term)
  #
  # SemanticFact struct has: id, timestamp, representation, embedding, evidence_ids, tags
  # ============================================================================
  defp import_semantic_facts(opts) do
    with_term_file(memory_store_path(), SemanticFact, opts, fn data ->
      semantics_map = normalize_world_map(g(data, :semantics, %{}))

      rows =
        Enum.flat_map(semantics_map, fn {world_id, world_sems} ->
          Enum.map(world_sems, fn {_id, sem} ->
            category = (g(sem, :tags) || []) |> List.first() || "general"

            %{
              world_id: world_id,
              content: to_string(g(sem, :representation) || ""),
              category: category,
              confidence: 0.8,
              embedding: g(sem, :embedding) || [],
              source_episodes: g(sem, :evidence_ids) || []
            }
          end)
        end)

      insert_rows(SemanticFact, rows, opts, "semantic_facts")
    end)
  end

  defp memory_store_path do
    Application.get_env(:brain, :memory_store_path) ||
      Path.join(brain_priv(), "data/memory_store.term")
  end

  # If the first value is a struct (flat episode map), wrap in "default" world
  defp normalize_world_map(map) when is_map(map) do
    case Map.values(map) |> List.first() do
      nil -> %{}
      v when is_struct(v) -> %{"default" => map}
      v when is_map(v) and not is_struct(v) -> map
      _ -> %{"default" => map}
    end
  end

  defp normalize_world_map(_), do: %{}

  # ============================================================================
  # Review Candidates (review_queue.term)
  #
  # File shape: %{candidates: [{id, candidate_map_or_struct}, ...], stats: ...}
  # ============================================================================
  defp import_review_candidates(opts) do
    with_term_file(review_queue_path(), ReviewCandidate, opts, fn data ->
      candidates = g(data, :candidates, [])

      rows =
        Enum.map(candidates, fn {id, cand} ->
          %{
            id: id,
            status: to_string(g(cand, :status) || :pending),
            finding: ensure_map(g(cand, :finding)),
            aggregate_confidence: g(cand, :aggregate_confidence) || 0.5,
            corroborating_sources: deep_to_maps(g(cand, :corroborating_sources) || []),
            conflicting_findings: deep_to_maps(g(cand, :conflicting_findings) || []),
            existing_contradictions: deep_to_maps(g(cand, :existing_contradictions) || []),
            reviewer_notes: safe_to_string(g(cand, :reviewer_notes)),
            reviewed_at: g(cand, :reviewed_at)
          }
        end)

      insert_rows(ReviewCandidate, rows, opts, "review_candidates")
    end)
  end

  defp review_queue_path do
    Application.get_env(:brain, :review_queue_path) ||
      Path.join(brain_priv(), "data/review_queue.term")
  end

  # ============================================================================
  # Learned Facts (data/facts/*.json)
  #
  # Each JSON: %{"facts" => [%{"id" => ..., "entity" => ..., ...}]}
  # ============================================================================
  defp import_learned_facts(opts) do
    facts_glob = facts_glob_path()
    files = Path.wildcard(facts_glob)

    cond do
      files == [] ->
        skip("no fact files at #{facts_glob}", opts)

      !opts[:force] and Repo.aggregate(LearnedFact, :count) > 0 ->
        skip("table has data", opts)

      true ->
        all_facts =
          Enum.flat_map(files, fn path ->
            with {:ok, content} <- File.read(path),
                 {:ok, data} <- Jason.decode(content) do
              Map.get(data, "facts", [])
            else
              _ -> []
            end
          end)

        rows =
          Enum.map(all_facts, fn f ->
            %{
              id: f["id"] || "fact_#{System.unique_integer([:positive])}",
              entity: f["entity"] || "unknown",
              entity_type: f["entity_type"],
              fact: f["fact"] || "",
              category: f["category"] || "learned",
              confidence: f["confidence"] || 1.0,
              verification_source: f["verification_source"]
            }
          end)

        insert_rows(LearnedFact, rows, opts, "learned_facts")
    end
  end

  defp facts_glob_path do
    dir = Application.get_env(:brain, :facts_dir)
    if dir, do: Path.join(dir, "*.json"), else: Path.join(File.cwd!(), "data/facts/*.json")
  end

  # ============================================================================
  # Source Reliability (source_reliability_learned.term)
  #
  # File shape: %{sources: %{domain => %SourceProfile{domain, factual_accuracy,
  #   bias_rating, trust_tier, notes, admin_decisions, last_updated}}}
  # ============================================================================
  defp import_source_reliability(opts) do
    with_term_file(source_reliability_path(), SourceReliability, opts, fn data ->
      sources = g(data, :sources, %{})

      rows =
        Enum.map(sources, fn {domain, profile} ->
          decisions = g(profile, :admin_decisions) || []

          %{
            domain: to_string(domain),
            reliability_score: g(profile, :factual_accuracy) || 0.5,
            bias_rating: safe_to_string(g(profile, :bias_rating)),
            trust_tier: safe_to_string(g(profile, :trust_tier)),
            confirmed_count: count_decisions(decisions, :approved),
            rejected_count: count_decisions(decisions, :rejected),
            admin_decisions: serialize_decisions(decisions),
            metadata: %{}
          }
        end)

      insert_rows(SourceReliability, rows, opts, "source_reliability")
    end)
  end

  defp count_decisions(decisions, type) when is_list(decisions) do
    Enum.count(decisions, fn d ->
      g(d, :decision) == type or g(d, "decision") == to_string(type)
    end)
  end

  defp count_decisions(_, _), do: 0

  defp serialize_decisions(decisions) when is_list(decisions) do
    Enum.map(decisions, fn d ->
      if is_map(d) do
        d
        |> Map.take([
          :decision,
          :timestamp,
          :candidate_id,
          :notes,
          "decision",
          "timestamp",
          "candidate_id",
          "notes"
        ])
        |> Enum.into(%{}, fn
          {k, v} when is_atom(k) -> {to_string(k), to_string(v)}
          {k, v} -> {k, v}
        end)
      else
        %{}
      end
    end)
  end

  defp serialize_decisions(_), do: []

  defp source_reliability_path do
    Application.get_env(:brain, :source_reliability_path) ||
      Path.join(brain_priv(), "data/source_reliability_learned.term")
  end

  # ============================================================================
  # Source Authority (source_authority_learned.term)
  #
  # File shape: %{version: ..., last_updated: ..., tracking: %{key => %{
  #   confirmed_count, contradicted_count, total_added, credibility, ...}}}
  # ============================================================================
  defp import_source_authority(opts) do
    with_term_file(source_authority_path(), SourceAuthority, opts, fn data ->
      tracking = g(data, :tracking, %{})

      rows =
        Enum.map(tracking, fn {key, t} ->
          %{
            authority_key: to_string(key),
            confirmed_count: g(t, :confirmed_count) || 0,
            contradicted_count: g(t, :contradicted_count) || 0,
            total_added: g(t, :total_added) || 0,
            credibility: g(t, :credibility),
            last_updated: g(t, :last_updated),
            metadata: %{}
          }
        end)

      insert_rows(SourceAuthority, rows, opts, "source_authority")
    end)
  end

  defp source_authority_path do
    Application.get_env(:brain, :source_authority_path) ||
      Path.join(brain_priv(), "data/source_authority_learned.term")
  end

  # ============================================================================
  # User Models (user_models.term)
  #
  # File shape: %{user_id => %Brain.Epistemic.Types.UserModel{...}}
  # UserModel struct: user_id, facts, interaction_patterns, epistemic_bounds,
  #                   provenance_map, disclosure_history, created_at, updated_at
  # ============================================================================
  defp import_user_models(opts) do
    with_term_file(user_models_path(), UserModel, opts, fn models ->
      rows =
        Enum.map(models, fn {user_id, model} ->
          %{
            user_id: to_string(user_id),
            facts: ensure_map(g(model, :facts)),
            interaction_patterns: ensure_map(g(model, :interaction_patterns)),
            epistemic_bounds: ensure_map(g(model, :epistemic_bounds)),
            provenance_map: ensure_map(g(model, :provenance_map)),
            disclosure_history: deep_to_maps(g(model, :disclosure_history) || [])
          }
        end)

      insert_rows(UserModel, rows, opts, "user_models")
    end)
  end

  defp user_models_path do
    Application.get_env(:brain, :user_models_path) ||
      Path.join(brain_priv(), "data/user_models.term")
  end

  # ============================================================================
  # Knowledge (knowledge/*.json per persona)
  #
  # Per-persona JSON with category keys ("people", "pets", "rooms", etc.)
  # ============================================================================
  defp import_knowledge(opts) do
    dir = knowledge_dir()

    cond do
      !File.dir?(dir) ->
        skip("knowledge dir not found: #{dir}", opts)

      !opts[:force] and Repo.aggregate(KnowledgeEntry, :count) > 0 ->
        skip("table has data", opts)

      true ->
        files = Path.wildcard(Path.join(dir, "*.json"))

        rows =
          Enum.flat_map(files, fn path ->
            persona = path |> Path.basename() |> Path.rootname()

            with {:ok, content} <- File.read(path),
                 {:ok, data} when is_map(data) <- Jason.decode(content) do
              flatten_knowledge(data, persona, "default")
            else
              _ -> []
            end
          end)

        insert_rows(KnowledgeEntry, rows, opts, "knowledge")
    end
  end

  defp flatten_knowledge(data, persona_name, world_id) do
    Enum.flat_map(data, fn
      {category, m} when is_map(m) ->
        Enum.map(m, fn {key, val} ->
          %{
            world_id: world_id,
            persona_name: persona_name,
            category: category,
            key: key,
            data: ensure_map(val)
          }
        end)

      {category, list} when is_list(list) ->
        list
        |> Enum.with_index()
        |> Enum.map(fn {item, i} ->
          %{
            world_id: world_id,
            persona_name: persona_name,
            category: category,
            key: "#{category}_#{i}",
            data: ensure_map(item)
          }
        end)

      _ ->
        []
    end)
  end

  defp knowledge_dir do
    Application.get_env(:brain, :knowledge_dir) || Path.join(brain_priv(), "knowledge")
  end

  # ============================================================================
  # Persona Memories (memory/*.json per persona)
  #
  # Per-persona JSON arrays of message maps:
  #   [%{"role" => "user", "text" => "...", "timestamp" => ...}, ...]
  # ============================================================================
  defp import_persona_memories(opts) do
    dir = memory_dir()

    cond do
      !File.dir?(dir) ->
        skip("memory dir not found: #{dir}", opts)

      !opts[:force] and Repo.aggregate(PersonaMemory, :count) > 0 ->
        skip("table has data", opts)

      true ->
        files = Path.wildcard(Path.join(dir, "*.json"))

        rows =
          Enum.flat_map(files, fn path ->
            persona = path |> Path.basename() |> Path.rootname()

            with {:ok, content} <- File.read(path),
                 {:ok, data} when is_list(data) <- Jason.decode(content) do
              Enum.map(data, fn msg ->
                %{
                  persona_name: persona,
                  role: msg["role"] || "unknown",
                  content: msg["text"] || msg["content"] || "",
                  context: %{"tags" => msg["tags"] || []},
                  message_timestamp: msg["timestamp"]
                }
              end)
            else
              _ -> []
            end
          end)

        insert_rows(PersonaMemory, rows, opts, "persona_memories")
    end
  end

  defp memory_dir do
    Application.get_env(:brain, :memory_dir) || Path.join(brain_priv(), "memory")
  end

  # ============================================================================
  # Shared helpers
  # ============================================================================

  # Generic pattern for .term file stores:
  # 1. Check file exists
  # 2. Check table empty (unless force)
  # 3. Read & deserialize
  # 4. Call callback with deserialized data
  defp with_term_file(path, schema, opts, callback) do
    cond do
      !File.exists?(path) ->
        skip("file not found: #{path}", opts)

      !opts[:force] and Repo.aggregate(schema, :count) > 0 ->
        skip("table has data", opts)

      true ->
        case File.read(path) do
          {:ok, binary} ->
            data = binary_to_term_safe(binary)
            callback.(data)

          {:error, reason} ->
            {:error, "could not read #{path}: #{inspect(reason)}"}
        end
    end
  end

  # Insert a list of attr maps via changeset. Returns {:ok, count}.
  defp insert_rows(_schema, rows, opts, label) when rows == [] do
    dry_run_log(label, 0, opts)
    {:ok, 0}
  end

  defp insert_rows(schema, rows, opts, label) do
    if opts[:dry_run] do
      dry_run_log(label, length(rows), opts)
      {:ok, length(rows)}
    else
      inserted =
        Enum.count(rows, fn attrs ->
          changeset = schema.changeset(struct(schema), attrs)

          case Repo.insert(changeset, on_conflict: :nothing) do
            {:ok, _} ->
              true

            {:error, cs} ->
              unless opts[:quiet] do
                errors =
                  Ecto.Changeset.traverse_errors(cs, fn {msg, o} ->
                    Enum.reduce(o, msg, fn {k, v}, acc ->
                      String.replace(acc, "%{#{k}}", to_string(v))
                    end)
                  end)

                Logger.warning("Atlas.Importer #{label} insert failed",
                  errors: inspect(errors),
                  attrs: inspect(Map.take(attrs, [:id, :subject, :entity, :domain, :user_id, :persona_name]))
                )
              end

              false
          end
        end)

      {:ok, inserted}
    end
  end

  # Safe map/struct field access — works with both atom and string keys,
  # and with structs (where Map.get/2 works on atom keys).
  defp g(map, key, default \\ nil)
  defp g(nil, _key, default), do: default

  defp g(map, key, default) when is_struct(map) and is_atom(key) do
    Map.get(map, key, default)
  end

  defp g(map, key, default) when is_map(map) and is_atom(key) do
    case Map.get(map, key) do
      nil -> Map.get(map, to_string(key), default)
      val -> val
    end
  end

  defp g(map, key, default) when is_map(map) and is_binary(key) do
    case Map.get(map, key) do
      nil ->
        try do
          Map.get(map, String.to_existing_atom(key), default)
        rescue
          ArgumentError -> default
        end

      val ->
        val
    end
  end

  defp ensure_map(v) when is_struct(v), do: Map.from_struct(v) |> deep_stringify()
  defp ensure_map(v) when is_map(v), do: deep_stringify(v)
  defp ensure_map(nil), do: %{}
  defp ensure_map([]), do: %{}
  defp ensure_map(v), do: %{"value" => v}

  defp safe_to_string(nil), do: nil
  defp safe_to_string(v), do: to_string(v)

  # Converts a 32-char hex string to UUID format (8-4-4-4-12) or returns
  # the value unchanged if it already looks like a UUID. Returns nil for
  # anything that isn't a valid hex/UUID string.
  defp normalize_uuid(nil), do: nil

  defp normalize_uuid(<<a::binary-size(8), b::binary-size(4), c::binary-size(4),
                        d::binary-size(4), e::binary-size(12)>>)
       when byte_size(a) == 8 do
    candidate = "#{a}-#{b}-#{c}-#{d}-#{e}"
    if String.match?(candidate, ~r/\A[0-9a-fA-F-]{36}\z/), do: candidate, else: nil
  end

  defp normalize_uuid(<<_::binary-size(36)>> = val) do
    if String.match?(val, ~r/\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\z/),
      do: val,
      else: nil
  end

  defp normalize_uuid(_), do: nil

  defp deep_to_maps(list) when is_list(list), do: Enum.map(list, &ensure_map/1)
  defp deep_to_maps(_), do: []

  # Recursively stringify atom keys in maps (Ecto JSONB needs string keys)
  defp deep_stringify(map) when is_struct(map), do: map |> Map.from_struct() |> deep_stringify()

  defp deep_stringify(map) when is_map(map) do
    Enum.into(map, %{}, fn
      {k, v} when is_atom(k) -> {to_string(k), deep_stringify(v)}
      {k, v} -> {k, deep_stringify(v)}
    end)
  end

  defp deep_stringify(list) when is_list(list), do: Enum.map(list, &deep_stringify/1)
  defp deep_stringify(v) when is_atom(v) and not is_nil(v) and not is_boolean(v), do: to_string(v)
  defp deep_stringify(v), do: v

  defp brain_priv do
    source_priv = Path.join(File.cwd!(), "apps/brain/priv")
    if File.dir?(source_priv), do: source_priv, else: Application.app_dir(:brain, "priv")
  rescue
    _ -> Path.join(File.cwd!(), "apps/brain/priv")
  end

  # Try :safe first; fall back to unrestricted for trusted project data with unknown atoms
  defp binary_to_term_safe(binary) do
    :erlang.binary_to_term(binary, [:safe])
  rescue
    ArgumentError -> :erlang.binary_to_term(binary)
  end

  defp skip(reason, opts) do
    unless opts[:quiet], do: Logger.info("Atlas.Importer skip", reason: reason)
    {:skip, reason}
  end

  defp dry_run_log(label, count, opts) do
    unless opts[:quiet], do: Logger.info("Atlas.Importer dry-run #{label}", count: count)
  end
end
