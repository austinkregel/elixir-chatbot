defmodule Atlas.ImporterTest do
  use Atlas.DataCase, async: false

  alias Atlas.Repo
  alias Atlas.Schemas.{
    Belief,
    Episode,
    SemanticFact,
    ReviewCandidate,
    LearnedFact,
    SourceReliability,
    SourceAuthority,
    UserModel,
    KnowledgeEntry,
    PersonaMemory
  }

  alias Atlas.Importer

  setup do
    tmp = Path.join(System.tmp_dir!(), "atlas_importer_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp)
    data_dir = Path.join(tmp, "data")
    File.mkdir_p!(data_dir)
    knowledge_dir = Path.join(tmp, "knowledge")
    File.mkdir_p!(knowledge_dir)
    memory_dir = Path.join(tmp, "memory")
    File.mkdir_p!(memory_dir)
    facts_dir = Path.join(tmp, "facts")
    File.mkdir_p!(facts_dir)
    secrets_dir = Path.join(tmp, "secrets")
    File.mkdir_p!(secrets_dir)

    write_all_fixtures(data_dir, knowledge_dir, memory_dir, facts_dir)

    Application.put_env(:brain, :belief_store_path, Path.join(data_dir, "belief_store.term"))
    Application.put_env(:brain, :memory_store_path, Path.join(data_dir, "memory_store.term"))
    Application.put_env(:brain, :review_queue_path, Path.join(data_dir, "review_queue.term"))
    Application.put_env(:brain, :source_reliability_path, Path.join(data_dir, "source_reliability_learned.term"))
    Application.put_env(:brain, :source_authority_path, Path.join(data_dir, "source_authority_learned.term"))
    Application.put_env(:brain, :user_models_path, Path.join(data_dir, "user_models.term"))
    Application.put_env(:brain, :knowledge_dir, knowledge_dir)
    Application.put_env(:brain, :memory_dir, memory_dir)
    Application.put_env(:brain, :facts_dir, facts_dir)
    Application.put_env(:brain, :secrets_path, secrets_dir)

    on_exit(fn ->
      for key <- ~w(belief_store_path memory_store_path review_queue_path
                     source_reliability_path source_authority_path user_models_path
                     knowledge_dir memory_dir facts_dir secrets_path)a do
        Application.delete_env(:brain, key)
      end

      File.rm_rf!(tmp)
    end)

    %{tmp: tmp, data_dir: data_dir}
  end

  # ── Beliefs ──────────────────────────────────────────────────────────────────

  describe "beliefs import" do
    test "imports beliefs from .term fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["beliefs"])
      assert {:ok, 2} = summary["beliefs"]
      assert Repo.aggregate(Belief, :count) == 2
    end

    test "skips retracted beliefs" do
      assert {:ok, _} = Importer.import_all(only: ["beliefs"])
      beliefs = Repo.all(Belief)
      refute Enum.any?(beliefs, &(&1.subject == "retracted_subject"))
    end
  end

  # ── Episodes ─────────────────────────────────────────────────────────────────

  describe "episodes import" do
    test "imports episodes from .term fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["episodes"])
      assert {:ok, 1} = summary["episodes"]
      assert Repo.aggregate(Episode, :count) == 1
    end
  end

  # ── Semantic Facts ───────────────────────────────────────────────────────────

  describe "semantic_facts import" do
    test "imports semantic facts from .term fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["semantic_facts"])
      assert {:ok, 1} = summary["semantic_facts"]
      assert Repo.aggregate(SemanticFact, :count) == 1
    end
  end

  # ── Review Candidates ───────────────────────────────────────────────────────

  describe "review_candidates import" do
    test "imports review candidates from .term fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["review_candidates"])
      assert {:ok, 1} = summary["review_candidates"]
      assert Repo.aggregate(ReviewCandidate, :count) == 1
    end
  end

  # ── Learned Facts ───────────────────────────────────────────────────────────

  describe "learned_facts import" do
    test "imports learned facts from JSON fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["learned_facts"])
      assert {:ok, 1} = summary["learned_facts"]
      assert Repo.aggregate(LearnedFact, :count) == 1
    end
  end

  # ── Source Reliability ──────────────────────────────────────────────────────

  describe "source_reliability import" do
    test "imports source reliability from .term fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["source_reliability"])
      assert {:ok, 1} = summary["source_reliability"]
      assert Repo.aggregate(SourceReliability, :count) == 1

      [sr] = Repo.all(SourceReliability)
      assert sr.domain == "example.com"
      assert sr.reliability_score == 0.8
    end
  end

  # ── Source Authority ────────────────────────────────────────────────────────

  describe "source_authority import" do
    test "imports source authority from .term fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["source_authority"])
      assert {:ok, 1} = summary["source_authority"]
      assert Repo.aggregate(SourceAuthority, :count) == 1

      [sa] = Repo.all(SourceAuthority)
      assert sa.authority_key == "source_1"
      assert sa.credibility == 0.9
    end
  end

  # ── User Models ─────────────────────────────────────────────────────────────

  describe "user_models import" do
    test "imports user models from .term fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["user_models"])
      assert {:ok, 1} = summary["user_models"]
      assert Repo.aggregate(UserModel, :count) == 1

      [um] = Repo.all(UserModel)
      assert um.user_id == "user_1"
    end
  end

  # ── Knowledge ───────────────────────────────────────────────────────────────

  describe "knowledge import" do
    test "imports knowledge entries from JSON fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["knowledge"])
      assert {:ok, 2} = summary["knowledge"]
      assert Repo.aggregate(KnowledgeEntry, :count) == 2
    end
  end

  # ── Persona Memories ────────────────────────────────────────────────────────

  describe "persona_memories import" do
    test "imports persona memories from JSON fixture" do
      assert {:ok, summary} = Importer.import_all(only: ["persona_memories"])
      assert {:ok, 1} = summary["persona_memories"]
      assert Repo.aggregate(PersonaMemory, :count) == 1
    end
  end

  # ── Cross-cutting behaviour ─────────────────────────────────────────────────

  describe "cross-cutting" do
    test "dry-run reports counts without inserting" do
      assert {:ok, summary} = Importer.import_all(only: ["knowledge"], dry_run: true)
      assert {:ok, 2} = summary["knowledge"]
      assert Repo.aggregate(KnowledgeEntry, :count) == 0
    end

    test "idempotency: running twice does not double counts" do
      assert {:ok, _} = Importer.import_all(only: ["knowledge"])
      assert Repo.aggregate(KnowledgeEntry, :count) == 2

      assert {:ok, _} = Importer.import_all(only: ["knowledge"], force: true)
      assert Repo.aggregate(KnowledgeEntry, :count) == 2
    end

    test "skips store when table has data and force is false" do
      %KnowledgeEntry{}
      |> KnowledgeEntry.changeset(%{persona_name: "pre", category: "people", key: "pre", data: %{}})
      |> Repo.insert!()

      assert {:ok, summary} = Importer.import_all(only: ["knowledge"])
      assert {:skip, "table has data"} = summary["knowledge"]
      assert Repo.aggregate(KnowledgeEntry, :count) == 1
    end

    test "force overrides table-has-data skip" do
      %KnowledgeEntry{}
      |> KnowledgeEntry.changeset(%{persona_name: "pre", category: "people", key: "pre", data: %{}})
      |> Repo.insert!()

      assert {:ok, summary} = Importer.import_all(only: ["knowledge"], force: true)
      assert {:ok, 2} = summary["knowledge"]
      assert Repo.aggregate(KnowledgeEntry, :count) >= 2
    end

    test "skip when file not found" do
      Application.put_env(:brain, :belief_store_path, "/nonexistent/belief_store.term")

      assert {:ok, summary} = Importer.import_all(only: ["beliefs"])
      assert {:skip, "file not found: " <> _} = summary["beliefs"]
    end
  end

  # ── Fixture generation ──────────────────────────────────────────────────────

  defp write_all_fixtures(data_dir, knowledge_dir, memory_dir, facts_dir) do
    # -- Belief store --
    # Matches Brain.Epistemic.BeliefStore persist_to_disk format:
    # %{beliefs: %{id => %Belief{}}, retracted: MapSet}
    # We use plain maps here since the importer uses generic map access.
    belief_1 = %{
      id: "belief_1",
      subject: :world,
      predicate: :capital,
      object: "Paris is the capital of France",
      confidence: 0.95,
      source: :learned,
      source_authority: :academic_expert,
      user_id: nil,
      node_id: nil,
      last_confirmed: nil,
      provenance: ["wikipedia"],
      volatility: 0.3,
      metadata: %{topic: "geography"},
      created_at: ~U[2026-01-01 00:00:00Z]
    }

    belief_2 = %{
      id: "belief_2",
      subject: :user,
      predicate: :name,
      object: "Alice",
      confidence: 0.9,
      source: :explicit,
      source_authority: nil,
      user_id: "u1",
      node_id: "n1",
      last_confirmed: ~U[2026-02-01 00:00:00Z],
      provenance: [],
      volatility: 0.5,
      metadata: %{},
      created_at: ~U[2026-01-15 00:00:00Z]
    }

    retracted_belief = %{
      id: "belief_retracted",
      subject: :world,
      predicate: :wrong,
      object: "wrong fact",
      confidence: 0.3,
      source: :inferred,
      source_authority: nil,
      user_id: nil,
      node_id: nil,
      last_confirmed: nil,
      provenance: [],
      volatility: 0.5,
      metadata: %{},
      created_at: ~U[2026-01-01 00:00:00Z]
    }

    belief_data = %{
      beliefs: %{
        "belief_1" => belief_1,
        "belief_2" => belief_2,
        "belief_retracted" => retracted_belief
      },
      by_user: %{},
      by_subject: %{},
      by_predicate: %{},
      retracted: MapSet.new(["belief_retracted"])
    }

    write_term(data_dir, "belief_store.term", belief_data)

    # -- Memory store (episodes + semantic facts) --
    episode = %{
      id: "ep1",
      timestamp: 1_000_000,
      state: "user asked about weather",
      action: "weather.query",
      outcome: "provided forecast",
      tags: ["weather"],
      embedding: [0.1, 0.2],
      semantic_id: nil
    }

    semantic = %{
      id: "sem1",
      timestamp: 1_000_001,
      representation: "Weather queries are common in the afternoon",
      embedding: [0.3, 0.4],
      evidence_ids: ["ep1"],
      tags: ["patterns"]
    }

    memory_data = %{
      episodes: %{"default" => %{"ep1" => episode}},
      semantics: %{"default" => %{"sem1" => semantic}}
    }

    write_term(data_dir, "memory_store.term", memory_data)

    # -- Review queue --
    review_data = %{
      candidates: [
        {"rev1", %{
          status: :pending,
          finding: %{topic: "Elixir", claim: "is functional"},
          aggregate_confidence: 0.88,
          corroborating_sources: [],
          conflicting_findings: [],
          existing_contradictions: [],
          reviewer_notes: nil,
          reviewed_at: nil
        }}
      ],
      stats: %{}
    }

    write_term(data_dir, "review_queue.term", review_data)

    # -- Source reliability --
    sr_data = %{
      sources: %{
        "example.com" => %{
          domain: "example.com",
          factual_accuracy: 0.8,
          bias_rating: :center,
          trust_tier: :verified,
          notes: nil,
          admin_decisions: [],
          last_updated: nil
        }
      }
    }

    write_term(data_dir, "source_reliability_learned.term", sr_data)

    # -- Source authority --
    sa_data = %{
      version: 1,
      last_updated: %{},
      tracking: %{
        source_1: %{
          confirmed_count: 5,
          contradicted_count: 1,
          total_added: 6,
          credibility: 0.9,
          last_updated: nil
        }
      }
    }

    write_term(data_dir, "source_authority_learned.term", sa_data)

    # -- User models --
    um_data = %{
      "user_1" => %{
        user_id: "user_1",
        facts: %{name: "Alice"},
        interaction_patterns: %{},
        epistemic_bounds: %{name: 0.9},
        provenance_map: %{name: :explicit},
        disclosure_history: [],
        created_at: ~U[2026-01-01 00:00:00Z],
        updated_at: ~U[2026-02-01 00:00:00Z]
      }
    }

    write_term(data_dir, "user_models.term", um_data)

    # -- Learned facts JSON --
    facts_json = %{
      "facts" => [
        %{
          "id" => "f1",
          "entity" => "Elixir",
          "entity_type" => "programming_language",
          "fact" => "runs on BEAM",
          "category" => "tech",
          "confidence" => 0.95
        }
      ]
    }

    File.write!(Path.join(facts_dir, "learned.json"), Jason.encode!(facts_json))

    # -- Knowledge per persona --
    knowledge = %{
      "people" => %{"alice" => %{"name" => "Alice"}},
      "pets" => %{"rex" => %{"name" => "Rex", "type" => "dog"}}
    }

    File.write!(Path.join(knowledge_dir, "TestPersona.json"), Jason.encode!(knowledge))

    # -- Persona memories --
    memories = [
      %{"role" => "user", "text" => "Hello there!", "timestamp" => 1_000_000, "tags" => ["greeting"]}
    ]

    File.write!(Path.join(memory_dir, "TestPersona.json"), Jason.encode!(memories))
  end

  defp write_term(dir, filename, data) do
    File.write!(Path.join(dir, filename), :erlang.term_to_binary(data))
  end
end
