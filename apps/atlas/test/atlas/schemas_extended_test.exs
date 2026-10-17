defmodule Atlas.SchemasExtendedTest do
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.{
    SourceReliability,
    UserModel,
    KnowledgeEntry,
    PersonaMemory,
    LearningSession,
    ResearchGoal,
    Investigation,
    Hypothesis,
    Evidence,
    EvaluationResult,
    IntentReviewCandidate
  }

  # Rows are rolled back with the test's transaction; no world directory is
  # created.
  @world_id "test_atlas_schemas_extended"

  describe "SourceReliability" do
    test "inserts with valid attributes" do
      attrs = %{
        domain: "example.com",
        reliability_score: 0.85,
        bias_rating: "low",
        trust_tier: "trusted"
      }

      assert {:ok, sr} = %SourceReliability{} |> SourceReliability.changeset(attrs) |> Repo.insert()
      assert sr.domain == "example.com"
      assert sr.reliability_score == 0.85
    end

    test "requires domain" do
      assert {:error, changeset} =
               %SourceReliability{} |> SourceReliability.changeset(%{}) |> Repo.insert()

      assert errors_on(changeset)[:domain]
    end

    test "enforces unique domain" do
      attrs = %{domain: "unique-domain.com"}
      {:ok, _} = %SourceReliability{} |> SourceReliability.changeset(attrs) |> Repo.insert()

      assert {:error, changeset} =
               %SourceReliability{} |> SourceReliability.changeset(attrs) |> Repo.insert()

      assert errors_on(changeset)[:domain]
    end

    test "queries by domain" do
      {:ok, _} =
        %SourceReliability{}
        |> SourceReliability.changeset(%{domain: "test.org"})
        |> Repo.insert()

      results = SourceReliability |> SourceReliability.for_domain("test.org") |> Repo.all()
      assert length(results) == 1
    end
  end

  describe "UserModel" do
    # A user model belongs to a web user: its user_id is the web user's id.
    defp web_user_id, do: Atlas.WebUsers.create!(@world_id).id

    test "inserts with valid attributes" do
      attrs = %{
        user_id: web_user_id(),
        facts: %{"name" => "Alice"},
        interaction_patterns: %{"greeting_style" => "formal"}
      }

      assert {:ok, um} = %UserModel{} |> UserModel.changeset(attrs) |> Repo.insert()
      assert um.facts == %{"name" => "Alice"}
    end

    test "is persisted and linked to its web user" do
      user = Atlas.WebUsers.create!(@world_id)

      {:ok, _} =
        %UserModel{}
        |> UserModel.changeset(%{user_id: user.id, facts: %{"location" => "Seattle"}})
        |> Repo.insert()

      [stored] = UserModel |> UserModel.for_user(user.id) |> Repo.all() |> Repo.preload(:web_user)
      assert stored.facts == %{"location" => "Seattle"}
      assert stored.web_user.id == user.id
    end

    test "has no world: a web user has one user model whichever world they select" do
      user = Atlas.WebUsers.create!(@world_id)
      {:ok, _} = %UserModel{} |> UserModel.changeset(%{user_id: user.id}) |> Repo.insert()

      Atlas.WebUsers.select_world!(user, "#{@world_id}_other")

      assert [_one] = UserModel |> UserModel.for_user(user.id) |> Repo.all()
      refute :world_id in UserModel.__schema__(:fields)
    end

    test "refuses a user id that is no web user's" do
      assert {:error, changeset} =
               %UserModel{} |> UserModel.changeset(%{user_id: Ecto.UUID.generate()}) |> Repo.insert()

      assert "does not exist" in errors_on(changeset)[:user_id]

      assert {:error, changeset} =
               %UserModel{} |> UserModel.changeset(%{user_id: "anonymous"}) |> Repo.insert()

      assert "is invalid" in errors_on(changeset)[:user_id]
    end

    test "a web user with a user model cannot be deleted" do
      user = Atlas.WebUsers.create!(@world_id)
      {:ok, _} = %UserModel{} |> UserModel.changeset(%{user_id: user.id}) |> Repo.insert()

      assert_raise Ecto.ConstraintError, ~r/atlas_user_models_user_id_fkey/, fn -> Repo.delete!(user) end
    end

    test "requires user_id" do
      assert {:error, changeset} = %UserModel{} |> UserModel.changeset(%{}) |> Repo.insert()
      assert errors_on(changeset)[:user_id]
    end

    test "enforces unique user_id" do
      uid = web_user_id()
      {:ok, _} = %UserModel{} |> UserModel.changeset(%{user_id: uid}) |> Repo.insert()

      assert {:error, changeset} =
               %UserModel{} |> UserModel.changeset(%{user_id: uid}) |> Repo.insert()

      assert errors_on(changeset)[:user_id]
    end

    test "queries by user_id" do
      uid = web_user_id()
      {:ok, _} = %UserModel{} |> UserModel.changeset(%{user_id: uid}) |> Repo.insert()
      results = UserModel |> UserModel.for_user(uid) |> Repo.all()
      assert length(results) == 1
    end
  end

  describe "KnowledgeEntry" do
    test "inserts with valid attributes" do
      attrs = %{
        world_id: @world_id,
        persona_name: "echo",
        category: "facts",
        key: "favorite_color",
        data: %{"value" => "blue"}
      }

      assert {:ok, ke} = %KnowledgeEntry{} |> KnowledgeEntry.changeset(attrs) |> Repo.insert()
      assert ke.world_id == @world_id
      assert ke.persona_name == "echo"
      assert ke.category == "facts"
    end

    test "requires world_id, persona_name, category, key" do
      assert {:error, changeset} = %KnowledgeEntry{} |> KnowledgeEntry.changeset(%{}) |> Repo.insert()
      assert "can't be blank" in errors_on(changeset)[:world_id]
      assert errors_on(changeset)[:persona_name]
      assert errors_on(changeset)[:category]
      assert errors_on(changeset)[:key]
    end

    test "queries for_persona" do
      {:ok, _} =
        %KnowledgeEntry{}
        |> KnowledgeEntry.changeset(%{world_id: @world_id, persona_name: "test_p", category: "c", key: "k"})
        |> Repo.insert()

      results = KnowledgeEntry |> KnowledgeEntry.for_persona("test_p") |> Repo.all()
      assert length(results) >= 1
    end

    test "queries for_world_persona" do
      {:ok, _} =
        %KnowledgeEntry{}
        |> KnowledgeEntry.changeset(%{
          world_id: "test_world",
          persona_name: "echo_wp",
          category: "c",
          key: "k"
        })
        |> Repo.insert()

      results = KnowledgeEntry |> KnowledgeEntry.for_world_persona("test_world", "echo_wp") |> Repo.all()
      assert length(results) >= 1
    end
  end

  describe "PersonaMemory" do
    test "inserts with valid attributes" do
      attrs = %{
        persona_name: "echo",
        role: "assistant",
        content: "Hello, how can I help?"
      }

      assert {:ok, pm} = %PersonaMemory{} |> PersonaMemory.changeset(attrs) |> Repo.insert()
      assert pm.role == "assistant"
    end

    test "requires persona_name, role, content" do
      assert {:error, changeset} = %PersonaMemory{} |> PersonaMemory.changeset(%{}) |> Repo.insert()
      assert errors_on(changeset)[:persona_name]
      assert errors_on(changeset)[:role]
      assert errors_on(changeset)[:content]
    end

    test "queries for_persona" do
      {:ok, _} =
        %PersonaMemory{}
        |> PersonaMemory.changeset(%{persona_name: "pm_test", role: "user", content: "Hi"})
        |> Repo.insert()

      results = PersonaMemory |> PersonaMemory.for_persona("pm_test") |> Repo.all()
      assert length(results) >= 1
    end
  end

  describe "LearningSession" do
    test "changeset requires a world" do
      cs = LearningSession.changeset(%LearningSession{}, %{topic: "no world"})
      refute cs.valid?
      assert "can't be blank" in errors_on(cs)[:world_id]
    end

    test "the table fills in no world for a row that leaves it out" do
      now = DateTime.utc_now()

      row = %{
        id: Ecto.UUID.bingenerate(),
        topic: "no world",
        status: "active",
        inserted_at: now,
        updated_at: now
      }

      error = assert_raise Postgrex.Error, fn -> Repo.insert_all("atlas_learning_sessions", [row]) end
      assert error.postgres.code == :not_null_violation
      assert error.postgres.column == "world_id"
    end

    test "changeset validates status inclusion" do
      cs = LearningSession.changeset(%LearningSession{}, %{world_id: @world_id, status: "invalid"})
      assert errors_on(cs)[:status]
    end

    test "changeset validates source_type inclusion" do
      cs = LearningSession.changeset(%LearningSession{}, %{world_id: @world_id, source_type: "invalid"})
      assert errors_on(cs)[:source_type]
    end

    test "changeset validates counts are non-negative" do
      cs = LearningSession.changeset(%LearningSession{}, %{world_id: @world_id, findings_count: -1})
      assert errors_on(cs)[:findings_count]
    end

    test "with_status query" do
      {:ok, _} =
        %LearningSession{}
        |> LearningSession.changeset(%{world_id: @world_id, status: "completed"})
        |> Repo.insert()

      results = LearningSession |> LearningSession.with_status("completed") |> Repo.all()
      assert Enum.all?(results, &(&1.status == "completed"))
    end

    test "active query" do
      {:ok, _} =
        %LearningSession{}
        |> LearningSession.changeset(%{world_id: @world_id, status: "active", topic: "active_test"})
        |> Repo.insert()

      results = LearningSession |> LearningSession.active() |> Repo.all()
      assert Enum.all?(results, &(&1.status == "active"))
    end

    test "recent_first query orders by started_at descending" do
      now = DateTime.utc_now()
      earlier = DateTime.add(now, -3600, :second)

      {:ok, _} =
        %LearningSession{}
        |> LearningSession.changeset(%{world_id: @world_id, topic: "older", started_at: earlier})
        |> Repo.insert()

      {:ok, _} =
        %LearningSession{}
        |> LearningSession.changeset(%{world_id: @world_id, topic: "newer", started_at: now})
        |> Repo.insert()

      results = LearningSession |> LearningSession.recent_first() |> Repo.all()
      if length(results) >= 2 do
        [first | _] = results
        assert first.topic == "newer"
      end
    end
  end

  describe "EvaluationResult" do
    test "inserts with valid attributes" do
      attrs = %{
        task: "intent",
        accuracy: 0.92,
        macro_f1: 0.88,
        total_examples: 500,
        per_class: %{"greeting" => %{"precision" => 0.95}}
      }

      assert {:ok, er} = %EvaluationResult{} |> EvaluationResult.changeset(attrs) |> Repo.insert()
      assert er.accuracy == 0.92
    end

    test "requires task, accuracy, macro_f1, total_examples" do
      assert {:error, changeset} =
               %EvaluationResult{} |> EvaluationResult.changeset(%{}) |> Repo.insert()

      assert errors_on(changeset)[:task]
      assert errors_on(changeset)[:accuracy]
      assert errors_on(changeset)[:macro_f1]
      assert errors_on(changeset)[:total_examples]
    end

    test "validates task inclusion" do
      attrs = %{task: "invalid", accuracy: 0.5, macro_f1: 0.5, total_examples: 10}

      assert {:error, changeset} =
               %EvaluationResult{} |> EvaluationResult.changeset(attrs) |> Repo.insert()

      assert errors_on(changeset)[:task]
    end
  end

  describe "IntentReviewCandidate" do
    test "inserts with valid attributes" do
      attrs = %{
        id: "irc_#{System.unique_integer([:positive])}",
        text: "What time is it in Tokyo?",
        status: "pending",
        predicted_intent: "time.query",
        best_score: 0.75,
        world_id: @world_id
      }

      assert {:ok, irc} =
               %IntentReviewCandidate{} |> IntentReviewCandidate.changeset(attrs) |> Repo.insert()

      assert irc.text == "What time is it in Tokyo?"
      assert irc.world_id == @world_id
    end

    test "requires id, text, predicted_intent, best_score, world_id" do
      assert {:error, changeset} =
               %IntentReviewCandidate{} |> IntentReviewCandidate.changeset(%{}) |> Repo.insert()

      assert errors_on(changeset)[:id]
      assert errors_on(changeset)[:text]
      assert errors_on(changeset)[:predicted_intent]
      assert errors_on(changeset)[:best_score]
      assert "can't be blank" in errors_on(changeset)[:world_id]
    end

    test "validates status inclusion" do
      attrs = %{
        world_id: @world_id,
        id: "irc_status",
        text: "test",
        status: "invalid",
        predicted_intent: "x",
        best_score: 0.5
      }

      assert {:error, changeset} =
               %IntentReviewCandidate{} |> IntentReviewCandidate.changeset(attrs) |> Repo.insert()

      assert errors_on(changeset)[:status]
    end
  end

  describe "ResearchGoal" do
    setup do
      {:ok, session} = %LearningSession{} |> LearningSession.changeset(%{world_id: @world_id}) |> Repo.insert()
      {:ok, session: session}
    end

    test "changeset validates priority", %{session: session} do
      cs = ResearchGoal.changeset(%ResearchGoal{}, %{session_id: session.id, topic: "T", priority: "bad"})
      assert errors_on(cs)[:priority]
    end

    test "changeset validates status", %{session: session} do
      cs = ResearchGoal.changeset(%ResearchGoal{}, %{session_id: session.id, topic: "T", status: "bad"})
      assert errors_on(cs)[:status]
    end

    test "with_status query", %{session: session} do
      {:ok, _} =
        %ResearchGoal{}
        |> ResearchGoal.changeset(%{session_id: session.id, topic: "T", status: "pending"})
        |> Repo.insert()

      results = ResearchGoal |> ResearchGoal.with_status("pending") |> Repo.all()
      assert length(results) >= 1
    end
  end

  describe "Investigation" do
    setup do
      {:ok, session} = %LearningSession{} |> LearningSession.changeset(%{world_id: @world_id}) |> Repo.insert()
      {:ok, session: session}
    end

    test "changeset validates status", %{session: session} do
      cs =
        Investigation.changeset(%Investigation{}, %{
          session_id: session.id,
          topic: "T",
          status: "bad"
        })

      assert errors_on(cs)[:status]
    end

    test "with_status query", %{session: session} do
      {:ok, _} =
        %Investigation{}
        |> Investigation.changeset(%{session_id: session.id, topic: "T", status: "planning"})
        |> Repo.insert()

      results = Investigation |> Investigation.with_status("planning") |> Repo.all()
      assert length(results) >= 1
    end
  end

  describe "Hypothesis" do
    setup do
      {:ok, session} = %LearningSession{} |> LearningSession.changeset(%{world_id: @world_id}) |> Repo.insert()

      {:ok, inv} =
        %Investigation{}
        |> Investigation.changeset(%{session_id: session.id, topic: "T"})
        |> Repo.insert()

      {:ok, investigation: inv}
    end

    test "validates confidence range", %{investigation: inv} do
      cs =
        Hypothesis.changeset(%Hypothesis{}, %{
          investigation_id: inv.id,
          claim: "C",
          confidence: 1.5
        })

      assert errors_on(cs)[:confidence]
    end

    test "promotable query", %{investigation: inv} do
      {:ok, _} =
        %Hypothesis{}
        |> Hypothesis.changeset(%{
          investigation_id: inv.id,
          claim: "Promotable",
          status: "supported",
          confidence: 0.9,
          source_count: 3
        })
        |> Repo.insert()

      results = Hypothesis |> Hypothesis.promotable() |> Repo.all()
      assert length(results) >= 1
    end
  end

  describe "Evidence" do
    setup do
      {:ok, session} = %LearningSession{} |> LearningSession.changeset(%{world_id: @world_id}) |> Repo.insert()

      {:ok, inv} =
        %Investigation{}
        |> Investigation.changeset(%{session_id: session.id, topic: "T"})
        |> Repo.insert()

      {:ok, investigation: inv}
    end

    test "validates evidence_type", %{investigation: inv} do
      cs =
        Evidence.changeset(%Evidence{}, %{
          investigation_id: inv.id,
          evidence_type: "invalid"
        })

      assert errors_on(cs)[:evidence_type]
    end

    test "queries by type", %{investigation: inv} do
      {:ok, _} =
        %Evidence{}
        |> Evidence.changeset(%{investigation_id: inv.id, evidence_type: "supporting"})
        |> Repo.insert()

      results = Evidence |> Evidence.with_type("supporting") |> Repo.all()
      assert length(results) >= 1
    end

    test "queries by entity", %{investigation: inv} do
      {:ok, _} =
        %Evidence{}
        |> Evidence.changeset(%{investigation_id: inv.id, entity: "Elixir"})
        |> Repo.insert()

      results = Evidence |> Evidence.for_entity("Elixir") |> Repo.all()
      assert length(results) >= 1
    end

    test "queries for hypothesis", %{investigation: inv} do
      {:ok, hyp} =
        %Hypothesis{}
        |> Hypothesis.changeset(%{investigation_id: inv.id, claim: "Test"})
        |> Repo.insert()

      {:ok, _} =
        %Evidence{}
        |> Evidence.changeset(%{investigation_id: inv.id, hypothesis_id: hyp.id})
        |> Repo.insert()

      results = Evidence |> Evidence.for_hypothesis(hyp.id) |> Repo.all()
      assert length(results) >= 1
    end
  end
end
