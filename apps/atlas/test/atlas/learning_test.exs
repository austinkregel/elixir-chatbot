defmodule Atlas.LearningTest do
  use Atlas.DataCase, async: false

  alias Atlas.Learning

  describe "sessions" do
    test "creates a session with valid attributes" do
      attrs = %{topic: "Elixir concurrency", source_type: "web"}
      assert {:ok, session} = Learning.create_session(attrs)
      assert session.topic == "Elixir concurrency"
      assert session.status == "active"
      assert session.source_type == "web"
    end

    test "creates a session with minimal attributes" do
      assert {:ok, session} = Learning.create_session(%{})
      assert session.status == "active"
    end

    test "rejects invalid status" do
      assert {:error, changeset} = Learning.create_session(%{status: "invalid"})
      assert errors_on(changeset)[:status]
    end

    test "rejects invalid source_type" do
      assert {:error, changeset} = Learning.create_session(%{source_type: "invalid"})
      assert errors_on(changeset)[:source_type]
    end

    test "updates a session by struct" do
      {:ok, session} = Learning.create_session(%{topic: "Original"})
      assert {:ok, updated} = Learning.update_session(session, %{topic: "Updated"})
      assert updated.topic == "Updated"
    end

    test "updates a session by ID" do
      {:ok, session} = Learning.create_session(%{topic: "By ID"})
      assert {:ok, updated} = Learning.update_session(session.id, %{status: "completed"})
      assert updated.status == "completed"
    end

    test "update_session returns error for nonexistent ID" do
      fake_id = Ecto.UUID.generate()
      assert {:error, :not_found} = Learning.update_session(fake_id, %{topic: "x"})
    end

    test "gets a session by ID" do
      {:ok, session} = Learning.create_session(%{topic: "Lookup"})
      found = Learning.get_session(session.id)
      assert found.id == session.id
    end

    test "returns nil for nonexistent session" do
      assert nil == Learning.get_session(Ecto.UUID.generate())
    end

    test "lists sessions with filters" do
      {:ok, _} = Learning.create_session(%{topic: "A", status: "active"})
      {:ok, _} = Learning.create_session(%{topic: "B", status: "completed"})

      all = Learning.list_sessions()
      assert length(all) >= 2

      active = Learning.list_sessions(status: "active")
      assert Enum.all?(active, &(&1.status == "active"))

      completed = Learning.list_sessions(status: "completed")
      assert Enum.all?(completed, &(&1.status == "completed"))
    end

    test "lists sessions with limit and offset" do
      for i <- 1..5, do: Learning.create_session(%{topic: "Session #{i}"})

      limited = Learning.list_sessions(limit: 2)
      assert length(limited) == 2

      offset = Learning.list_sessions(limit: 2, offset: 2)
      assert length(offset) == 2
    end

    test "session_with_details returns session with associations" do
      {:ok, session} = Learning.create_session(%{topic: "Detailed"})
      assert {:ok, detailed} = Learning.session_with_details(session.id)
      assert detailed.id == session.id
      assert is_list(detailed.goals)
      assert is_list(detailed.investigations)
    end

    test "session_with_details returns error for nonexistent" do
      assert {:error, :not_found} = Learning.session_with_details(Ecto.UUID.generate())
    end
  end

  describe "goals" do
    setup do
      {:ok, session} = Learning.create_session(%{topic: "Goals test"})
      {:ok, session: session}
    end

    test "creates a goal", %{session: session} do
      attrs = %{session_id: session.id, topic: "Learn OTP"}
      assert {:ok, goal} = Learning.create_goal(attrs)
      assert goal.topic == "Learn OTP"
      assert goal.status == "pending"
      assert goal.priority == "normal"
    end

    test "validates goal priority", %{session: session} do
      attrs = %{session_id: session.id, topic: "X", priority: "invalid"}
      assert {:error, changeset} = Learning.create_goal(attrs)
      assert errors_on(changeset)[:priority]
    end

    test "updates goal status", %{session: session} do
      {:ok, goal} = Learning.create_goal(%{session_id: session.id, topic: "Update me"})
      assert {:ok, updated} = Learning.update_goal_status(goal.id, "completed")
      assert updated.status == "completed"
    end

    test "update_goal_status returns error for nonexistent" do
      assert {:error, :not_found} = Learning.update_goal_status(Ecto.UUID.generate(), "completed")
    end

    test "lists goals for session", %{session: session} do
      {:ok, _} = Learning.create_goal(%{session_id: session.id, topic: "G1"})
      {:ok, _} = Learning.create_goal(%{session_id: session.id, topic: "G2"})

      goals = Learning.goals_for_session(session.id)
      assert length(goals) == 2
    end
  end

  describe "investigations" do
    setup do
      {:ok, session} = Learning.create_session(%{topic: "Investigations test"})
      {:ok, session: session}
    end

    test "creates an investigation", %{session: session} do
      attrs = %{session_id: session.id, topic: "BEAM internals"}
      assert {:ok, inv} = Learning.create_investigation(attrs)
      assert inv.topic == "BEAM internals"
      assert inv.status == "planning"
    end

    test "concludes an investigation", %{session: session} do
      {:ok, inv} = Learning.create_investigation(%{session_id: session.id, topic: "Test"})

      attrs = %{conclusion: "hypotheses_supported"}
      assert {:ok, concluded} = Learning.conclude_investigation(inv.id, attrs)
      assert concluded.status == "concluded"
      assert concluded.conclusion == "hypotheses_supported"
      assert concluded.concluded_at != nil
    end

    test "conclude_investigation returns error for nonexistent" do
      assert {:error, :not_found} = Learning.conclude_investigation(Ecto.UUID.generate(), %{})
    end

    test "lists investigations for session", %{session: session} do
      {:ok, _} = Learning.create_investigation(%{session_id: session.id, topic: "I1"})
      {:ok, _} = Learning.create_investigation(%{session_id: session.id, topic: "I2"})

      invs = Learning.investigations_for_session(session.id)
      assert length(invs) == 2
    end
  end

  describe "hypotheses" do
    setup do
      {:ok, session} = Learning.create_session(%{topic: "Hypotheses test"})
      {:ok, inv} = Learning.create_investigation(%{session_id: session.id, topic: "Test"})
      {:ok, investigation: inv}
    end

    test "creates a hypothesis", %{investigation: inv} do
      attrs = %{investigation_id: inv.id, claim: "GenServers are supervised"}
      assert {:ok, hyp} = Learning.create_hypothesis(attrs)
      assert hyp.claim == "GenServers are supervised"
      assert hyp.status == "untested"
      assert hyp.confidence == 0.0
    end

    test "validates hypothesis status", %{investigation: inv} do
      attrs = %{investigation_id: inv.id, claim: "X", status: "invalid"}
      assert {:error, changeset} = Learning.create_hypothesis(attrs)
      assert errors_on(changeset)[:status]
    end

    test "validates confidence range", %{investigation: inv} do
      attrs = %{investigation_id: inv.id, claim: "X", confidence: 1.5}
      assert {:error, changeset} = Learning.create_hypothesis(attrs)
      assert errors_on(changeset)[:confidence]
    end

    test "updates a hypothesis", %{investigation: inv} do
      {:ok, hyp} = Learning.create_hypothesis(%{investigation_id: inv.id, claim: "Test"})

      assert {:ok, updated} =
               Learning.update_hypothesis(hyp.id, %{
                 status: "supported",
                 confidence: 0.9,
                 confidence_level: "high"
               })

      assert updated.status == "supported"
      assert updated.confidence == 0.9
    end

    test "update_hypothesis returns error for nonexistent" do
      assert {:error, :not_found} = Learning.update_hypothesis(Ecto.UUID.generate(), %{})
    end

    test "lists hypotheses for investigation", %{investigation: inv} do
      {:ok, _} = Learning.create_hypothesis(%{investigation_id: inv.id, claim: "H1"})
      {:ok, _} = Learning.create_hypothesis(%{investigation_id: inv.id, claim: "H2"})

      hyps = Learning.hypotheses_for_investigation(inv.id)
      assert length(hyps) == 2
    end
  end

  describe "evidence" do
    setup do
      {:ok, session} = Learning.create_session(%{topic: "Evidence test"})
      {:ok, inv} = Learning.create_investigation(%{session_id: session.id, topic: "Test"})
      {:ok, session: session, investigation: inv}
    end

    test "creates evidence", %{investigation: inv} do
      attrs = %{
        investigation_id: inv.id,
        claim: "Processes are lightweight",
        entity: "Erlang",
        source_url: "https://erlang.org",
        confidence: 0.85
      }

      assert {:ok, ev} = Learning.create_evidence(attrs)
      assert ev.claim == "Processes are lightweight"
      assert ev.confidence == 0.85
    end

    test "validates evidence_type", %{investigation: inv} do
      attrs = %{investigation_id: inv.id, evidence_type: "invalid"}
      assert {:error, changeset} = Learning.create_evidence(attrs)
      assert errors_on(changeset)[:evidence_type]
    end

    test "lists evidence for investigation", %{investigation: inv} do
      {:ok, _} = Learning.create_evidence(%{investigation_id: inv.id, claim: "E1"})
      {:ok, _} = Learning.create_evidence(%{investigation_id: inv.id, claim: "E2"})

      evidence = Learning.evidence_for_investigation(inv.id)
      assert length(evidence) == 2
    end

    test "lists all evidence for session", %{session: session, investigation: inv} do
      {:ok, _} = Learning.create_evidence(%{investigation_id: inv.id, claim: "E1"})

      {:ok, inv2} = Learning.create_investigation(%{session_id: session.id, topic: "I2"})
      {:ok, _} = Learning.create_evidence(%{investigation_id: inv2.id, claim: "E2"})

      all = Learning.all_evidence_for_session(session.id)
      assert length(all) == 2
    end
  end
end
