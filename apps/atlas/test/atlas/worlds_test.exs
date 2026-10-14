defmodule Atlas.WorldsTest do
  @moduledoc """
  Purging a world deletes every row it holds in a world-scoped table and drops
  its graphs; another world's rows and graphs are left as they were. Every row
  and graph a test writes is inside its transaction and gone with it.
  """

  use Atlas.DataCase, async: false

  alias Atlas.Graph
  alias Atlas.Learning
  alias Atlas.Schemas.{ResearchGoal, Investigation, Hypothesis, Evidence}
  alias Atlas.Worlds

  defp test_world(name), do: "test_#{name}_#{System.unique_integer([:positive])}"

  # One row of every world-scoped table in the world, and a learning session
  # with a goal, an investigation, a hypothesis and evidence under it.
  defp populate!(world_id) do
    %Atlas.Schemas.Episode{}
    |> Atlas.Schemas.Episode.changeset(%{world_id: world_id, state: "hello", action: "greet"})
    |> Repo.insert!()

    %Atlas.Schemas.SemanticFact{}
    |> Atlas.Schemas.SemanticFact.changeset(%{world_id: world_id, content: "greetings", category: "facts", confidence: 0.5})
    |> Repo.insert!()

    %Atlas.Schemas.Belief{}
    |> Atlas.Schemas.Belief.changeset(%{
      world_id: world_id,
      subject: "user",
      predicate: "likes",
      object: "tea",
      confidence: 0.5,
      source: "test"
    })
    |> Repo.insert!()

    %Atlas.Schemas.KnowledgeEntry{}
    |> Atlas.Schemas.KnowledgeEntry.changeset(%{world_id: world_id, persona_name: "echo", category: "facts", key: "k"})
    |> Repo.insert!()

    %Atlas.Schemas.Credential{}
    |> Atlas.Schemas.Credential.changeset(%{world: world_id, service: "weather", key: "api_key", encrypted_value: "v"})
    |> Repo.insert!()

    Atlas.Scheduler.insert_item!(%{
      world_id: world_id,
      kind: "alarm",
      time: "7am",
      status: "set",
      resolution: "recorded_before_resolution"
    })
    Atlas.WorldServiceDrivers.put!(world_id, "scheduling", "local_scheduler")
    # One metric window with one value and one record in it.
    :ok =
      Atlas.Metrics.write!([
        %{
          scope: {:world, world_id},
          family: "semantic_facts",
          values: [{"retrievals", :sum, 1}],
          records: [{"last_turn", %{"facts" => 1}}]
        }
      ])

    %Atlas.Schemas.IntentReviewCandidate{}
    |> Atlas.Schemas.IntentReviewCandidate.changeset(%{
      id: "irc_#{System.unique_integer([:positive])}",
      text: "what time is it",
      status: "pending",
      predicted_intent: "time.query",
      best_score: 0.4,
      world_id: world_id
    })
    |> Repo.insert!()

    %Atlas.Schemas.ReviewCandidate{}
    |> Atlas.Schemas.ReviewCandidate.changeset(%{
      id: "rc_#{System.unique_integer([:positive])}",
      world_id: world_id,
      status: "pending",
      finding: %{"claim" => "tea is hot"},
      aggregate_confidence: 0.5
    })
    |> Repo.insert!()

    {:ok, _case} =
      Atlas.Verification.save_case(%{
        subsystem: "speech_act",
        name: "a plain greeting",
        world_id: world_id,
        input: %{text: "hello there"},
        expected: %{category: "expressive"}
      })

    {:ok, session} = Learning.create_session(%{world_id: world_id, topic: "tea"})
    {:ok, _goal} = Learning.create_goal(%{session_id: session.id, topic: "tea"})
    {:ok, investigation} = Learning.create_investigation(%{session_id: session.id, topic: "tea"})
    {:ok, hypothesis} = Learning.create_hypothesis(%{investigation_id: investigation.id, claim: "tea is hot"})

    {:ok, _evidence} =
      Learning.create_evidence(%{investigation_id: investigation.id, hypothesis_id: hypothesis.id, claim: "it steams"})

    Graph.ensure_world_graphs!(world_id)
    {:ok, _} = Graph.add_node(Graph.world_graph("knowledge_graph", world_id), "Thing", %{name: "tea"})

    %{session: session, investigation: investigation}
  end

  defp count(schema, column, world_id) do
    schema |> where([row], field(row, ^column) == ^world_id) |> Repo.aggregate(:count)
  end

  describe "purge!/1" do
    test "deletes every world-scoped row and drops the graphs of the world, and leaves another world's" do
      world = test_world("purged")
      other = test_world("kept")
      %{session: session, investigation: investigation} = populate!(world)
      %{session: other_session} = populate!(other)

      purged = Worlds.purge!(world)

      assert purged.graphs == Graph.world_graphs(world)
      assert Map.keys(purged.rows) |> Enum.sort() == Worlds.world_tables() |> Enum.map(&elem(&1, 0)) |> Enum.sort()
      assert Enum.all?(purged.rows, fn {_schema, deleted} -> deleted == 1 end)

      for {schema, column} <- Worlds.world_tables() do
        assert count(schema, column, world) == 0, "#{inspect(schema)} still holds a row of #{world}"
        assert count(schema, column, other) == 1, "#{inspect(schema)} lost #{other}'s row"
      end

      # Checked by name as well, so a table dropped from the world tables
      # cannot pass the loop above by no longer being in it.
      assert count(Atlas.Schemas.ReviewCandidate, :world_id, world) == 0
      assert count(Atlas.Schemas.ReviewCandidate, :world_id, other) == 1
      assert count(Atlas.Schemas.WorldServiceDriver, :world_id, world) == 0
      assert count(Atlas.Schemas.WorldServiceDriver, :world_id, other) == 1
      for schema <- [Atlas.Schemas.MetricValue, Atlas.Schemas.MetricRecord, Atlas.Schemas.MetricWindow] do
        assert count(schema, :world_id, world) == 0
        assert count(schema, :world_id, other) == 1
      end

      assert Atlas.Metrics.windows({:world, world}, "semantic_facts", 10) == []

      assert [%{values: %{"retrievals" => 1}, records: %{"last_turn" => %{record: %{"facts" => 1}}}}] =
               Atlas.Metrics.windows({:world, other}, "semantic_facts", 10)

      refute Enum.any?(Graph.world_graphs(world), &Graph.graph_exists?/1)
      assert Enum.all?(Graph.world_graphs(other), &Graph.graph_exists?/1)
      assert {:ok, 1} = Graph.count_nodes(Graph.world_graph("knowledge_graph", other))

      # A session's goals, investigations, hypotheses and evidence go with it.
      assert Learning.session_with_details(session.id) == {:error, :not_found}
      assert Repo.all(from(g in ResearchGoal, where: g.session_id == ^session.id)) == []
      assert Repo.all(from(i in Investigation, where: i.session_id == ^session.id)) == []
      assert Repo.all(from(h in Hypothesis, where: h.investigation_id == ^investigation.id)) == []
      assert Repo.all(from(e in Evidence, where: e.investigation_id == ^investigation.id)) == []

      assert {:ok, %{goals: [_], investigations: [%{hypotheses: [_], evidence: [_]}]}} =
               Learning.session_with_details(other_session.id)
    end

    test "purging a world with nothing in Atlas deletes nothing and drops no graph" do
      purged = Worlds.purge!(test_world("empty"))

      assert purged.graphs == []
      assert Enum.all?(purged.rows, fn {_schema, deleted} -> deleted == 0 end)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/needs the world to purge/, fn -> Worlds.purge!(nil) end
      assert_raise ArgumentError, ~r/needs the world to purge/, fn -> Worlds.purge!("") end
    end
  end
end
