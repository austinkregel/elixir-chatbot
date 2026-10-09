defmodule Atlas.SchemasTest do
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.{Credential, Belief, Episode, SemanticFact, ReviewCandidate, LearnedFact}

  # Rows are rolled back with the test's transaction; no world directory is
  # created.
  @world_id "test_atlas_schemas"

  describe "Credential" do
    test "inserts and retrieves a credential" do
      attrs = %{
        world: @world_id,
        service: "weather",
        key: "api_key",
        encrypted_value: "encrypted_data_here"
      }

      assert {:ok, credential} =
               %Credential{}
               |> Credential.changeset(attrs)
               |> Repo.insert()

      assert credential.service == "weather"
      assert credential.key == "api_key"

      found = Repo.get!(Credential, credential.id)
      assert found.world == @world_id
    end

    test "enforces unique constraint on (world, service, key)" do
      attrs = %{world: @world_id, service: "weather", key: "api_key", encrypted_value: "v1"}

      assert {:ok, _} =
               %Credential{} |> Credential.changeset(attrs) |> Repo.insert()

      assert {:error, changeset} =
               %Credential{} |> Credential.changeset(attrs) |> Repo.insert()

      assert errors_on(changeset)[:world] || errors_on(changeset)[:service] ||
               errors_on(changeset)[:key]
    end

    test "the same service and key may be stored in two worlds" do
      attrs = %{service: "weather", key: "api_key", encrypted_value: "v1"}

      assert {:ok, _} =
               %Credential{} |> Credential.changeset(Map.put(attrs, :world, @world_id)) |> Repo.insert()

      assert {:ok, _} =
               %Credential{}
               |> Credential.changeset(Map.put(attrs, :world, "#{@world_id}_other"))
               |> Repo.insert()
    end

    test "requires a world" do
      changeset =
        Credential.changeset(%Credential{}, %{service: "weather", key: "api_key", encrypted_value: "v1"})

      refute changeset.valid?
      assert "can't be blank" in errors_on(changeset)[:world]
    end
  end

  describe "Belief" do
    test "inserts and queries a belief" do
      attrs = %{
        world_id: "test_schemas",
        subject: "Paris",
        predicate: "is_a",
        object: "city",
        confidence: 0.95,
        source: "conversation"
      }

      assert {:ok, belief} =
               %Belief{} |> Belief.changeset(attrs) |> Repo.insert()

      assert belief.subject == "Paris"
      assert belief.world_id == "test_schemas"
      assert belief.confidence == 0.95

      active_beliefs = Belief |> Belief.active() |> Repo.all()
      assert length(active_beliefs) >= 1
    end

    test "a belief names its world, and is found only in that world" do
      attrs = %{subject: "Paris", predicate: "is_a", object: "city", confidence: 0.9, source: "conversation"}

      refute Belief.changeset(%Belief{}, attrs).valid?

      {:ok, _} = %Belief{} |> Belief.changeset(Map.put(attrs, :world_id, "test_belief_a")) |> Repo.insert()

      assert [%Belief{world_id: "test_belief_a"}] = Belief |> Belief.in_world("test_belief_a") |> Repo.all()
      assert [] = Belief |> Belief.in_world("test_belief_b") |> Repo.all()
    end
  end

  describe "Episode" do
    test "inserts and queries an episode" do
      attrs = %{
        world_id: @world_id,
        state: "user asked about weather",
        action: "retrieved weather data",
        outcome: "provided forecast",
        tags: ["weather", "query"]
      }

      assert {:ok, episode} =
               %Episode{} |> Episode.changeset(attrs) |> Repo.insert()

      assert episode.tags == ["weather", "query"]

      world_episodes = Episode |> Episode.for_world(@world_id) |> Repo.all()
      assert Enum.map(world_episodes, & &1.id) == [episode.id]
    end

    test "requires a world" do
      changeset =
        Episode.changeset(%Episode{}, %{state: "user asked about weather", action: "retrieved weather data"})

      refute changeset.valid?
      assert "can't be blank" in errors_on(changeset)[:world_id]
    end

    test "the table fills in no world for a row that leaves it out" do
      now = DateTime.utc_now()

      row = %{
        id: Ecto.UUID.bingenerate(),
        state: "user asked about weather",
        action: "retrieved weather data",
        outcome: "",
        inserted_at: now,
        updated_at: now
      }

      error = assert_raise Postgrex.Error, fn -> Repo.insert_all("atlas_episodes", [row]) end
      assert error.postgres.code == :not_null_violation
      assert error.postgres.column == "world_id"
    end
  end

  describe "SemanticFact" do
    test "inserts and queries a semantic fact" do
      attrs = %{
        world_id: @world_id,
        content: "Weather queries are common in the afternoon",
        category: "patterns",
        confidence: 0.8
      }

      assert {:ok, fact} =
               %SemanticFact{} |> SemanticFact.changeset(attrs) |> Repo.insert()

      assert fact.category == "patterns"
      assert fact.world_id == @world_id
    end

    test "requires a world" do
      changeset =
        SemanticFact.changeset(%SemanticFact{}, %{
          content: "Weather queries are common in the afternoon",
          category: "patterns",
          confidence: 0.8
        })

      refute changeset.valid?
      assert "can't be blank" in errors_on(changeset)[:world_id]
    end
  end

  describe "ReviewCandidate" do
    test "inserts and queries a review candidate" do
      attrs = %{
        id: "review_#{System.unique_integer([:positive])}",
        world_id: "default",
        status: "pending",
        finding: %{"topic" => "Elixir", "claim" => "is functional"},
        aggregate_confidence: 0.88
      }

      assert {:ok, candidate} =
               %ReviewCandidate{} |> ReviewCandidate.changeset(attrs) |> Repo.insert()

      assert candidate.status == "pending"
      assert candidate.world_id == "default"

      pending = ReviewCandidate |> ReviewCandidate.pending() |> Repo.all()
      assert length(pending) >= 1
    end

    test "every status a reviewer or the queue can give is valid, and no other" do
      for status <- ~w(pending approved rejected deferred auto_approved) do
        attrs = %{
          id: "review_#{System.unique_integer([:positive])}",
          world_id: "default",
          status: status,
          finding: %{"claim" => "is functional"},
          aggregate_confidence: 0.88
        }

        assert {:ok, %ReviewCandidate{status: ^status}} =
                 %ReviewCandidate{} |> ReviewCandidate.changeset(attrs) |> Repo.insert()
      end

      changeset =
        ReviewCandidate.changeset(%ReviewCandidate{}, %{
          id: "review_#{System.unique_integer([:positive])}",
          world_id: "default",
          status: "postponed",
          finding: %{"claim" => "is functional"},
          aggregate_confidence: 0.88
        })

      refute changeset.valid?
      assert {"is invalid", _} = changeset.errors[:status]
    end

    test "a candidate without a world is invalid" do
      attrs = %{
        id: "review_#{System.unique_integer([:positive])}",
        status: "pending",
        finding: %{"claim" => "is functional"},
        aggregate_confidence: 0.88
      }

      changeset = ReviewCandidate.changeset(%ReviewCandidate{}, attrs)

      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:world_id]
    end
  end

  describe "LearnedFact" do
    test "inserts and queries a learned fact" do
      attrs = %{
        id: "fact_#{System.unique_integer([:positive])}",
        entity: "Elixir",
        entity_type: "programming_language",
        fact: "Elixir runs on the BEAM virtual machine",
        confidence: 0.95
      }

      assert {:ok, fact} =
               %LearnedFact{} |> LearnedFact.changeset(attrs) |> Repo.insert()

      assert fact.entity == "Elixir"

      elixir_facts = LearnedFact |> LearnedFact.for_entity("Elixir") |> Repo.all()
      assert length(elixir_facts) >= 1
    end
  end
end
