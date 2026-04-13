defmodule Atlas.SchemasTest do
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.{Credential, Belief, Episode, SemanticFact, ReviewCandidate, LearnedFact}

  describe "Credential" do
    test "inserts and retrieves a credential" do
      attrs = %{
        world: "default",
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
      assert found.world == "default"
    end

    test "enforces unique constraint on (world, service, key)" do
      attrs = %{service: "weather", key: "api_key", encrypted_value: "v1"}

      assert {:ok, _} =
               %Credential{} |> Credential.changeset(attrs) |> Repo.insert()

      assert {:error, changeset} =
               %Credential{} |> Credential.changeset(attrs) |> Repo.insert()

      assert errors_on(changeset)[:world] || errors_on(changeset)[:service] ||
               errors_on(changeset)[:key]
    end
  end

  describe "Belief" do
    test "inserts and queries a belief" do
      attrs = %{
        subject: "Paris",
        predicate: "is_a",
        object: "city",
        confidence: 0.95,
        source: "conversation"
      }

      assert {:ok, belief} =
               %Belief{} |> Belief.changeset(attrs) |> Repo.insert()

      assert belief.subject == "Paris"
      assert belief.confidence == 0.95

      active_beliefs = Belief |> Belief.active() |> Repo.all()
      assert length(active_beliefs) >= 1
    end
  end

  describe "Episode" do
    test "inserts and queries an episode" do
      attrs = %{
        state: "user asked about weather",
        action: "retrieved weather data",
        outcome: "provided forecast",
        tags: ["weather", "query"]
      }

      assert {:ok, episode} =
               %Episode{} |> Episode.changeset(attrs) |> Repo.insert()

      assert episode.tags == ["weather", "query"]

      world_episodes = Episode |> Episode.for_world("default") |> Repo.all()
      assert length(world_episodes) >= 1
    end
  end

  describe "SemanticFact" do
    test "inserts and queries a semantic fact" do
      attrs = %{
        content: "Weather queries are common in the afternoon",
        category: "patterns",
        confidence: 0.8
      }

      assert {:ok, fact} =
               %SemanticFact{} |> SemanticFact.changeset(attrs) |> Repo.insert()

      assert fact.category == "patterns"
    end
  end

  describe "ReviewCandidate" do
    test "inserts and queries a review candidate" do
      attrs = %{
        id: "review_#{System.unique_integer([:positive])}",
        status: "pending",
        finding: %{"topic" => "Elixir", "claim" => "is functional"},
        aggregate_confidence: 0.88
      }

      assert {:ok, candidate} =
               %ReviewCandidate{} |> ReviewCandidate.changeset(attrs) |> Repo.insert()

      assert candidate.status == "pending"

      pending = ReviewCandidate |> ReviewCandidate.pending() |> Repo.all()
      assert length(pending) >= 1
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
