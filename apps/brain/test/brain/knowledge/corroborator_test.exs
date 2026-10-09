defmodule Brain.Knowledge.CorroboratorTest do
  alias Brain.Knowledge.Types
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Knowledge.Corroborator
  alias Types.{Finding, SourceInfo}

  # "default" is the world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  describe "corroborate/2" do
    test "groups similar claims from different sources" do
      findings = [
        build_finding("Paris is the capital of France", "France", "source1.com"),
        build_finding("France's capital city is Paris", "France", "source2.com"),
        build_finding("Berlin is the capital of Germany", "Germany", "source1.com")
      ]

      {:ok, all_candidates} =
        Corroborator.corroborate(findings, world_id: @world_id, include_uncorroborated: true)

      assert all_candidates != []
      {:ok, filtered_candidates} = Corroborator.corroborate(findings, world_id: @world_id)

      paris_candidates =
        Enum.filter(all_candidates, fn c ->
          String.contains?(c.finding.claim, "Paris") or
            String.contains?(c.finding.claim, "capital of France")
        end)

      assert paris_candidates != []

      berlin_in_filtered =
        Enum.find(filtered_candidates, &String.contains?(&1.finding.claim, "Berlin"))

      if berlin_in_filtered do
        assert berlin_in_filtered.corroborating_sources == []
      end
    end

    test "includes uncorroborated when option set" do
      findings = [build_finding("Single source claim", "Entity", "source1.com")]

      {:ok, candidates} =
        Corroborator.corroborate(findings, world_id: @world_id, include_uncorroborated: true)

      assert length(candidates) == 1
    end

    test "returns empty list for empty input" do
      {:ok, candidates} = Corroborator.corroborate([], world_id: @world_id)
      assert candidates == []
    end

    test "raises without a :world_id, whatever the findings" do
      findings = [
        build_finding("Paris is the capital of France", "France", "source1.com"),
        build_finding("France's capital city is Paris", "France", "source2.com")
      ]

      assert_raise ArgumentError, ~r/corroborate requires a :world_id option/, fn ->
        Corroborator.corroborate(findings, include_uncorroborated: true)
      end

      assert_raise ArgumentError, ~r/corroborate requires a :world_id option/, fn ->
        Corroborator.corroborate([], world_id: nil)
      end

      assert_raise ArgumentError, ~r/corroborate requires a :world_id option/, fn ->
        Corroborator.corroborate(findings, world_id: "")
      end
    end

    test "respects custom similarity threshold" do
      findings = [
        build_finding("The population is 100", "City", "source1.com"),
        build_finding("The population is around 100", "City", "source2.com")
      ]

      {:ok, candidates} =
        Corroborator.corroborate(findings, world_id: @world_id, similarity_threshold: 0.5)

      assert candidates != []
    end
  end

  describe "compare_claims/3" do
    test "similar claims have high similarity" do
      {:ok, similarity} =
        Corroborator.compare_claims(
          @world_id,
          "Paris is the capital of France",
          "The capital of France is Paris"
        )

      assert similarity > 0.5
    end

    test "different claims have low similarity" do
      {:ok, similarity} =
        Corroborator.compare_claims(
          @world_id,
          "Paris is the capital of France",
          "Tokyo is a large city in Japan"
        )

      assert similarity < 0.5
    end

    test "compares in the named world's embedder" do
      ensure_started(World.Manager)
      setup_world_sandbox()
      {:ok, world} = create_test_world("corroborator_embedder")

      # The world's vocabulary knows only "paris", so both claims embed as the
      # same vector. Token overlap ({paris, capital} against {paris, city})
      # would be 1/3.
      :ok = World.Embedder.load_model(world.id, %{vocabulary: %{"paris" => 0}, idf_weights: %{"paris" => 1.0}})

      assert {:ok, 1.0} = Corroborator.compare_claims(world.id, "Paris capital", "Paris city")
    end

    test "raises for a world that holds no embedder" do
      # World.Manager holds the table world embedders are loaded into.
      ensure_started(World.Manager)
      world_id = "test_corroborator_no_embedder_#{System.unique_integer([:positive])}"

      assert_raise RuntimeError, ~r/world #{inspect(world_id)} has no embedder vocabulary/, fn ->
        Corroborator.compare_claims(world_id, "Paris is the capital of France", "Paris is in France")
      end
    end
  end

  describe "test_hypotheses/3" do
    test "records the evidence in the named world's embedder and concludes" do
      ensure_started(World.Manager)
      setup_world_sandbox()
      {:ok, paris_world} = create_test_world("corroborator_hypotheses_paris_only")
      {:ok, capital_city_world} = create_test_world("corroborator_hypotheses_capital_and_city")

      # In the first world "Paris capital" and "Paris city" are the same
      # vector; in the second they are orthogonal.
      :ok = World.Embedder.load_model(paris_world.id, %{vocabulary: %{"paris" => 0}, idf_weights: %{"paris" => 1.0}})

      :ok =
        World.Embedder.load_model(capital_city_world.id, %{
          vocabulary: %{"capital" => 0, "city" => 1},
          idf_weights: %{"capital" => 1.0, "city" => 1.0}
        })

      findings = [
        build_finding("Paris city", "Paris", "source1.com"),
        build_finding("Paris city", "Paris", "source2.com")
      ]

      investigation = Types.Investigation.new("Paris", hypotheses: [Types.Hypothesis.new("Paris capital")])

      {:ok, concluded} = Corroborator.test_hypotheses(paris_world.id, investigation, findings)
      assert [%{status: :supported, supporting_evidence: [_, _]}] = concluded.hypotheses
      assert concluded.status == :concluded

      {:ok, concluded} = Corroborator.test_hypotheses(capital_city_world.id, investigation, findings)
      assert [%{status: :inconclusive, supporting_evidence: []}] = concluded.hypotheses
    end

    test "requires a world" do
      investigation = Types.Investigation.new("Paris", hypotheses: [Types.Hypothesis.new("Paris capital")])

      assert_raise FunctionClauseError, fn ->
        Corroborator.test_hypotheses("", investigation, [])
      end
    end
  end

  describe "find_conflicts/2" do
    test "detects negation conflicts" do
      finding = build_finding("Paris is the capital", "Paris", "source1.com")

      existing = [build_finding("Paris is not the capital", "Paris", "source2.com")]

      conflicts = Corroborator.find_conflicts(finding, existing)

      assert length(conflicts) == 1
    end

    test "detects number disagreements" do
      finding = build_finding("The population is 14 million", "Tokyo", "source1.com")

      existing = [build_finding("The population is 37 million", "Tokyo", "source2.com")]

      conflicts = Corroborator.find_conflicts(finding, existing)

      assert length(conflicts) == 1
    end

    test "does not flag similar claims as conflicts" do
      finding = build_finding("Paris is the capital of France", "Paris", "source1.com")

      existing = [build_finding("Paris is the capital of France", "Paris", "source2.com")]

      conflicts = Corroborator.find_conflicts(finding, existing)

      assert conflicts == []
    end
  end

  defp build_finding(claim, entity, domain) do
    source = SourceInfo.new("https://#{domain}/article", reliability_score: 0.7)

    Finding.new(claim, entity, source, entity_type: "location", confidence: 0.7)
  end
end