defmodule Brain.Knowledge.InvestigationTest do
  @moduledoc """
  `Investigation.record_evidence/3` associates a finding with a hypothesis by
  the similarity of their claims in the named world's embedder
  (`Brain.Worlds.Embedder`), and nowhere else: while the world's vocabulary is being
  rebuilt it waits for the rebuild to finish, and a world that cannot embed
  raises naming the world.

  The worlds are `test_` worlds, cleaned up after each test.
  """

  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Knowledge.Types.{Finding, Hypothesis, Investigation, SourceInfo}
  alias Brain.Memory.Store
  alias Brain.Worlds.Embedder

  # Both claims hold "paris" and one other word. In a world whose vocabulary
  # is only "paris" they embed as the same vector (similarity 1.0); in one
  # whose vocabulary is only "capital" and "city" they are orthogonal (0.0).
  # Their token overlap ({paris, capital} against {paris, city}) is 1/3.
  @hypothesis_claim "Paris capital"
  @finding_claim "Paris city"
  @paris_only %{vocabulary: %{"paris" => 0}, idf_weights: %{"paris" => 1.0}}
  @capital_and_city %{
    vocabulary: %{"capital" => 0, "city" => 1},
    idf_weights: %{"capital" => 1.0, "city" => 1.0}
  }

  setup tags do
    # A rebuild reads the world's episodes from Atlas, inside this test's
    # sandbox transaction.
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    start_world_test_services()
    setup_world_sandbox()

    :ok
  end

  describe "record_evidence/3" do
    test "compares claims in the named world's embedder" do
      {:ok, paris_world} = create_test_world("investigation_paris_only")
      {:ok, capital_city_world} = create_test_world("investigation_capital_and_city")
      :ok = Embedder.load_model(paris_world.id, @paris_only)
      :ok = Embedder.load_model(capital_city_world.id, @capital_and_city)

      finding = build_finding(@finding_claim)

      recorded = Investigation.record_evidence(paris_world.id, investigation(), [finding])
      assert [%Hypothesis{supporting_evidence: [^finding]}] = recorded.hypotheses
      assert recorded.evidence == [finding]
      assert recorded.status == :gathering_evidence

      recorded = Investigation.record_evidence(capital_city_world.id, investigation(), [finding])
      assert [%Hypothesis{supporting_evidence: [], contradicting_evidence: []}] = recorded.hypotheses
      assert recorded.evidence == [finding]
    end

    test "raises naming a world that holds no embedder" do
      world_id = "test_investigation_no_embedder_#{System.unique_integer([:positive])}"

      assert_raise RuntimeError, ~r/world #{inspect(world_id)} has no embedder vocabulary/, fn ->
        Investigation.record_evidence(world_id, investigation(), [build_finding(@finding_claim)])
      end
    end

    test "waits while the world's vocabulary is being rebuilt, then compares in it" do
      {:ok, world} = create_test_world("investigation_rebuilding")
      :ok = Embedder.load_model(world.id, @paris_only)
      store = Process.whereis(Store)
      finding = build_finding(@finding_claim)

      # Hold the store so the rebuild waits in its first phase, reading the
      # world's episodes.
      :ok = :sys.suspend(store)
      rebuild = Task.async(fn -> Embedder.build_vocabulary(world.id, force: true) end)

      record =
        try do
          eventually(
            fn -> Embedder.get_status(world.id).phase end,
            fn phase -> phase == :loading_episodes end
          )

          record = Task.async(fn -> Investigation.record_evidence(world.id, investigation(), [finding]) end)

          # Waiting for the rebuild, not compared without the vocabulary.
          assert Task.yield(record, 200) == nil
          record
        after
          :ok = :sys.resume(store)
        end

      # No episodes to rebuild from: the world keeps its vocabulary, in which
      # the claims are the same vector.
      assert {:ok, :kept} = Task.await(rebuild)
      assert [%Hypothesis{supporting_evidence: [^finding]}] = Task.await(record).hypotheses
    end

    test "requires a world" do
      assert_raise FunctionClauseError, fn ->
        Investigation.record_evidence("", investigation(), [build_finding(@finding_claim)])
      end

      assert_raise FunctionClauseError, fn ->
        Investigation.record_evidence(nil, investigation(), [build_finding(@finding_claim)])
      end
    end
  end

  defp investigation do
    Investigation.new("Paris", hypotheses: [Hypothesis.new(@hypothesis_claim)])
  end

  defp build_finding(claim) do
    source = SourceInfo.new("https://source1.com/article", reliability_score: 0.7)
    Finding.new(claim, "Paris", source, entity_type: "location", confidence: 0.7, world_id: "test_investigation")
  end
end
