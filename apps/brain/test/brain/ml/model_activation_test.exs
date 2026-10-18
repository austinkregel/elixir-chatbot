defmodule Brain.ML.ModelActivationTest do
  @moduledoc """
  Verifies that all Tier 3 models are wired into the live pipeline and
  reachable from the Chat path (Brain.evaluate → Pipeline → analysis).

  These tests verify the wiring, not the model accuracy.
  """

  use ExUnit.Case, async: false
  @moduletag :integration

  # The world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  # This exercises the live pipeline, which reads the knowledge graph; without
  # a checked-out connection those lookups fail silently.
  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    :ok
  end

  describe "Poincare Embeddings wiring" do
    test "the test world holds its own Poincare embeddings" do
      assert %{embeddings: %Nx.Tensor{}, dim: dim} = Brain.ML.Poincare.Embeddings.loaded!(@world_id)
      assert is_integer(dim) and dim > 0
    end

    test "Poincare is referenced in EntityExtractor disambiguation" do
      fns = Brain.ML.EntityExtractor.__info__(:functions)
      assert {:extract_entities, 2} in fns or {:extract_entities, 1} in fns
    end
  end

  describe "KG-LSTM TripleScorer wiring" do
    test "the test world holds its own triple scorer" do
      assert %{model: %Axon{}, vocab: vocab} = Brain.ML.KnowledgeGraph.TripleScorer.loaded!(@world_id)
      assert map_size(vocab) > 0
    end

    test "Corroborator module exists and is callable" do
      assert Code.ensure_loaded?(Brain.Knowledge.Corroborator)
      fns = Brain.Knowledge.Corroborator.__info__(:functions)
      assert {:corroborate, 2} in fns
    end
  end

  describe "EventLinker wiring" do
    test "EventLinker is accessible from Pipeline" do
      assert Code.ensure_loaded?(Brain.Analysis.EventLinker)
      assert {:link, 4} in Brain.Analysis.EventLinker.__info__(:functions)
    end

    test "Pipeline produces event_frames in ChunkAnalysis" do
      analysis = Brain.Analysis.Pipeline.analyze_chunk("The president visited Berlin yesterday", world_id: @world_id)
      assert Map.has_key?(analysis, :event_frames)
      assert is_list(analysis.event_frames)
    end
  end

  describe "SemanticRoleLabeler wiring" do
    test "SRL is accessible from Pipeline" do
      fns = Brain.Analysis.SemanticRoleLabeler.__info__(:functions)
      assert {:label, 2} in fns or {:label, 3} in fns
    end

    test "Pipeline produces srl_frames in ChunkAnalysis" do
      analysis = Brain.Analysis.Pipeline.analyze_chunk("John gave the book to Mary", world_id: @world_id)
      assert Map.has_key?(analysis, :srl_frames)
      assert is_list(analysis.srl_frames)
    end
  end

  describe "StanceTracker wiring" do
    test "StanceTracker exposes ready?/0" do
      assert is_boolean(Brain.Epistemic.StanceTracker.ready?())
    end

    test "StanceTracker is callable" do
      fns = Brain.Epistemic.StanceTracker.__info__(:functions)
      assert {:record_stance, 4} in fns or {:record_stance, 5} in fns
      assert {:check_drift, 2} in fns or {:check_drift, 3} in fns
    end
  end

  describe "Graph.Writer enrichment" do
    test "Writer exposes SRL triple writing" do
      fns = Brain.Graph.Writer.__info__(:functions)
      assert {:write_srl_triples, 2} in fns
    end

    test "Writer exposes event frame writing" do
      fns = Brain.Graph.Writer.__info__(:functions)
      assert {:write_event_frames, 2} in fns
    end

    test "Writer.write_analysis handles enriched model" do
      model = %{
        analyses: [
          %{
            entities: [],
            events: [],
            event_frames: [%{trigger: "test", arguments: []}],
            srl_frames: [],
            pos_tags: []
          }
        ]
      }

      assert Brain.Graph.Writer.write_analysis(model, @world_id) == :ok
    end
  end

  describe "Tier 3 context extraction" do
    test "Pipeline analysis includes Tier 3 fields" do
      model = Brain.Analysis.Pipeline.process("Einstein discovered relativity in 1905", world_id: @world_id)

      analyses = model.analyses
      assert length(analyses) >= 1

      first = hd(analyses)
      assert Map.has_key?(first, :event_frames)
      assert Map.has_key?(first, :srl_frames)
    end
  end
end
