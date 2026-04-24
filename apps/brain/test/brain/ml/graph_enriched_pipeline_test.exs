defmodule Brain.ML.GraphEnrichedPipelineTest do
  @moduledoc """
  Cross-capability integration test that exercises Tier 3 model
  capabilities together: Poincare, EventLinker, KG TripleScorer, and SRL.

  Verifies that each phase's output can be consumed by subsequent phases
  and that no phase produces errors that block downstream processing.
  """

  use ExUnit.Case, async: false
  @moduletag :integration

  alias Brain.ML.Poincare
  alias Brain.ML.KnowledgeGraph.TripleScorer
  alias Brain.Analysis.{EventLinker, SemanticRoleLabeler}
  alias Brain.Epistemic.StanceTracker

  @test_sentences [
    "The president visited Berlin yesterday",
    "John likes coffee and tea",
    "She ran quickly in the park",
    "Einstein discovered relativity",
    "The company hired new engineers",
    "Mary gave the book to John",
    "It rained heavily in Seattle",
    "The teacher explained quantum physics",
    "Dogs are a type of animal",
    "Paris is the capital of France"
  ]

  describe "cross-capability pipeline" do
    test "Phase 1: Poincare distance functions are consistent" do
      pairs = [
        {"dog", "animal"},
        {"cat", "animal"},
        {"animal", "entity"},
        {"city", "location"},
        {"location", "entity"}
      ]

      {:ok, embeddings, entity_to_idx, _idx_to_entity} = Poincare.Embeddings.train(pairs,
        dim: 5,
        epochs: 50,
        learning_rate: 0.01
      )

      # All embeddings should be inside the Poincare ball
      for {_name, idx} <- entity_to_idx do
        emb = embeddings[idx]
        norm = Nx.sqrt(Nx.sum(Nx.pow(emb, 2))) |> Nx.to_number()
        assert norm < 1.0, "Embedding should be inside ball, norm: #{norm}"
      end

      # Distances should be valid floats
      dog_idx = Map.fetch!(entity_to_idx, "dog")
      cat_idx = Map.fetch!(entity_to_idx, "cat")
      dist = Poincare.Distance.distance(embeddings[dog_idx], embeddings[cat_idx]) |> Nx.to_number()
      assert is_float(dist)
      assert dist > 0.0
    end

    test "Phase 2: EventLinker processes events without errors" do
      events = [
        %{action: %{verb: "visited"}, source_tokens: [0, 2, 3]},
        %{action: %{verb: "discovered"}, source_tokens: [0, 1, 2]}
      ]

      entities = [
        %{text: "President", type: :person, start_pos: 0},
        %{text: "Berlin", type: :location, start_pos: 3},
        %{text: "yesterday", type: :temporal, start_pos: 4}
      ]

      tokens = Enum.map(0..4, &%{text: "t#{&1}", normalized: "t#{&1}"})
      pos_tags = List.duplicate("NN", 5)

      frames = EventLinker.link(events, entities, tokens, pos_tags)

      assert length(frames) == 2
      for frame <- frames do
        assert is_binary(frame.trigger)
        assert is_list(frame.arguments)
        assert is_list(frame.temporal_relations)
      end
    end

    test "Phase 3: Triple scoring produces valid scores" do
      triples = [
        {"dog", "IS_A", "animal"},
        {"cat", "IS_A", "animal"},
        {"Paris", "CAPITAL_OF", "France"},
        {"Berlin", "CAPITAL_OF", "Germany"},
        {"Einstein", "BORN_IN", "Germany"},
        {"Newton", "BORN_IN", "England"}
      ]

      {:ok, model, params, vocab, _config} = TripleScorer.train(triples,
        epochs: 30,
        neg_ratio: 3,
        embedding_dim: 16,
        hidden_dim: 32
      )

      for {h, r, t} <- triples do
        text = "[HEAD] #{h} [REL] #{r} [TAIL] #{t}"
        {input, mask} = TripleScorer.encode_single_public(text, vocab)
        output = Axon.predict(model, params, %{"input" => input, "mask" => mask})
        score = Nx.squeeze(output) |> Nx.to_number()

        assert score > 0.0 and score < 1.0,
          "Score for #{h}-#{r}-#{t} should be in (0,1), got #{score}"
      end
    end

    test "Phase 4: SRL produces frames from BIO tags" do
      for sentence <- @test_sentences do
        tokens = String.split(sentence)
        bio_tags = generate_simple_bio_tags(tokens)

        frames = SemanticRoleLabeler.label(tokens, bio_tags)
        assert is_list(frames)

        for frame <- frames do
          assert is_binary(frame.predicate)
          assert is_list(frame.arguments)
        end
      end
    end

    test "Phase 4b: SRL frames convert to triples without errors" do
      tokens = ["John", "visited", "Berlin", "yesterday"]
      bio_tags = ["B-ARG0", "B-V", "B-ARG1", "B-ARGM-TMP"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      triples = SemanticRoleLabeler.to_triples(frames)

      assert length(triples) >= 1
      for {subj, pred, obj} <- triples do
        assert is_binary(subj)
        assert is_binary(pred)
        assert is_binary(obj)
      end
    end

    test "Phase 5A: Stance tracker records and detects drift" do
      {:ok, tracker} = StanceTracker.start_link(name: :"integration_tracker_#{:rand.uniform(100000)}")

      StanceTracker.record_stance("test_conv", "climate", 0.3, :system, tracker)
      StanceTracker.record_stance("test_conv", "climate", 0.8, :user, tracker)
      StanceTracker.record_stance("test_conv", "climate", 0.7, :system, tracker)
      Process.sleep(100)

      {:ok, drift_info} = StanceTracker.check_drift("test_conv", "climate", tracker)
      assert is_map(drift_info)
      assert drift_info.absolute_drift > 0.3
    end

    test "All phases produce compatible output formats for graph writing" do
      # SRL frames produce triples
      tokens = ["John", "gave", "Mary", "a", "book"]
      bio_tags = ["B-ARG0", "B-V", "B-ARG2", "B-ARG1", "I-ARG1"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      triples = SemanticRoleLabeler.to_triples(frames)

      # Triples can be scored
      assert length(triples) >= 1
      for {s, p, o} <- triples do
        assert is_binary(s) and is_binary(p) and is_binary(o)
      end

      # Event frames have the right structure for graph writer
      events = [%{action: %{verb: "gave"}, source_tokens: [0, 1, 2]}]
      entities = [%{text: "John", type: :person, start_pos: 0}]
      event_frames = EventLinker.link(events, entities, tokens, List.duplicate("NN", 5))

      for frame <- event_frames do
        assert Map.has_key?(frame, :trigger)
        assert Map.has_key?(frame, :arguments)
        for arg <- frame.arguments do
          assert Map.has_key?(arg, :role)
          assert Map.has_key?(arg, :text)
        end
      end
    end
  end

  defp generate_simple_bio_tags(tokens) do
    # Simple heuristic for test: first token = ARG0, verbs = V, rest = ARG1
    Enum.with_index(tokens)
    |> Enum.map(fn {token, idx} ->
      downcased = String.downcase(token)
      cond do
        idx == 0 -> "B-ARG0"
        downcased in ~w(visited discovered gave ran explained hired is are) -> "B-V"
        idx > 0 and idx < length(tokens) - 1 -> "B-ARG1"
        true -> "O"
      end
    end)
  end
end
