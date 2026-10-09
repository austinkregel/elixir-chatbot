defmodule Brain.Analysis.NoveltyDetectorTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.NoveltyDetector

  describe "maybe_kg_downweight/4" do
    setup do
      ensure_started(World.Manager)
      setup_world_sandbox()

      {:ok, with_belief} = create_test_world("novelty_with_belief")
      {:ok, without_belief} = create_test_world("novelty_without_belief")

      # Both worlds embed with a vocabulary that knows the belief's words, so the
      # utterance and the belief's text embed identically.
      for world <- [with_belief, without_belief] do
        :ok =
          World.Embedder.load_model(world.id, %{
            vocabulary: %{"dogs" => 0, "animals" => 1},
            idf_weights: %{"dogs" => 1.0, "animals" => 1.0}
          })
      end

      {:ok, _belief_id} = Brain.Epistemic.BeliefStore.add_belief(with_belief.id, "dogs", :are, "animals")

      {:ok, with_belief: with_belief.id, without_belief: without_belief.id}
    end

    test "downweights against the beliefs held in the named world, and only those",
         %{with_belief: with_belief, without_belief: without_belief} do
      # Identical embeddings: similarity 1.0, so the score is halved.
      assert_in_delta NoveltyDetector.maybe_kg_downweight(0.8, "dogs are animals", nil, with_belief), 0.4, 1.0e-9
      assert NoveltyDetector.maybe_kg_downweight(0.8, "dogs are animals", nil, without_belief) == 0.8
    end

    test "is_novel?/3 passes its :world_id to the downweight", %{with_belief: with_belief} do
      assert {:novel, downweighted} = NoveltyDetector.is_novel?(0.3, 0.1, text: "dogs are animals", world_id: with_belief)
      assert {:novel, raw} = NoveltyDetector.is_novel?(0.3, 0.1)
      assert_in_delta downweighted, raw * 0.5, 1.0e-9
    end

    test "a text without a world raises" do
      assert_raise ArgumentError, ~r/NoveltyDetector needs a :world_id/, fn ->
        NoveltyDetector.is_novel?(0.3, 0.1, text: "dogs are animals")
      end
    end

    test "no text needs no world" do
      assert NoveltyDetector.maybe_kg_downweight(0.8, "", nil, nil) == 0.8
    end
  end

  describe "is_novel?/3" do
    test "returns novel for low confidence" do
      assert {:novel, _score} = NoveltyDetector.is_novel?(0.3, 0.5)
    end

    test "returns novel for small margin" do
      assert {:novel, _score} = NoveltyDetector.is_novel?(0.7, 0.1)
    end

    test "returns not_novel for high confidence and large margin" do
      assert :not_novel = NoveltyDetector.is_novel?(0.8, 0.5)
    end

    test "respects custom thresholds" do
      assert :not_novel = NoveltyDetector.is_novel?(0.6, 0.3, novelty_threshold: 0.4, margin_threshold: 0.2)
      assert {:novel, _score} = NoveltyDetector.is_novel?(0.3, 0.1, novelty_threshold: 0.5, margin_threshold: 0.2)
    end

    test "graph-unknown entities boost novelty score" do
      entities = [%{value: "Zorblax", entity_type: "technology", confidence: 0.8, graph_known: false}]

      assert {:novel, score_with_graph} = NoveltyDetector.is_novel?(0.8, 0.5, entities: entities)
      assert score_with_graph > 0.0
    end

    test "graph-known entities do not trigger novelty alone" do
      entities = [%{value: "Elixir", entity_type: "technology", confidence: 0.9, graph_known: true}]

      assert :not_novel = NoveltyDetector.is_novel?(0.8, 0.5, entities: entities)
    end
  end

  describe "is_substantive?/2" do
    test "returns true for directive speech acts" do
      speech_act = %{category: :directive, sub_type: :command}
      assert NoveltyDetector.is_substantive?(speech_act, "smarthome.lights.switch.on")
    end

    test "returns true for assertive speech acts" do
      speech_act = %{category: :assertive, sub_type: :statement}
      assert NoveltyDetector.is_substantive?(speech_act, "unknown")
    end

    test "returns false for well-handled expressives" do
      speech_act = %{category: :expressive, sub_type: :greeting}
      refute NoveltyDetector.is_substantive?(speech_act, "smalltalk.greetings.hello")
    end

    test "returns true for other expressives" do
      speech_act = %{category: :expressive, sub_type: :general}
      assert NoveltyDetector.is_substantive?(speech_act, "unknown")
    end
  end

  describe "is_researchable?/3" do
    test "rejects short social inputs like greetings and introductions" do
      entities = [%{value: "Austin", entity_type: "person", confidence: 0.8}]
      speech_act = %{category: :expressive, sub_type: :greeting}

      refute NoveltyDetector.is_researchable?("Hello I'm Austin", entities, speech_act)
    end

    test "rejects inputs with too few tokens" do
      entities = [%{value: "London", entity_type: "location", confidence: 0.9}]
      speech_act = %{category: :directive, sub_type: :query}

      refute NoveltyDetector.is_researchable?("weather London", entities, speech_act)
    end

    test "accepts directive inputs with location entities" do
      entities = [%{value: "London", entity_type: "location", confidence: 0.9}]
      speech_act = %{category: :directive, sub_type: :query}

      assert NoveltyDetector.is_researchable?(
               "What is the weather like in London today",
               entities,
               speech_act
             )
    end

    test "accepts assertive claims about domain concepts" do
      entities = [%{value: "Elixir", entity_type: "technology", confidence: 0.7}]
      speech_act = %{category: :assertive, sub_type: :claim}

      assert NoveltyDetector.is_researchable?(
               "Elixir uses the BEAM virtual machine for concurrency",
               entities,
               speech_act
             )
    end

    test "rejects inputs with only person entities and no knowledge-oriented speech act" do
      entities = [%{value: "John", entity_type: "person", confidence: 0.9}]
      speech_act = %{category: :expressive, sub_type: :general}

      refute NoveltyDetector.is_researchable?(
               "My name is John and I like coding",
               entities,
               speech_act
             )
    end

    test "accepts inputs with knowledge-oriented speech act even without entities" do
      entities = []
      speech_act = %{category: :directive, sub_type: :query}

      assert NoveltyDetector.is_researchable?(
               "How does photosynthesis work in plants",
               entities,
               speech_act
             )
    end

    test "rejects inputs with no entities and non-knowledge speech act" do
      entities = []
      speech_act = %{category: :expressive, sub_type: :general}

      refute NoveltyDetector.is_researchable?(
               "I am feeling pretty good today actually",
               entities,
               speech_act
             )
    end

    test "rejects inputs with only low-confidence entities" do
      entities = [%{value: "thing", entity_type: "object", confidence: 0.1}]
      speech_act = %{category: :assertive, sub_type: nil}

      refute NoveltyDetector.is_researchable?(
               "That thing over there is pretty cool",
               entities,
               speech_act
             )
    end

    test "accepts directive questions about organizations" do
      entities = [%{value: "NASA", entity_type: "organization", confidence: 0.95}]
      speech_act = %{category: :directive, sub_type: :query}

      assert NoveltyDetector.is_researchable?(
               "What missions has NASA planned for next year",
               entities,
               speech_act
             )
    end

    test "graph-unknown entities make input researchable" do
      entities = [%{value: "Zorblax", entity_type: "technology", confidence: 0.8, graph_known: false}]
      speech_act = %{category: :directive, sub_type: :query}

      assert NoveltyDetector.is_researchable?(
               "Tell me about the Zorblax framework features",
               entities,
               speech_act
             )
    end
  end
end
