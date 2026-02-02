defmodule Brain.Epistemic.SelfKnowledgeFlowTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.SelfKnowledgeAnalyzer
  alias Brain.Epistemic.{UserModelStore, BeliefStore}
  alias Brain.Epistemic.Types.SelfKnowledgeAssessment
  alias Brain.Response.Synthesizer

  setup do
    # Start common test services (including IntentClassifierSimple)
    start_test_services()

    # Start stores under ExUnit supervision
    ensure_started(UserModelStore)
    ensure_started(BeliefStore)

    # Clear before each test
    UserModelStore.clear_all()
    BeliefStore.clear()

    :ok
  end

  describe "SelfKnowledgeAnalyzer.is_self_knowledge_query?/1" do
    test "detects 'what do you know about me'" do
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("What do you know about me?")
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("what do you know about me")
    end

    test "detects 'do you remember me'" do
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("Do you remember me?")
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("do you remember anything about me")
    end

    test "detects privacy probes" do
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("Are you tracking me?")
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("are you monitoring me")
    end

    test "detects variations" do
      # These variations work with both classifier and keyword fallback
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("What have you learned about me?")
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("Tell me what you know about me")
      # Note: "How much do you actually know?" requires the trained classifier
      # because the keyword fallback needs "about me" pattern
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?("How much do you know about me?")
    end

    test "does not match regular queries" do
      refute SelfKnowledgeAnalyzer.is_self_knowledge_query?("What's the weather?")
      refute SelfKnowledgeAnalyzer.is_self_knowledge_query?("Hello!")
      refute SelfKnowledgeAnalyzer.is_self_knowledge_query?("Play some music")
    end
  end

  describe "SelfKnowledgeAnalyzer.detect_query_type/1" do
    test "identifies self_query type" do
      {:ok, type, confidence} =
        SelfKnowledgeAnalyzer.detect_query_type("What do you know about me?")

      assert type == :self_query
      # Confidence varies based on whether classifier or fallback is used
      assert confidence >= 0.5
    end

    test "identifies memory_check type" do
      {:ok, type, _confidence} = SelfKnowledgeAnalyzer.detect_query_type("Do you remember me?")

      assert type == :memory_check
    end

    test "identifies privacy_probe type" do
      {:ok, type, _confidence} = SelfKnowledgeAnalyzer.detect_query_type("Are you tracking me?")

      assert type == :privacy_probe
    end

    test "returns no_match for regular queries" do
      result = SelfKnowledgeAnalyzer.detect_query_type("What's the weather?")

      assert result == :no_match
    end
  end

  describe "SelfKnowledgeAnalyzer.build_self_knowledge_assessment/1" do
    test "builds assessment from user model" do
      # Set up user with some facts
      UserModelStore.update_fact("user1", :name, "Alice", :explicit, 0.95)
      UserModelStore.update_fact("user1", :location, "Seattle", :explicit, 0.85)
      UserModelStore.update_fact("user1", :occupation, "Developer", :inferred, 0.6)
      UserModelStore.update_fact("user1", :hobby, "Hiking", :assumed, 0.3)

      assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment("user1")

      assert assessment.user_id == "user1"
      assert length(assessment.discloseable) >= 1
      assert assessment.total_facts == 4
    end

    test "returns empty assessment for unknown user" do
      assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment("unknown_user")

      assert assessment.user_id == "unknown_user"
      assert assessment.discloseable == []
      assert assessment.inferred_uncertain == []
    end
  end

  describe "Synthesizer.synthesize_self_knowledge_response/2" do
    test "generates response with no knowledge" do
      assessment = SelfKnowledgeAssessment.new("user1")

      response = Synthesizer.synthesize_self_knowledge_response(assessment)

      assert is_binary(response)
      assert String.length(response) > 0
      # Should indicate lack of knowledge
      assert response =~ ~r/don't|haven't|much|yet|getting to know/i
    end

    test "generates response with knowledge" do
      # Build assessment with some facts
      UserModelStore.update_fact("user1", :name, "Alice", :explicit, 0.95)
      UserModelStore.update_fact("user1", :location, "Seattle", :explicit, 0.85)

      assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment("user1")
      response = Synthesizer.synthesize_self_knowledge_response(assessment)

      assert is_binary(response)
      assert String.length(response) > 0
    end

    test "includes hedging for uncertain knowledge" do
      UserModelStore.update_fact("user1", :occupation, "Developer", :inferred, 0.5)

      assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment("user1")
      response = Synthesizer.synthesize_self_knowledge_response(assessment)

      # Should have some hedging language or acknowledge limited knowledge
      assert response =~
               ~r/remember|think|impression|might|could|correct|learned|getting|know|yet|shared/i
    end
  end

  describe "Synthesizer.determine_rhetorical_strategy/2" do
    test "returns :no_knowledge for empty assessment" do
      assessment = SelfKnowledgeAssessment.new("user1")

      strategy = Synthesizer.determine_rhetorical_strategy(assessment, %{})

      assert strategy == :no_knowledge
    end

    test "returns :limited_knowledge for few facts" do
      assessment = %SelfKnowledgeAssessment{
        user_id: "user1",
        discloseable: [%{key: :name, value: "Alice", confidence: 0.9, provenance: :explicit}],
        inferred_uncertain: [],
        should_avoid: [],
        total_facts: 1,
        assessment_timestamp: DateTime.utc_now()
      }

      strategy = Synthesizer.determine_rhetorical_strategy(assessment, %{})

      assert strategy == :limited_knowledge
    end

    test "returns :moderate_knowledge for several facts" do
      facts =
        for i <- 1..4,
            do: %{key: :"fact_#{i}", value: "value", confidence: 0.8, provenance: :explicit}

      assessment = %SelfKnowledgeAssessment{
        user_id: "user1",
        discloseable: facts,
        inferred_uncertain: [],
        should_avoid: [],
        total_facts: 4,
        assessment_timestamp: DateTime.utc_now()
      }

      strategy = Synthesizer.determine_rhetorical_strategy(assessment, %{})

      assert strategy == :moderate_knowledge
    end
  end

  describe "end-to-end self-knowledge query flow" do
    test "complete flow from query to response" do
      # 1. Set up user knowledge
      user_id = "test_user_#{:rand.uniform(10000)}"
      UserModelStore.update_fact(user_id, :name, "Bob", :explicit, 0.95)
      UserModelStore.update_fact(user_id, :location, "Portland", :explicit, 0.85)
      UserModelStore.update_fact(user_id, :interest, "AI", :inferred, 0.7)

      # 2. Detect meta-cognitive query
      query = "What do you know about me?"
      assert SelfKnowledgeAnalyzer.is_self_knowledge_query?(query)

      # 3. Build assessment
      assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment(user_id)
      assert assessment.total_facts >= 2

      # 4. Generate response
      response =
        Synthesizer.synthesize_self_knowledge_response(assessment,
          context: %{user_initiated: true}
        )

      assert is_binary(response)
      assert String.length(response) > 20

      # 5. Response should have appropriate structure
      # Not too confident
      refute response =~ "I know that you are"
      # Should have some acknowledgment of knowledge or hedging
      assert response =~
               ~r/correct|wrong|off|mixed|remember|recall|think|mentioned|know|conversations|shared/i
    end
  end
end
