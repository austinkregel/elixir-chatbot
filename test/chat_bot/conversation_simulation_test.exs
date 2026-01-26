defmodule ChatBot.ConversationSimulationTest do
  @moduledoc """
  Tests multi-turn conversations with controlled datasets.

  These tests simulate realistic conversation flows and verify:
  - Context is maintained across turns
  - User facts are extracted and remembered
  - Meta-cognitive queries return appropriate responses
  - The epistemic system correctly models user knowledge

  To run with a clean slate, each test clears the epistemic stores.
  """
  use ExUnit.Case, async: false

  alias ChatBot.Brain
  alias ChatBot.Epistemic.{BeliefStore, UserModelStore}
  alias ChatBot.Epistemic.Types.Config
  alias ChatBot.Analysis.SelfKnowledgeAnalyzer
  import ChatBot.TestHelpers

  # Test fixtures - controlled conversation scenarios
  @introduction_scenario [
    {"Hello, my name is Alex", :greeting_with_name},
    {"I'm from Seattle", :location_statement},
    {"I work at a tech company", :work_statement}
  ]

  @preference_scenario [
    {"I like coffee", :preference_statement},
    {"I prefer dark mode", :preference_statement},
    {"I love hiking on weekends", :preference_statement}
  ]

  @meta_cognitive_queries [
    "What do you know about me?",
    "Do you remember my name?",
    "What have you learned about me?"
  ]

  setup do
    start_brain_services()

    # Ensure epistemic stores are started and cleared
    ensure_epistemic_stores_started()

    # Create a conversation for the test
    {:ok, conversation_id} = Brain.create_conversation()

    # Generate a unique user_id for this test
    user_id = "test_user_#{:rand.uniform(100_000)}"

    %{
      conversation_id: conversation_id,
      user_id: user_id
    }
  end

  describe "introduction conversation flow" do
    test "extracts and remembers user name from introduction", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      # User introduces themselves
      {:ok, response1} = Brain.evaluate(conv_id, "Hello, my name is Alex", user_id: user_id)

      # Positive: Should recognize greeting/introduction
      assert response1 =~ ~r/hello|hi|hey|nice|meet|welcome|alex/i,
             "Expected greeting/introduction response, got: #{response1}"

      # Negative: Regression test
      refute response1 =~ ~r/bye|goodbye/i

      # Verify the name was potentially extracted (if epistemic system is enabled)
      if Config.enabled?() and Config.auto_extraction_enabled?() do
        # Give a moment for async processing
        Process.sleep(50)

        case UserModelStore.get(user_id) do
          nil ->
            # Model not created yet, that's okay for some flows
            :ok

          model ->
            # If model exists, check if name was captured
            # (depends on extraction patterns)
            assert is_map(model.facts)
        end
      end
    end

    test "maintains context across multiple turns", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      # Turn 1: Greeting
      {:ok, response1} = Brain.evaluate(conv_id, "Hello!", user_id: user_id)
      # Positive: Should be greeting response (broad patterns)
      assert response1 =~ ~r/hello|hi|hey|welcome|nice|meet|how.*you|good|day|going|wuz/i,
             "Expected greeting response, got: #{response1}"

      # Turn 2: Statement
      {:ok, response2} = Brain.evaluate(conv_id, "I like coffee", user_id: user_id)
      # Positive: Should produce a non-empty response (acknowledgment or conversation)
      assert String.length(response2) > 0,
             "Expected non-empty response, got empty"

      # Turn 3: Question
      {:ok, response3} = Brain.evaluate(conv_id, "What's the weather like?", user_id: user_id)
      # Positive: Should produce a non-empty response
      assert String.length(response3) > 0,
             "Expected non-empty response, got empty"

      # Negative: Ensure responses are different (regression test)
      refute response1 == response2
      refute response2 == response3
    end

    test "full introduction scenario", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      responses =
        for {input, _type} <- @introduction_scenario do
          {:ok, response} = Brain.evaluate(conv_id, input, user_id: user_id)
          {input, response}
        end

      # All turns should produce responses
      assert length(responses) == length(@introduction_scenario)

      for {_input, response} <- responses do
        assert is_binary(response)
        assert String.length(response) > 0
      end
    end
  end

  describe "preference learning flow" do
    test "processes preference statements", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      for {input, _type} <- @preference_scenario do
        {:ok, response} = Brain.evaluate(conv_id, input, user_id: user_id)

        # Positive: Should produce a non-empty response (acknowledgment or conversation)
        assert String.length(response) > 0,
               "Expected non-empty response to preference statement, got empty"

        # Negative: Regression test
        refute response =~ ~r/bye|goodbye|see you later/i
      end
    end
  end

  describe "meta-cognitive query flow" do
    test "responds to self-knowledge queries with learned facts", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      # First, provide some information
      {:ok, _} = Brain.evaluate(conv_id, "Hello, I'm from Portland", user_id: user_id)
      {:ok, _} = Brain.evaluate(conv_id, "I like hiking", user_id: user_id)

      # Verify facts were stored in user model
      model = UserModelStore.get(user_id)
      assert model != nil, "User model should exist after providing facts"

      # Check that location was captured (might be under :location or :city)
      # Values are normalized to lowercase during extraction
      location_value =
        Map.get(model.facts, :location) ||
          Map.get(model.facts, :city) ||
          Map.get(model.facts, :from)

      assert location_value != nil,
             "Expected location fact to be stored. Got facts: #{inspect(model.facts)}"

      assert String.downcase(to_string(location_value)) == "portland",
             "Expected Portland to be stored. Got: #{inspect(location_value)}"

      # Check that hiking preference was captured
      hiking_value =
        Map.get(model.facts, :likes) ||
          Map.get(model.facts, :hobby) ||
          Map.get(model.facts, :preference)

      # Also check if hiking appears anywhere in the facts
      hiking_in_facts =
        hiking_value != nil or
          Enum.any?(Map.values(model.facts), fn v ->
            is_binary(v) and String.contains?(String.downcase(v), "hiking")
          end)

      assert hiking_in_facts,
             "Expected hiking to be stored in user model. Got facts: #{inspect(model.facts)}"

      # Now ask what the bot knows
      {:ok, response} = Brain.evaluate(conv_id, "What do you know about me?", user_id: user_id)

      assert String.length(response) > 0
      # Should not be a farewell
      refute response =~ ~r/bye|goodbye|see you later/i

      # Response should mention Portland (the location we provided)
      response_lower = String.downcase(response)

      assert String.contains?(response_lower, "portland") or
               String.contains?(response_lower, "location") or
               String.contains?(response_lower, "from"),
             "Expected response to mention Portland or location. Got: #{response}"

      # Response should mention hiking (the preference we provided)
      assert String.contains?(response_lower, "hiking") or
               String.contains?(response_lower, "like") or
               String.contains?(response_lower, "hobby") or
               String.contains?(response_lower, "enjoy"),
             "Expected response to mention hiking or preferences. Got: #{response}"
    end

    test "detects meta-cognitive intents correctly" do
      # These should be detected as self-knowledge queries
      for query <- @meta_cognitive_queries do
        assert SelfKnowledgeAnalyzer.is_self_knowledge_query?(query),
               "Expected '#{query}' to be detected as self-knowledge query"
      end

      # These should NOT be detected as self-knowledge queries
      refute SelfKnowledgeAnalyzer.is_self_knowledge_query?("What's the weather?")
      refute SelfKnowledgeAnalyzer.is_self_knowledge_query?("Hello!")
      refute SelfKnowledgeAnalyzer.is_self_knowledge_query?("Play some music")
    end

    test "builds assessment from user model", %{user_id: user_id} do
      # Directly add facts to the user model
      UserModelStore.update_fact(user_id, :name, "TestUser", :explicit, 0.95)
      UserModelStore.update_fact(user_id, :location, "TestCity", :explicit, 0.85)

      # Build assessment
      assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment(user_id)

      assert assessment.user_id == user_id
      assert assessment.total_facts >= 2
    end
  end

  describe "conversation with epistemic memory" do
    test "fact persists across conversation turns", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      # Store a fact directly
      UserModelStore.update_fact(user_id, :test_fact, "test_value", :explicit, 0.9)

      # Have a conversation
      {:ok, _} = Brain.evaluate(conv_id, "Hello!", user_id: user_id)
      {:ok, _} = Brain.evaluate(conv_id, "How are you?", user_id: user_id)

      # Fact should still exist
      {:ok, model} = UserModelStore.get_or_create(user_id)
      assert Map.has_key?(model.facts, :test_fact)
    end

    test "multiple users have isolated knowledge", %{conversation_id: _conv_id} do
      user1 = "isolation_test_user_1"
      user2 = "isolation_test_user_2"

      # User 1 states a preference
      UserModelStore.update_fact(user1, :favorite_color, "blue", :explicit, 0.9)

      # User 2 states different preference
      UserModelStore.update_fact(user2, :favorite_color, "red", :explicit, 0.9)

      # Verify isolation - facts map stores values directly
      {:ok, model1} = UserModelStore.get_or_create(user1)
      {:ok, model2} = UserModelStore.get_or_create(user2)

      assert model1.facts[:favorite_color] == "blue"
      assert model2.facts[:favorite_color] == "red"

      # Confidence is stored in epistemic_bounds
      assert model1.epistemic_bounds[:favorite_color] == 0.9
      assert model2.epistemic_bounds[:favorite_color] == 0.9
    end
  end

  describe "fact extraction assertions" do
    @describetag :fact_extraction

    test "extracts name from 'my name is X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "my name is Alex", user_id: user_id)

      # Assert the name was extracted and stored
      assert_fact_extracted(user_id, :name, "alex")
    end

    test "extracts name from 'call me X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "call me Jordan", user_id: user_id)

      assert_fact_extracted(user_id, :name, "jordan")
    end

    test "extracts location from 'I'm from X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "I'm from Seattle", user_id: user_id)

      assert_fact_extracted(user_id, :location, "seattle")
    end

    test "extracts location from 'I live in X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "I live in Portland", user_id: user_id)

      assert_fact_extracted(user_id, :location, "portland")
    end

    test "extracts preference from 'I like X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "I like coffee", user_id: user_id)

      assert_fact_extracted(user_id, :likes, "coffee")
    end

    test "extracts preference from 'I prefer X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "I prefer tea", user_id: user_id)

      assert_fact_extracted(user_id, :preference, "tea")
    end

    test "extracts workplace from 'I work at X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "I work at Google", user_id: user_id)

      assert_fact_extracted(user_id, :workplace, "google")
    end

    test "extracts workplace from 'I work for X'", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _response} = Brain.evaluate(conv_id, "I work for Microsoft", user_id: user_id)

      assert_fact_extracted(user_id, :workplace, "microsoft")
    end

    test "extracts multiple facts from conversation", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      # User shares multiple pieces of information
      {:ok, _} = Brain.evaluate(conv_id, "my name is Sam", user_id: user_id)
      {:ok, _} = Brain.evaluate(conv_id, "I'm from Denver", user_id: user_id)
      {:ok, _} = Brain.evaluate(conv_id, "I like hiking", user_id: user_id)

      # All facts should be stored
      assert_fact_extracted(user_id, :name, "sam")
      assert_fact_extracted(user_id, :location, "denver")
      assert_fact_extracted(user_id, :likes, "hiking")

      # Verify the user model has multiple facts
      {:ok, model} = UserModelStore.get_or_create(user_id)
      assert map_size(model.facts) >= 3
    end

    test "facts have appropriate confidence levels", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, _} = Brain.evaluate(conv_id, "my name is Taylor", user_id: user_id)

      {:ok, model} = UserModelStore.get_or_create(user_id)

      # Explicit statements should have high confidence
      if Map.has_key?(model.epistemic_bounds, :name) do
        assert model.epistemic_bounds[:name] >= 0.8,
               "Explicit name statement should have high confidence"
      end
    end
  end

  describe "controlled scenario testing" do
    @tag :scenario
    test "complete conversation scenario with fact verification", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      # Scenario: User introduces themselves, states preferences, then asks what bot knows

      # Phase 1: Introduction
      {:ok, r1} = Brain.evaluate(conv_id, "Hi there! my name is Jordan", user_id: user_id)
      assert String.length(r1) > 0

      # Phase 2: Share information
      {:ok, r2} = Brain.evaluate(conv_id, "I'm from Chicago", user_id: user_id)
      assert String.length(r2) > 0

      {:ok, r3} = Brain.evaluate(conv_id, "I like playing guitar", user_id: user_id)
      assert String.length(r3) > 0

      # Verify facts were extracted
      assert_fact_extracted(user_id, :name, "jordan")
      assert_fact_extracted(user_id, :location, "chicago")
      assert_fact_extracted(user_id, :likes, "playing guitar")

      # Phase 3: Test recall (if epistemic enabled)
      if Config.enabled?() do
        {:ok, r4} = Brain.evaluate(conv_id, "What do you know about me?", user_id: user_id)
        assert String.length(r4) > 0
        # Should not be dismissive
        refute r4 =~ ~r/don't know anything|no idea who you are/i

        # Response should reference at least some of the learned facts
        r4_lower = String.downcase(r4)

        facts_mentioned =
          String.contains?(r4_lower, "jordan") or String.contains?(r4_lower, "name") or
            (String.contains?(r4_lower, "chicago") or String.contains?(r4_lower, "location")) or
            (String.contains?(r4_lower, "guitar") or String.contains?(r4_lower, "like"))

        assert facts_mentioned,
               "Expected response to mention at least one learned fact (jordan/chicago/guitar). Got: #{r4}"
      end

      # Phase 4: Normal conversation continues
      {:ok, r5} = Brain.evaluate(conv_id, "What's the weather like today?", user_id: user_id)
      assert String.length(r5) > 0

      # Phase 5: Farewell
      {:ok, r6} = Brain.evaluate(conv_id, "Thanks, goodbye!", user_id: user_id)
      assert String.length(r6) > 0

      # Final verification: all facts still present after conversation
      {:ok, final_model} = UserModelStore.get_or_create(user_id)

      assert map_size(final_model.facts) >= 3,
             "Expected at least 3 facts, got #{map_size(final_model.facts)}: #{inspect(final_model.facts)}"
    end
  end

  # Helper functions

  defp ensure_epistemic_stores_started do
    # Start stores under ExUnit supervision
    ensure_started(UserModelStore)
    ensure_started(BeliefStore)

    # Clear data before each test
    UserModelStore.clear_all()
    BeliefStore.clear()

    :ok
  end

  # Asserts that a fact was extracted and stored in the user's model.
  # The expected_value is compared case-insensitively and with trimming.
  defp assert_fact_extracted(user_id, predicate, expected_value) do
    # Give a moment for any async processing
    Process.sleep(10)

    case UserModelStore.get(user_id) do
      nil ->
        flunk("No user model found for #{user_id}. Expected fact #{predicate}: #{expected_value}")

      model ->
        actual_value = Map.get(model.facts, predicate)

        if actual_value == nil do
          available_facts = Map.keys(model.facts)

          flunk("""
          Fact not found in user model.
            Expected: #{predicate} = "#{expected_value}"
            Available facts: #{inspect(available_facts)}
            Full model.facts: #{inspect(model.facts)}
          """)
        else
          # Normalize for comparison
          normalized_actual = normalize_for_comparison(actual_value)
          normalized_expected = normalize_for_comparison(expected_value)

          assert String.contains?(normalized_actual, normalized_expected),
                 """
                 Fact value mismatch for #{predicate}.
                   Expected to contain: "#{expected_value}"
                   Actual value: "#{actual_value}"
                 """
        end
    end
  end

  defp normalize_for_comparison(value) when is_binary(value) do
    value
    |> String.downcase()
    |> String.trim()
  end

  defp normalize_for_comparison(value) do
    value
    |> to_string()
    |> normalize_for_comparison()
  end
end
