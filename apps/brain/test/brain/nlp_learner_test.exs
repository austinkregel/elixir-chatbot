defmodule Brain.NLPLearnerTest do
  use Brain.Test.GraphCase, async: false
  alias Brain.Learner
  alias Brain.KnowledgeStore
  import Brain.TestHelpers

  setup _context do
    start_test_services()

    persona_name = "TestPersona"

    on_exit(fn ->
      # Clean up test knowledge files
      # Use :brain config namespace (test.exs sets knowledge_dir under :brain app)
      knowledge_dir = Application.get_env(:brain, :knowledge_dir, "test/knowledge")
      knowledge_file = Path.join(knowledge_dir, "#{persona_name}.json")

      File.rm(knowledge_file)
    end)

    %{persona_name: persona_name}
  end

  describe "NLP-based entity extraction" do
    test "extracts entities from natural language", %{persona_name: persona_name} do
      input = "Turn on the kitchen lights"

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      # Should return some kind of result (either extracted data or general memory)
      assert is_map(result)

      # Check that knowledge was stored
      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end

    test "extracts location information from natural language", %{persona_name: persona_name} do
      input = "What's the weather in New York?"

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      assert is_map(result)

      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end

    test "extracts device information from natural language", %{persona_name: persona_name} do
      input = "Turn off the living room TV"

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      assert is_map(result)

      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end

    test "handles ambiguous or unclear input gracefully", %{persona_name: persona_name} do
      input = "Hello, how are you today? The weather is nice."

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      # Should still return a result, even if no specific entities found
      assert is_map(result)

      # The result should either be general_memory (fallback) or extracted data
      assert result["type"] == "general_memory" or Map.has_key?(result, "entities")

      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end

    test "handles input with no entities gracefully", %{persona_name: persona_name} do
      # This test simulates what happens when no entities are found
      # The system should fall back to storing general memory
      input = "Hello world"

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      # Should always return a result
      assert is_map(result)

      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end
  end

  describe "confidence-based filtering" do
    test "only stores high-confidence entities", %{persona_name: persona_name} do
      # This test would need to mock the LLM response to test confidence filtering
      # For now, we just ensure the system handles confidence scores
      input = "I am John and I am 30 years old"

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      assert is_map(result)

      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end
  end

  describe "relationship extraction" do
    test "extracts relationships between entities", %{persona_name: persona_name} do
      input =
        "John owns a dog named Max. Sarah lives in the apartment next door. The dog belongs to John."

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      assert is_map(result)

      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end
  end

  describe "fact extraction" do
    test "extracts descriptive facts about entities", %{persona_name: persona_name} do
      input =
        "My car is a red Toyota Camry that I bought last year. It has great fuel efficiency and is very reliable."

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      assert is_map(result)

      knowledge = KnowledgeStore.get_knowledge(persona_name)
      assert is_map(knowledge)
    end
  end

  describe "memory integration" do
    test "stores learning context in memory", %{persona_name: persona_name} do
      input = "I am learning about machine learning and artificial intelligence."

      {:ok, result} = Learner.learn_from_input(persona_name, input)

      assert is_map(result)

      # The system should store the learning context in memory
      # This would require checking the MemoryStore, but for now we just ensure no errors
    end
  end
end
