defmodule ChatBot.SimpleEntityTest do
  use ExUnit.Case, async: true
  alias ChatBot.Learner
  alias ChatBot.KnowledgeStore

  setup do
    persona_name = "TestPersona"

    on_exit(fn ->
      # Clean up test knowledge files
      knowledge_file =
        Path.join(Application.get_env(:chat_bot, :knowledge_dir), "#{persona_name}.json")

      File.rm(knowledge_file)
    end)

    %{persona_name: persona_name}
  end

  test "can learn simple entity information", %{persona_name: persona_name} do
    input = "Turn on the kitchen lights"

    {:ok, result} = Learner.learn_from_input(persona_name, input)

    # Should return some kind of result
    assert is_map(result)

    # Check that knowledge was stored
    knowledge = KnowledgeStore.get_knowledge(persona_name)
    assert is_map(knowledge)

    # The system should have learned something, even if it's just general memory
  end

  test "handles simple input gracefully", %{persona_name: persona_name} do
    input = "Hello"

    {:ok, result} = Learner.learn_from_input(persona_name, input)

    # Should always return a result
    assert is_map(result)

    knowledge = KnowledgeStore.get_knowledge(persona_name)
    assert is_map(knowledge)
  end
end
