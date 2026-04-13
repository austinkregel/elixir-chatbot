defmodule Brain.Subprocesses.ConversationSubprocessTest do
  use ExUnit.Case, async: false
  alias Brain.Subprocesses.ConversationSubprocess
  import Brain.TestHelpers

  setup_all do
    Brain.TestHelpers.require_services!(:brain)
    :ok
  end

  setup do
    subprocess_id = "test_conv_#{:rand.uniform(1000)}"
    memory_snapshot = %{knowledge: %{}, global_memory: []}

    # Don't pass a conversation_id so the subprocess creates a real one
    # via Brain.create_conversation(), which registers it with Brain
    {:ok, pid} =
      ConversationSubprocess.start_link(
        subprocess_id: subprocess_id,
        memory_snapshot: memory_snapshot
      )

    on_exit(fn ->
      DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, pid)
    end)

    %{subprocess_id: subprocess_id, pid: pid}
  end

  test "starts and returns conversation state", %{subprocess_id: subprocess_id} do
    state = ConversationSubprocess.get_conversation_state(subprocess_id)

    assert state.subprocess_id == subprocess_id
    assert is_binary(state.conversation_id)
    assert is_integer(state.uptime)
    assert state.is_shutting_down == false
    assert state.is_interrupted == false
  end

  test "evaluates input and maintains conversation memory", %{subprocess_id: subprocess_id} do
    # First input
    {:ok, response1} = ConversationSubprocess.evaluate_input(subprocess_id, "Hello there!")
    assert is_binary(response1)
    assert_response_intent(response1, "smalltalk.greetings")

    # Second input (should have context)
    {:ok, response2} = ConversationSubprocess.evaluate_input(subprocess_id, "How are you?")
    assert is_binary(response2)
    assert_response_intent(response2, "smalltalk.")

    # Verify conversation state shows messages
    state = ConversationSubprocess.get_conversation_state(subprocess_id)
    assert state.message_count == 2
  end

  test "handles urgent interrupts", %{subprocess_id: subprocess_id} do
    # Send urgent interrupt
    ConversationSubprocess.handle_urgent_interrupt(subprocess_id, "user_request", %{
      reason: "stop"
    })

    # Verify interrupt was handled
    state = ConversationSubprocess.get_conversation_state(subprocess_id)
    assert state.is_interrupted == true

    # Try to evaluate input after interrupt
    {:error, reason} = ConversationSubprocess.evaluate_input(subprocess_id, "Hello")
    assert reason == "Conversation interrupted"
  end

  test "handles learning summaries", %{subprocess_id: subprocess_id} do
    summary = "User learned about Phoenix LiveView"

    # This should not crash
    ConversationSubprocess.send_learning_summary(subprocess_id, summary)

    # Verify subprocess is still running
    state = ConversationSubprocess.get_conversation_state(subprocess_id)
    assert state.subprocess_id == subprocess_id
  end
end
