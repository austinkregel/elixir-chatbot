defmodule Brain.Subprocesses.HttpSubprocessTest do
  use ExUnit.Case, async: false
  alias Brain.Subprocesses.HttpSubprocess
  import Brain.TestHelpers

  setup_all do
    Brain.TestHelpers.require_services!(:brain)
    :ok
  end

  setup do
    subprocess_id = "test_http_#{:rand.uniform(1000)}"
    port = 7878 + :rand.uniform(100)
    memory_snapshot = %{knowledge: %{}, global_memory: []}

    {:ok, pid} =
      HttpSubprocess.start_link(
        subprocess_id: subprocess_id,
        port: port,
        memory_snapshot: memory_snapshot
      )

    on_exit(fn ->
      DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, pid)
    end)

    %{subprocess_id: subprocess_id, port: port, pid: pid}
  end

  test "starts and returns status", %{subprocess_id: subprocess_id} do
    status = HttpSubprocess.get_status(subprocess_id)

    assert status.subprocess_id == subprocess_id
    assert is_integer(status.port)
    assert is_integer(status.uptime)
    assert status.is_shutting_down == false
  end

  test "creates and manages conversations", %{subprocess_id: subprocess_id} do
    {:ok, conversation_id} = HttpSubprocess.create_conversation(subprocess_id)
    assert is_binary(conversation_id)
    conversations = HttpSubprocess.get_conversations(subprocess_id)
    assert length(conversations) == 1
    assert hd(conversations).id == conversation_id

    {:ok, response} =
      HttpSubprocess.route_to_conversation(subprocess_id, conversation_id, "Hello")

    assert is_binary(response)
    assert_response_intent(response, "smalltalk.greetings")
    :ok = HttpSubprocess.end_conversation(subprocess_id, conversation_id)
    conversations = HttpSubprocess.get_conversations(subprocess_id)
    assert conversations == []
  end

  test "handles learning summaries", %{subprocess_id: subprocess_id} do
    conversation_id = "test_conv_123"
    summary = "User learned about Elixir programming"
    HttpSubprocess.send_learning_summary(subprocess_id, conversation_id, summary)
    status = HttpSubprocess.get_status(subprocess_id)
    assert status.subprocess_id == subprocess_id
  end

  test "handles non-existent conversation", %{subprocess_id: subprocess_id} do
    non_existent_id = "non_existent_123"

    {:error, reason} =
      HttpSubprocess.route_to_conversation(subprocess_id, non_existent_id, "Hello")

    assert reason == "Conversation not found"

    {:error, reason} = HttpSubprocess.end_conversation(subprocess_id, non_existent_id)
    assert reason == "Conversation not found"
  end
end
