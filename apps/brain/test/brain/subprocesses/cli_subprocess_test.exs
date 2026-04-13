defmodule Brain.Subprocesses.CliSubprocessTest do
  use ExUnit.Case, async: false
  alias Brain.Subprocesses.CliSubprocess
  import Brain.TestHelpers

  setup_all do
    Brain.TestHelpers.require_services!(:brain)
    :ok
  end

  setup do
    subprocess_id = "test_cli_#{:rand.uniform(1000)}"
    memory_snapshot = %{knowledge: %{}, global_memory: []}

    {:ok, pid} =
      CliSubprocess.start_link(
        subprocess_id: subprocess_id,
        memory_snapshot: memory_snapshot
      )

    on_exit(fn ->
      DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, pid)
    end)

    %{subprocess_id: subprocess_id, pid: pid}
  end

  test "starts and returns status", %{subprocess_id: subprocess_id} do
    status = CliSubprocess.get_status(subprocess_id)

    assert status.subprocess_id == subprocess_id
    assert is_integer(status.uptime)
    assert status.is_shutting_down == false
    assert status.is_interrupted == false
  end

  test "executes commands and maintains CLI memory", %{subprocess_id: subprocess_id} do
    # Execute help command
    {:ok, help_text} = CliSubprocess.execute_command(subprocess_id, "help")
    assert is_binary(help_text)
    assert String.contains?(help_text, "help")

    # Execute status command
    {:ok, status_text} = CliSubprocess.execute_command(subprocess_id, "status")
    assert is_binary(status_text)
    assert String.contains?(status_text, subprocess_id)

    # Verify CLI state shows commands
    status = CliSubprocess.get_status(subprocess_id)
    assert status.command_count == 2
  end

  test "handles conversation commands", %{subprocess_id: subprocess_id} do
    # Create conversation
    {:ok, result} = CliSubprocess.execute_command(subprocess_id, "create my_chat")
    assert is_binary(result)
    assert String.contains?(result, "Created conversation")

    # Extract the real conversation ID from "Created conversation 'my_chat' with Brain ID: <id>"
    [_, conv_id] = Regex.run(~r/Brain ID: (\S+)/, result)

    # Send message using the real conversation ID
    {:ok, result} = CliSubprocess.execute_command(subprocess_id, "send #{conv_id} Hello there!")
    assert is_binary(result)
    assert_response_intent(result, "smalltalk.greetings")

    # End conversation using the real conversation ID
    {:ok, result} = CliSubprocess.execute_command(subprocess_id, "end #{conv_id}")
    assert is_binary(result)
    assert String.contains?(result, "Ended conversation")
  end

  test "handles urgent interrupts", %{subprocess_id: subprocess_id} do
    # Send urgent interrupt
    CliSubprocess.handle_urgent_interrupt(subprocess_id, "user_request", %{reason: "stop"})

    # Verify interrupt was handled
    status = CliSubprocess.get_status(subprocess_id)
    assert status.is_interrupted == true

    # Try to execute command after interrupt
    {:error, reason} = CliSubprocess.execute_command(subprocess_id, "help")
    assert reason == "CLI interrupted"
  end

  test "provides help information", %{subprocess_id: subprocess_id} do
    help_text = CliSubprocess.get_help(subprocess_id)

    assert is_binary(help_text)
    assert String.contains?(help_text, "help")
    assert String.contains?(help_text, "status")
    assert String.contains?(help_text, "conversations")
    assert String.contains?(help_text, "create")
    assert String.contains?(help_text, "send")
  end

  test "handles unknown commands", %{subprocess_id: subprocess_id} do
    {:ok, result} = CliSubprocess.execute_command(subprocess_id, "unknown_command")
    assert is_binary(result)
    assert String.contains?(result, "Unknown command")
    assert String.contains?(result, "help")
  end
end
