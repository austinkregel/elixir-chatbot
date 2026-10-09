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
        world_id: "default",
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

    # Brain owns the id, and the conversation runs in the subprocess's world
    assert {:ok, %{world_id: "default"}} = Brain.get_conversation(conv_id)

    # Send message using the real conversation ID
    {:ok, result} = CliSubprocess.execute_command(subprocess_id, "send #{conv_id} Hello there!")
    assert is_binary(result)
    assert_response_intent(result, "smalltalk.greetings")

    # End conversation using the real conversation ID
    {:ok, result} = CliSubprocess.execute_command(subprocess_id, "end #{conv_id}")
    assert is_binary(result)
    assert String.contains?(result, "Ended conversation")
  end

  test "chat commands in one session use the conversation the session started with",
       %{subprocess_id: subprocess_id} do
    %{conversation_id: conversation_id} = CliSubprocess.get_status(subprocess_id)
    assert {:ok, %{world_id: "default"}} = Brain.get_conversation(conversation_id)

    {:ok, _} = CliSubprocess.execute_command(subprocess_id, "chat Hello there!")
    {:ok, _} = CliSubprocess.execute_command(subprocess_id, "chat How are you?")

    assert CliSubprocess.get_status(subprocess_id).conversation_id == conversation_id

    {:ok, conversation} = Brain.get_conversation(conversation_id)
    user_turns = conversation.memory |> Enum.filter(&(&1[:role] == "user")) |> Enum.map(& &1[:content])
    assert user_turns == ["Hello there!", "How are you?"]
  end

  test "exiting a session ends its conversation" do
    {:ok, subprocess_id} = Brain.start_cli_subprocess(world_id: "default")
    %{conversation_id: conversation_id} = CliSubprocess.get_status(subprocess_id)
    assert {:ok, _} = Brain.get_conversation(conversation_id)

    assert :ok = Brain.stop_subprocess(subprocess_id)

    assert {:error, _} = Brain.get_conversation(conversation_id)
  end

  describe "quit and exit" do
    for command <- ["quit", "exit"] do
      test "#{command} ends the session's conversation and the subprocess stays stopped" do
        {:ok, subprocess_id} = Brain.start_cli_subprocess(world_id: "default")
        %{conversation_id: conversation_id} = CliSubprocess.get_status(subprocess_id)
        pid = session_pid!(subprocess_id)
        ref = Process.monitor(pid)

        assert {:ok, "Goodbye!"} = CliSubprocess.execute_command(subprocess_id, unquote(command))

        assert_receive {:DOWN, ^ref, :process, ^pid, :normal}
        assert {:error, _} = Brain.get_conversation(conversation_id)
        assert_stays_stopped(subprocess_id)
      end
    end

    test "an interrupted session still ends on quit" do
      {:ok, subprocess_id} = Brain.start_cli_subprocess(world_id: "default")
      %{conversation_id: conversation_id} = CliSubprocess.get_status(subprocess_id)
      pid = session_pid!(subprocess_id)
      ref = Process.monitor(pid)

      CliSubprocess.handle_urgent_interrupt(subprocess_id, "user_request", %{reason: "stop"})
      assert {:error, "CLI interrupted"} = CliSubprocess.execute_command(subprocess_id, "help")

      assert {:ok, "Goodbye!"} = CliSubprocess.execute_command(subprocess_id, "quit")

      assert_receive {:DOWN, ^ref, :process, ^pid, :normal}
      assert {:error, _} = Brain.get_conversation(conversation_id)
      assert_stays_stopped(subprocess_id)
    end

    test "a session that crashes is restarted" do
      {:ok, subprocess_id} = Brain.start_cli_subprocess(world_id: "default")
      pid = session_pid!(subprocess_id)
      ref = Process.monitor(pid)

      Process.exit(pid, :kill)
      assert_receive {:DOWN, ^ref, :process, ^pid, :killed}

      restarted =
        eventually(
          fn -> Registry.lookup(Brain.SubprocessRegistry, {:cli_subprocess, subprocess_id}) end,
          &match?([{_pid, _}], &1)
        )

      [{restarted_pid, _}] = restarted
      assert restarted_pid != pid
      on_exit(fn -> DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, restarted_pid) end)
    end

    test "a session restarted after a crash goes on with the conversation it had" do
      {:ok, subprocess_id} = Brain.start_cli_subprocess(world_id: "default")
      %{conversation_id: conversation_id} = CliSubprocess.get_status(subprocess_id)
      {:ok, _} = CliSubprocess.execute_command(subprocess_id, "chat Hello there!")
      conversations_before = length(Brain.get_conversations())
      pid = session_pid!(subprocess_id)
      ref = Process.monitor(pid)

      # Killed, so its terminate/2 does not run and nothing ends the conversation.
      Process.exit(pid, :kill)
      assert_receive {:DOWN, ^ref, :process, ^pid, :killed}

      [{restarted_pid, _}] =
        eventually(
          fn -> Registry.lookup(Brain.SubprocessRegistry, {:cli_subprocess, subprocess_id}) end,
          &match?([{restarted, _}] when restarted != pid, &1)
        )

      on_exit(fn -> DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, restarted_pid) end)

      assert CliSubprocess.get_status(subprocess_id).conversation_id == conversation_id
      assert length(Brain.get_conversations()) == conversations_before

      {:ok, _} = CliSubprocess.execute_command(subprocess_id, "chat How are you?")
      {:ok, conversation} = Brain.get_conversation(conversation_id)
      user_turns = conversation.memory |> Enum.filter(&(&1[:role] == "user")) |> Enum.map(& &1[:content])
      assert user_turns == ["Hello there!", "How are you?"]

      # Stopping the restarted session ends the one conversation.
      assert :ok = Brain.stop_subprocess(subprocess_id)
      assert {:error, _} = Brain.get_conversation(conversation_id)
    end

    test "a session restarted after its conversation was ended starts a new one" do
      {:ok, subprocess_id} = Brain.start_cli_subprocess(world_id: "default")
      %{conversation_id: conversation_id} = CliSubprocess.get_status(subprocess_id)
      :ok = Brain.end_conversation(conversation_id)
      pid = session_pid!(subprocess_id)
      ref = Process.monitor(pid)

      Process.exit(pid, :kill)
      assert_receive {:DOWN, ^ref, :process, ^pid, :killed}

      [{restarted_pid, _}] =
        eventually(
          fn -> Registry.lookup(Brain.SubprocessRegistry, {:cli_subprocess, subprocess_id}) end,
          &match?([{restarted, _}] when restarted != pid, &1)
        )

      on_exit(fn -> DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, restarted_pid) end)

      %{conversation_id: new_conversation_id} = CliSubprocess.get_status(subprocess_id)
      assert new_conversation_id != conversation_id
      assert {:ok, %{world_id: "default"}} = Brain.get_conversation(new_conversation_id)

      assert :ok = Brain.stop_subprocess(subprocess_id)
      assert {:error, _} = Brain.get_conversation(new_conversation_id)
    end

    test "a session restarted after a crash is still listed, and stop_subprocess stops it" do
      {:ok, subprocess_id} = Brain.start_cli_subprocess(world_id: "default")
      pid = session_pid!(subprocess_id)
      ref = Process.monitor(pid)

      Process.exit(pid, :kill)
      assert_receive {:DOWN, ^ref, :process, ^pid, :killed}

      [{restarted_pid, _}] =
        eventually(
          fn -> Registry.lookup(Brain.SubprocessRegistry, {:cli_subprocess, subprocess_id}) end,
          &match?([{restarted, _}] when restarted != pid, &1)
        )

      # The restarted session has told Brain its pid once `get_status/1`
      # answers: it cast that from `init/1`, before it could take a call.
      %{subprocess_id: ^subprocess_id} = CliSubprocess.get_status(subprocess_id)
      assert subprocess_id in Brain.list_subprocesses().cli

      restarted_ref = Process.monitor(restarted_pid)
      assert :ok = Brain.stop_subprocess(subprocess_id)

      assert_receive {:DOWN, ^restarted_ref, :process, ^restarted_pid, _reason}
      assert_stays_stopped(subprocess_id)
    end
  end

  defp session_pid!(subprocess_id) do
    [{pid, _}] = Registry.lookup(Brain.SubprocessRegistry, {:cli_subprocess, subprocess_id})
    pid
  end

  # The supervisor started no replacement, and Brain no longer lists the
  # session. `which_children/1` returns once the supervisor has handled the
  # session's exit, and a replacement would register under the same id.
  defp assert_stays_stopped(subprocess_id) do
    DynamicSupervisor.which_children(Brain.Subprocesses.Supervisor)
    assert Registry.lookup(Brain.SubprocessRegistry, {:cli_subprocess, subprocess_id}) == []

    eventually(fn -> Brain.list_subprocesses().cli end, &(subprocess_id not in &1))
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
