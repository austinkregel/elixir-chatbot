defmodule Brain.Subprocesses.CliSubprocess do
  @moduledoc """
  CLI subprocess GenServer for handling command-line interactions.
  Provides a command-line interface for the chat bot functionality.

  A CLI session has one Brain conversation, created in the subprocess's world
  when the session starts. Every `chat` command uses it, and it is ended
  (`Brain.end_conversation/2`) when the session exits. `quit` or `exit` ends
  the session: the subprocess stops and is not restarted.
  """

  # Transient: a session that ends with `quit` or `exit` stops normally and is
  # not restarted, while one that crashes is.
  use GenServer, restart: :transient
  require Logger

  # Client API

  @doc "Starts the subprocess. `opts` must carry `:world_id`, the world its conversations run in; absent raises."
  def start_link(opts) do
    world_id = world_id!(opts)
    subprocess_id = Keyword.get(opts, :subprocess_id, generate_id())
    memory_snapshot = Keyword.get(opts, :memory_snapshot, %{})

    GenServer.start_link(__MODULE__, {subprocess_id, memory_snapshot, world_id},
      name: via_tuple(subprocess_id)
    )
  end

  defp world_id!(opts) do
    case Keyword.fetch(opts, :world_id) do
      {:ok, world_id} when is_binary(world_id) and world_id != "" ->
        world_id

      {:ok, other} ->
        raise ArgumentError, "CliSubprocess option :world_id must be a world id, got: #{inspect(other)}"

      :error ->
        raise ArgumentError, "CliSubprocess needs :world_id, the world its conversations run in"
    end
  end

  @doc "Returns true if the CLI subprocess is ready to accept requests."
  def ready?(subprocess_id) do
    try do
      GenServer.call(via_tuple(subprocess_id), :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  def execute_command(subprocess_id, command) do
    GenServer.call(via_tuple(subprocess_id), {:execute_command, command}, 30_000)
  end

  def get_status(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_status, 5_000)
  end

  def get_help(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_help, 5_000)
  end

  def handle_urgent_interrupt(subprocess_id, reason, data) do
    GenServer.cast(via_tuple(subprocess_id), {:urgent_interrupt, reason, data})
  end

  # Server Callbacks

  @impl true
  def init({subprocess_id, memory_snapshot, world_id}) do
    user_id = "cli_#{subprocess_id}"

    # Exits are trapped so `terminate/2` runs when the session is stopped, and
    # ends the session's conversation.
    Process.flag(:trap_exit, true)

    # Initialize state
    state = %{
      subprocess_id: subprocess_id,
      world_id: world_id,
      user_id: user_id,
      conversation_id: nil,
      memory_snapshot: memory_snapshot,
      cli_memory: [],
      learning_data: %{
        interactions: [],
        new_knowledge: %{},
        insights: []
      },
      start_time: System.system_time(:millisecond),
      is_shutting_down: false,
      is_interrupted: false
    }

    Logger.info("CLI subprocess started", %{
      subprocess_id: subprocess_id,
      memory_size: map_size(memory_snapshot)
    })

    {:ok, state, {:continue, :start_conversation}}
  end

  # Created after `init/1` returns: the subprocess may be started from inside
  # the Brain server (`Brain.start_cli_subprocess/1`), which cannot answer this
  # call until the start has finished.
  @impl true
  def handle_continue(:start_conversation, state) do
    case Brain.create_conversation(world_id: state.world_id) do
      {:ok, conversation_id} ->
        Logger.info("CLI session conversation started", %{
          subprocess_id: state.subprocess_id,
          conversation_id: conversation_id
        })

        {:noreply, %{state | conversation_id: conversation_id}}

      {:error, reason} ->
        raise "CLI subprocess #{state.subprocess_id}: the session's conversation in world " <>
                "#{inspect(state.world_id)} was not created: #{inspect(reason)}"
    end
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call({:execute_command, command}, _from, state) do
    # An interrupted session still ends on `quit` or `exit`.
    if state.is_interrupted and not quit_command?(command) do
      {:reply, {:error, "CLI interrupted"}, state}
    else
      # Process the command
      result = process_cli_command(command, state)

      # Update CLI memory
      interaction = %{
        command: command,
        result: result,
        timestamp: System.system_time(:millisecond)
      }

      updated_memory = state.cli_memory ++ [interaction]

      # Add to learning data
      updated_learning_data = %{
        state.learning_data
        | interactions: state.learning_data.interactions ++ [interaction]
      }

      updated_state = %{
        state
        | cli_memory: updated_memory,
          learning_data: updated_learning_data
      }

      Logger.info("CLI subprocess executed command", %{
        subprocess_id: state.subprocess_id,
        command: command,
        result_type: if(is_binary(result), do: "text", else: "data")
      })

      # `quit` and `exit` end the session: the subprocess stops normally, which
      # ends its conversation (`terminate/2`) before the reply is sent, and a
      # normal stop is not restarted.
      if quit_command?(command) do
        {:stop, :normal, {:ok, result}, updated_state}
      else
        {:reply, {:ok, result}, updated_state}
      end
    end
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    status = %{
      subprocess_id: state.subprocess_id,
      conversation_id: state.conversation_id,
      command_count: length(state.cli_memory),
      learning_interactions: length(state.learning_data.interactions),
      uptime: System.system_time(:millisecond) - state.start_time,
      is_shutting_down: state.is_shutting_down,
      is_interrupted: state.is_interrupted
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call(:get_help, _from, state) do
    help_text = """
    Chat Bot CLI Commands:

    help                    - Show this help message
    status                  - Show subprocess status
    conversations           - List active conversations
    create <name>           - Create a new conversation
    end <id>                - End a conversation
    send <id> <message>     - Send message to conversation
    interrupt               - Send urgent interrupt
    emergency               - Send urgent emergency
    quit, exit              - Exit CLI

    Examples:
      create my_chat
      send abc123 Hello there!
      end abc123
    """

    {:reply, help_text, state}
  end

  @impl true
  def handle_cast({:urgent_interrupt, reason, data}, state) do
    Logger.warning("CLI subprocess handling urgent interrupt", %{
      subprocess_id: state.subprocess_id,
      reason: reason
    })

    # Mark as interrupted
    updated_state = %{state | is_interrupted: true}

    # Add interrupt to CLI memory
    interrupt_entry = %{
      type: "interrupt",
      reason: reason,
      data: data,
      timestamp: System.system_time(:millisecond)
    }

    updated_memory = state.cli_memory ++ [interrupt_entry]
    final_state = %{updated_state | cli_memory: updated_memory}

    {:noreply, final_state}
  end

  @impl true
  def terminate(reason, state) do
    Logger.info("CLI subprocess shutting down", %{
      subprocess_id: state.subprocess_id,
      reason: reason,
      uptime: System.system_time(:millisecond) - state.start_time,
      commands: length(state.cli_memory)
    })

    end_session_conversation(state)
  end

  # No conversation when the session stopped before its conversation was created.
  defp end_session_conversation(%{conversation_id: nil}), do: :ok

  defp end_session_conversation(%{conversation_id: conversation_id} = state) do
    case Brain.end_conversation(conversation_id) do
      :ok ->
        Logger.info("CLI session conversation ended", %{
          subprocess_id: state.subprocess_id,
          conversation_id: conversation_id
        })

        :ok

      {:error, reason} ->
        Logger.error("CLI session conversation was not ended", %{
          subprocess_id: state.subprocess_id,
          conversation_id: conversation_id,
          reason: reason
        })

        {:error, reason}
    end
  end

  # Private Functions

  defp quit_command?(command) do
    case String.split(command, " ", parts: 2) do
      [cmd | _] -> cmd in ["quit", "exit"]
    end
  end

  defp via_tuple(subprocess_id) do
    {:via, Registry, {Brain.SubprocessRegistry, {:cli_subprocess, subprocess_id}}}
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp process_cli_command(command, state) do
    # Parse and execute CLI commands
    case String.split(command, " ", parts: 2) do
      ["help"] ->
        """
        Available commands:
        - help: Show this help
        - status: Show subprocess status
        - conversations: List conversations
        - create <name>: Create conversation
        - send <id> <message>: Send message to conversation via Brain
        - chat <message>: Send message using default conversation
        - end <id>: End conversation
        - interrupt: Send interrupt
        - quit: Exit CLI
        """

      ["status"] ->
        """
        CLI Subprocess Status:
        - ID: #{state.subprocess_id}
        - Commands executed: #{length(state.cli_memory)}
        - Uptime: #{System.system_time(:millisecond) - state.start_time}ms
        - Interrupted: #{state.is_interrupted}
        - Conversation ID: #{state.conversation_id}
        """

      ["conversations"] ->
        "Conversations: (This would list active conversations)"

      ["create", name] ->
        case Brain.create_conversation(world_id: state.world_id) do
          {:ok, id} ->
            "Created conversation '#{name}' with Brain ID: #{id}"

          {:error, reason} ->
            "Failed to create conversation '#{name}': #{inspect(reason)}"
        end

      ["send", rest] ->
        case String.split(rest, " ", parts: 2) do
          [conversation_id, message] ->
            # Route through Brain.evaluate for full NLP processing
            case Brain.evaluate(conversation_id, message, user_id: state.user_id) do
              {:ok, %{response: response}} when is_binary(response) ->
                response

              {:ok, response} when is_binary(response) ->
                response

              {:ok, %{response: nil}} ->
                "(no response - deferred)"

              {:ok, nil} ->
                "(no response - deferred)"

              {:error, reason} ->
                "Error: #{inspect(reason)}"
            end

          _ ->
            "Usage: send <conversation_id> <message>"
        end

      ["chat", message] ->
        # The session's conversation, created when the session started
        case Brain.evaluate(state.conversation_id, message, user_id: state.user_id) do
          {:ok, %{response: response}} when is_binary(response) ->
            response

          {:ok, response} when is_binary(response) ->
            response

          {:ok, %{response: nil}} ->
            "(no response - deferred)"

          {:ok, nil} ->
            "(no response - deferred)"

          {:error, reason} ->
            "Error: #{inspect(reason)}"
        end

      ["end", conversation_id] ->
        case Brain.end_conversation(conversation_id) do
          :ok -> "Ended conversation #{conversation_id}"
          {:error, reason} -> "Failed to end conversation: #{inspect(reason)}"
        end

      ["interrupt"] ->
        "Sent urgent interrupt signal"

      ["emergency"] ->
        "Sent urgent emergency signal"

      ["quit", _] ->
        "Goodbye!"

      ["exit", _] ->
        "Goodbye!"

      [cmd] when cmd in ["quit", "exit"] ->
        "Goodbye!"

      _ ->
        "Unknown command: #{command}. Type 'help' for available commands."
    end
  end

end
