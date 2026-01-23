defmodule ChatBot.Subprocesses.ConversationSubprocess do
  @moduledoc """
  Conversation subprocess GenServer for handling individual conversations.
  Manages conversation state, learning, and communication with the Brain.
  """

  use GenServer
  require Logger

  # Client API

  def start_link(opts \\ []) do
    subprocess_id = Keyword.get(opts, :subprocess_id, generate_id())
    conversation_id = Keyword.get(opts, :conversation_id)
    memory_snapshot = Keyword.get(opts, :memory_snapshot, %{})

    GenServer.start_link(__MODULE__, {subprocess_id, conversation_id, memory_snapshot},
      name: via_tuple(subprocess_id)
    )
  end

  def evaluate_input(subprocess_id, input) do
    GenServer.call(via_tuple(subprocess_id), {:evaluate_input, input})
  end

  def get_conversation_state(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_conversation_state)
  end

  def send_learning_summary(subprocess_id, summary) do
    GenServer.cast(via_tuple(subprocess_id), {:send_learning_summary, summary})
  end

  def handle_urgent_interrupt(subprocess_id, reason, data) do
    GenServer.cast(via_tuple(subprocess_id), {:urgent_interrupt, reason, data})
  end

  # Server Callbacks

  @impl true
  def init({subprocess_id, conversation_id, memory_snapshot}) do
    # Initialize state
    state = %{
      subprocess_id: subprocess_id,
      conversation_id: conversation_id,
      memory_snapshot: memory_snapshot,
      conversation_memory: [],
      learning_data: %{
        interactions: [],
        new_knowledge: %{},
        insights: []
      },
      start_time: System.system_time(:millisecond),
      is_shutting_down: false,
      is_interrupted: false
    }

    Logger.info("Conversation subprocess started", %{
      subprocess_id: subprocess_id,
      conversation_id: conversation_id,
      memory_size: map_size(memory_snapshot)
    })

    {:ok, state}
  end

  @impl true
  def handle_call({:evaluate_input, input}, _from, state) do
    if state.is_interrupted do
      {:reply, {:error, "Conversation interrupted"}, state}
    else
      # Process the input
      response = process_conversation_input(input, state)

      # Update conversation memory
      interaction = %{
        input: input,
        response: response,
        timestamp: System.system_time(:millisecond)
      }

      updated_memory = state.conversation_memory ++ [interaction]

      # Add to learning data
      updated_learning_data = %{
        state.learning_data
        | interactions: state.learning_data.interactions ++ [interaction]
      }

      updated_state = %{
        state
        | conversation_memory: updated_memory,
          learning_data: updated_learning_data
      }

      Logger.info("Conversation subprocess processed input", %{
        subprocess_id: state.subprocess_id,
        conversation_id: state.conversation_id,
        input_length: String.length(input),
        response_length: String.length(response)
      })

      {:reply, {:ok, response}, updated_state}
    end
  end

  @impl true
  def handle_call(:get_conversation_state, _from, state) do
    state_info = %{
      subprocess_id: state.subprocess_id,
      conversation_id: state.conversation_id,
      message_count: length(state.conversation_memory),
      learning_interactions: length(state.learning_data.interactions),
      uptime: System.system_time(:millisecond) - state.start_time,
      is_shutting_down: state.is_shutting_down,
      is_interrupted: state.is_interrupted
    }

    {:reply, state_info, state}
  end

  @impl true
  def handle_cast({:send_learning_summary, summary}, state) do
    Logger.info("Conversation subprocess received learning summary", %{
      subprocess_id: state.subprocess_id,
      conversation_id: state.conversation_id,
      summary_length: String.length(summary)
    })

    # Process learning summary
    insight = %{
      conversation_id: state.conversation_id,
      summary: summary,
      timestamp: System.system_time(:millisecond)
    }

    updated_learning_data = %{
      state.learning_data
      | insights: state.learning_data.insights ++ [insight]
    }

    {:noreply, %{state | learning_data: updated_learning_data}}
  end

  @impl true
  def handle_cast({:urgent_interrupt, reason, data}, state) do
    Logger.warning("Conversation subprocess handling urgent interrupt", %{
      subprocess_id: state.subprocess_id,
      conversation_id: state.conversation_id,
      reason: reason
    })

    # Mark as interrupted
    updated_state = %{state | is_interrupted: true}

    # Add interrupt to conversation memory
    interrupt_entry = %{
      type: "interrupt",
      reason: reason,
      data: data,
      timestamp: System.system_time(:millisecond)
    }

    updated_memory = state.conversation_memory ++ [interrupt_entry]
    final_state = %{updated_state | conversation_memory: updated_memory}

    {:noreply, final_state}
  end

  @impl true
  def terminate(reason, state) do
    Logger.info("Conversation subprocess shutting down", %{
      subprocess_id: state.subprocess_id,
      conversation_id: state.conversation_id,
      reason: reason,
      uptime: System.system_time(:millisecond) - state.start_time,
      interactions: length(state.learning_data.interactions)
    })

    :ok
  end

  # Private Functions

  defp via_tuple(subprocess_id) do
    {:via, Registry, {ChatBot.SubprocessRegistry, {:conversation_subprocess, subprocess_id}}}
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp process_conversation_input(input, state) do
    # Simple response generation - in a real implementation, this would call the Brain
    # and use the conversation memory for context
    context =
      if length(state.conversation_memory) > 0 do
        " (with #{length(state.conversation_memory)} previous messages)"
      else
        " (new conversation)"
      end

    "Conversation subprocess received: #{input}#{context}. This is a simplified response."
  end
end
