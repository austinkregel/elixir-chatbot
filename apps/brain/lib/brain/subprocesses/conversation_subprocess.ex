defmodule Brain.Subprocesses.ConversationSubprocess do
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
    user_id = Keyword.get(opts, :user_id) || "conv_#{subprocess_id}"

    GenServer.start_link(__MODULE__, {subprocess_id, conversation_id, memory_snapshot, user_id},
      name: via_tuple(subprocess_id)
    )
  end

  @doc "Returns true if the Conversation subprocess is ready to accept requests."
  def ready?(subprocess_id) do
    try do
      GenServer.call(via_tuple(subprocess_id), :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  def evaluate_input(subprocess_id, input) do
    GenServer.call(via_tuple(subprocess_id), {:evaluate_input, input}, 60_000)
  end

  def get_conversation_state(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_conversation_state, 5_000)
  end

  def send_learning_summary(subprocess_id, summary) do
    GenServer.cast(via_tuple(subprocess_id), {:send_learning_summary, summary})
  end

  def handle_urgent_interrupt(subprocess_id, reason, data) do
    GenServer.cast(via_tuple(subprocess_id), {:urgent_interrupt, reason, data})
  end

  # Server Callbacks

  @impl true
  def init(args) do
    {subprocess_id, conversation_id, memory_snapshot, user_id} =
      case args do
        {sid, cid, snap} -> {sid, cid, snap, "conv_#{elem(args, 0)}"}
        {sid, cid, snap, uid} -> {sid, cid, snap, uid}
      end
    # Ensure a Brain-level conversation exists so evaluate/3 works
    resolved_conversation_id =
      if conversation_id do
        conversation_id
      else
        case Brain.create_conversation() do
          {:ok, id} -> id
          _ -> generate_id()
        end
      end

    # Initialize state
    state = %{
      subprocess_id: subprocess_id,
      conversation_id: resolved_conversation_id,
      memory_snapshot: memory_snapshot,
      user_id: user_id,
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
      conversation_id: resolved_conversation_id,
      memory_size: map_size(memory_snapshot)
    })

    {:ok, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
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
    {:via, Registry, {Brain.SubprocessRegistry, {:conversation_subprocess, subprocess_id}}}
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp process_conversation_input(input, state) do
    # Route through the main Brain.evaluate pipeline for full NLP processing
    conv_id = state.conversation_id
    user_id = Map.get(state, :user_id, "conv_#{state.subprocess_id}")

    case Brain.evaluate(conv_id, input, user_id: user_id) do
      {:ok, %{response: response}} when is_binary(response) ->
        response

      {:ok, response} when is_binary(response) ->
        response

      {:ok, %{response: nil}} ->
        ""

      {:ok, nil} ->
        # ResponseGate deferred - no response needed
        ""

      {:error, reason} ->
        Logger.warning("Brain.evaluate failed in conversation subprocess",
          reason: inspect(reason),
          conversation_id: conv_id,
          input: input
        )

        "I'm sorry, I wasn't able to process that right now."
    end
  end
end
