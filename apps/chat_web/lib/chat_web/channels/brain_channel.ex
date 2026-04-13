defmodule ChatWeb.BrainChannel do
  @moduledoc """
  Phoenix Channel for real-time brain communication.
  Handles WebSocket connections for the chat bot brain.
  """

  use ChatWeb, :channel
  alias Phoenix.PubSub
  require Logger

  @impl true
  def join("brain:status", _payload, socket) do
    Logger.info("Client joined brain status channel", %{client_id: socket.id})
    {:ok, socket}
  end

  def join("brain:learning", _payload, socket) do
    Logger.info("Client joined brain learning channel", %{client_id: socket.id})
    {:ok, socket}
  end

  def join("brain:conversations", _payload, socket) do
    Logger.info("Client joined brain conversations channel", %{client_id: socket.id})
    PubSub.subscribe(Brain.PubSub, "brain:analysis")
    {:ok, socket}
  end

  def join("urgent:interrupt", _payload, socket) do
    Logger.info("Client joined urgent interrupt channel", %{client_id: socket.id})
    {:ok, socket}
  end

  def join("urgent:emergency", _payload, socket) do
    Logger.info("Client joined urgent emergency channel", %{client_id: socket.id})
    {:ok, socket}
  end

  def join(_room, _payload, _socket) do
    {:error, %{reason: "unauthorized"}}
  end

  @impl true
  def handle_in("evaluate", %{"conversation_id" => conversation_id, "input" => input} = payload, socket) do
    user_id = Map.get(payload, "user_id") || "ws_#{conversation_id}"
    message_id = :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)

    Logger.info("Received evaluation request", %{
      conversation_id: conversation_id,
      input: String.slice(input, 0, 100)
    })

    eval_opts = [
      user_id: user_id,
      progress: %{conversation_id: conversation_id, message_id: message_id}
    ]

    case Brain.evaluate(conversation_id, input, eval_opts) do
      {:ok, %{response: response, context: context, processing_method: method}} ->
        push(socket, "conversation_result", %{
          conversation_id: conversation_id,
          response: response,
          context: context,
          processing_method: method,
          message_id: message_id,
          timestamp: System.system_time(:millisecond)
        })

        {:noreply, socket}

      {:ok, response} when is_binary(response) ->
        push(socket, "conversation_result", %{
          conversation_id: conversation_id,
          response: response,
          message_id: message_id,
          timestamp: System.system_time(:millisecond)
        })

        {:noreply, socket}

      {:error, reason} ->
        push(socket, "conversation_error", %{
          conversation_id: conversation_id,
          error: reason,
          message_id: message_id,
          timestamp: System.system_time(:millisecond)
        })

        {:noreply, socket}
    end
  end

  def handle_in("create_conversation", payload, socket) do
    world_id = Map.get(payload || %{}, "world_id", "default")

    Logger.info("Received create conversation request", %{
      client_id: socket.id,
      world_id: world_id
    })

    case Brain.create_conversation(world_id: world_id) do
      {:ok, conversation_id} ->
        push(socket, "create_conversation_result", %{
          conversation_id: conversation_id,
          world_id: world_id,
          success: true,
          timestamp: System.system_time(:millisecond)
        })

        {:noreply, socket}

      {:error, reason} ->
        push(socket, "create_conversation_error", %{
          error: reason,
          timestamp: System.system_time(:millisecond)
        })

        {:noreply, socket}
    end
  end

  def handle_in("end_conversation", %{"conversation_id" => conversation_id}, socket) do
    Logger.info("Received end conversation request", %{
      conversation_id: conversation_id,
      client_id: socket.id
    })

    case Brain.end_conversation(conversation_id) do
      :ok ->
        push(socket, "end_conversation_result", %{
          conversation_id: conversation_id,
          success: true,
          timestamp: System.system_time(:millisecond)
        })

        {:noreply, socket}

      {:error, reason} ->
        push(socket, "end_conversation_error", %{
          conversation_id: conversation_id,
          error: reason,
          timestamp: System.system_time(:millisecond)
        })

        {:noreply, socket}
    end
  end

  def handle_in("get_conversations", _payload, socket) do
    Logger.info("Received get conversations request", %{client_id: socket.id})

    conversations = Brain.get_conversations()

    push(socket, "conversations_result", %{
      conversations: conversations,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, socket}
  end

  def handle_in("get_status", _payload, socket) do
    Logger.info("Received get status request", %{client_id: socket.id})

    status = Brain.get_status()

    push(socket, "status_result", %{
      status: status,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, socket}
  end

  def handle_in("urgent_interrupt", %{"reason" => reason, "data" => data}, socket) do
    Logger.warning("Received urgent interrupt", %{
      reason: reason,
      client_id: socket.id
    })

    Brain.handle_urgent_interrupt(reason, data)

    push(socket, "interrupt_acknowledged", %{
      reason: reason,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, socket}
  end

  def handle_in("urgent_emergency", %{"reason" => reason, "data" => data}, socket) do
    Logger.error("Received urgent emergency", %{
      reason: reason,
      client_id: socket.id
    })

    Brain.handle_urgent_emergency(reason, data)

    push(socket, "emergency_acknowledged", %{
      reason: reason,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, socket}
  end

  def handle_in(_event, _payload, socket) do
    Logger.warning("Unknown event received", %{client_id: socket.id})
    {:noreply, socket}
  end

  @impl true
  def handle_info({:analysis_progress, payload}, socket) do
    push(socket, "analysis_progress", payload)
    {:noreply, socket}
  end

  def handle_info(%Phoenix.Socket.Broadcast{event: "analysis_progress", payload: payload}, socket) do
    push(socket, "analysis_progress", payload)
    {:noreply, socket}
  end

  def handle_info(_msg, socket) do
    {:noreply, socket}
  end

  @impl true
  def terminate(reason, socket) do
    Logger.info("Client disconnected from brain channel", %{
      reason: reason,
      client_id: socket.id
    })

    :ok
  end
end
