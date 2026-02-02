defmodule Brain.Subprocesses.HttpSubprocess do
  @moduledoc """
  HTTP subprocess GenServer for handling web requests.
  Provides HTTP API endpoints for the chat bot functionality.
  """

  use GenServer
  require Logger

  # Client API

  def start_link(opts \\ []) do
    subprocess_id = Keyword.get(opts, :subprocess_id, generate_id())
    port = Keyword.get(opts, :port, 7878)
    memory_snapshot = Keyword.get(opts, :memory_snapshot, %{})

    GenServer.start_link(__MODULE__, {subprocess_id, port, memory_snapshot},
      name: via_tuple(subprocess_id)
    )
  end

  def get_status(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_status)
  end

  def get_conversations(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_conversations)
  end

  def create_conversation(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :create_conversation)
  end

  def end_conversation(subprocess_id, conversation_id) do
    GenServer.call(via_tuple(subprocess_id), {:end_conversation, conversation_id})
  end

  def route_to_conversation(subprocess_id, conversation_id, input) do
    GenServer.call(via_tuple(subprocess_id), {:route_to_conversation, conversation_id, input})
  end

  def send_learning_summary(subprocess_id, conversation_id, summary) do
    GenServer.cast(via_tuple(subprocess_id), {:send_learning_summary, conversation_id, summary})
  end

  # Server Callbacks

  @impl true
  def init({subprocess_id, port, memory_snapshot}) do
    # Initialize state
    state = %{
      subprocess_id: subprocess_id,
      port: port,
      memory_snapshot: memory_snapshot,
      conversations: %{},
      learning_data: %{
        requests: [],
        new_knowledge: %{},
        insights: []
      },
      start_time: System.system_time(:millisecond),
      is_shutting_down: false,
      http_server: nil
    }

    # Start HTTP server
    case start_http_server(port, subprocess_id) do
      {:ok, http_server} ->
        Logger.info("HTTP subprocess started", %{
          subprocess_id: subprocess_id,
          port: port,
          memory_size: map_size(memory_snapshot)
        })

        {:ok, %{state | http_server: http_server}}

      {:error, reason} ->
        Logger.error("Failed to start HTTP server", %{
          subprocess_id: subprocess_id,
          port: port,
          reason: reason
        })

        {:stop, reason}
    end
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    status = %{
      subprocess_id: state.subprocess_id,
      port: state.port,
      conversations: map_size(state.conversations),
      learning_requests: length(state.learning_data.requests),
      uptime: System.system_time(:millisecond) - state.start_time,
      is_shutting_down: state.is_shutting_down
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call(:get_conversations, _from, state) do
    conversations =
      state.conversations
      |> Map.values()
      |> Enum.map(fn conv ->
        %{
          id: conv.id,
          message_count: length(conv.messages),
          created_at: conv.created_at,
          last_activity: conv.last_activity
        }
      end)

    {:reply, conversations, state}
  end

  @impl true
  def handle_call(:create_conversation, _from, state) do
    conversation_id = generate_conversation_id()

    conversation = %{
      id: conversation_id,
      messages: [],
      created_at: System.system_time(:millisecond),
      last_activity: System.system_time(:millisecond)
    }

    updated_state = %{
      state
      | conversations: Map.put(state.conversations, conversation_id, conversation)
    }

    Logger.info("HTTP subprocess created conversation", %{
      subprocess_id: state.subprocess_id,
      conversation_id: conversation_id
    })

    {:reply, {:ok, conversation_id}, updated_state}
  end

  @impl true
  def handle_call({:end_conversation, conversation_id}, _from, state) do
    case Map.pop(state.conversations, conversation_id) do
      {nil, _} ->
        {:reply, {:error, "Conversation not found"}, state}

      {_conversation, updated_conversations} ->
        Logger.info("HTTP subprocess ended conversation", %{
          subprocess_id: state.subprocess_id,
          conversation_id: conversation_id
        })

        {:reply, :ok, %{state | conversations: updated_conversations}}
    end
  end

  @impl true
  def handle_call({:route_to_conversation, conversation_id, input}, _from, state) do
    case Map.get(state.conversations, conversation_id) do
      nil ->
        {:reply, {:error, "Conversation not found"}, state}

      conversation ->
        # Process the input (simplified for now)
        response = process_http_input(input, conversation)

        # Update conversation
        updated_conversation = %{
          conversation
          | messages:
              conversation.messages ++
                [%{role: "user", content: input}, %{role: "assistant", content: response}],
            last_activity: System.system_time(:millisecond)
        }

        updated_state = %{
          state
          | conversations: Map.put(state.conversations, conversation_id, updated_conversation)
        }

        # Add to learning data
        learning_entry = %{
          conversation_id: conversation_id,
          input: input,
          response: response,
          timestamp: System.system_time(:millisecond)
        }

        updated_learning_data = %{
          state.learning_data
          | requests: state.learning_data.requests ++ [learning_entry]
        }

        final_state = %{updated_state | learning_data: updated_learning_data}

        {:reply, {:ok, response}, final_state}
    end
  end

  @impl true
  def handle_cast({:send_learning_summary, conversation_id, summary}, state) do
    Logger.info("HTTP subprocess received learning summary", %{
      subprocess_id: state.subprocess_id,
      conversation_id: conversation_id,
      summary_length: String.length(summary)
    })

    # Process learning summary (simplified for now)
    updated_learning_data = %{
      state.learning_data
      | insights:
          state.learning_data.insights ++
            [
              %{
                conversation_id: conversation_id,
                summary: summary,
                timestamp: System.system_time(:millisecond)
              }
            ]
    }

    {:noreply, %{state | learning_data: updated_learning_data}}
  end

  @impl true
  def terminate(reason, state) do
    Logger.info("HTTP subprocess shutting down", %{
      subprocess_id: state.subprocess_id,
      reason: reason,
      uptime: System.system_time(:millisecond) - state.start_time
    })

    # Stop HTTP server if running
    if state.http_server do
      DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, state.http_server)
    end

    :ok
  end

  # Private Functions

  defp via_tuple(subprocess_id) do
    {:via, Registry, {Brain.SubprocessRegistry, {:http_subprocess, subprocess_id}}}
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp generate_conversation_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp start_http_server(port, subprocess_id) do
    # Create a simple HTTP handler
    handler = fn conn ->
      case conn.request_path do
        "/status" ->
          status = get_status(subprocess_id)

          conn
          |> Plug.Conn.put_resp_content_type("application/json")
          |> Plug.Conn.send_resp(200, Jason.encode!(status))

        "/conversations" ->
          case conn.method do
            "GET" ->
              conversations = get_conversations(subprocess_id)

              conn
              |> Plug.Conn.put_resp_content_type("application/json")
              |> Plug.Conn.send_resp(200, Jason.encode!(conversations))

            "POST" ->
              case create_conversation(subprocess_id) do
                {:ok, conversation_id} ->
                  conn
                  |> Plug.Conn.put_resp_content_type("application/json")
                  |> Plug.Conn.send_resp(201, Jason.encode!(%{conversation_id: conversation_id}))

                {:error, reason} ->
                  conn
                  |> Plug.Conn.put_resp_content_type("application/json")
                  |> Plug.Conn.send_resp(400, Jason.encode!(%{error: reason}))
              end

            _ ->
              conn
              |> Plug.Conn.send_resp(405, "Method not allowed")
          end

        path ->
          if String.starts_with?(path, "/conversations/") do
            # Extract conversation ID and handle conversation-specific requests
            conversation_id = String.replace_prefix(path, "/conversations/", "")

            case conn.method do
              "POST" ->
                # Handle message to conversation
                {:ok, body, _conn} = Plug.Conn.read_body(conn)
                input_data = Jason.decode!(body)
                input = input_data["input"]

                case route_to_conversation(subprocess_id, conversation_id, input) do
                  {:ok, response} ->
                    conn
                    |> Plug.Conn.put_resp_content_type("application/json")
                    |> Plug.Conn.send_resp(200, Jason.encode!(%{response: response}))

                  {:error, reason} ->
                    conn
                    |> Plug.Conn.put_resp_content_type("application/json")
                    |> Plug.Conn.send_resp(400, Jason.encode!(%{error: reason}))
                end

              "DELETE" ->
                # End conversation
                case end_conversation(subprocess_id, conversation_id) do
                  :ok ->
                    conn
                    |> Plug.Conn.send_resp(204, "")

                  {:error, reason} ->
                    conn
                    |> Plug.Conn.put_resp_content_type("application/json")
                    |> Plug.Conn.send_resp(400, Jason.encode!(%{error: reason}))
                end

              _ ->
                conn
                |> Plug.Conn.send_resp(405, "Method not allowed")
            end
          else
            conn
            |> Plug.Conn.send_resp(404, "Not found")
          end
      end
    end

    # Start Bandit HTTP server
    Bandit.start_link(
      scheme: :http,
      port: port,
      plug: handler
    )
  end

  defp process_http_input(input, _conversation) do
    # Simple response generation - in a real implementation, this would call the Brain
    "HTTP Subprocess received: #{input}. This is a simplified response."
  end
end
