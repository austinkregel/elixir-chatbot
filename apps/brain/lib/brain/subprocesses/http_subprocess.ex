defmodule Brain.Subprocesses.HttpSubprocess do
  @moduledoc "HTTP subprocess GenServer for handling web requests.\nProvides HTTP API endpoints for the chat bot functionality.\n"

  alias Plug.Conn
  use GenServer
  require Logger

  def start_link(opts \\ []) do
    subprocess_id = Keyword.get(opts, :subprocess_id, generate_id())
    port = Keyword.get(opts, :port, 7878)
    memory_snapshot = Keyword.get(opts, :memory_snapshot, %{})

    GenServer.start_link(__MODULE__, {subprocess_id, port, memory_snapshot},
      name: via_tuple(subprocess_id)
    )
  end

  @doc "Returns true if the HTTP subprocess is ready to accept requests."
  def ready?(subprocess_id) do
    try do
      GenServer.call(via_tuple(subprocess_id), :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  def get_status(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_status, 5_000)
  end

  def get_conversations(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :get_conversations, 5_000)
  end

  def create_conversation(subprocess_id) do
    GenServer.call(via_tuple(subprocess_id), :create_conversation, 30_000)
  end

  def end_conversation(subprocess_id, conversation_id) do
    GenServer.call(via_tuple(subprocess_id), {:end_conversation, conversation_id}, 5_000)
  end

  def route_to_conversation(subprocess_id, conversation_id, input) do
    GenServer.call(via_tuple(subprocess_id), {:route_to_conversation, conversation_id, input}, 60_000)
  end

  def send_learning_summary(subprocess_id, conversation_id, summary) do
    GenServer.cast(via_tuple(subprocess_id), {:send_learning_summary, conversation_id, summary})
  end

  @impl true
  def init({subprocess_id, port, memory_snapshot}) do
    user_id = "http_#{subprocess_id}"

    state = %{
      subprocess_id: subprocess_id,
      user_id: user_id,
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
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
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
    conversation_id =
      case Brain.create_conversation() do
        {:ok, id} -> id
        _ -> generate_conversation_id()
      end

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
        response = process_http_input(input, conversation, state.user_id)

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

    if state.http_server do
      DynamicSupervisor.terminate_child(Brain.Subprocesses.Supervisor, state.http_server)
    end

    :ok
  end

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
    handler = fn conn ->
      case conn.request_path do
        "/status" ->
          status = get_status(subprocess_id)

          conn
          |> Conn.put_resp_content_type("application/json")
          |> Conn.send_resp(200, Jason.encode!(status))

        "/conversations" ->
          case conn.method do
            "GET" ->
              conversations = get_conversations(subprocess_id)

              conn
              |> Conn.put_resp_content_type("application/json")
              |> Conn.send_resp(200, Jason.encode!(conversations))

            "POST" ->
              case create_conversation(subprocess_id) do
                {:ok, conversation_id} ->
                  conn
                  |> Conn.put_resp_content_type("application/json")
                  |> Conn.send_resp(201, Jason.encode!(%{conversation_id: conversation_id}))

                {:error, reason} ->
                  conn
                  |> Conn.put_resp_content_type("application/json")
                  |> Conn.send_resp(400, Jason.encode!(%{error: reason}))
              end

            _ ->
              conn
              |> Conn.send_resp(405, "Method not allowed")
          end

        path ->
          if String.starts_with?(path, "/conversations/") do
            conversation_id = String.replace_prefix(path, "/conversations/", "")

            case conn.method do
              "POST" ->
                {:ok, body, _conn} = Conn.read_body(conn)
                input_data = Jason.decode!(body)
                input = input_data["input"]

                case route_to_conversation(subprocess_id, conversation_id, input) do
                  {:ok, response} ->
                    conn
                    |> Conn.put_resp_content_type("application/json")
                    |> Conn.send_resp(200, Jason.encode!(%{response: response}))

                  {:error, reason} ->
                    conn
                    |> Conn.put_resp_content_type("application/json")
                    |> Conn.send_resp(400, Jason.encode!(%{error: reason}))
                end

              "DELETE" ->
                case end_conversation(subprocess_id, conversation_id) do
                  :ok ->
                    conn
                    |> Conn.send_resp(204, "")

                  {:error, reason} ->
                    conn
                    |> Conn.put_resp_content_type("application/json")
                    |> Conn.send_resp(400, Jason.encode!(%{error: reason}))
                end

              _ ->
                conn
                |> Conn.send_resp(405, "Method not allowed")
            end
          else
            conn
            |> Conn.send_resp(404, "Not found")
          end
      end
    end

    Bandit.start_link(
      scheme: :http,
      port: port,
      plug: handler
    )
  end

  defp process_http_input(input, conversation, user_id) do
    case Brain.evaluate(conversation.id, input, user_id: user_id) do
      {:ok, %{response: response}} when is_binary(response) ->
        response

      {:ok, response} when is_binary(response) ->
        response

      {:ok, %{response: nil}} ->
        ""

      {:ok, nil} ->
        ""

      {:error, reason} ->
        Logger.warning("Brain.evaluate failed in HTTP subprocess",
          reason: inspect(reason),
          input: input
        )

        "I'm sorry, I wasn't able to process that right now."
    end
  end
end
