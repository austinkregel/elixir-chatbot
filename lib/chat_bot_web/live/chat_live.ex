defmodule ChatBotWeb.ChatLive do
  @moduledoc """
  LiveView for the chat interface.
  Provides real-time chat functionality with the AI brain.
  """

  use ChatBotWeb, :live_view
  require Logger

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Subscribe to brain channels for real-time updates
      Phoenix.PubSub.subscribe(ChatBot.PubSub, "brain:status")
      Phoenix.PubSub.subscribe(ChatBot.PubSub, "brain:learning")
      Phoenix.PubSub.subscribe(ChatBot.PubSub, "brain:conversations")
      Phoenix.PubSub.subscribe(ChatBot.PubSub, "brain:analysis")

      # Start periodic system status polling (every 2 seconds)
      :timer.send_interval(2_000, self(), :refresh_system_status)
    end

    # Get initial status with longer timeout
    status = GenServer.call(ChatBot.Brain, :get_status, 60_000)
    conversations = GenServer.call(ChatBot.Brain, :get_conversations, 60_000)
    knowledge = get_combined_knowledge(status.name)

    # Get cognitive memory stats
    memory_stats = get_cognitive_memory_stats()

    # Get system status
    system_status = ChatBot.SystemStatus.get_all()

    # Generate a session user_id for epistemic tracking
    user_id = "web_user_" <> (:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower))

    socket =
      socket
      |> assign(:status, status)
      |> assign(:conversations, conversations)
      |> assign(:current_conversation_id, nil)
      |> assign(:messages, [])
      |> assign(:input_text, "")
      |> assign(:is_connected, true)
      |> assign(:error_message, nil)
      |> assign(:knowledge, knowledge)
      |> assign(:show_knowledge, false)
      |> assign(:show_processing, true)
      |> assign(:expanded_traces, MapSet.new())
      |> assign(:analysis_logs, %{})
      |> assign(:analysis_details, %{})
      |> assign(:dev_panel_tabs, %{})
      |> assign(:memory_stats, memory_stats)
      |> assign(:system_status, system_status)
      |> assign(:user_id, user_id)
      |> assign(:selected_message_id, nil)

    {:ok, socket}
  end

  @impl true
  def handle_event("send_message", %{"input" => input}, socket) do
    if socket.assigns.current_conversation_id do
      # Send message to existing conversation
      send_message(socket.assigns.current_conversation_id, input, socket)
    else
      # Create new conversation first
      case ChatBot.Brain.create_conversation() do
        {:ok, conversation_id} ->
          socket =
            socket
            |> assign(:current_conversation_id, conversation_id)
            |> assign(:conversations, [
              %{
                id: conversation_id,
                message_count: 0,
                created_at: System.system_time(:millisecond),
                last_activity: System.system_time(:millisecond)
              }
              | socket.assigns.conversations
            ])

          send_message(conversation_id, input, socket)

        {:error, reason} ->
          socket = assign(socket, :error_message, "Failed to create conversation: #{reason}")
          {:noreply, socket}
      end
    end
  end

  def handle_event("input_change", %{"value" => value}, socket) do
    {:noreply, assign(socket, :input_text, value)}
  end

  def handle_event("toggle_knowledge", _params, socket) do
    {:noreply, assign(socket, :show_knowledge, !socket.assigns.show_knowledge)}
  end

  def handle_event("toggle_processing", _params, socket) do
    {:noreply, assign(socket, :show_processing, !socket.assigns.show_processing)}
  end

  def handle_event("toggle_trace", %{"message_id" => message_id}, socket) do
    expanded = socket.assigns.expanded_traces

    new_expanded =
      if MapSet.member?(expanded, message_id) do
        MapSet.delete(expanded, message_id)
      else
        MapSet.put(expanded, message_id)
      end

    {:noreply, assign(socket, :expanded_traces, new_expanded)}
  end

  def handle_event("set_dev_tab", %{"message_id" => message_id, "tab" => tab}, socket) do
    tabs = socket.assigns.dev_panel_tabs || %{}
    {:noreply, assign(socket, :dev_panel_tabs, Map.put(tabs, message_id, tab))}
  end

  def handle_event("select_message", %{"message_id" => message_id}, socket) do
    # Toggle selection: if already selected, deselect; otherwise select
    new_selected =
      if socket.assigns.selected_message_id == message_id do
        nil
      else
        message_id
      end

    {:noreply, assign(socket, :selected_message_id, new_selected)}
  end

  def handle_event("select_conversation", %{"conversation_id" => conversation_id}, socket) do
    case ChatBot.Brain.get_conversation(conversation_id) do
      {:ok, conversation} ->
        messages = to_display_messages(conversation_id, Map.get(conversation, :memory, []))

        socket =
          socket
          |> assign(:current_conversation_id, conversation_id)
          |> assign(:messages, messages)
          |> assign(:expanded_traces, MapSet.new())
          |> assign(:analysis_logs, %{})
          |> assign(:analysis_details, %{})
          |> assign(:dev_panel_tabs, %{})
          |> assign(:error_message, nil)
          |> assign(:selected_message_id, nil)

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, assign(socket, :error_message, "Failed to load conversation: #{reason}")}
    end
  end

  def handle_event("new_conversation", _params, socket) do
    socket =
      socket
      |> assign(:current_conversation_id, nil)
      |> assign(:messages, [])
      |> assign(:expanded_traces, MapSet.new())
      |> assign(:analysis_logs, %{})
      |> assign(:analysis_details, %{})
      |> assign(:dev_panel_tabs, %{})
      |> assign(:error_message, nil)
      |> assign(:selected_message_id, nil)

    {:noreply, socket}
  end

  def handle_event("end_conversation", _params, socket) do
    if socket.assigns.current_conversation_id do
      ChatBot.Brain.end_conversation(socket.assigns.current_conversation_id)

      socket =
        socket
        |> assign(:current_conversation_id, nil)
        |> assign(:messages, [])
        |> assign(
          :conversations,
          Enum.reject(
            socket.assigns.conversations,
            &(&1.id == socket.assigns.current_conversation_id)
          )
        )
        |> assign(:error_message, nil)

      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end

  @impl true
  def handle_info(
        {:conversation_result, %{conversation_id: conversation_id, response: response}},
        socket
      ) do
    if socket.assigns.current_conversation_id == conversation_id do
      new_message = %{
        id: generate_message_id(),
        role: "assistant",
        content: response,
        timestamp: System.system_time(:millisecond)
      }

      socket =
        socket
        |> assign(:messages, socket.assigns.messages ++ [new_message])
        |> assign(:input_text, "")
        |> assign(:error_message, nil)

      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end

  def handle_info(
        {:conversation_error, %{conversation_id: conversation_id, error: error}},
        socket
      ) do
    if socket.assigns.current_conversation_id == conversation_id do
      socket = assign(socket, :error_message, "Error: #{error}")
      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end

  def handle_info(
        %Phoenix.Socket.Broadcast{
          topic: "brain:analysis",
          event: "analysis_progress",
          payload: payload
        },
        socket
      ) do
    message_id = Map.get(payload, :message_id) || Map.get(payload, "message_id")
    logs = socket.assigns.analysis_logs || %{}
    details = socket.assigns.analysis_details || %{}

    socket =
      if is_binary(message_id) do
        updated =
          Map.update(logs, message_id, [payload], fn existing ->
            (existing ++ [payload]) |> Enum.take(-200)
          end)

        updated_details =
          Map.update(details, message_id, update_analysis_details(%{}, payload), fn existing ->
            update_analysis_details(existing, payload)
          end)

        socket
        |> assign(:analysis_logs, updated)
        |> assign(:analysis_details, updated_details)
      else
        socket
      end

    {:noreply, socket}
  end

  def handle_info(
        {:evaluation_complete,
         %{conversation_id: conversation_id, message_id: message_id, result: result}},
        socket
      ) do
    case result do
      {:ok, nil} ->
        # Response was deferred (e.g., backchannel, gratitude loop, continuation)
        # No assistant message to display - just clear input
        socket =
          if socket.assigns.current_conversation_id == conversation_id do
            socket
            |> assign(:input_text, "")
            |> assign(:error_message, nil)
          else
            socket
          end

        {:noreply, socket}

      {:ok, response} when is_binary(response) and response != "" ->
        assistant_message = %{
          id: generate_message_id(),
          role: "assistant",
          content: response,
          timestamp: System.system_time(:millisecond),
          trace: nil
        }

        socket =
          if socket.assigns.current_conversation_id == conversation_id do
            socket
            |> assign(:messages, socket.assigns.messages ++ [assistant_message])
            |> assign(:input_text, "")
            |> assign(:error_message, nil)
          else
            socket
          end

        {:noreply, socket}

      {:ok, ""} ->
        # Empty response - treat same as nil (deferred)
        socket =
          if socket.assigns.current_conversation_id == conversation_id do
            socket
            |> assign(:input_text, "")
            |> assign(:error_message, nil)
          else
            socket
          end

        {:noreply, socket}

      {:error, reason} ->
        socket =
          if socket.assigns.current_conversation_id == conversation_id do
            assign(socket, :error_message, "Error: #{reason}")
          else
            socket
          end

        {:noreply, socket}

      other ->
        Logger.warning("Unexpected evaluation result", %{
          message_id: message_id,
          result: inspect(other)
        })

        {:noreply, socket}
    end
  end

  def handle_info(
        %Phoenix.Socket.Broadcast{
          topic: "brain:learning",
          event: "learning_processed",
          payload: _data
        },
        socket
      ) do
    # Update status, knowledge, and memory stats when learning is processed
    status = ChatBot.Brain.get_status()
    knowledge = get_combined_knowledge(status.name)
    memory_stats = get_cognitive_memory_stats()

    {:noreply,
     socket
     |> assign(:status, status)
     |> assign(:knowledge, knowledge)
     |> assign(:memory_stats, memory_stats)}
  end

  def handle_info(
        %Phoenix.Socket.Broadcast{
          topic: "brain:status",
          event: "interrupt_acknowledged",
          payload: data
        },
        socket
      ) do
    # Handle interrupt acknowledgment
    socket = assign(socket, :error_message, "System interrupted: #{data.reason}")
    {:noreply, socket}
  end

  def handle_info(
        %Phoenix.Socket.Broadcast{
          topic: "brain:status",
          event: "emergency_acknowledged",
          payload: data
        },
        socket
      ) do
    # Handle emergency acknowledgment
    socket = assign(socket, :error_message, "Emergency: #{data.reason}")
    {:noreply, socket}
  end

  def handle_info(:refresh_system_status, socket) do
    # Refresh system status for background processes
    system_status = ChatBot.SystemStatus.get_all()
    memory_stats = get_cognitive_memory_stats()

    {:noreply,
     socket
     |> assign(:system_status, system_status)
     |> assign(:memory_stats, memory_stats)}
  end

  # Private Functions

  defp send_message(conversation_id, input, socket) do
    message_id = generate_message_id()

    # Add user message to the display immediately (analysis streams in via PubSub)
    user_message = %{
      id: message_id,
      role: "user",
      content: input,
      timestamp: System.system_time(:millisecond),
      trace: nil
    }

    socket =
      socket
      |> assign(:messages, socket.assigns.messages ++ [user_message])
      |> assign(:error_message, nil)

    view_pid = self()

    user_id = socket.assigns.user_id

    Task.start(fn ->
      result =
        ChatBot.Brain.evaluate(conversation_id, input,
          user_id: user_id,
          progress: %{conversation_id: conversation_id, message_id: message_id}
        )

      send(
        view_pid,
        {:evaluation_complete,
         %{conversation_id: conversation_id, message_id: message_id, result: result}}
      )
    end)

    {:noreply, socket}
  end

  defp generate_message_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  def step_label(step) when is_atom(step) do
    step |> Atom.to_string() |> String.replace("_", " ")
  end

  def step_label(step) when is_binary(step), do: String.replace(step, "_", " ")
  def step_label(_), do: "progress"

  def strategy_badge_variant(:can_respond), do: :success
  def strategy_badge_variant(:needs_clarification), do: :warning
  def strategy_badge_variant(:partial_response_with_clarification), do: :info
  def strategy_badge_variant(:cannot_respond), do: :error
  def strategy_badge_variant(_), do: :default

  defp update_analysis_details(details, payload) when is_map(details) and is_map(payload) do
    step = Map.get(payload, :step) || Map.get(payload, "step")
    ts = Map.get(payload, :timestamp) || Map.get(payload, "timestamp")
    chunk_index = Map.get(payload, :chunk_index) || Map.get(payload, "chunk_index")

    details =
      details
      |> Map.put_new(:started_at, ts)
      |> Map.update(:steps, [payload], fn existing ->
        (existing ++ [payload]) |> Enum.take(-300)
      end)

    details =
      case step do
        :pipeline_start ->
          Map.put(
            details,
            :text_length,
            Map.get(payload, :text_length) || Map.get(payload, "text_length")
          )

        :chunking_complete ->
          Map.put(
            details,
            :chunk_count,
            Map.get(payload, :chunk_count) || Map.get(payload, "chunk_count")
          )

        :strategy_determined ->
          details
          |> Map.put(
            :overall_strategy,
            Map.get(payload, :overall_strategy) || Map.get(payload, "overall_strategy")
          )
          |> Map.put(:strategy_reasoning, %{
            chunk_strategies:
              Map.get(payload, :chunk_strategies) || Map.get(payload, "chunk_strategies") || [],
            has_expressives:
              Map.get(payload, :has_expressives) || Map.get(payload, "has_expressives"),
            has_substantive:
              Map.get(payload, :has_substantive) || Map.get(payload, "has_substantive"),
            missing_slots_count:
              Map.get(payload, :missing_slots_count) || Map.get(payload, "missing_slots_count") ||
                0,
            missing_slots:
              Map.get(payload, :missing_slots) || Map.get(payload, "missing_slots") || [],
            decision_reason:
              Map.get(payload, :decision_reason) || Map.get(payload, "decision_reason"),
            suggested_prompts:
              Map.get(payload, :suggested_prompts) || Map.get(payload, "suggested_prompts") || []
          })

        :pipeline_complete ->
          Map.put(
            details,
            :elapsed_ms,
            Map.get(payload, :elapsed_ms) || Map.get(payload, "elapsed_ms")
          )

        :racing_complete ->
          Map.put(details, :racing, %{
            fast_path: Map.get(payload, :fast_path) || Map.get(payload, "fast_path"),
            fast_path_source:
              Map.get(payload, :fast_path_source) || Map.get(payload, "fast_path_source"),
            early_exit: Map.get(payload, :early_exit) || Map.get(payload, "early_exit"),
            elapsed_ms: Map.get(payload, :elapsed_ms) || Map.get(payload, "elapsed_ms"),
            results: Map.get(payload, :results) || Map.get(payload, "results") || [],
            alternatives:
              Map.get(payload, :alternatives) || Map.get(payload, "alternatives") || []
          })

        :memory_query ->
          Map.put(details, :memory, %{
            query_text: Map.get(payload, :query_text) || Map.get(payload, "query_text"),
            match_count: Map.get(payload, :match_count) || Map.get(payload, "match_count") || 0,
            top_similarity:
              Map.get(payload, :top_similarity) || Map.get(payload, "top_similarity") || 0.0,
            matches: Map.get(payload, :matches) || Map.get(payload, "matches") || []
          })

        :response_generated ->
          Map.put(details, :response, %{
            response_type:
              Map.get(payload, :response_type) || Map.get(payload, "response_type"),
            strategy: Map.get(payload, :strategy) || Map.get(payload, "strategy"),
            method: Map.get(payload, :method) || Map.get(payload, "method"),
            intent: Map.get(payload, :intent) || Map.get(payload, "intent"),
            entities_count:
              Map.get(payload, :entities_count) || Map.get(payload, "entities_count") || 0,
            nlp_confidence:
              Map.get(payload, :nlp_confidence) || Map.get(payload, "nlp_confidence"),
            base_method: Map.get(payload, :base_method) || Map.get(payload, "base_method"),
            prompts_count:
              Map.get(payload, :prompts_count) || Map.get(payload, "prompts_count") || 0
          })

        _ ->
          details
      end

    if is_integer(chunk_index) do
      chunks = Map.get(details, :chunks, %{})
      chunk = Map.get(chunks, chunk_index, %{}) |> Map.put_new(:index, chunk_index)

      chunk =
        case step do
          :chunk_start ->
            chunk
            |> Map.put(
              :chunk_length,
              Map.get(payload, :chunk_length) || Map.get(payload, "chunk_length")
            )
            |> Map.put(
              :chunk_text,
              Map.get(payload, :chunk_text) || Map.get(payload, "chunk_text")
            )

          :discourse_complete ->
            Map.put(chunk, :discourse, %{
              addressee: Map.get(payload, :addressee) || Map.get(payload, "addressee"),
              confidence: Map.get(payload, :confidence) || Map.get(payload, "confidence")
            })

          :speech_act_complete ->
            Map.put(chunk, :speech_act, %{
              category: Map.get(payload, :category) || Map.get(payload, "category"),
              sub_type: Map.get(payload, :sub_type) || Map.get(payload, "sub_type"),
              confidence: Map.get(payload, :confidence) || Map.get(payload, "confidence"),
              is_question: Map.get(payload, :is_question) || Map.get(payload, "is_question")
            })

          :anaphora_resolved ->
            Map.put(chunk, :anaphora, %{
              resolved_count:
                Map.get(payload, :resolved_count) || Map.get(payload, "resolved_count") || 0,
              entities: Map.get(payload, :entities) || Map.get(payload, "entities") || []
            })

          :entities_filtered ->
            Map.put(chunk, :entity_filtering, %{
              original_count:
                Map.get(payload, :original_count) || Map.get(payload, "original_count") || 0,
              filtered_count:
                Map.get(payload, :filtered_count) || Map.get(payload, "filtered_count") || 0,
              excluded_types:
                Map.get(payload, :excluded_types) || Map.get(payload, "excluded_types") || []
            })

          :entities_extracted ->
            chunk
            |> Map.put(
              :entities,
              Map.get(payload, :entities) || Map.get(payload, "entities") || []
            )
            |> Map.put(
              :entity_count,
              Map.get(payload, :entity_count) || Map.get(payload, "entity_count")
            )

          :intent_determined ->
            chunk
            |> Map.put(:intent, Map.get(payload, :intent) || Map.get(payload, "intent"))
            |> Map.put(
              :intent_method,
              Map.get(payload, :intent_method) || Map.get(payload, "intent_method")
            )
            |> Map.put(
              :intent_confidence,
              Map.get(payload, :intent_confidence) || Map.get(payload, "intent_confidence")
            )

          :slots_detected ->
            Map.put(chunk, :slots_detected, %{
              missing_required:
                Map.get(payload, :missing_required) || Map.get(payload, "missing_required") || [],
              filled_count: Map.get(payload, :filled_count) || Map.get(payload, "filled_count"),
              filled_slots:
                Map.get(payload, :filled_slots) || Map.get(payload, "filled_slots") || %{}
            })

          :context_resolved ->
            Map.put(chunk, :context_resolved, %{
              all_required_filled:
                Map.get(payload, :all_required_filled) || Map.get(payload, "all_required_filled"),
              missing_required:
                Map.get(payload, :missing_required) || Map.get(payload, "missing_required") || [],
              filled_slots:
                Map.get(payload, :filled_slots) || Map.get(payload, "filled_slots") || %{}
            })

          :chunk_complete ->
            chunk
            |> Map.put(
              :response_strategy,
              Map.get(payload, :response_strategy) || Map.get(payload, "response_strategy")
            )
            |> Map.put(
              :confidence,
              Map.get(payload, :confidence) || Map.get(payload, "confidence")
            )

          _ ->
            chunk
        end

      Map.put(details, :chunks, Map.put(chunks, chunk_index, chunk))
    else
      details
    end
  end

  defp to_display_messages(conversation_id, memory) when is_list(memory) do
    now = System.system_time(:millisecond)
    base = now - max(length(memory) - 1, 0) * 1_000

    memory
    |> Enum.with_index()
    |> Enum.map(fn {entry, idx} ->
      role = Map.get(entry, :role) || Map.get(entry, "role") || "system"

      content =
        Map.get(entry, :content) || Map.get(entry, "content") || Map.get(entry, "text") || ""

      timestamp =
        Map.get(entry, :timestamp) ||
          Map.get(entry, "timestamp") ||
          get_in(entry, [:context, :timestamp]) ||
          get_in(entry, ["context", "timestamp"]) ||
          base + idx * 1_000

      id =
        Map.get(entry, :id) ||
          Map.get(entry, "id") ||
          "#{conversation_id}:#{idx}"

      %{
        id: id,
        role: role,
        content: content,
        timestamp: timestamp,
        trace: nil
      }
    end)
    # Filter out messages with nil or empty content (deferred responses)
    |> Enum.filter(fn msg ->
      msg.content != nil and msg.content != ""
    end)
  end

  # Component for rendering processing trace
  attr :trace, :map, required: true

  def processing_trace(assigns) do
    ~H"""
    <div class="bg-base-200 rounded-lg p-3 text-xs border border-base-300 shadow-sm">
      <!-- Multi-chunk header -->
      <%= if (@trace.chunk_count || 1) > 1 do %>
        <div class="flex items-center justify-between mb-3 pb-2 border-b border-base-300">
          <div class="flex items-center gap-2">
            <.icon name="hero-document-text" class="w-4 h-4 text-info" />
            <span class="font-semibold text-base-content">
              {@trace.chunk_count} utterances detected
            </span>
          </div>
          <div class="flex items-center gap-2 text-base-content/60">
            <span>{@trace.total_processing_ms || 0}ms total</span>
            <span class={[
              "badge badge-xs",
              strategy_badge_class(@trace.overall_strategy)
            ]}>
              {format_strategy(@trace.overall_strategy)}
            </span>
          </div>
        </div>
        
    <!-- Each chunk -->
        <div class="space-y-3">
          <%= for chunk <- @trace.chunks || [] do %>
            <.chunk_trace chunk={chunk} />
          <% end %>
        </div>
      <% else %>
        <!-- Single chunk - show full details -->
        <.single_chunk_trace trace={@trace} />
      <% end %>
    </div>
    """
  end

  # Component for a single chunk in multi-chunk view
  attr :chunk, :map, required: true

  defp chunk_trace(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-lg p-2 border border-base-300">
      <!-- Chunk header with text preview -->
      <div class="flex items-start justify-between gap-2 mb-2">
        <div class="flex-1">
          <div class="text-base-content/60 text-xs mb-1">
            Chunk {@chunk.index + 1}
          </div>
          <div class="text-sm text-base-content italic truncate" title={@chunk.text}>
            "{@chunk.text}"
          </div>
        </div>
        <div class="flex items-center gap-1 shrink-0">
          <span class="font-medium text-base-content">
            {@chunk.intent || "Unknown"}
          </span>
          <.confidence_badge level={@chunk.confidence_level} confidence={@chunk.confidence} />
          <%= if @chunk.fast_path do %>
            <span class="badge badge-success badge-xs">⚡</span>
          <% end %>
        </div>
      </div>
      
    <!-- Compact details row -->
      <div class="flex flex-wrap items-center gap-2 text-base-content/60">
        <!-- Entities -->
        <%= if length(@chunk.entities || []) > 0 do %>
          <div class="flex items-center gap-1">
            <.icon name="hero-tag" class="w-3 h-3" />
            <%= for entity <- Enum.take(@chunk.entities, 3) do %>
              <span class="badge badge-outline badge-xs">{entity.value}</span>
            <% end %>
            <%= if length(@chunk.entities) > 3 do %>
              <span class="text-xs">+{length(@chunk.entities) - 3}</span>
            <% end %>
          </div>
        <% end %>
        
    <!-- Missing slots -->
        <%= if length(@chunk.slots_missing || []) > 0 do %>
          <div class="flex items-center gap-1 text-warning">
            <.icon name="hero-exclamation-triangle" class="w-3 h-3" />
            <span>Missing: {Enum.join(@chunk.slots_missing, ", ")}</span>
          </div>
        <% end %>
        
    <!-- Alternatives (collapsed) -->
        <%= if length(@chunk.alternatives || []) > 0 do %>
          <div class="flex items-center gap-1">
            <span class="text-base-content/40">Also:</span>
            <%= for alt <- Enum.take(@chunk.alternatives, 2) do %>
              <span class="text-xs">{alt.intent}</span>
            <% end %>
          </div>
        <% end %>
        
    <!-- Backtrack indicator -->
        <%= if @chunk.backtrack_count > 0 do %>
          <span class="badge badge-warning badge-xs">↩{@chunk.backtrack_count}</span>
        <% end %>
        
    <!-- Time -->
        <span class="ml-auto">{@chunk.racing_ms}ms</span>
      </div>
    </div>
    """
  end

  # Component for single chunk (full detail view)
  attr :trace, :map, required: true

  defp single_chunk_trace(assigns) do
    ~H"""
    <div>
      <!-- Header with intent and confidence -->
      <div class="flex items-center justify-between mb-2 pb-2 border-b border-base-300">
        <div class="flex items-center gap-2">
          <span class="font-semibold text-base-content">
            {@trace.intent || "Unknown"}
          </span>
          <.confidence_badge level={@trace.confidence_level} confidence={@trace.confidence} />
        </div>
        <div class="flex items-center gap-2 text-base-content/60">
          <%= if @trace.fast_path do %>
            <span class="badge badge-success badge-xs">Fast Path</span>
          <% end %>
          <%= if @trace.racing_ms do %>
            <span>{@trace.racing_ms}ms</span>
          <% end %>
        </div>
      </div>
      
    <!-- Racing Analyzers -->
      <%= if length(@trace.analyzers || []) > 0 do %>
        <div class="mb-3">
          <div class="font-semibold text-base-content/70 mb-1 flex items-center gap-1">
            <.icon name="hero-scale" class="w-3 h-3" /> Racing Analyzers
          </div>
          <div class="space-y-1">
            <%= for {analyzer, idx} <- Enum.with_index(@trace.analyzers) do %>
              <div class="flex items-center gap-2">
                <div class="w-24 truncate text-base-content/60">{analyzer.analyzer}</div>
                <div class="flex-1">
                  <div class="flex items-center gap-1">
                    <div
                      class={["h-1.5 rounded-full", activation_color(analyzer.calibrated)]}
                      style={"width: #{analyzer.calibrated * 100}%"}
                    >
                    </div>
                    <span class="text-base-content/50 w-10">
                      {format_percent(analyzer.calibrated)}
                    </span>
                  </div>
                </div>
                <%= if idx == 0 do %>
                  <span class="badge badge-primary badge-xs">Winner</span>
                <% end %>
              </div>
            <% end %>
          </div>
        </div>
      <% end %>
      
    <!-- Alternatives -->
      <%= if length(@trace.alternatives || []) > 0 do %>
        <div class="mb-3">
          <div class="font-semibold text-base-content/70 mb-1 flex items-center gap-1">
            <.icon name="hero-arrows-right-left" class="w-3 h-3" /> Also Considered
          </div>
          <div class="flex flex-wrap gap-1">
            <%= for alt <- @trace.alternatives do %>
              <span class="badge badge-ghost badge-sm">
                {alt.intent}
                <span class="opacity-60 ml-1">{format_percent(alt.activation)}</span>
              </span>
            <% end %>
          </div>
        </div>
      <% end %>
      
    <!-- Entities & Slots -->
      <div class="grid grid-cols-2 gap-3 mb-3">
        <!-- Entities Found -->
        <div>
          <div class="font-semibold text-base-content/70 mb-1 flex items-center gap-1">
            <.icon name="hero-tag" class="w-3 h-3" /> Entities
          </div>
          <%= if length(@trace.entities || []) > 0 do %>
            <div class="space-y-0.5">
              <%= for entity <- @trace.entities do %>
                <% 
                  confidence = Map.get(entity, :confidence)
                  confidence_percent = if confidence, do: Float.round(confidence * 100, 1), else: nil
                  confidence_variant = cond do
                    confidence && confidence >= 0.8 -> :success
                    confidence && confidence >= 0.6 -> :warning
                    confidence -> :error
                    true -> :default
                  end
                %>
                <div class="flex items-center gap-1">
                  <span class="badge badge-outline badge-xs">{entity.type}</span>
                  <%= if confidence_percent do %>
                    <.badge variant={confidence_variant} size={:xs}>
                      {confidence_percent}%
                    </.badge>
                  <% end %>
                  <span class="text-base-content/80 truncate">{entity.value}</span>
                </div>
              <% end %>
            </div>
          <% else %>
            <span class="text-base-content/40 italic">None detected</span>
          <% end %>
        </div>
        
    <!-- Slots -->
        <div>
          <div class="font-semibold text-base-content/70 mb-1 flex items-center gap-1">
            <.icon name="hero-puzzle-piece" class="w-3 h-3" /> Slots
          </div>
          <%= if map_size(@trace.slots_filled || %{}) > 0 || length(@trace.slots_missing || []) > 0 do %>
            <div class="space-y-0.5">
              <%= for {slot, value} <- @trace.slots_filled || %{} do %>
                <div class="flex items-center gap-1">
                  <span class="text-success">✓</span>
                  <span class="text-base-content/60">{slot}:</span>
                  <span class="text-base-content/80">{value}</span>
                </div>
              <% end %>
              <%= for slot <- @trace.slots_missing || [] do %>
                <div class="flex items-center gap-1 text-warning">
                  <span>✗</span>
                  <span>{slot}</span>
                  <span class="text-base-content/40">(missing)</span>
                </div>
              <% end %>
            </div>
          <% else %>
            <span class="text-base-content/40 italic">None required</span>
          <% end %>
        </div>
      </div>
      
    <!-- Backtracking -->
      <%= if @trace.backtrack_count > 0 do %>
        <div class="mb-2 p-2 bg-warning/10 rounded border border-warning/30">
          <div class="flex items-center gap-2">
            <.icon name="hero-arrow-path" class="w-4 h-4 text-warning" />
            <span class="text-warning font-medium">
              Backtracked {@trace.backtrack_count}x
            </span>
            <%= if @trace.backtrack_reason do %>
              <span class="text-base-content/60">- {@trace.backtrack_reason}</span>
            <% end %>
          </div>
        </div>
      <% end %>
      
    <!-- Clarification Needed -->
      <%= if @trace.needs_clarification && @trace.clarification do %>
        <div class="p-2 bg-info/10 rounded border border-info/30">
          <div class="flex items-center gap-2">
            <.icon name="hero-question-mark-circle" class="w-4 h-4 text-info" />
            <span class="text-info">{@trace.clarification}</span>
          </div>
        </div>
      <% end %>
      
    <!-- Stability Footer -->
      <div class="mt-2 pt-2 border-t border-base-300 flex items-center justify-between text-base-content/50">
        <div class="flex items-center gap-2">
          <span>Total Activation: {format_percent(@trace.total_activation)}</span>
          <%= if @trace.was_normalized do %>
            <span class="badge badge-warning badge-xs">Normalized</span>
          <% end %>
        </div>
        <span class="capitalize">{@trace.source}</span>
      </div>
    </div>
    """
  end

  attr :level, :atom, required: true
  attr :confidence, :string, required: true

  defp confidence_badge(assigns) do
    badge_class =
      case assigns.level do
        :high -> "badge-success"
        :medium -> "badge-info"
        :low -> "badge-warning"
        _ -> "badge-error"
      end

    assigns = assign(assigns, :badge_class, badge_class)

    ~H"""
    <span class={["badge badge-sm", @badge_class]}>
      {@confidence}
    </span>
    """
  end

  defp activation_color(value) when value >= 0.7, do: "bg-success"
  defp activation_color(value) when value >= 0.4, do: "bg-info"
  defp activation_color(value) when value >= 0.2, do: "bg-warning"
  defp activation_color(_), do: "bg-error"

  defp format_percent(nil), do: "0%"
  defp format_percent(value) when is_float(value), do: "#{round(value * 100)}%"
  defp format_percent(value) when is_integer(value), do: "#{value}%"
  defp format_percent(_), do: "0%"

  # Helper for compact badge display
  def confidence_badge_class(:high), do: "badge-success"
  def confidence_badge_class(:medium), do: "badge-info"
  def confidence_badge_class(:low), do: "badge-warning"
  def confidence_badge_class(_), do: "badge-error"

  # Make these public so they can be used in the template
  def strategy_badge_class(:can_respond), do: "badge-success"
  def strategy_badge_class(:partial_response_with_clarification), do: "badge-info"
  def strategy_badge_class(:needs_clarification), do: "badge-warning"
  def strategy_badge_class(_), do: "badge-error"

  def format_strategy(:can_respond), do: "Ready"
  def format_strategy(:partial_response_with_clarification), do: "Partial"
  def format_strategy(:needs_clarification), do: "Need Info"
  def format_strategy(:low_confidence), do: "Low Conf"
  def format_strategy(:response_optional), do: "Optional"
  def format_strategy(:response_deferred), do: "Deferred"
  def format_strategy(nil), do: "Unknown"
  def format_strategy(other), do: to_string(other)

  defp get_cognitive_memory_stats do
    if Process.whereis(ChatBot.Memory.Store) != nil do
      ChatBot.Memory.Store.stats()
    else
      %{episode_count: 0, semantic_count: 0, episode_index_size: 0, semantic_index_size: 0}
    end
  rescue
    _ -> %{episode_count: 0, semantic_count: 0, episode_index_size: 0, semantic_index_size: 0}
  end

  # Combines knowledge from KnowledgeStore and UserModelStore
  defp get_combined_knowledge(persona_name) do
    # Get structured knowledge from KnowledgeStore
    base_knowledge = ChatBot.KnowledgeStore.get_knowledge(persona_name)

    # Get user facts from UserModelStore
    user_facts = get_all_user_facts()

    # Merge user facts into the knowledge structure
    Map.put(base_knowledge, "user_facts", user_facts)
  end

  # Gets all user facts from UserModelStore for display
  defp get_all_user_facts do
    if Process.whereis(ChatBot.Epistemic.UserModelStore) do
      case ChatBot.Epistemic.UserModelStore.list_all_users() do
        {:ok, user_ids} ->
          user_ids
          |> Enum.map(fn user_id ->
            case ChatBot.Epistemic.UserModelStore.get(user_id) do
              nil ->
                nil

              model ->
                %{
                  "user_id" => user_id,
                  "facts" => format_user_facts(model.facts),
                  "confidence" => format_epistemic_bounds(model.epistemic_bounds),
                  "interaction_count" => map_size(model.interaction_patterns),
                  "last_seen" => model.updated_at
                }
            end
          end)
          |> Enum.reject(&is_nil/1)

        _ ->
          []
      end
    else
      []
    end
  rescue
    _ -> []
  end

  defp format_user_facts(facts) when is_map(facts) do
    facts
    |> Enum.map(fn {k, v} -> %{"key" => to_string(k), "value" => to_string(v)} end)
  end

  defp format_user_facts(_), do: []

  defp format_epistemic_bounds(bounds) when is_map(bounds) do
    bounds
    |> Enum.map(fn {k, v} -> %{"key" => to_string(k), "confidence" => v} end)
  end

  defp format_epistemic_bounds(_), do: []
end
