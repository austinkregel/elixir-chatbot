defmodule ChatWeb.ChatLive do
  @moduledoc "LiveView for the chat interface.\nProvides real-time chat functionality with the AI brain.\nUses global world context from WorldContext hook.\n"

  alias Brain.Epistemic.UserModelStore
  alias Brain.Epistemic.SourceAuthority
  alias Brain.KnowledgeStore
  alias Brain.Memory.Store
  alias Brain.SystemStatus
  alias Phoenix.PubSub
  use ChatWeb, :live_view
  require Logger

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      PubSub.subscribe(Brain.PubSub, "brain:status")
      PubSub.subscribe(Brain.PubSub, "brain:learning")
      PubSub.subscribe(Brain.PubSub, "brain:conversations")
      PubSub.subscribe(Brain.PubSub, "brain:analysis")
      PubSub.subscribe(Brain.PubSub, "world_models:status")
      :timer.send_interval(2000, self(), :refresh_system_status)
    end

    status = GenServer.call(Brain, :get_status, 60_000)
    conversations = GenServer.call(Brain, :get_conversations, 60_000)
    knowledge = get_combined_knowledge(status.name)
    memory_stats = get_cognitive_memory_stats()
    system_status = SystemStatus.get_all()
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
      |> assign(:world_models_loading, false)
      |> assign(:world_models_status, get_world_models_status(socket))
      |> assign(:inspector_correction_form, nil)
      |> assign(:authority_profiles, [])

    {:ok, socket}
  end

  defp get_world_models_status(socket) do
    world_id = Map.get(socket.assigns, :current_world_id, "default")
    SystemStatus.get_world_models_status(world_id)
  end

  @impl true
  def handle_params(params, _uri, socket) do
    conversation_id = params["conversation_id"]

    socket =
      cond do
        is_nil(conversation_id) and is_nil(socket.assigns.current_conversation_id) ->
          socket

        conversation_id == socket.assigns.current_conversation_id ->
          socket

        conversation_id ->
          case Brain.get_conversation(conversation_id) do
            {:ok, conversation} ->
              messages = to_display_messages(conversation_id, Map.get(conversation, :memory, []))

              socket
              |> assign(:current_conversation_id, conversation_id)
              |> assign(:messages, messages)
              |> assign(:expanded_traces, MapSet.new())
              |> assign(:analysis_logs, %{})
              |> assign(:analysis_details, %{})
              |> assign(:dev_panel_tabs, %{})
              |> assign(:error_message, nil)
              |> assign(:selected_message_id, nil)

            {:error, _reason} ->
              socket
              |> put_flash(:error, "Conversation not found")
              |> push_navigate(to: ~p"/chat")
          end

        true ->
          socket
          |> assign(:current_conversation_id, nil)
          |> assign(:messages, [])
          |> assign(:expanded_traces, MapSet.new())
          |> assign(:analysis_logs, %{})
          |> assign(:analysis_details, %{})
          |> assign(:dev_panel_tabs, %{})
          |> assign(:error_message, nil)
          |> assign(:selected_message_id, nil)
      end

    {:noreply, socket}
  end

  @impl true
  def handle_event("send_message", %{"input" => input}, socket) do
    if socket.assigns.current_conversation_id do
      send_message(socket.assigns.current_conversation_id, input, socket)
    else
      world_id = socket.assigns.current_world_id

      case Brain.create_conversation(world_id: world_id) do
        {:ok, conversation_id} ->
          socket =
            socket
            |> assign(:current_conversation_id, conversation_id)
            |> assign(:conversations, [
              %{
                id: conversation_id,
                world_id: world_id,
                message_count: 0,
                created_at: System.system_time(:millisecond),
                last_activity: System.system_time(:millisecond)
              }
              | socket.assigns.conversations
            ])
            |> push_patch(to: ~p"/chat/#{conversation_id}", replace: true)

          send_message(conversation_id, input, socket)

        {:error, reason} ->
          socket = assign(socket, :error_message, "Failed to create conversation: #{reason}")
          {:noreply, socket}
      end
    end
  end

  def handle_event("input_change", %{"input" => value}, socket) do
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
    new_selected =
      if socket.assigns.selected_message_id == message_id do
        nil
      else
        message_id
      end

    {:noreply, assign(socket, :selected_message_id, new_selected)}
  end

  def handle_event("select_conversation", %{"conversation_id" => conversation_id}, socket) do
    {:noreply, push_patch(socket, to: ~p"/chat/#{conversation_id}")}
  end

  def handle_event("new_conversation", _params, socket) do
    socket =
      socket
      |> assign(:selected_message_id, nil)
      |> push_patch(to: ~p"/chat")

    {:noreply, socket}
  end

  def handle_event("switch_world", %{"world_id" => _world_id}, socket) do
    {:noreply, reset_for_world_change(socket)}
  end

  def handle_event("refresh_worlds", _params, socket) do
    {:noreply, socket}
  end

  def handle_event("end_conversation", _params, socket) do
    if socket.assigns.current_conversation_id do
      Brain.end_conversation(socket.assigns.current_conversation_id)

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

  # ---- Inspector belief actions ----

  def handle_event("inspector_confirm_belief", %{"belief_id" => belief_id}, socket) do
    alias Brain.Epistemic.BeliefStore

    case BeliefStore.confirm_belief(belief_id) do
      {:ok, _updated} ->
        {:noreply, put_flash(socket, :info, "Belief confirmed")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to confirm: #{inspect(reason)}")}
    end
  end

  def handle_event("inspector_retract_belief", %{"belief_id" => belief_id}, socket) do
    alias Brain.Epistemic.BeliefStore

    case BeliefStore.retract_belief(belief_id) do
      :ok ->
        {:noreply, put_flash(socket, :info, "Belief retracted")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to retract: #{inspect(reason)}")}
    end
  end

  def handle_event("show_correction_form", _params, socket) do
    form = %{"subject" => "self", "predicate" => "", "object" => "", "authority" => "mentor"}

    authority_profiles =
      if SourceAuthority.ready?() do
        try do
          SourceAuthority.list_profiles()
        catch
          :exit, _ -> []
        end
      else
        []
      end

    socket =
      socket
      |> assign(:inspector_correction_form, form)
      |> assign(:authority_profiles, authority_profiles)

    {:noreply, socket}
  end

  def handle_event("cancel_correction_form", _params, socket) do
    {:noreply, assign(socket, :inspector_correction_form, nil)}
  end

  def handle_event("submit_correction_belief", params, socket) do
    alias Brain.Epistemic.BeliefStore

    subject = (params["subject"] || "self") |> String.trim()
    predicate = (params["predicate"] || "") |> String.trim()
    object = (params["object"] || "") |> String.trim()
    authority = (params["authority"] || "mentor") |> String.trim()

    if predicate != "" and object != "" do
      subject_atom =
        case subject do
          "user" -> :user
          "world" -> :world
          _ -> :self
        end

      predicate_atom =
        try do
          String.to_existing_atom(predicate)
        rescue
          _ -> String.to_atom(predicate)
        end

      authority_atom =
        try do
          String.to_existing_atom(authority)
        rescue
          _ -> String.to_atom(authority)
        end

      case BeliefStore.add_belief_with_authority(subject_atom, predicate_atom, object, authority_atom) do
        {:ok, _id} ->
          socket =
            socket
            |> assign(:inspector_correction_form, nil)
            |> put_flash(:info, "Guided belief added (#{authority})")

          {:noreply, socket}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Predicate and object are required")}
    end
  end

  defp reset_for_world_change(socket) do
    socket
    |> assign(:current_conversation_id, nil)
    |> assign(:messages, [])
    |> assign(:selected_message_id, nil)
  end

  @impl true
  def handle_info(
        {:conversation_result, %{conversation_id: conversation_id, response: response}},
        socket
      ) do
    if socket.assigns.current_conversation_id == conversation_id do
      {content, context} =
        case response do
          %{response: text, context: ctx} -> {text, ctx}
          text when is_binary(text) -> {text, nil}
          _ -> {inspect(response), nil}
        end

      new_message = %{
        id: generate_message_id(),
        role: "assistant",
        content: content,
        timestamp: System.system_time(:millisecond),
        context: context
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

  def handle_info({:world_context_changed, world_id}, socket) do
    world_models_status = SystemStatus.get_world_models_status(world_id)

    socket =
      socket
      |> reset_for_world_change()
      |> assign(:world_models_status, world_models_status)
      |> assign(:world_models_loading, false)

    {:noreply, socket}
  end

  def handle_info({:analysis_progress, payload}, socket) do
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
        %Phoenix.Socket.Broadcast{
          topic: "brain:analysis",
          event: "analysis_progress",
          payload: payload
        },
        socket
      ) do
    handle_info({:analysis_progress, payload}, socket)
  end

  def handle_info(
        {:evaluation_complete,
         %{conversation_id: conversation_id, message_id: message_id, result: result}},
        socket
      ) do
    case result do
      {:ok, nil} ->
        socket =
          if socket.assigns.current_conversation_id == conversation_id do
            socket
            |> assign(:input_text, "")
            |> assign(:error_message, nil)
          else
            socket
          end

        {:noreply, socket}

      {:ok, %{response: response} = enriched} when is_binary(response) and response != "" ->
        context = Map.get(enriched, :context, %{})
        processing_method = Map.get(enriched, :processing_method)

        assistant_message = %{
          id: generate_message_id(),
          role: "assistant",
          content: response,
          timestamp: System.system_time(:millisecond),
          trace: nil,
          context: context,
          processing_method: processing_method
        }

        socket =
          if socket.assigns.current_conversation_id == conversation_id do
            socket
            |> assign(:messages, socket.assigns.messages ++ [assistant_message])
            |> assign(:input_text, "")
            |> assign(:error_message, nil)
            |> merge_context_into_analysis_details(message_id, context)
          else
            socket
          end

        {:noreply, socket}

      {:ok, %{response: response}} when is_nil(response) or response == "" ->
        socket =
          if socket.assigns.current_conversation_id == conversation_id do
            socket
            |> assign(:input_text, "")
            |> assign(:error_message, nil)
          else
            socket
          end

        {:noreply, socket}

      # Backwards compatibility: plain string response
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

  def handle_info({:learning_processed, _data}, socket) do
    status = Brain.get_status()
    knowledge = get_combined_knowledge(status.name)
    memory_stats = get_cognitive_memory_stats()

    {:noreply,
     socket
     |> assign(:status, status)
     |> assign(:knowledge, knowledge)
     |> assign(:memory_stats, memory_stats)}
  end

  def handle_info({:interrupt_acknowledged, data}, socket) do
    socket = assign(socket, :error_message, "System interrupted: #{data.reason}")
    {:noreply, socket}
  end

  def handle_info({:emergency_acknowledged, data}, socket) do
    socket = assign(socket, :error_message, "Emergency: #{data.reason}")
    {:noreply, socket}
  end

  def handle_info(%Phoenix.Socket.Broadcast{event: event, payload: payload}, socket)
      when event in ["learning_processed", "interrupt_acknowledged", "emergency_acknowledged"] do
    handle_info({String.to_atom(event), payload}, socket)
  end

  def handle_info(:refresh_system_status, socket) do
    system_status = SystemStatus.get_all()
    memory_stats = get_cognitive_memory_stats()

    {:noreply,
     socket
     |> assign(:system_status, system_status)
     |> assign(:memory_stats, memory_stats)}
  end

  def handle_info({:world_models_loading, world_id}, socket) do
    if world_id == socket.assigns.current_world_id do
      {:noreply, assign(socket, :world_models_loading, true)}
    else
      {:noreply, socket}
    end
  end

  def handle_info({:world_models_loaded, world_id, _status}, socket) do
    if world_id == socket.assigns.current_world_id do
      world_models_status = get_world_models_status(socket)

      {:noreply,
       socket
       |> assign(:world_models_loading, false)
       |> assign(:world_models_status, world_models_status)}
    else
      {:noreply, socket}
    end
  end

  def handle_info({:world_models_error, world_id, reason}, socket) do
    if world_id == socket.assigns.current_world_id do
      Logger.warning("World models error for #{world_id}: #{inspect(reason)}")

      {:noreply,
       socket
       |> assign(:world_models_loading, false)
       |> put_flash(:warning, "World models failed to load. Using default models.")}
    else
      {:noreply, socket}
    end
  end

  defp send_message(conversation_id, input, socket) do
    message_id = generate_message_id()

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
      |> assign(:input_text, "")
      |> assign(:error_message, nil)

    view_pid = self()

    user_id = socket.assigns.user_id

    Task.start(fn ->
      result =
        Brain.evaluate(conversation_id, input,
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

  defp merge_context_into_analysis_details(socket, message_id, context)
       when is_binary(message_id) and is_map(context) do
    details = socket.assigns.analysis_details || %{}

    existing = Map.get(details, message_id, %{})

    merged =
      existing
      |> Map.put_new(:intent, Map.get(context, :intent))
      |> Map.put_new(:entities, Map.get(context, :entities, []))
      |> Map.put_new(:speech_act, Map.get(context, :speech_act))
      |> Map.put_new(:slots, Map.get(context, :slots, %{}))
      |> Map.put_new(:missing_slots, Map.get(context, :missing_slots, []))

    assign(socket, :analysis_details, Map.put(details, message_id, merged))
  end

  defp merge_context_into_analysis_details(socket, _message_id, _context), do: socket

  defp generate_message_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  def step_label(step) when is_atom(step) do
    step |> Atom.to_string() |> String.replace("_", " ")
  end

  def step_label(step) when is_binary(step) do
    String.replace(step, "_", " ")
  end

  def step_label(_) do
    "progress"
  end

  @doc false
  def stage_icon(step) do
    case step do
      :pipeline_start -> "▶"
      :chunking_complete -> "✂"
      :chunk_start -> "📦"
      :discourse_complete -> "🗣"
      :speech_act_complete -> "💬"
      :sentiment_complete -> "😊"
      :entities_extracted -> "🏷"
      :events_extracted -> "📅"
      :intent_determined -> "🎯"
      :entities_filtered -> "🔍"
      :fact_verification -> "✓"
      :slots_detected -> "🧩"
      :context_resolved -> "🔗"
      :anaphora_resolved -> "↩"
      :chunk_complete -> "✅"
      :strategy_determined -> "📋"
      :pipeline_complete -> "⏹"
      :racing_complete -> "⚡"
      :memory_query -> "🧠"
      :response_generated -> "💭"
      :followup_detected -> "↪"
      :meta_cognitive_query -> "🪞"
      :fast_path_used -> "⚡"
      :fast_path_bypassed -> "⏭"
      :fast_path_miss -> "⏭"
      :response_gate_start -> "🚦"
      :response_gate_complete -> "🚦"
      :nlp_pipeline_start -> "⚙"
      :nlp_pipeline_complete -> "⚙"
      :learning_complete -> "📚"
      _ -> "•"
    end
  end

  @doc false
  def stage_color(step) do
    case step do
      s when s in [:pipeline_start, :pipeline_complete] ->
        "bg-primary/10"

      s when s in [:chunk_start, :chunk_complete] ->
        "bg-info/10"

      s
      when s in [
             :discourse_complete,
             :speech_act_complete,
             :sentiment_complete,
             :intent_determined
           ] ->
        "bg-base-200/50"

      s when s in [:entities_extracted, :entities_filtered, :events_extracted] ->
        "bg-success/5"

      :fact_verification ->
        "bg-violet-500/10"

      s when s in [:slots_detected, :context_resolved, :anaphora_resolved] ->
        "bg-warning/5"

      s when s in [:strategy_determined, :response_gate_start, :response_gate_complete] ->
        "bg-info/5"

      s when s in [:racing_complete, :fast_path_used, :fast_path_bypassed, :fast_path_miss] ->
        "bg-amber-500/10"

      :memory_query ->
        "bg-cyan-500/10"

      :response_generated ->
        "bg-success/10"

      s when s in [:nlp_pipeline_start, :nlp_pipeline_complete] ->
        "bg-indigo-500/5"

      :learning_complete ->
        "bg-purple-500/5"

      s when s in [:followup_detected, :meta_cognitive_query] ->
        "bg-orange-500/10"

      _ ->
        "bg-base-200/50"
    end
  end

  @doc false
  def stage_detail(step) when is_map(step) do
    step_name = step[:step] || step["step"]

    case step_name do
      :pipeline_start ->
        len = step[:text_length] || step["text_length"]
        if len, do: "#{len} chars", else: ""

      :chunking_complete ->
        count = step[:chunk_count] || step["chunk_count"]
        if count, do: "#{count} chunk(s)", else: ""

      :chunk_start ->
        text = step[:chunk_text] || step["chunk_text"]
        if text, do: "\"#{String.slice(text, 0, 40)}#{if String.length(text || "") > 40, do: "…", else: ""}\"", else: ""

      :discourse_complete ->
        addr = step[:addressee] || step["addressee"]
        conf = step[:confidence] || step["confidence"]
        parts = []
        parts = if addr, do: parts ++ ["→ #{addr}"], else: parts
        parts = if is_number(conf), do: parts ++ ["#{Float.round(conf * 100, 1)}%"], else: parts
        Enum.join(parts, " ")

      :speech_act_complete ->
        cat = step[:category] || step["category"]
        sub = step[:sub_type] || step["sub_type"]
        is_q = step[:is_question] || step["is_question"]

        parts = []
        parts = if cat, do: parts ++ ["#{cat}"], else: parts
        parts = if sub, do: parts ++ ["/#{sub}"], else: parts
        parts = if is_q, do: parts ++ ["(question)"], else: parts
        Enum.join(parts, "")

      :sentiment_complete ->
        label = step[:label] || step["label"]
        conf = step[:confidence] || step["confidence"]
        parts = []
        parts = if label, do: parts ++ ["#{label}"], else: parts
        parts = if is_number(conf), do: parts ++ ["#{Float.round(conf * 100, 1)}%"], else: parts
        Enum.join(parts, " ")

      :entities_extracted ->
        count = step[:entity_count] || step["entity_count"] || 0
        "#{count} entity(ies)"

      :events_extracted ->
        count = step[:event_count] || step["event_count"] || 0
        "#{count} event(s)"

      :intent_determined ->
        intent = step[:intent] || step["intent"]
        method = step[:intent_method] || step["intent_method"]
        conf = step[:intent_confidence] || step["intent_confidence"]
        parts = []
        parts = if intent, do: parts ++ ["#{intent}"], else: parts
        parts = if method, do: parts ++ ["via #{method}"], else: parts
        parts = if is_number(conf), do: parts ++ ["(#{Float.round(conf * 100, 1)}%)"], else: parts
        Enum.join(parts, " ")

      :entities_filtered ->
        orig = step[:original_count] || step["original_count"] || 0
        filt = step[:filtered_count] || step["filtered_count"] || 0
        "#{orig} → #{filt}"

      :fact_verification ->
        status = step[:epistemic_status] || step["epistemic_status"]
        belief_count = step[:related_beliefs_count] || step["related_beliefs_count"] || 0
        parts = []
        parts = if status, do: parts ++ ["#{status}"], else: parts
        parts = if belief_count > 0, do: parts ++ ["#{belief_count} belief(s)"], else: parts
        Enum.join(parts, ", ")

      :slots_detected ->
        filled = step[:filled_count] || step["filled_count"] || 0
        missing = length(step[:missing_required] || step["missing_required"] || [])
        "#{filled} filled, #{missing} missing"

      :context_resolved ->
        all_filled = step[:all_required_filled] || step["all_required_filled"]
        if all_filled, do: "all required filled", else: "missing required slots"

      :strategy_determined ->
        strategy = step[:overall_strategy] || step["overall_strategy"]
        if strategy, do: "#{step_label(strategy)}", else: ""

      :pipeline_complete ->
        elapsed = step[:elapsed_ms] || step["elapsed_ms"]
        if elapsed, do: "#{elapsed}ms total", else: ""

      :racing_complete ->
        fast = step[:fast_path] || step["fast_path"]
        elapsed = step[:elapsed_ms] || step["elapsed_ms"]
        parts = []
        parts = if fast, do: parts ++ ["fast path hit"], else: parts ++ ["no fast path"]
        parts = if elapsed, do: parts ++ ["#{elapsed}ms"], else: parts
        Enum.join(parts, ", ")

      :memory_query ->
        count = step[:match_count] || step["match_count"] || 0
        sim = step[:top_similarity] || step["top_similarity"] || 0.0
        sim_str = if is_number(sim), do: "#{Float.round(sim * 100, 1)}%", else: "-"
        "#{count} match(es), top #{sim_str}"

      :response_generated ->
        type = step[:response_type] || step["response_type"]
        strategy = step[:strategy] || step["strategy"]
        parts = []
        parts = if type, do: parts ++ ["#{type}"], else: parts
        parts = if strategy, do: parts ++ ["(#{step_label(strategy)})"], else: parts
        Enum.join(parts, " ")

      :followup_detected ->
        prev = step[:previous_intent] || step["previous_intent"]
        if prev, do: "continuing #{prev}", else: ""

      :fast_path_used ->
        intent = step[:intent] || step["intent"]
        source = step[:source] || step["source"]
        parts = []
        parts = if intent, do: parts ++ ["#{intent}"], else: parts
        parts = if source, do: parts ++ ["via #{source}"], else: parts
        Enum.join(parts, " ")

      :fast_path_bypassed ->
        reason = step[:reason] || step["reason"]
        if reason, do: "#{reason}", else: "needs full analysis"

      :fast_path_miss ->
        "no heuristic match"

      :response_gate_complete ->
        decision = step[:decision] || step["decision"]
        if decision, do: "#{decision}", else: ""

      :nlp_pipeline_complete ->
        method = step[:method] || step["method"]
        intent = step[:final_intent] || step["final_intent"] || step[:nlp_intent] || step["nlp_intent"]
        parts = []
        parts = if method, do: parts ++ ["#{method}"], else: parts
        parts = if intent, do: parts ++ ["→ #{intent}"], else: parts
        Enum.join(parts, " ")

      :learning_complete ->
        "knowledge updated"

      _ ->
        ""
    end
  end

  def stage_detail(_), do: ""

  def strategy_badge_variant(:can_respond) do
    :success
  end

  def strategy_badge_variant(:needs_clarification) do
    :warning
  end

  def strategy_badge_variant(:partial_response_with_clarification) do
    :info
  end

  def strategy_badge_variant(:cannot_respond) do
    :error
  end

  def strategy_badge_variant(_) do
    :default
  end

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
            response_type: Map.get(payload, :response_type) || Map.get(payload, "response_type"),
            strategy: Map.get(payload, :strategy) || Map.get(payload, "strategy"),
            method: Map.get(payload, :method) || Map.get(payload, "method"),
            intent: Map.get(payload, :intent) || Map.get(payload, "intent"),
            entities_count:
              Map.get(payload, :entities_count) || Map.get(payload, "entities_count") || 0,
            nlp_confidence:
              Map.get(payload, :nlp_confidence) || Map.get(payload, "nlp_confidence"),
            base_method: Map.get(payload, :base_method) || Map.get(payload, "base_method"),
            prompts_count:
              Map.get(payload, :prompts_count) || Map.get(payload, "prompts_count") || 0,
            response_path: Map.get(payload, :response_path) || Map.get(payload, "response_path"),
            reason: Map.get(payload, :reason) || Map.get(payload, "reason"),
            source: Map.get(payload, :source) || Map.get(payload, "source")
          })

        :followup_detected ->
          Map.put(details, :followup, %{
            previous_intent:
              Map.get(payload, :previous_intent) || Map.get(payload, "previous_intent"),
            previous_entities:
              Map.get(payload, :previous_entities) || Map.get(payload, "previous_entities") || 0
          })

        :meta_cognitive_query ->
          Map.put(
            details,
            :meta_cognitive,
            Map.get(payload, :query_type) || Map.get(payload, "query_type")
          )

        :fast_path_used ->
          Map.put(details, :fast_path, %{
            used: true,
            intent: Map.get(payload, :intent) || Map.get(payload, "intent"),
            source: Map.get(payload, :source) || Map.get(payload, "source"),
            activation: Map.get(payload, :activation) || Map.get(payload, "activation"),
            domain: Map.get(payload, :domain) || Map.get(payload, "domain")
          })

        :fast_path_bypassed ->
          Map.put(details, :fast_path, %{
            used: false,
            bypassed: true,
            intent: Map.get(payload, :intent) || Map.get(payload, "intent"),
            domain: Map.get(payload, :domain) || Map.get(payload, "domain"),
            reason: Map.get(payload, :reason) || Map.get(payload, "reason")
          })

        :fast_path_miss ->
          Map.put(details, :fast_path, %{used: false, bypassed: false})

        :response_gate_start ->
          details

        :response_gate_complete ->
          Map.put(details, :response_gate, %{
            decision: Map.get(payload, :decision) || Map.get(payload, "decision"),
            confidence: Map.get(payload, :confidence) || Map.get(payload, "confidence"),
            reason: Map.get(payload, :reason) || Map.get(payload, "reason")
          })

        :nlp_pipeline_start ->
          details

        :nlp_pipeline_complete ->
          Map.put(details, :nlp_pipeline, %{
            method: Map.get(payload, :method) || Map.get(payload, "method"),
            reason: Map.get(payload, :reason) || Map.get(payload, "reason"),
            nlp_confidence:
              Map.get(payload, :nlp_confidence) || Map.get(payload, "nlp_confidence"),
            nlp_intent: Map.get(payload, :nlp_intent) || Map.get(payload, "nlp_intent"),
            final_intent: Map.get(payload, :final_intent) || Map.get(payload, "final_intent"),
            entities_count:
              Map.get(payload, :entities_count) || Map.get(payload, "entities_count") || 0
          })

        :learning_complete ->
          details

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

          :sentiment_complete ->
            Map.put(chunk, :sentiment, %{
              label: Map.get(payload, :label) || Map.get(payload, "label"),
              confidence: Map.get(payload, :confidence) || Map.get(payload, "confidence")
            })

          :events_extracted ->
            Map.put(chunk, :events, %{
              event_count: Map.get(payload, :event_count) || Map.get(payload, "event_count") || 0,
              events: Map.get(payload, :events) || Map.get(payload, "events") || []
            })

          :fact_verification ->
            chunk
            |> Map.put(
              :epistemic_status,
              Map.get(payload, :epistemic_status) || Map.get(payload, "epistemic_status")
            )
            |> Map.put(
              :fact_verification,
              Map.get(payload, :fact_verification) || Map.get(payload, "fact_verification")
            )
            |> Map.put(
              :related_beliefs_count,
              Map.get(payload, :related_beliefs_count) ||
                Map.get(payload, "related_beliefs_count") || 0
            )
            |> Map.put(
              :related_beliefs,
              Map.get(payload, :related_beliefs) ||
                Map.get(payload, "related_beliefs") || []
            )

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
    base = now - max(length(memory) - 1, 0) * 1000

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
          base + idx * 1000

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
    |> Enum.filter(fn msg ->
      msg.content != nil and msg.content != ""
    end)
  end

  attr(:trace, :map, required: true)

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

  attr(:chunk, :map, required: true)

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

  attr(:trace, :map, required: true)

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
                <% confidence = Map.get(entity, :confidence)
                confidence_percent = if confidence, do: Float.round(confidence * 100, 1), else: nil

                confidence_variant =
                  cond do
                    confidence && confidence >= 0.8 -> :success
                    confidence && confidence >= 0.6 -> :warning
                    confidence -> :error
                    true -> :default
                  end %>
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

  attr(:level, :atom, required: true)
  attr(:confidence, :string, required: true)

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

  defp activation_color(value) when value >= 0.7 do
    "bg-success"
  end

  defp activation_color(value) when value >= 0.4 do
    "bg-info"
  end

  defp activation_color(value) when value >= 0.2 do
    "bg-warning"
  end

  defp activation_color(_) do
    "bg-error"
  end

  defp format_percent(nil) do
    "0%"
  end

  defp format_percent(value) when is_float(value) do
    "#{round(value * 100)}%"
  end

  defp format_percent(value) when is_integer(value) do
    "#{value}%"
  end

  defp format_percent(_) do
    "0%"
  end

  def confidence_badge_class(:high) do
    "badge-success"
  end

  def confidence_badge_class(:medium) do
    "badge-info"
  end

  def confidence_badge_class(:low) do
    "badge-warning"
  end

  def confidence_badge_class(_) do
    "badge-error"
  end

  def strategy_badge_class(:can_respond) do
    "badge-success"
  end

  def strategy_badge_class(:partial_response_with_clarification) do
    "badge-info"
  end

  def strategy_badge_class(:needs_clarification) do
    "badge-warning"
  end

  def strategy_badge_class(_) do
    "badge-error"
  end

  def format_strategy(:can_respond) do
    "Ready"
  end

  def format_strategy(:partial_response_with_clarification) do
    "Partial"
  end

  def format_strategy(:needs_clarification) do
    "Need Info"
  end

  def format_strategy(:low_confidence) do
    "Low Conf"
  end

  def format_strategy(:response_optional) do
    "Optional"
  end

  def format_strategy(:response_deferred) do
    "Deferred"
  end

  def format_strategy(nil) do
    "Unknown"
  end

  def format_strategy(other) do
    to_string(other)
  end

  defp get_cognitive_memory_stats do
    if Process.whereis(Brain.Memory.Store) != nil do
      Store.stats()
    else
      %{episode_count: 0, semantic_count: 0, episode_index_size: 0, semantic_index_size: 0}
    end
  rescue
    _ -> %{episode_count: 0, semantic_count: 0, episode_index_size: 0, semantic_index_size: 0}
  end

  defp get_combined_knowledge(persona_name) do
    base_knowledge = KnowledgeStore.get_knowledge(persona_name)
    user_facts = get_all_user_facts()
    Map.put(base_knowledge, "user_facts", user_facts)
  end

  defp get_all_user_facts do
    if Process.whereis(Brain.Epistemic.UserModelStore) do
      case UserModelStore.list_all_users() do
        {:ok, user_ids} ->
          user_ids
          |> Enum.map(fn user_id ->
            case UserModelStore.get(user_id) do
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

  defp format_user_facts(_) do
    []
  end

  defp format_epistemic_bounds(bounds) when is_map(bounds) do
    bounds
    |> Enum.map(fn {k, v} -> %{"key" => to_string(k), "confidence" => v} end)
  end

  defp format_epistemic_bounds(_) do
    []
  end

  @doc "Returns a list of data stores that were accessed for a given response type.\n"
  def get_stores_accessed(response_type) do
    case response_type do
      :domain ->
        [
          %{name: "FactDatabase", purpose: "Query facts for domain response"},
          %{name: "KnowledgeStore", purpose: "Query world knowledge"}
        ]

      :memory_augmented ->
        [
          %{name: "Memory.Store", purpose: "Query similar episodes"},
          %{name: "Embedder", purpose: "Generate TF-IDF embeddings"}
        ]

      :template ->
        [%{name: "TemplateStore", purpose: "Load response templates"}]

      :conditional_template ->
        [
          %{name: "TemplateStore", purpose: "Load templates with conditions"},
          %{name: "Embedder", purpose: "Semantic ranking of templates"}
        ]

      :blended ->
        [
          %{name: "TemplateStore", purpose: "Segment templates into chunks"},
          %{name: "ChunkCompatibility", purpose: "Score chunk compatibility"},
          %{name: "Embedder", purpose: "Embed query for chunk selection"}
        ]

      :smalltalk ->
        [%{name: "TemplateStore", purpose: "Load smalltalk templates"}]

      :expressive ->
        [%{name: "TemplateStore", purpose: "Load expressive templates"}]

      :fast_path ->
        [
          %{name: "HeuristicStore", purpose: "Match learned heuristics"},
          %{name: "Memory.Store", purpose: "Query similar episodes"}
        ]

      :clarification ->
        []

      :fallback ->
        []

      :deferred ->
        []

      _ ->
        []
    end
  end

  # Epistemic status helpers for processing inspector
  def epistemic_status_class(:verified), do: "text-success font-medium"
  def epistemic_status_class(:contradicted), do: "text-error font-medium"
  def epistemic_status_class(:uncertain), do: "text-warning font-medium"
  def epistemic_status_class(:unchecked), do: "text-base-content/60"
  def epistemic_status_class(_), do: "text-base-content/60"

  def sentiment_label_class(:positive), do: "text-success font-medium"
  def sentiment_label_class(:negative), do: "text-error font-medium"
  def sentiment_label_class(:neutral), do: "text-base-content/70"
  def sentiment_label_class("positive"), do: "text-success font-medium"
  def sentiment_label_class("negative"), do: "text-error font-medium"
  def sentiment_label_class("neutral"), do: "text-base-content/70"
  def sentiment_label_class(_), do: "text-base-content/60"

  def format_verification(nil), do: "-"
  def format_verification({:verified, conf}) when is_number(conf), do: "Verified (#{Float.round(conf * 100, 1)}%)"
  def format_verification({:contradicted, beliefs}) when is_list(beliefs), do: "Contradicted (#{length(beliefs)} conflicts)"
  def format_verification({:uncertain, reason}), do: "Uncertain: #{reason}"
  def format_verification(_), do: "-"

  @doc false
  def collect_all_beliefs(dev) when is_map(dev) do
    chunks = Map.get(dev, :chunks, %{})

    chunks
    |> Enum.sort_by(fn {i, _} -> i end)
    |> Enum.flat_map(fn {_idx, chunk} ->
      status = chunk[:epistemic_status]
      beliefs = chunk[:related_beliefs] || []
      Enum.map(beliefs, fn b -> Map.put(b, :_chunk_status, status) end)
    end)
    |> Enum.uniq_by(fn b -> b[:id] || b end)
  end

  def collect_all_beliefs(_), do: []

  @doc false
  def collect_epistemic_statuses(dev) when is_map(dev) do
    chunks = Map.get(dev, :chunks, %{})

    chunks
    |> Enum.map(fn {_idx, chunk} -> chunk[:epistemic_status] end)
    |> Enum.reject(&is_nil/1)
  end

  def collect_epistemic_statuses(_), do: []

  @doc false
  def worst_epistemic_status(statuses) when is_list(statuses) do
    priority = %{contradicted: 3, uncertain: 2, verified: 1, unchecked: 0}

    statuses
    |> Enum.max_by(fn s -> Map.get(priority, s, 0) end, fn -> :unchecked end)
  end

  def worst_epistemic_status(_), do: :unchecked

  # ── Authority helpers (shared with template) ──────────────────

  defp group_authority_profiles(profiles) do
    profiles
    |> Enum.group_by(fn p -> p.profile.category end)
    |> Enum.sort_by(fn {cat, _} ->
      case cat do
        "professional" -> 0
        "academic" -> 1
        "personal" -> 2
        "unknown" -> 3
        "entertainment" -> 4
        _ -> 5
      end
    end)
  end

  defp authority_badge_class("professional"), do: "badge-info"
  defp authority_badge_class("personal"), do: "badge-secondary"
  defp authority_badge_class("academic"), do: "badge-accent"
  defp authority_badge_class("entertainment"), do: "badge-warning"
  defp authority_badge_class("unknown"), do: "badge-ghost"
  defp authority_badge_class(_), do: "badge-ghost"

  defp authority_category(authority_key) do
    if SourceAuthority.ready?() do
      case SourceAuthority.get_profile(authority_key) do
        nil -> "unknown"
        profile -> profile.category
      end
    else
      "unknown"
    end
  rescue
    _ -> "unknown"
  end
end