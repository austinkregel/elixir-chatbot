defmodule ChatBot.Brain do
  @moduledoc """
  The Brain GenServer manages the AI personality, subprocesses, and global memory.
  This is the core component that orchestrates all chat bot functionality.
  """

  use GenServer
  require Logger

  # Client API

  def start_link(artifact_path) do
    GenServer.start_link(__MODULE__, artifact_path, name: __MODULE__)
  end

  def evaluate(conversation_id, input) do
    evaluate(conversation_id, input, [])
  end

  def evaluate(conversation_id, input, opts) when is_list(opts) do
    timeout = Keyword.get(opts, :timeout, 90_000)
    opts = Keyword.delete(opts, :timeout)
    GenServer.call(__MODULE__, {:evaluate, conversation_id, input, opts}, timeout)
  end

  def create_conversation do
    GenServer.call(__MODULE__, :create_conversation)
  end

  def end_conversation(conversation_id) do
    GenServer.call(__MODULE__, {:end_conversation, conversation_id})
  end

  def get_status do
    GenServer.call(__MODULE__, :get_status)
  end

  def get_conversations do
    GenServer.call(__MODULE__, :get_conversations)
  end

  def get_conversation(conversation_id) do
    GenServer.call(__MODULE__, {:get_conversation, conversation_id})
  end

  def handle_urgent_interrupt(reason, data \\ %{}) do
    GenServer.cast(__MODULE__, {:urgent_interrupt, reason, data})
  end

  def handle_urgent_emergency(reason, data \\ %{}) do
    GenServer.cast(__MODULE__, {:urgent_emergency, reason, data})
  end

  def start_http_subprocess(opts \\ []) do
    GenServer.call(__MODULE__, {:start_http_subprocess, opts})
  end

  def start_conversation_subprocess(conversation_id, opts \\ []) do
    GenServer.call(__MODULE__, {:start_conversation_subprocess, conversation_id, opts})
  end

  def start_cli_subprocess(opts \\ []) do
    GenServer.call(__MODULE__, {:start_cli_subprocess, opts})
  end

  def stop_subprocess(subprocess_id) do
    GenServer.call(__MODULE__, {:stop_subprocess, subprocess_id})
  end

  def list_subprocesses do
    GenServer.call(__MODULE__, :list_subprocesses)
  end

  def reset_state do
    GenServer.call(__MODULE__, :reset_state)
  end

  # Server Callbacks

  @impl true
  def init(artifact_path) do
    # Load the personality artifact
    artifact = load_artifact(artifact_path)
    persona = create_personality(artifact)

    # Load existing knowledge and memory
    knowledge = ChatBot.KnowledgeStore.load_knowledge(persona.name)
    memory = ChatBot.MemoryStore.load_all(persona.name)

    # Update persona with loaded knowledge
    updated_persona = Map.put(persona, :knowledge, knowledge)

    # Initialize state
    state = %{
      artifact_path: artifact_path,
      artifact: artifact,
      persona: updated_persona,
      active_conversations: %{},
      global_memory: memory,
      learning_queue: [],
      is_shutting_down: false,
      pending_requests: %{},
      subprocesses: %{
        http: %{},
        conversation: %{},
        cli: %{}
      }
    }

    Logger.info("Brain initialized", %{
      name: persona.name,
      traits: persona.traits,
      memory_size: length(state.global_memory),
      knowledge_size: map_size(knowledge)
    })

    {:ok, state}
  end

  @impl true
  def handle_call({:evaluate, conversation_id, input}, from, state) do
    handle_call({:evaluate, conversation_id, input, []}, from, state)
  end

  def handle_call({:evaluate, conversation_id, input, opts}, _from, state) do
    case Map.get(state.active_conversations, conversation_id) do
      nil ->
        {:reply, {:error, "Conversation not found"}, state}

      conversation ->
        now = System.system_time(:millisecond)

        # Process with classical NLP
        {response, processing_method, context} =
          if Application.get_env(:chat_bot, :ml)[:enabled] do
            try_classical_nlp_first(state.persona, input, conversation.memory, opts)
          else
            # ML disabled - use simple fallback
            {simple_fallback_response(state.persona, input), :simple, %{}}
          end

        # Build context snapshot for conversation memory
        context_snapshot = build_context_snapshot(context)

        # Update conversation memory with context
        user_message_id =
          get_in(opts, [:progress, :message_id]) ||
            get_in(opts, [:progress, "message_id"]) ||
            generate_message_id()

        user_message = %{
          id: user_message_id,
          role: "user",
          content: input,
          timestamp: now,
          context: context_snapshot
        }

        assistant_message = %{
          id: generate_message_id(),
          role: "assistant",
          content: response,
          timestamp: System.system_time(:millisecond),
          processing_method: processing_method
        }

        updated_conversation =
          conversation
          |> Map.put(
            :memory,
            conversation.memory ++
              [
                user_message,
                assistant_message
              ]
          )
          # Track active context for follow-up detection (use Map.put since key may not exist)
          |> Map.put(:active_context, context_snapshot)
          |> Map.put(:last_activity, System.system_time(:millisecond))

        # Add to learning queue
        learning_entry = %{
          conversation_id: conversation_id,
          timestamp: System.system_time(:millisecond),
          input: input,
          response: response
        }

        updated_state = %{
          state
          | active_conversations:
              Map.put(state.active_conversations, conversation_id, updated_conversation),
            learning_queue: state.learning_queue ++ [learning_entry]
        }

        # Process learning queue asynchronously
        send(self(), :process_learning_queue)

        {:reply, {:ok, response}, updated_state}
    end
  end

  @impl true
  def handle_call(:create_conversation, _from, state) do
    conversation_id = generate_conversation_id()

    conversation = %{
      id: conversation_id,
      memory: [],
      active_context: nil,
      created_at: System.system_time(:millisecond),
      last_activity: System.system_time(:millisecond)
    }

    updated_state = %{
      state
      | active_conversations: Map.put(state.active_conversations, conversation_id, conversation)
    }

    Logger.info("Conversation created", %{conversation_id: conversation_id})

    {:reply, {:ok, conversation_id}, updated_state}
  end

  @impl true
  def handle_call({:end_conversation, conversation_id}, _from, state) do
    case Map.pop(state.active_conversations, conversation_id) do
      {nil, _} ->
        {:reply, {:error, "Conversation not found"}, state}

      {_conversation, updated_conversations} ->
        Logger.info("Conversation ended", %{conversation_id: conversation_id})
        {:reply, :ok, %{state | active_conversations: updated_conversations}}
    end
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    status = %{
      name: state.persona.name,
      traits: state.persona.traits,
      active_conversations: map_size(state.active_conversations),
      global_memory_size: length(state.global_memory),
      learning_queue_size: length(state.learning_queue),
      is_shutting_down: state.is_shutting_down
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call(:get_conversations, _from, state) do
    conversations =
      state.active_conversations
      |> Map.values()
      |> Enum.map(fn conv ->
        %{
          id: conv.id,
          message_count: length(conv.memory),
          created_at: conv.created_at,
          last_activity: conv.last_activity
        }
      end)

    {:reply, conversations, state}
  end

  def handle_call({:get_conversation, conversation_id}, _from, state) do
    case Map.get(state.active_conversations, conversation_id) do
      nil -> {:reply, {:error, "Conversation not found"}, state}
      conversation -> {:reply, {:ok, conversation}, state}
    end
  end

  @impl true
  def handle_call({:start_http_subprocess, opts}, _from, state) do
    memory_snapshot = %{
      knowledge: state.persona.knowledge || %{},
      global_memory: state.global_memory
    }

    case ChatBot.Subprocesses.Supervisor.start_http_subprocess(
           Keyword.put(opts, :memory_snapshot, memory_snapshot)
         ) do
      {:ok, pid, subprocess_id} ->
        updated_subprocesses = %{
          state.subprocesses
          | http:
              Map.put(state.subprocesses.http, subprocess_id, %{
                pid: pid,
                started_at: System.system_time(:millisecond)
              })
        }

        Logger.info("HTTP subprocess started", %{
          subprocess_id: subprocess_id,
          pid: pid
        })

        {:reply, {:ok, subprocess_id}, %{state | subprocesses: updated_subprocesses}}

      {:error, reason} ->
        Logger.error("Failed to start HTTP subprocess", %{reason: reason})
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:start_conversation_subprocess, conversation_id, opts}, _from, state) do
    memory_snapshot = %{
      knowledge: state.persona.knowledge || %{},
      global_memory: state.global_memory
    }

    case ChatBot.Subprocesses.Supervisor.start_conversation_subprocess(
           Keyword.put(opts, [:conversation_id, :memory_snapshot], [
             conversation_id,
             memory_snapshot
           ])
         ) do
      {:ok, pid, subprocess_id} ->
        updated_subprocesses = %{
          state.subprocesses
          | conversation:
              Map.put(state.subprocesses.conversation, subprocess_id, %{
                pid: pid,
                conversation_id: conversation_id,
                started_at: System.system_time(:millisecond)
              })
        }

        Logger.info("Conversation subprocess started", %{
          subprocess_id: subprocess_id,
          conversation_id: conversation_id,
          pid: pid
        })

        {:reply, {:ok, subprocess_id}, %{state | subprocesses: updated_subprocesses}}

      {:error, reason} ->
        Logger.error("Failed to start conversation subprocess", %{reason: reason})
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:start_cli_subprocess, opts}, _from, state) do
    memory_snapshot = %{
      knowledge: state.persona.knowledge || %{},
      global_memory: state.global_memory
    }

    case ChatBot.Subprocesses.Supervisor.start_cli_subprocess(
           Keyword.put(opts, :memory_snapshot, memory_snapshot)
         ) do
      {:ok, pid, subprocess_id} ->
        updated_subprocesses = %{
          state.subprocesses
          | cli:
              Map.put(state.subprocesses.cli, subprocess_id, %{
                pid: pid,
                started_at: System.system_time(:millisecond)
              })
        }

        Logger.info("CLI subprocess started", %{
          subprocess_id: subprocess_id,
          pid: pid
        })

        {:reply, {:ok, subprocess_id}, %{state | subprocesses: updated_subprocesses}}

      {:error, reason} ->
        Logger.error("Failed to start CLI subprocess", %{reason: reason})
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:stop_subprocess, subprocess_id}, _from, state) do
    # Find and stop the subprocess
    subprocess_info = find_subprocess_by_id(state.subprocesses, subprocess_id)

    case subprocess_info do
      {type, info} ->
        case ChatBot.Subprocesses.Supervisor.stop_subprocess(info.pid) do
          :ok ->
            updated_subprocesses = remove_subprocess(state.subprocesses, type, subprocess_id)
            Logger.info("Subprocess stopped", %{type: type, subprocess_id: subprocess_id})
            {:reply, :ok, %{state | subprocesses: updated_subprocesses}}

          {:error, reason} ->
            Logger.error("Failed to stop subprocess", %{
              subprocess_id: subprocess_id,
              reason: reason
            })

            {:reply, {:error, reason}, state}
        end

      nil ->
        {:reply, {:error, "Subprocess not found"}, state}
    end
  end

  @impl true
  def handle_call(:list_subprocesses, _from, state) do
    subprocesses = %{
      http: Map.keys(state.subprocesses.http),
      conversation: Map.keys(state.subprocesses.conversation),
      cli: Map.keys(state.subprocesses.cli)
    }

    {:reply, subprocesses, state}
  end

  @impl true
  def handle_call(:reset_state, _from, state) do
    # Reset the Brain to a clean state (useful for testing)
    reset_state = %{
      state
      | is_shutting_down: false,
        active_conversations: %{},
        learning_queue: []
    }

    Logger.info("Brain state reset")
    {:reply, :ok, reset_state}
  end

  @impl true
  def handle_cast({:urgent_interrupt, reason, data}, state) do
    Logger.warning("Handling urgent interrupt", %{reason: reason, data: data})

    # Stop all active conversations
    updated_conversations =
      state.active_conversations
      |> Enum.map(fn {id, conversation} ->
        Logger.info("Interrupting conversation", %{conversation_id: id})

        {id,
         %{
           conversation
           | memory:
               conversation.memory ++ [%{role: "system", content: "Conversation interrupted"}]
         }}
      end)
      |> Map.new()

    updated_state = %{state | active_conversations: updated_conversations}

    # Broadcast interrupt acknowledgment (if PubSub is running)
    safe_pubsub_broadcast("brain:status", "interrupt_acknowledged", %{
      reason: reason,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, updated_state}
  end

  @impl true
  def handle_cast({:urgent_emergency, reason, data}, state) do
    Logger.error("Handling urgent emergency", %{reason: reason, data: data})

    # Emergency shutdown of all conversations
    updated_state = %{state | active_conversations: %{}, is_shutting_down: true}

    # Broadcast emergency acknowledgment (if PubSub is running)
    safe_pubsub_broadcast("brain:status", "emergency_acknowledged", %{
      reason: reason,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, updated_state}
  end

  @impl true
  def handle_info(:process_learning_queue, state) do
    if length(state.learning_queue) > 0 do
      # Process learning entries
      {processed_entries, remaining_queue} = Enum.split(state.learning_queue, 5)

      # Add to global memory
      new_memory_entries =
        processed_entries
        |> Enum.map(fn entry ->
          %{
            conversation_id: entry.conversation_id,
            timestamp: entry.timestamp,
            summary: create_learning_summary(entry.input, entry.response)
          }
        end)

      # Also store in cognitive memory system for embedding-based retrieval
      store_in_cognitive_memory(processed_entries)

      updated_state = %{
        state
        | global_memory: state.global_memory ++ new_memory_entries,
          learning_queue: remaining_queue
      }

      Logger.info("Learning queue processed", %{
        processed: length(processed_entries),
        remaining: length(remaining_queue),
        global_memory_size: length(updated_state.global_memory)
      })

      # Broadcast learning update
      safe_pubsub_broadcast("brain:learning", "learning_processed", %{
        processed_count: length(processed_entries),
        global_memory_size: length(updated_state.global_memory),
        timestamp: System.system_time(:millisecond)
      })

      # Schedule next processing if there are more entries
      if length(remaining_queue) > 0 do
        Process.send_after(self(), :process_learning_queue, 1000)
      end

      {:noreply, updated_state}
    else
      {:noreply, state}
    end
  end

  # Private Functions

  defp build_context_snapshot(context) do
    %{
      intent: Map.get(context, :intent),
      entities: Map.get(context, :entities, []),
      slots: Map.get(context, :slots, %{}),
      missing_slots: Map.get(context, :missing_slots, []),
      timestamp: System.system_time(:millisecond)
    }
  end

  defp find_subprocess_by_id(subprocesses, subprocess_id) do
    cond do
      Map.has_key?(subprocesses.http, subprocess_id) ->
        {:http, Map.get(subprocesses.http, subprocess_id)}

      Map.has_key?(subprocesses.conversation, subprocess_id) ->
        {:conversation, Map.get(subprocesses.conversation, subprocess_id)}

      Map.has_key?(subprocesses.cli, subprocess_id) ->
        {:cli, Map.get(subprocesses.cli, subprocess_id)}

      true ->
        nil
    end
  end

  defp remove_subprocess(subprocesses, type, subprocess_id) do
    case type do
      :http ->
        %{subprocesses | http: Map.delete(subprocesses.http, subprocess_id)}

      :conversation ->
        %{subprocesses | conversation: Map.delete(subprocesses.conversation, subprocess_id)}

      :cli ->
        %{subprocesses | cli: Map.delete(subprocesses.cli, subprocess_id)}
    end
  end

  defp load_artifact(path) do
    case File.read(path) do
      {:ok, content} ->
        Jason.decode!(content)

      {:error, _} ->
        # Fallback to default artifact
        %{
          "name" => "Echo",
          "traits" => ["cheerful"],
          "system_prompt" => "You are a helpful AI assistant named Echo."
        }
    end
  end

  defp create_personality(artifact) do
    %{
      name: artifact["name"] || "Echo",
      traits: artifact["traits"] || ["cheerful"],
      system_prompt: artifact["system_prompt"] || "You are a helpful AI assistant."
    }
  end

  defp try_classical_nlp_first(persona, input, memory, opts) do
    # First, check if this is a follow-up to a previous message
    previous_context = get_previous_context(memory)

    if ChatBot.Analysis.FollowupDetector.is_followup?(input, previous_context) do
      # This is providing context for the previous intent
      Logger.info("Detected follow-up message", %{
        input: input,
        previous_intent: previous_context[:intent]
      })

      handle_followup_message(persona, input, previous_context)
    else
      # Normal processing - run the analysis pipeline
      process_new_message(persona, input, memory, opts)
    end
  end

  defp process_new_message(persona, input, memory, opts) do
    # Run the analysis pipeline to build an internal model
    analysis_model = run_analysis_pipeline(input, memory, opts)

    Logger.debug("Analysis pipeline complete", %{
      strategy: analysis_model.overall_strategy,
      chunks: length(analysis_model.chunks),
      prompts: analysis_model.suggested_prompts
    })

    # Check if we need clarification before processing
    case analysis_model.overall_strategy do
      :needs_clarification ->
        # Return a clarification request instead of trying to respond
        prompts = analysis_model.suggested_prompts
        response = build_clarification_response(prompts, persona)
        context = extract_context_from_analysis(analysis_model)
        {response, :clarification_needed, context}

      :partial_response_with_clarification ->
        # Respond to what we can, then ask for clarification on what's missing
        {base_response, _status, context} =
          try_nlp_with_analysis(persona, input, memory, analysis_model)

        prompts = analysis_model.suggested_prompts
        clarification = build_clarification_addendum(prompts)

        combined_response =
          if clarification != "" do
            "#{base_response} #{clarification}"
          else
            base_response
          end

        {combined_response, :partial_with_clarification, context}

      :defer_to_user ->
        # Bot wasn't addressed - acknowledge but don't try to respond substantively
        {simple_acknowledgment(persona), :not_addressed, %{}}

      :cannot_respond ->
        # Cannot respond with classical NLP - use simple fallback
        {simple_fallback_response(persona, input), :cannot_respond, %{}}

      _ ->
        # Can respond (fully or partially) - proceed with NLP pipeline
        try_nlp_with_analysis(persona, input, memory, analysis_model)
    end
  end

  defp get_previous_context(memory) do
    memory
    |> Enum.reverse()
    |> Enum.find(fn m -> Map.has_key?(m, :context) and m[:context] != nil end)
    |> case do
      nil -> nil
      msg -> msg[:context]
    end
  end

  defp handle_followup_message(persona, input, previous_context) do
    # Extract entities from the follow-up
    entities = ChatBot.ML.EntityExtractor.extract_entities(input)

    Logger.info("Extracted entities from follow-up", %{
      entities_count: length(entities),
      entities: Enum.map(entities, & &1.entity)
    })

    # Get carried context with previous info
    carried = ChatBot.Analysis.FollowupDetector.get_carried_context(input, previous_context)

    # Merge with previous context
    merged_context = ChatBot.Analysis.FollowupDetector.merge_with_previous(carried, entities)

    Logger.info("Merged context", %{
      intent: merged_context.intent,
      all_filled: merged_context.all_required_filled,
      missing: merged_context.missing_slots
    })

    # Generate response with complete context
    if merged_context.all_required_filled do
      response = generate_intent_response(merged_context, persona)

      context = %{
        intent: merged_context.intent,
        entities: merged_context.entities,
        slots: merged_context.slots,
        missing_slots: []
      }

      {response, :followup_completed, context}
    else
      # Still missing slots - ask for clarification
      prompt = generate_followup_clarification(merged_context)

      context = %{
        intent: merged_context.intent,
        entities: merged_context.entities,
        slots: merged_context.slots,
        missing_slots: merged_context.missing_slots
      }

      {prompt, :followup_needs_more, context}
    end
  end

  defp extract_context_from_analysis(analysis_model) do
    best_analysis =
      analysis_model.analyses
      |> Enum.max_by(& &1.confidence, fn -> nil end)

    if best_analysis do
      %{
        intent: best_analysis.intent,
        entities: best_analysis.entities || [],
        slots: Map.get(best_analysis, :slots, %{}) |> extract_filled_slots(),
        missing_slots: Map.get(best_analysis, :missing_context, [])
      }
    else
      %{}
    end
  end

  defp extract_filled_slots(slots) when is_map(slots) do
    case Map.get(slots, :filled_slots) do
      nil -> slots
      filled -> filled
    end
  end

  defp extract_filled_slots(_), do: %{}

  # Extract the actual value from a slot, which may be a map with :value key
  defp get_slot_value(slots, key, default \\ nil) do
    case Map.get(slots, key) do
      nil -> default
      %{value: value} -> value
      value when is_binary(value) -> value
      value -> "#{inspect(value)}"
    end
  end

  defp generate_intent_response(context, persona) do
    # Generate response based on completed intent
    intent = context.intent
    slots = context.slots

    case intent do
      "weather.query" ->
        location = get_slot_value(slots, "location", "your location")
        "Let me check the weather for #{location}. One moment please..."

      "device.control" ->
        device = get_slot_value(slots, "device", "device")
        action = get_slot_value(slots, "action", "control")
        "I'll #{action} the #{device} for you."

      "music.play" ->
        artist = get_slot_value(slots, "music-artist") || get_slot_value(slots, "song") || "music"
        "Playing #{artist} for you now."

      _ ->
        # Fall back to general response - extract values from slot maps
        entities =
          Enum.map(slots, fn {k, v} ->
            value = case v do
              %{value: val} -> val
              val -> val
            end
            %{entity: k, value: value}
          end)

        generate_classical_response(intent, entities, persona)
    end
  end

  defp generate_followup_clarification(context) do
    missing = context.missing_slots

    case missing do
      ["location" | _] ->
        "What location would you like the weather for?"

      ["device" | _] ->
        "Which device would you like me to control?"

      ["action" | _] ->
        "What would you like me to do?"

      [slot | _] ->
        readable = slot |> String.replace("-", " ") |> String.replace("_", " ")
        "Could you please specify the #{readable}?"

      [] ->
        "I need a bit more information. Could you elaborate?"
    end
  end

  defp run_analysis_pipeline(input, memory, opts) do
    # Build conversation history from memory for context resolution
    # Now supports both old format (with :entities) and new format (with :context)
    history =
      memory
      |> Enum.filter(fn m ->
        Map.has_key?(m, :entities) or Map.has_key?(m, :context)
      end)
      |> Enum.take(5)
      |> Enum.map(fn m ->
        case Map.get(m, :context) do
          nil ->
            # Old format
            %{
              entities: Map.get(m, :entities, %{}),
              intent: Map.get(m, :intent),
              timestamp: Map.get(m, :timestamp, 0)
            }

          context ->
            # New format with context snapshot
            %{
              entities: build_entities_map(Map.get(context, :entities, [])),
              intent: Map.get(context, :intent),
              timestamp: Map.get(context, :timestamp, 0),
              missing_slots: Map.get(context, :missing_slots, [])
            }
        end
      end)

    pipeline_opts =
      Keyword.merge(opts,
        participants: [:user, :bot],
        conversation_history: history,
        user_profile: %{}
      )

    ChatBot.Analysis.Pipeline.process(input, pipeline_opts)
  end

  defp safe_pubsub_broadcast(topic, event, payload) when is_binary(topic) and is_binary(event) do
    if Process.whereis(ChatBot.PubSub) != nil do
      Phoenix.PubSub.broadcast(ChatBot.PubSub, topic, %Phoenix.Socket.Broadcast{
        topic: topic,
        event: event,
        payload: payload
      })
    end

    :ok
  rescue
    _ -> :ok
  end

  defp build_entities_map(entities) when is_list(entities) do
    # Convert list of entities to a map keyed by entity type
    Enum.reduce(entities, %{}, fn entity, acc ->
      entity_type = entity[:entity] || entity["entity"]
      entity_value = entity[:value] || entity["value"]
      if entity_type, do: Map.put(acc, entity_type, entity_value), else: acc
    end)
  end

  defp build_entities_map(entities) when is_map(entities), do: entities
  defp build_entities_map(_), do: %{}

  defp try_nlp_with_analysis(persona, input, _memory, analysis_model) do
    # Collect ALL intents from all chunks, prioritizing substantive ones
    all_intents =
      analysis_model.analyses
      |> Enum.map(fn analysis ->
        {analysis.intent, analysis.speech_act, analysis.confidence}
      end)
      |> Enum.filter(fn {intent, _, _} -> intent != nil and intent != "" end)

    # Prioritize substantive intents (questions, commands) over expressives (greetings)
    substantive_intents =
      all_intents
      |> Enum.filter(fn {intent, speech_act, _} ->
        # Not a greeting/farewell/thanks
        not String.contains?(intent || "", "greeting") and
          not String.contains?(intent || "", "bye") and
          speech_act.category in [:directive, :assertive]
      end)

    # Pick the best substantive intent, or fall back to any intent
    {analysis_intent, _, _} =
      case substantive_intents do
        [first | _] -> first
        [] -> List.first(all_intents) || {nil, nil, 0}
      end

    # Use the best analysis for other attributes
    best_analysis =
      analysis_model.analyses
      |> Enum.filter(&(&1.response_strategy == :can_respond))
      |> Enum.max_by(& &1.confidence, fn -> List.first(analysis_model.analyses) end)

    # Collect entities from ALL chunks
    analysis_entities =
      analysis_model.analyses
      |> Enum.flat_map(fn analysis ->
        (analysis.entities || [])
        |> Enum.map(fn e ->
          %{
            entity: e["type"] || e[:entity] || e["entity"],
            value: e["name"] || e[:value] || e["value"],
            confidence: e["confidence"] || e[:confidence] || 0.8
          }
        end)
      end)
      |> Enum.uniq_by(fn e -> {e.entity, e.value} end)

    # Extract slot information for context
    slots_info = if best_analysis, do: Map.get(best_analysis, :slots), else: nil
    missing_slots = if best_analysis, do: Map.get(best_analysis, :missing_context, []), else: []

    # Try classical NLP pipeline for additional processing
    case ChatBot.ML.NLPPipeline.process(input) do
      {:ok, %{confidence: conf, intent: nlp_intent, entities: nlp_entities}} ->
        # Merge analysis intent with NLP intent (prefer analysis if both present)
        intent = analysis_intent || nlp_intent
        entities = merge_entities(analysis_entities, nlp_entities)

        Logger.info("Using analysis-enhanced NLP", %{
          analysis_intent: analysis_intent,
          nlp_intent: nlp_intent,
          final_intent: intent,
          confidence: conf,
          entities_count: length(entities)
        })

        # Build context for storage
        context = %{
          intent: intent,
          entities: entities,
          slots: extract_filled_slots(slots_info),
          missing_slots: missing_slots
        }

        # Learn from extraction
        ChatBot.Learner.learn_from_classical_extraction(persona.name, entities, input)

        # Generate response with analysis context
        response = generate_analysis_response(intent, entities, analysis_model, persona)

        method =
          if ChatBot.ML.NLPPipeline.should_use_classical_result?(conf) or analysis_intent != nil do
            :analysis_enhanced
          else
            :classical_low_confidence
          end

        {response, method, context}

      {:error, reason} ->
        Logger.warning("NLP pipeline failed", %{reason: reason})

        # Build context even on error
        context = %{
          intent: analysis_intent,
          entities: analysis_entities,
          slots: extract_filled_slots(slots_info),
          missing_slots: missing_slots
        }

        # Fall back to analysis-only response if we have a good analysis
        if analysis_intent && best_analysis.confidence > 0.5 do
          response =
            generate_analysis_response(
              analysis_intent,
              analysis_entities,
              analysis_model,
              persona
            )

          {response, :analysis_only, context}
        else
          {simple_fallback_response(persona, input), :classical_error, context}
        end
    end
  end

  defp merge_entities(analysis_entities, nlp_entities) do
    # Combine entities, preferring analysis entities for duplicates
    analysis_types = Enum.map(analysis_entities, & &1.entity) |> MapSet.new()

    unique_nlp =
      nlp_entities
      |> Enum.reject(fn e -> MapSet.member?(analysis_types, e.entity) end)

    analysis_entities ++ unique_nlp
  end

  defp build_clarification_response(prompts, _persona) do
    case prompts do
      [] ->
        "I'm not sure I understand. Could you please provide more details?"

      [single_prompt] ->
        single_prompt

      multiple ->
        # Combine multiple prompts
        first = List.first(multiple)
        rest_count = length(multiple) - 1

        "#{first} (I also have #{rest_count} more question#{if rest_count > 1, do: "s", else: ""})"
    end
  end

  defp build_clarification_addendum(prompts) do
    # Build a follow-up question to append to a partial response
    case prompts do
      [] ->
        ""

      [single_prompt] ->
        "By the way, #{String.downcase(String.first(single_prompt))}#{String.slice(single_prompt, 1..-1//1)}"

      [first | _rest] ->
        "Also, #{String.downcase(String.first(first))}#{String.slice(first, 1..-1//1)}"
    end
  end

  defp simple_acknowledgment(_persona) do
    "I noticed you said something, but I'm not sure if you were talking to me. Let me know if you need anything!"
  end

  defp generate_analysis_response(intent, entities, analysis_model, persona) do
    # Analyze all speech acts in the message
    speech_acts =
      analysis_model.analyses
      |> Enum.map(& &1.speech_act)

    # Find unique expressive types (avoid duplicate greetings)
    expressives =
      speech_acts
      |> Enum.filter(&(&1.category == :expressive))
      |> Enum.uniq_by(& &1.sub_type)

    directives = Enum.filter(speech_acts, &(&1.category == :directive))

    # Check if there's substantive content (questions, commands, or known intent)
    has_substantive_content =
      length(directives) > 0 or
        (intent != nil and intent != "" and
           not String.starts_with?(intent || "", "smalltalk.greetings"))

    # Build response parts
    response_parts = []

    # Add ONE expressive acknowledgment if present
    response_parts =
      if length(expressives) > 0 do
        expressive = List.first(expressives)
        expressive_response = generate_expressive_part(expressive)

        if expressive_response do
          [expressive_response | response_parts]
        else
          response_parts
        end
      else
        response_parts
      end

    # Add substantive response for directives/questions/commands
    response_parts =
      if has_substantive_content do
        substantive_response = generate_classical_response(intent, entities, persona)
        [substantive_response | response_parts]
      else
        response_parts
      end

    # Combine response parts (filter out nils)
    valid_parts =
      response_parts
      |> Enum.reverse()
      |> Enum.filter(&(&1 != nil and &1 != ""))

    case valid_parts do
      [] ->
        # No specific response parts - use fallback
        generate_classical_response(intent, entities, persona)

      [single] ->
        single

      parts ->
        # Join multiple parts with space
        Enum.join(parts, " ")
    end
  end

  defp generate_expressive_part(speech_act) do
    case speech_act.sub_type do
      :greeting ->
        Enum.random(["Hello!", "Hi there!", "Hey!"])

      :farewell ->
        Enum.random(["Goodbye!", "See you!", "Take care!"])

      :thanks ->
        Enum.random(["You're welcome!", "Happy to help!", "No problem!"])

      :apology ->
        Enum.random(["No worries!", "That's fine.", "Don't worry about it!"])

      _ ->
        nil
    end
  end

  defp generate_classical_response(intent, entities, persona) do
    # First, try domain-specific response generation
    case generate_domain_response(intent, entities) do
      {:ok, response} ->
        response

      :not_handled ->
        # Fall back to smalltalk responses
        generate_smalltalk_response(intent, entities, persona)
    end
  end

  # Domain-specific response handlers
  defp generate_domain_response("weather.query", entities) do
    location = find_entity_value(entities, "location")

    response =
      if location do
        "Let me check the weather for #{location}. The current conditions are partly cloudy with a temperature around 72°F."
      else
        "What location would you like the weather for?"
      end

    {:ok, response}
  end

  defp generate_domain_response("weather" <> _, entities) do
    generate_domain_response("weather.query", entities)
  end

  defp generate_domain_response("music.play", entities) do
    artist = find_entity_value(entities, "music-artist")
    song = find_entity_value(entities, "song")

    response =
      cond do
        artist -> "Playing music by #{artist} for you now."
        song -> "Playing #{song} for you now."
        true -> "What would you like me to play?"
      end

    {:ok, response}
  end

  defp generate_domain_response("device.control", entities) do
    device = find_entity_value(entities, "device")
    action = find_entity_value(entities, "action") || find_entity_value(entities, "locks-status")

    response =
      cond do
        device && action -> "I'll #{action} the #{device} for you."
        device -> "What would you like me to do with the #{device}?"
        true -> "Which device would you like me to control?"
      end

    {:ok, response}
  end

  defp generate_domain_response("news.query", entities) do
    topic = find_entity_value(entities, "topic")

    response =
      if topic do
        "Here are the latest headlines about #{topic}."
      else
        "Here are today's top headlines."
      end

    {:ok, response}
  end

  defp generate_domain_response("reminder.create", entities) do
    content = find_entity_value(entities, "content")
    date = find_entity_value(entities, "date")

    response =
      cond do
        content && date -> "I'll remind you about #{content} on #{date}."
        content -> "When would you like to be reminded about #{content}?"
        true -> "What would you like me to remind you about?"
      end

    {:ok, response}
  end

  defp generate_domain_response(_intent, _entities) do
    :not_handled
  end

  defp find_entity_value(entities, entity_type) when is_list(entities) do
    entity = Enum.find(entities, fn e ->
      e_type = e[:entity] || e["entity"]
      e_type == entity_type
    end)

    if entity do
      entity[:value] || entity["value"]
    else
      nil
    end
  end

  defp find_entity_value(_, _), do: nil

  defp generate_smalltalk_response(intent, entities, _persona) do
    # Load custom responses from Companion data
    responses_path =
      Path.join(
        Application.get_env(:chat_bot, :ml)[:training_data_path],
        "customSmalltalkResponses_en.json"
      )

    case File.read(responses_path) do
      {:ok, content} ->
        decoded = Jason.decode!(content)
        responses_map = smalltalk_to_map(decoded)

        # Try to find a matching response for the intent
        answers =
          Map.get(responses_map, intent) ||
            Map.get(responses_map, choose_smalltalk_action(intent, entities))

        cond do
          is_list(answers) and length(answers) > 0 ->
            template = Enum.random(answers)

            Enum.reduce(entities, template, fn entity, acc ->
              entity_type = entity[:entity] || entity["entity"] || ""
              entity_value = entity[:value] || entity["value"] || ""
              String.replace(acc, "@#{entity_type}", "#{entity_value}")
            end)

          is_binary(answers) ->
            Enum.reduce(entities, answers, fn entity, acc ->
              entity_type = entity[:entity] || entity["entity"] || ""
              entity_value = entity[:value] || entity["value"] || ""
              String.replace(acc, "@#{entity_type}", "#{entity_value}")
            end)

          true ->
            # Generic response when no smalltalk match
            if intent && intent != "" do
              "I understood that you're asking about #{humanize_intent(intent)}#{format_entities(entities)}."
            else
              "I'm not sure I understand. Could you rephrase that?"
            end
        end

      {:error, _} ->
        # Fallback response
        "I understood that you're asking about #{humanize_intent(intent)}#{format_entities(entities)}."
    end
  end

  defp humanize_intent(nil), do: "something"
  defp humanize_intent(""), do: "something"

  defp humanize_intent(intent) when is_binary(intent) do
    intent
    |> String.replace(".", " ")
    |> String.replace("_", " ")
    |> String.replace("smalltalk ", "")
  end

  defp smalltalk_to_map(%{} = map), do: map

  defp smalltalk_to_map(list) when is_list(list) do
    # Convert array of %{"action" => action, "customAnswers" => [...] } into map
    Enum.reduce(list, %{}, fn item, acc ->
      action = Map.get(item, "action")
      answers = Map.get(item, "customAnswers", [])

      if is_binary(action) and is_list(answers) do
        Map.put(acc, action, answers)
      else
        acc
      end
    end)
  end

  defp smalltalk_to_map(_), do: %{}

  defp choose_smalltalk_action(intent, _entities) do
    # Heuristic fallbacks for greetings/smalltalk
    cond do
      String.starts_with?(to_string(intent), "smalltalk.") -> intent
      true -> "smalltalk.greetings.hello"
    end
  end

  defp format_entities([]), do: ""

  defp format_entities(entities) do
    entity_str =
      entities
      |> Enum.map(fn e -> "#{e.entity}: #{e.value}" end)
      |> Enum.join(", ")

    " (with #{entity_str})"
  end

  defp simple_fallback_response(persona, input) do
    case persona.traits do
      ["cheerful"] ->
        "Hello! I'm #{persona.name}, and I'm happy to help! You said: #{input}"

      _ ->
        "I'm #{persona.name}. You said: #{input}"
    end
  end

  defp generate_conversation_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp generate_message_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp create_learning_summary(input, response) do
    # Simple learning summary for storing conversation context
    "User: #{String.slice(input, 0, 50)}... | Assistant: #{String.slice(response, 0, 50)}..."
  end

  defp store_in_cognitive_memory(entries) do
    # Store conversation entries in the cognitive memory system
    # This allows embedding-based retrieval for future classification
    if Process.whereis(ChatBot.Memory.Store) != nil do
      Enum.each(entries, fn entry ->
        # Determine tags from NLP analysis if possible
        tags = extract_tags_for_memory(entry.input)

        ChatBot.Memory.Think.think(:add_episode, %{
          state: entry.input,
          action: "conversation",
          outcome: entry.response,
          tags: ["conversation", entry.conversation_id | tags]
        })
      end)
    end
  rescue
    e ->
      Logger.warning("Failed to store in cognitive memory: #{inspect(e)}")
  end

  defp extract_tags_for_memory(input) do
    # Try to get intent classification for tagging
    case ChatBot.ML.IntentClassifierSimple.classify(input) do
      {:ok, %{intent: intent}} when is_binary(intent) ->
        [intent]

      _ ->
        []
    end
  rescue
    _ -> []
  end
end
