defmodule ChatBot.Brain do
  @moduledoc """
  The Brain GenServer manages the AI personality, subprocesses, and global memory.
  This is the core component that orchestrates all chat bot functionality.
  """

  use GenServer
  require Logger

  alias ChatBot.Analysis.{SelfKnowledgeAnalyzer, Progress}
  alias ChatBot.Epistemic.{UserModelStore, BeliefStore}
  alias ChatBot.Epistemic.Types.{Belief, Config}
  alias ChatBot.Response.{Synthesizer, TemplateStore}

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

    # Wrap with telemetry span for async, non-blocking metrics
    ChatBot.Telemetry.span(:brain_evaluate, %{conversation_id: conversation_id}, fn ->
      GenServer.call(__MODULE__, {:evaluate, conversation_id, input, opts}, timeout)
    end)
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

        # Extract and store beliefs from entities (epistemic integration)
        user_id = Keyword.get(opts, :user_id)
        entities = Map.get(context, :entities, [])
        extract_and_store_beliefs(input, entities, user_id, conversation_id)

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
    user_id = Keyword.get(opts, :user_id)

    # First, check for meta-cognitive queries (epistemic self-knowledge)
    if Config.enabled?() and SelfKnowledgeAnalyzer.is_self_knowledge_query?(input) do
      handle_meta_cognitive_query(persona, input, user_id, opts)
    else
      # Standard processing - run the analysis pipeline
      process_standard_message(persona, input, memory, opts)
    end
  end

  defp process_standard_message(persona, input, memory, opts) do
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

        Progress.report(opts, :response_generated, %{
          response_type: :clarification,
          strategy: :needs_clarification,
          prompts_count: length(prompts)
        })

        {response, :clarification_needed, context}

      :partial_response_with_clarification ->
        # Respond to what we can, then ask for clarification on what's missing
        {base_response, response_method, context} =
          try_nlp_with_analysis(persona, input, memory, analysis_model, opts)

        prompts = analysis_model.suggested_prompts
        clarification = build_clarification_addendum(prompts)

        combined_response =
          if clarification != "" do
            "#{base_response} #{clarification}"
          else
            base_response
          end

        Progress.report(opts, :response_generated, %{
          response_type: :partial_with_clarification,
          strategy: :partial_response_with_clarification,
          base_method: response_method,
          prompts_count: length(prompts)
        })

        {combined_response, :partial_with_clarification, context}

      :defer_to_user ->
        # Bot wasn't addressed - acknowledge but don't try to respond substantively
        Progress.report(opts, :response_generated, %{
          response_type: :acknowledgment,
          strategy: :defer_to_user
        })

        {simple_acknowledgment(persona), :not_addressed, %{}}

      :cannot_respond ->
        # Cannot respond with classical NLP - use simple fallback
        Progress.report(opts, :response_generated, %{
          response_type: :fallback,
          strategy: :cannot_respond
        })

        {simple_fallback_response(persona, input), :cannot_respond, %{}}

      _ ->
        # Can respond (fully or partially) - proceed with NLP pipeline
        try_nlp_with_analysis(persona, input, memory, analysis_model, opts)
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
            value =
              case v do
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

  defp try_nlp_with_analysis(persona, input, _memory, analysis_model, opts) do
    # Collect ALL intents from all chunks with their analysis reference
    all_intents_with_analysis =
      analysis_model.analyses
      |> Enum.map(fn analysis ->
        {analysis.intent, analysis.speech_act, analysis.confidence, analysis}
      end)
      |> Enum.filter(fn {intent, _, _, _} -> intent != nil and intent != "" end)

    # Prioritize substantive intents (questions, commands) over expressives (greetings)
    substantive_intents =
      all_intents_with_analysis
      |> Enum.filter(fn {intent, speech_act, _, _} ->
        # Not a greeting/farewell/thanks
        not String.contains?(intent || "", "greeting") and
          not String.contains?(intent || "", "bye") and
          speech_act.category in [:directive, :assertive]
      end)

    # Pick the best substantive intent, or fall back to any intent
    # IMPORTANT: Use entities from the SAME chunk as the selected intent
    {analysis_intent, _, _, intent_analysis} =
      case substantive_intents do
        [first | _] -> first
        [] -> List.first(all_intents_with_analysis) || {nil, nil, 0, nil}
      end

    # Use the intent's chunk for entities, NOT the highest confidence chunk
    # This prevents entities from one chunk (e.g., "I'm Austin" greeting)
    # from filling slots in another chunk (e.g., weather query)
    best_analysis =
      if intent_analysis do
        intent_analysis
      else
        # Fallback to highest confidence if no intent match
        analysis_model.analyses
        |> Enum.filter(&(&1.response_strategy == :can_respond))
        |> Enum.max_by(& &1.confidence, fn -> List.first(analysis_model.analyses) end)
      end

    # Extract entities from the chunk that contains the selected intent
    analysis_entities =
      if best_analysis do
        (best_analysis.entities || [])
        |> Enum.map(fn e ->
          %{
            entity: e["type"] || e[:entity] || e["entity"],
            value: e["name"] || e[:value] || e["value"],
            confidence: e["confidence"] || e[:confidence] || 0.8
          }
        end)
      else
        []
      end

    Logger.debug("Entity selection for intent", %{
      intent: analysis_intent,
      entities: Enum.map(analysis_entities, & &1[:entity]),
      chunk_index: best_analysis && best_analysis.index
    })

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

        # Determine response type (domain vs smalltalk)
        {response, response_type} =
          generate_analysis_response_with_type(intent, entities, analysis_model, persona)

        method =
          if ChatBot.ML.NLPPipeline.should_use_classical_result?(conf) or analysis_intent != nil do
            :analysis_enhanced
          else
            :classical_low_confidence
          end

        # Report response generation details
        Progress.report(opts, :response_generated, %{
          response_type: response_type,
          strategy: :can_respond,
          method: method,
          intent: intent,
          entities_count: length(entities),
          nlp_confidence: conf
        })

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
          {response, response_type} =
            generate_analysis_response_with_type(
              analysis_intent,
              analysis_entities,
              analysis_model,
              persona
            )

          Progress.report(opts, :response_generated, %{
            response_type: response_type,
            strategy: :can_respond,
            method: :analysis_only,
            intent: analysis_intent,
            entities_count: length(analysis_entities),
            nlp_error: reason
          })

          {response, :analysis_only, context}
        else
          Progress.report(opts, :response_generated, %{
            response_type: :fallback,
            strategy: :cannot_respond,
            method: :classical_error,
            nlp_error: reason
          })

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

  defp generate_expressive_part(speech_act) do
    # Map speech act sub_types to intent names for template lookup
    intent_name =
      case speech_act.sub_type do
        :greeting -> "smalltalk.greetings.hello"
        :farewell -> "smalltalk.greetings.bye"
        :thanks -> "smalltalk.appraisal.thank_you"
        :apology -> "smalltalk.dialog.sorry"
        :how_are_you -> "smalltalk.greetings.how_are_you"
        _ -> nil
      end

    # Try to get a template from the store
    if intent_name && TemplateStore.ready?() do
      case TemplateStore.get_random_template(intent_name) do
        nil -> generate_expressive_fallback(speech_act.sub_type)
        template -> template
      end
    else
      generate_expressive_fallback(speech_act.sub_type)
    end
  end

  defp generate_expressive_fallback(sub_type) do
    case sub_type do
      :greeting -> Enum.random(["Hello!", "Hi there!", "Hey!"])
      :farewell -> Enum.random(["Goodbye!", "See you!", "Take care!"])
      :thanks -> Enum.random(["You're welcome!", "Happy to help!", "No problem!"])
      :apology -> Enum.random(["No worries!", "That's fine.", "Don't worry about it!"])
      _ -> nil
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

  # Version that returns both response and type for progress reporting
  defp generate_classical_response_with_type(intent, entities, persona) do
    case generate_domain_response(intent, entities) do
      {:ok, response} ->
        {response, :domain}

      :not_handled ->
        response = generate_smalltalk_response(intent, entities, persona)
        {response, :smalltalk}
    end
  end

  # Generate response and return type for progress reporting
  defp generate_analysis_response_with_type(intent, entities, analysis_model, persona) do
    speech_acts =
      analysis_model.analyses
      |> Enum.map(& &1.speech_act)

    expressives =
      speech_acts
      |> Enum.filter(&(&1.category == :expressive))
      |> Enum.uniq_by(& &1.sub_type)

    directives = Enum.filter(speech_acts, &(&1.category == :directive))

    has_substantive_content =
      length(directives) > 0 or
        (intent != nil and intent != "" and
           not String.starts_with?(intent || "", "smalltalk.greetings"))

    response_parts = []
    response_types = []

    # Add expressive acknowledgment if present
    {response_parts, response_types} =
      if length(expressives) > 0 do
        expressive = List.first(expressives)
        expressive_response = generate_expressive_part(expressive)

        if expressive_response do
          {[expressive_response | response_parts], [:expressive | response_types]}
        else
          {response_parts, response_types}
        end
      else
        {response_parts, response_types}
      end

    # Add substantive response for directives/questions/commands
    {response_parts, response_types} =
      if has_substantive_content do
        {substantive_response, response_type} =
          generate_classical_response_with_type(intent, entities, persona)

        {[substantive_response | response_parts], [response_type | response_types]}
      else
        {response_parts, response_types}
      end

    valid_parts =
      response_parts
      |> Enum.reverse()
      |> Enum.filter(&(&1 != nil and &1 != ""))

    # Determine primary response type
    primary_type =
      cond do
        :domain in response_types -> :domain
        :smalltalk in response_types -> :smalltalk
        :expressive in response_types -> :expressive
        true -> :fallback
      end

    response =
      case valid_parts do
        [] -> generate_classical_response(intent, entities, persona)
        [single] -> single
        parts -> Enum.join(parts, " ")
      end

    {response, primary_type}
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
    entity =
      Enum.find(entities, fn e ->
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
    # First, try to get a response from the TemplateStore (loaded from intent files)
    template_response = try_template_store_response(intent, entities)

    if template_response do
      template_response
    else
      # Fall back to custom smalltalk responses file
      generate_smalltalk_from_file(intent, entities)
    end
  end

  defp try_template_store_response(intent, entities) do
    if TemplateStore.ready?() do
      case TemplateStore.get_random_template(intent) do
        nil ->
          # Try parent intent (e.g., "smalltalk.greetings" from "smalltalk.greetings.hello")
          nil

        template ->
          # Substitute slots with entity values
          TemplateStore.substitute_slots(template, entities)
      end
    else
      nil
    end
  end

  defp generate_smalltalk_from_file(intent, entities) do
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

  # ============================================================================
  # Epistemic System Integration
  # ============================================================================

  defp handle_meta_cognitive_query(_persona, input, user_id, _opts) do
    Logger.info("Handling meta-cognitive query", %{input: input, user_id: user_id})

    # Build self-knowledge assessment
    assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment(user_id)

    # Synthesize response using the epistemic response synthesizer
    response =
      Synthesizer.synthesize_self_knowledge_response(assessment,
        context: %{
          relationship_duration: :new,
          user_initiated: true
        }
      )

    # Record this disclosure
    if user_id do
      disclosed_keys =
        (assessment.discloseable ++ assessment.inferred_uncertain)
        |> Enum.map(& &1.key)

      UserModelStore.record_disclosure(user_id, disclosed_keys, %{
        query: input,
        timestamp: DateTime.utc_now()
      })
    end

    context = %{
      intent: "meta.self_query",
      entities: [],
      slots: %{},
      missing_slots: [],
      epistemic_assessment: true
    }

    {response, :epistemic_response, context}
  end

  @doc false
  def extract_and_store_beliefs(input, entities, user_id, conversation_id) do
    if Config.auto_extraction_enabled?() and user_id do
      # Extract potential beliefs from entities
      Enum.each(entities, fn entity ->
        entity_type = entity[:entity] || entity["entity"]
        entity_value = entity[:value] || entity["value"]

        if entity_type && entity_value do
          # Determine if this is a user fact
          if is_user_fact?(entity_type) do
            # Create and store belief
            belief =
              Belief.new(:user, normalize_predicate(entity_type), entity_value,
                source: :explicit,
                confidence: 0.85,
                user_id: user_id,
                provenance: [
                  "conversation:#{conversation_id}",
                  "input:#{String.slice(input, 0, 50)}"
                ]
              )

            BeliefStore.add_belief(belief)

            # Also update user model
            UserModelStore.update_fact(
              user_id,
              normalize_predicate(entity_type),
              entity_value,
              :explicit,
              0.85
            )

            Logger.debug("Extracted belief from conversation", %{
              predicate: entity_type,
              value: entity_value,
              user_id: user_id
            })
          end
        end
      end)

      # Look for self-referential statements
      extract_self_referential_facts(input, user_id, conversation_id)
    end
  rescue
    e ->
      Logger.warning("Failed to extract beliefs: #{inspect(e)}")
  end

  defp is_user_fact?(entity_type) do
    user_fact_types = [
      "location",
      "city",
      "country",
      "timezone",
      "name",
      "person",
      "occupation",
      "company",
      "preference",
      "hobby",
      "interest"
    ]

    entity_type_str = to_string(entity_type) |> String.downcase()
    Enum.any?(user_fact_types, &String.contains?(entity_type_str, &1))
  end

  defp normalize_predicate(predicate) when is_atom(predicate), do: predicate

  defp normalize_predicate(predicate) when is_binary(predicate) do
    predicate
    |> String.downcase()
    |> String.replace([" ", "-"], "_")
    |> String.to_atom()
  end

  defp normalize_predicate(_), do: :unknown

  defp extract_self_referential_facts(input, user_id, conversation_id) do
    # Expand contractions first for simpler pattern matching
    # "I'm" → "I am", "don't" → "do not", etc.
    expanded = ChatBot.ML.Tokenizer.expand_contractions(input)
    tokens = ChatBot.ML.Tokenizer.tokenize_normalized(expanded)

    # Check for various self-referential patterns (all in canonical form now)
    extract_location_facts(tokens, expanded, user_id, conversation_id)
    extract_name_facts(tokens, expanded, user_id, conversation_id)
    extract_preference_facts(tokens, expanded, user_id, conversation_id)
    extract_work_facts(tokens, expanded, user_id, conversation_id)
  end

  defp extract_location_facts(tokens, _input, user_id, conversation_id) do
    # Pattern: "i am from X", "i live in X"
    # Contractions are already expanded, so we only need canonical patterns
    cond do
      # "i am from" (handles both "I'm from" and "I am from")
      has_sequence?(tokens, ["i", "am", "from"]) ->
        value = extract_after_sequence(tokens, ["from"])
        store_fact_if_valid(user_id, :location, value, conversation_id)

      # "i live in"
      has_sequence?(tokens, ["i", "live", "in"]) ->
        value = extract_after_sequence(tokens, ["in"])
        store_fact_if_valid(user_id, :location, value, conversation_id)

      true ->
        :ok
    end
  end

  defp extract_name_facts(tokens, _input, user_id, conversation_id) do
    # Pattern: "my name is X", "i'm X" (when short), "call me X"
    cond do
      has_sequence?(tokens, ["my", "name", "is"]) ->
        value = extract_after_sequence(tokens, ["is"])
        store_fact_if_valid(user_id, :name, value, conversation_id)

      has_sequence?(tokens, ["call", "me"]) ->
        value = extract_after_sequence(tokens, ["me"])
        store_fact_if_valid(user_id, :name, value, conversation_id)

      true ->
        :ok
    end
  end

  defp extract_preference_facts(tokens, _input, user_id, conversation_id) do
    # Pattern: "i like X", "i prefer X", "i love X"
    cond do
      has_sequence?(tokens, ["i", "like"]) ->
        value = extract_after_sequence(tokens, ["like"])
        store_fact_if_valid(user_id, :likes, value, conversation_id)

      has_sequence?(tokens, ["i", "prefer"]) ->
        value = extract_after_sequence(tokens, ["prefer"])
        store_fact_if_valid(user_id, :preference, value, conversation_id)

      has_sequence?(tokens, ["i", "love"]) ->
        value = extract_after_sequence(tokens, ["love"])
        store_fact_if_valid(user_id, :likes, value, conversation_id)

      true ->
        :ok
    end
  end

  defp extract_work_facts(tokens, _input, user_id, conversation_id) do
    # Pattern: "i work at X", "i work for X"
    cond do
      has_sequence?(tokens, ["i", "work", "at"]) ->
        value = extract_after_sequence(tokens, ["at"])
        store_fact_if_valid(user_id, :workplace, value, conversation_id)

      has_sequence?(tokens, ["i", "work", "for"]) ->
        value = extract_after_sequence(tokens, ["for"])
        store_fact_if_valid(user_id, :workplace, value, conversation_id)

      true ->
        :ok
    end
  end

  defp has_sequence?(tokens, sequence) do
    # Check if tokens contain the sequence in order
    sequence_len = length(sequence)

    tokens
    |> Enum.chunk_every(sequence_len, 1, :discard)
    |> Enum.any?(&(&1 == sequence))
  end

  defp extract_after_sequence(tokens, marker_sequence) do
    # Find the marker sequence and return tokens after it
    marker_len = length(marker_sequence)

    case find_sequence_index(tokens, marker_sequence) do
      nil ->
        nil

      idx ->
        tokens
        |> Enum.drop(idx + marker_len)
        # Take up to 5 tokens
        |> Enum.take(5)
        |> Enum.join(" ")
    end
  end

  defp find_sequence_index(tokens, sequence) do
    sequence_len = length(sequence)

    tokens
    |> Enum.chunk_every(sequence_len, 1, :discard)
    |> Enum.with_index()
    |> Enum.find_value(fn {chunk, idx} ->
      if chunk == sequence, do: idx, else: nil
    end)
  end

  defp store_fact_if_valid(user_id, predicate, value, conversation_id) do
    clean_value = if value, do: String.trim(value), else: ""

    if String.length(clean_value) > 0 and String.length(clean_value) < 50 do
      belief =
        Belief.new(:user, predicate, clean_value,
          source: :explicit,
          confidence: 0.9,
          user_id: user_id,
          provenance: ["self_statement", "conversation:#{conversation_id}"]
        )

      BeliefStore.add_belief(belief)
      UserModelStore.update_fact(user_id, predicate, clean_value, :explicit, 0.9)

      Logger.debug("Extracted self-referential fact", %{
        predicate: predicate,
        value: clean_value
      })
    end
  end
end
