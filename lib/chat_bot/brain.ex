defmodule ChatBot.Brain do
  @moduledoc """
  The Brain GenServer manages the AI personality, subprocesses, and global memory.
  This is the core component that orchestrates all chat bot functionality.
  """

  use GenServer
  require Logger

  alias ChatBot.Analysis.{
    SelfKnowledgeAnalyzer,
    RacingAnalyzer,
    OutcomeLearner,
    Interpretation,
    Progress,
    ResponseGate,
    SlotDetector,
    IntentRegistry
  }

  alias ChatBot.Epistemic.{UserModelStore, BeliefStore}
  alias ChatBot.Epistemic.Types.{Belief, Config}
  alias ChatBot.Response.{Synthesizer, Generator}
  alias ChatBot.Learning.WorldContext

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the Brain GenServer.

  ## Arguments
    - `artifact_path` - Path to the personality artifact file
    - `opts` - Options including:
      - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(artifact_path, opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, artifact_path, name: name)
  end

  @doc """
  Evaluates user input in a conversation.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
    - `:timeout` - Call timeout in ms (default: 90_000)
  """
  def evaluate(conversation_id, input, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    timeout = Keyword.get(opts, :timeout, 90_000)
    opts = opts |> Keyword.delete(:server) |> Keyword.delete(:timeout)

    # Wrap with telemetry span for async, non-blocking metrics
    ChatBot.Telemetry.span(:brain_evaluate, %{conversation_id: conversation_id}, fn ->
      GenServer.call(server, {:evaluate, conversation_id, input, opts}, timeout)
    end)
  end

  @doc """
  Creates a new conversation.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
    - `:world_id` - The training world to use for this conversation (default: "default")
  """
  def create_conversation(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:create_conversation, opts})
  end

  @doc """
  Ends a conversation.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def end_conversation(conversation_id, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:end_conversation, conversation_id})
  end

  @doc """
  Gets the Brain's status.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def get_status(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :get_status)
  end

  @doc """
  Gets all active conversations.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def get_conversations(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :get_conversations)
  end

  @doc """
  Gets a specific conversation.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def get_conversation(conversation_id, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:get_conversation, conversation_id})
  end

  @doc """
  Handles an urgent interrupt.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def handle_urgent_interrupt(reason, data \\ %{}, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.cast(server, {:urgent_interrupt, reason, data})
  end

  @doc """
  Handles an urgent emergency.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def handle_urgent_emergency(reason, data \\ %{}, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.cast(server, {:urgent_emergency, reason, data})
  end

  @doc """
  Starts an HTTP subprocess.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def start_http_subprocess(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:start_http_subprocess, opts})
  end

  @doc """
  Starts a conversation subprocess.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def start_conversation_subprocess(conversation_id, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:start_conversation_subprocess, conversation_id, opts})
  end

  @doc """
  Starts a CLI subprocess.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def start_cli_subprocess(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:start_cli_subprocess, opts})
  end

  @doc """
  Stops a subprocess.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def stop_subprocess(subprocess_id, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:stop_subprocess, subprocess_id})
  end

  @doc """
  Lists all subprocesses.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def list_subprocesses(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :list_subprocesses)
  end

  @doc """
  Resets the Brain state. Useful for testing.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def reset_state(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :reset_state)
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

        # Get world_id from conversation (default to "default")
        world_id = Map.get(conversation, :world_id, "default")

        # Add world_id to options for downstream processing
        opts_with_world = Keyword.put(opts, :world_id, world_id)

        # Process with classical NLP
        {response, processing_method, context} =
          if Application.get_env(:chat_bot, :ml)[:enabled] do
            try_classical_nlp_first(state.persona, input, conversation.memory, opts_with_world)
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

        # Handle nil responses (deferred by ResponseGate)
        # Still store user message for speech act history, but no assistant message
        {new_messages, learning_response} =
          if response == nil do
            # Response deferred - only add user message
            {[user_message], nil}
          else
            # Normal response - add both messages
            assistant_message = %{
              id: generate_message_id(),
              role: "assistant",
              content: response,
              timestamp: System.system_time(:millisecond),
              processing_method: processing_method
            }

            {[user_message, assistant_message], response}
          end

        updated_conversation =
          conversation
          |> Map.put(:memory, conversation.memory ++ new_messages)
          # Track active context for follow-up detection (use Map.put since key may not exist)
          |> Map.put(:active_context, context_snapshot)
          |> Map.put(:last_activity, System.system_time(:millisecond))

        # Add to learning queue only if we responded
        updated_learning_queue =
          if learning_response != nil do
            learning_entry = %{
              conversation_id: conversation_id,
              world_id: world_id,
              timestamp: System.system_time(:millisecond),
              input: input,
              response: learning_response
            }

            state.learning_queue ++ [learning_entry]
          else
            state.learning_queue
          end

        # Extract and store beliefs from entities (epistemic integration)
        user_id = Keyword.get(opts, :user_id)
        entities = Map.get(context, :entities, [])
        extract_and_store_beliefs(input, entities, user_id, conversation_id)

        # Learn from this interaction for future fast-path (heuristic learning)
        # Only learn when we actually responded (not deferred)
        if learning_response != nil and processing_method != :response_deferred do
          Task.start(fn ->
            interpretation = build_interpretation_from_context(input, context)

            OutcomeLearner.learn_from_outcome(interpretation, learning_response,
              user_id: user_id,
              cohort_id: nil
            )
          end)
        end

        updated_state = %{
          state
          | active_conversations:
              Map.put(state.active_conversations, conversation_id, updated_conversation),
            learning_queue: updated_learning_queue
        }

        # Process learning queue asynchronously (only if there are entries)
        if length(updated_learning_queue) > length(state.learning_queue) do
          send(self(), :process_learning_queue)
        end

        {:reply, {:ok, response}, updated_state}
    end
  end

  @impl true
  def handle_call({:create_conversation, opts}, _from, state) do
    conversation_id = generate_conversation_id()
    world_id = Keyword.get(opts, :world_id, "default")

    conversation = %{
      id: conversation_id,
      world_id: world_id,
      memory: [],
      active_context: nil,
      created_at: System.system_time(:millisecond),
      last_activity: System.system_time(:millisecond)
    }

    updated_state = %{
      state
      | active_conversations: Map.put(state.active_conversations, conversation_id, conversation)
    }

    Logger.info("Conversation created", %{
      conversation_id: conversation_id,
      world_id: world_id
    })

    {:reply, {:ok, conversation_id}, updated_state}
  end

  # Backward compatibility for old create_conversation calls
  @impl true
  def handle_call(:create_conversation, from, state) do
    handle_call({:create_conversation, []}, from, state)
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
      # Store speech act for response optionality history-based reasoning
      speech_act: Map.get(context, :speech_act),
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
      # Check for fast-path via RacingAnalyzer before running full pipeline
      # This uses heuristics and memory similarity for quick responses
      case RacingAnalyzer.check_fast_path(input, user_id, nil) do
        {:fast_path, interpretation} ->
          Logger.debug("Fast path hit", %{
            intent: interpretation.intent,
            source: interpretation.source,
            activation: interpretation.activation
          })

          handle_fast_path_response(persona, interpretation, memory, opts)

        :no_match ->
          # No fast path - run standard analysis pipeline
          process_standard_message(persona, input, memory, opts)
      end
    end
  end

  # Handles responses when RacingAnalyzer finds a fast-path match
  defp handle_fast_path_response(_persona, interpretation, _memory, opts) do
    intent = interpretation.intent
    entities = interpretation.entities || []

    # Generate response using existing Generator
    # Generator.generate always returns {:ok, response, type} with fallback if needed
    {:ok, response, response_type} = Generator.generate(intent, entities, nil)

    context = %{
      intent: intent,
      source: interpretation.source,
      fast_path: true,
      activation: interpretation.activation,
      entities: entities
    }

    Logger.info("Fast path response generated", %{
      intent: intent,
      response_type: response_type
    })

    Progress.report(opts, :response_generated, %{
      response_type: :fast_path,
      strategy: :heuristic_match,
      intent: intent,
      source: interpretation.source
    })

    {response, :fast_path, context}
  end

  defp process_standard_message(persona, input, memory, opts) do
    # Run the analysis pipeline to build an internal model
    analysis_model = run_analysis_pipeline(input, memory, opts)

    Logger.debug("Analysis pipeline complete", %{
      strategy: analysis_model.overall_strategy,
      chunks: length(analysis_model.chunks),
      prompts: analysis_model.suggested_prompts
    })

    # NEW: Check if response is optional (after analysis, before response generation)
    # This evaluates gratitude loops, backchannels, compliments, continuations, etc.
    case ResponseGate.evaluate(analysis_model, memory, opts) do
      {:defer, reason} ->
        # Response not needed - return nil
        Logger.info("Response deferred by ResponseGate", reason)

        Progress.report(opts, :response_generated, %{
          response_type: :deferred,
          strategy: :response_optional,
          reason: reason[:reason]
        })

        context = extract_context_from_analysis(analysis_model)
        {nil, :response_deferred, Map.put(context, :defer_reason, reason)}

      {:optional, confidence, reason} ->
        # Response is situational - Brain decides based on confidence threshold
        defer_threshold = get_defer_threshold(opts)

        if confidence >= defer_threshold do
          Logger.info("Response optional, deferring", %{
            confidence: confidence,
            threshold: defer_threshold,
            reason: reason[:reason]
          })

          Progress.report(opts, :response_generated, %{
            response_type: :optional_deferred,
            strategy: :response_optional,
            confidence: confidence,
            reason: reason[:reason]
          })

          context = extract_context_from_analysis(analysis_model)
          {nil, :response_optional, Map.put(context, :defer_reason, reason)}
        else
          # Low confidence - still respond but record for learning
          Logger.debug("Response optional but proceeding", %{
            confidence: confidence,
            threshold: defer_threshold,
            reason: reason[:reason]
          })

          proceed_with_standard_response(persona, input, memory, analysis_model, opts)
        end

      {:respond, _reason} ->
        # Normal flow - proceed with response generation
        proceed_with_standard_response(persona, input, memory, analysis_model, opts)
    end
  end

  # Standard response generation after ResponseGate approves
  defp proceed_with_standard_response(persona, input, memory, analysis_model, opts) do
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

  # Get the threshold for deferring optional responses
  defp get_defer_threshold(opts) do
    Keyword.get(opts, :defer_threshold, 0.7)
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
    # Run full analysis pipeline to get proper discourse/speech_act context for disambiguation
    # This ensures entities are disambiguated correctly even in follow-up messages
    analysis_model = run_analysis_pipeline(input, [], [])

    # Extract entities from the best analysis chunk with proper disambiguation context
    best_analysis =
      analysis_model.analyses
      |> Enum.max_by(& &1.confidence, fn -> nil end)

    entities =
      if best_analysis do
        (best_analysis.entities || [])
        |> Enum.map(fn e ->
          %{
            entity_type: e["type"] || e[:entity_type] || e["entity_type"],
            value: e["name"] || e[:value] || e["value"],
            confidence: e["confidence"] || e[:confidence] || 0.8
          }
        end)
      else
        # Fallback: extract without context if analysis failed
        ChatBot.ML.EntityExtractor.extract_entities(input)
      end

    Logger.info("Extracted entities from follow-up", %{
      entities_count: length(entities),
      entities: Enum.map(entities, & &1.entity_type),
      used_pipeline: best_analysis != nil
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
        missing_slots: [],
        # Followup inherits speech_act from original context if available
        speech_act: Map.get(merged_context, :speech_act)
      }

      {response, :followup_completed, context}
    else
      # Still missing slots - ask for clarification
      prompt = generate_followup_clarification(merged_context)

      context = %{
        intent: merged_context.intent,
        entities: merged_context.entities,
        slots: merged_context.slots,
        missing_slots: merged_context.missing_slots,
        # Followup inherits speech_act from original context if available
        speech_act: Map.get(merged_context, :speech_act)
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
        missing_slots: Map.get(best_analysis, :missing_context, []),
        # Include speech act for response optionality reasoning
        speech_act: extract_speech_act_info(best_analysis.speech_act)
      }
    else
      %{}
    end
  end

  # Extract speech act info for storage in conversation memory
  defp extract_speech_act_info(nil), do: nil

  defp extract_speech_act_info(speech_act) when is_map(speech_act) do
    %{
      category: Map.get(speech_act, :category),
      sub_type: Map.get(speech_act, :sub_type),
      confidence: Map.get(speech_act, :confidence),
      is_question: Map.get(speech_act, :is_question, false)
    }
  end

  defp extract_speech_act_info(_), do: nil

  defp extract_filled_slots(slots) when is_map(slots) do
    case Map.get(slots, :filled_slots) do
      nil -> slots
      filled -> filled
    end
  end

  defp extract_filled_slots(_), do: %{}

  defp generate_intent_response(context, _persona) do
    # Delegate to Generator for unified response generation
    intent = context.intent
    entities = slots_to_entities(context.slots)
    {:ok, response, _type} = Generator.generate(intent, entities, nil)
    response
  end

  defp slots_to_entities(slots) when is_map(slots) do
    Enum.map(slots, fn {k, v} ->
      value =
        case v do
          %{value: val} -> val
          val -> val
        end

      %{entity: k, value: value}
    end)
  end

  defp slots_to_entities(_), do: []

  defp generate_followup_clarification(context) do
    # Use centralized clarification prompts from IntentRegistry via SlotDetector
    case context.missing_slots do
      [] ->
        "I need a bit more information. Could you elaborate?"

      [slot | _] ->
        SlotDetector.get_clarification_prompt(slot, context.intent)
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
      entity_type = entity[:entity_type]
      entity_value = entity[:value]
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
        # Not a greeting/farewell/thanks - use registry for classification
        not IntentRegistry.greeting?(intent) and
          not IntentRegistry.farewell?(intent) and
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
            entity_type: e["type"] || e[:entity_type] || e["entity_type"],
            value: e["name"] || e[:value] || e["value"],
            confidence: e["confidence"] || e[:confidence] || 0.8
          }
        end)
      else
        []
      end

    Logger.debug("Entity selection for intent", %{
      intent: analysis_intent,
      entities: Enum.map(analysis_entities, & &1[:entity_type]),
      chunk_index: best_analysis && best_analysis.index
    })

    # Extract slot information for context
    slots_info = if best_analysis, do: Map.get(best_analysis, :slots), else: nil
    missing_slots = if best_analysis, do: Map.get(best_analysis, :missing_context, []), else: []

    # Extract discourse and speech_act context for disambiguation
    disambiguation_opts =
      if best_analysis do
        [
          discourse: Map.get(best_analysis, :discourse),
          speech_act: Map.get(best_analysis, :speech_act)
        ]
      else
        []
      end

    # Collect ALL entities from ALL chunks for conflict detection
    # This prevents "Austin" being used as location when it was identified as person in another chunk
    all_analysis_entities =
      analysis_model.analyses
      |> Enum.flat_map(fn analysis ->
        (analysis.entities || [])
        |> Enum.map(fn e ->
          %{
            entity_type: e["type"] || e[:entity_type] || e["entity_type"],
            value: e["name"] || e[:value] || e["value"],
            confidence: e["confidence"] || e[:confidence] || 0.8
          }
        end)
      end)

    # For multi-chunk inputs, skip NLPPipeline entirely.
    # The per-chunk analysis is more accurate because each chunk gets proper context.
    # NLPPipeline processes the whole text with only one chunk's context, which can cause
    # cross-chunk entity bleeding (e.g., "Austin" as person in greeting being used as
    # location for a weather query in the same message).
    num_chunks = length(analysis_model.analyses)

    {intent, entities, method} =
      if num_chunks > 1 do
        # Multi-chunk: use only per-chunk analysis results
        Logger.debug("Multi-chunk input: skipping NLPPipeline, using only per-chunk analysis", %{
          num_chunks: num_chunks,
          analysis_entity_count: length(analysis_entities),
          selected_intent: analysis_intent
        })

        {analysis_intent, analysis_entities, :analysis_only}
      else
        # Single chunk: safe to use NLPPipeline for additional processing
        case ChatBot.ML.NLPPipeline.process(input, disambiguation_opts) do
          {:ok, %{confidence: conf, intent: nlp_intent, entities: nlp_entities}} ->
            # Merge analysis intent with NLP intent (prefer analysis if both present)
            intent = analysis_intent || nlp_intent

            entities =
              merge_entities(analysis_entities, nlp_entities, all_analysis_entities, intent)

            method =
              if ChatBot.ML.NLPPipeline.should_use_classical_result?(conf) or
                   analysis_intent != nil do
                :analysis_enhanced
              else
                :classical_low_confidence
              end

            {intent, entities, method}

          {:error, reason} ->
            Logger.warning("NLP pipeline failed", %{reason: reason})
            {analysis_intent, analysis_entities, :analysis_only}
        end
      end

    Logger.info("Processing with analysis", %{
      analysis_intent: analysis_intent,
      final_intent: intent,
      method: method,
      entities_count: length(entities),
      num_chunks: num_chunks
    })

    # Build context for storage (include speech_act for response optionality)
    speech_act_info =
      if(best_analysis, do: extract_speech_act_info(best_analysis.speech_act), else: nil)

    context = %{
      intent: intent,
      entities: entities,
      slots: extract_filled_slots(slots_info),
      missing_slots: missing_slots,
      speech_act: speech_act_info
    }

    # Learn from conversation - extracts both entities AND facts from assertive statements
    analysis_for_learning = %{
      entities: entities,
      speech_act: speech_act_info,
      intent: intent
    }

    ChatBot.Learner.learn_from_conversation(persona.name, input, analysis_for_learning)

    # Determine response type (domain vs smalltalk)
    {response, response_type} =
      generate_analysis_response_with_type(intent, entities, analysis_model, persona, input)

    # Report response generation details
    Progress.report(opts, :response_generated, %{
      response_type: response_type,
      strategy: :can_respond,
      method: method,
      intent: intent,
      entities_count: length(entities)
    })

    {response, method, context}
  end

  defp merge_entities(analysis_entities, nlp_entities, all_analysis_entities, intent) do
    # Combine entities, preferring analysis entities for duplicates
    # Analysis entities are extracted per-chunk with proper context disambiguation
    # NLP entities are extracted globally and may have wrong context

    # Get types from the selected chunk's entities
    analysis_types = Enum.map(analysis_entities, & &1.entity_type) |> MapSet.new()

    # Get entity types allowed by the intent's slot schema
    # This prevents entities extracted from other chunks from bleeding into the wrong intent
    allowed_types =
      if intent do
        SlotDetector.get_entity_types_for_intent(intent)
      else
        MapSet.new()
      end

    # Get normalized values from ALL analysis chunks to detect cross-chunk conflicts
    # e.g., if "Austin" was identified as "person" in chunk 1, don't add it as "location"
    # for chunk 3's weather query
    all_entities_for_conflict_check =
      if length(all_analysis_entities) > 0, do: all_analysis_entities, else: analysis_entities

    analysis_values =
      all_entities_for_conflict_check
      |> Enum.map(fn e ->
        value = e[:value] || e["value"] || ""
        String.downcase(to_string(value))
      end)
      |> MapSet.new()

    unique_nlp =
      nlp_entities
      |> Enum.reject(fn e ->
        e_type = e.entity_type
        e_value = String.downcase(to_string(e[:value] || e["value"] || ""))

        # Reject if same type already exists in selected chunk
        same_type = MapSet.member?(analysis_types, e_type)

        # Also reject if the same value was extracted by analysis in ANY chunk with a DIFFERENT type
        # This prevents cross-chunk entity bleeding (e.g., "Austin" as person in greeting
        # shouldn't become "Austin" as location for weather query)
        value_conflict = MapSet.member?(analysis_values, e_value) and not same_type

        # Additionally, reject if the entity type is not allowed by the intent's slot schema
        # This prevents "person" entities from filling "location" slots, etc.
        type_not_allowed =
          MapSet.size(allowed_types) > 0 and not MapSet.member?(allowed_types, e_type)

        same_type or value_conflict or type_not_allowed
      end)

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

  # Generate response and return type for progress reporting
  defp generate_analysis_response_with_type(
         intent,
         entities,
         analysis_model,
         _persona,
         query_text
       ) do
    # Delegate to Generator for unified analysis response generation
    Generator.generate_from_analysis(analysis_model, intent, entities, query_text)
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
    # Using WorldContext for inheritance-aware data access
    if Process.whereis(ChatBot.Memory.Store) != nil do
      Enum.each(entries, fn entry ->
        # Determine tags from NLP analysis if possible
        tags = extract_tags_for_memory(entry.input)
        world_id = Map.get(entry, :world_id, "default")

        # Use WorldContext.add_episode for world-scoped storage with inheritance
        WorldContext.add_episode(
          world_id,
          entry.input,
          "conversation",
          entry.response,
          ["conversation", entry.conversation_id | tags]
        )
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
      epistemic_assessment: true,
      # Meta queries are directives (questions) - always expect response
      speech_act: %{
        category: :directive,
        sub_type: :request_information,
        confidence: 0.9,
        is_question: true
      }
    }

    {response, :epistemic_response, context}
  end

  @doc false
  def extract_and_store_beliefs(input, entities, user_id, conversation_id) do
    if Config.auto_extraction_enabled?() and user_id do
      # Extract potential beliefs from entities
      Enum.each(entities, fn entity ->
        entity_type = entity[:entity_type]
        entity_value = entity[:value]

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

  # Builds an Interpretation struct from the context for OutcomeLearner
  defp build_interpretation_from_context(input, context) do
    intent = Map.get(context, :intent)
    activation = Map.get(context, :activation, 0.7)
    source = Map.get(context, :source, :pipeline)
    entities = Map.get(context, :entities, [])

    # Determine the source type for the Interpretation
    source_atom =
      case source do
        :fast_path -> :heuristic
        :heuristic_match -> :heuristic
        :memory_match -> :memory_match
        :pattern_recognition -> :pattern_recognition
        _ -> :model
      end

    Interpretation.new(intent, input, activation, source_atom)
    |> Interpretation.with_entities(entities)
  end
end
