defmodule Brain do
  @moduledoc "The Brain GenServer manages the AI personality, subprocesses, and global memory.\nThis is the core component that orchestrates all chat bot functionality.\n"

  # World.Context and World.Manager are in a sibling umbrella app that depends on :brain.
  # They are available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Context}
  @compile {:no_warn_undefined, World.Manager}

  alias Brain.ML.Tokenizer
  alias Brain.Learner
  alias Brain.ML.NLPPipeline
  alias Phoenix.PubSub
  alias Brain.Analysis.Pipeline
  alias Brain.Analysis.FollowupDetector
  alias Brain.ML.EntityExtractor
  alias Brain.MemoryStore
  alias Brain.KnowledgeStore
  alias Brain.Telemetry
  alias Brain.Response
  alias Brain.Epistemic.Types
  alias Brain.Epistemic
  alias Brain.Analysis
  use GenServer
  require Logger

  alias Analysis.{
    SelfKnowledgeAnalyzer,
    ConsistencyChecker,
    Interpretation,
    Progress,
    ResponseGate,
    SlotDetector,
    ChunkPriority,
    ChunkProfile,
    FeatureExtractor
  }

  alias Brain.Analysis.OutcomeLearner

  alias Epistemic.{UserModelStore, BeliefStore}
  alias Types.{Belief, Config}
  alias Response.{Synthesizer, Generator}
  alias World.Context, as: WorldContext
  alias World.Manager, as: WorldManager

  @doc "Returns the priv directory path for the brain app.\n\nUses `:code.priv_dir(:brain)` which correctly resolves the path\nregardless of which directory the application is run from.\n"
  def priv_dir do
    :code.priv_dir(:brain) |> to_string()
  end

  @doc "Returns a path within the brain priv directory.\n\n## Examples\n\n    Brain.priv_path(\"ml_models/classifier.term\")\n    #=> \"/path/to/apps/brain/priv/ml_models/classifier.term\"\n"
  def priv_path(subpath) when is_binary(subpath) do
    Path.join(priv_dir(), subpath)
  end

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

    Telemetry.span(:brain_evaluate, %{conversation_id: conversation_id}, fn ->
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

  @impl true
  def init(artifact_path) do
    artifact = load_artifact(artifact_path)
    persona = create_personality(artifact)
    knowledge = KnowledgeStore.load_knowledge(persona.name)
    memory = MemoryStore.load_all(persona.name)
    updated_persona = Map.put(persona, :knowledge, knowledge)

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
        do_evaluate(conversation_id, conversation, input, opts, state)
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

    Brain.Graph.Writer.write_conversation(conversation)

    {:reply, {:ok, conversation_id}, updated_state}
  end

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
          world_id: Map.get(conv, :world_id, "default"),
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

    case Brain.Subprocesses.Supervisor.start_http_subprocess(
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

    case Brain.Subprocesses.Supervisor.start_conversation_subprocess(
           opts
           |> Keyword.put(:conversation_id, conversation_id)
           |> Keyword.put(:memory_snapshot, memory_snapshot)
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

    case Brain.Subprocesses.Supervisor.start_cli_subprocess(
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
    subprocess_info = find_subprocess_by_id(state.subprocesses, subprocess_id)

    case subprocess_info do
      {type, info} ->
        case Brain.Subprocesses.Supervisor.stop_subprocess(info.pid) do
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

    safe_pubsub_broadcast("brain:status", "interrupt_acknowledged", %{
      reason: reason,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, updated_state}
  end

  @impl true
  def handle_cast({:urgent_emergency, reason, data}, state) do
    Logger.error("Handling urgent emergency", %{reason: reason, data: data})
    updated_state = %{state | active_conversations: %{}, is_shutting_down: true}

    safe_pubsub_broadcast("brain:status", "emergency_acknowledged", %{
      reason: reason,
      timestamp: System.system_time(:millisecond)
    })

    {:noreply, updated_state}
  end

  @impl true
  def handle_info(:process_learning_queue, state) do
    if state.learning_queue != [] do
      {processed_entries, remaining_queue} = Enum.split(state.learning_queue, 5)

      new_memory_entries =
        processed_entries
        |> Enum.map(fn entry ->
          %{
            conversation_id: entry.conversation_id,
            timestamp: entry.timestamp,
            summary: create_learning_summary(entry.input, entry.response)
          }
        end)

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

      safe_pubsub_broadcast("brain:learning", "learning_processed", %{
        processed_count: length(processed_entries),
        global_memory_size: length(updated_state.global_memory),
        timestamp: System.system_time(:millisecond)
      })

      if remaining_queue != [] do
        Process.send_after(self(), :process_learning_queue, 1000)
      end

      {:noreply, updated_state}
    else
      {:noreply, state}
    end
  end

  defp build_context_snapshot(context) do
    %{
      intent: Map.get(context, :intent),
      entities: Map.get(context, :entities, []),
      slots: Map.get(context, :slots, %{}),
      missing_slots: Map.get(context, :missing_slots, []),
      speech_act: Map.get(context, :speech_act),
      timestamp: System.system_time(:millisecond)
    }
  end

  defp maybe_put_ouro_messages(result, nil), do: result
  defp maybe_put_ouro_messages(result, messages), do: Map.put(result, :ouro_messages, messages)

  defp maybe_include_analysis(result, context, opts) do
    if Keyword.get(opts, :include_analysis, false) do
      Map.put(result, :analysis_model, Map.get(context, :analysis_model))
    else
      result
    end
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

  defp do_evaluate(conversation_id, conversation, input, opts, state) do
    now = System.system_time(:millisecond)
    world_id = Map.get(conversation, :world_id, "default")
    opts_with_world = Keyword.put(opts, :world_id, world_id)
    ml_config = Application.get_env(:brain, :ml) || Application.get_env(:chat_bot, :ml) || []

    {response, processing_method, context} =
      if ml_config[:enabled] do
        try_classical_nlp_first(state.persona, input, conversation.memory, opts_with_world)
      else
        {simple_fallback_response(state.persona, input), :simple, %{}}
      end

    {ouro_messages, response} =
      case response do
        {:ouro_dry_run, messages} -> {messages, nil}
        other -> {nil, other}
      end

    context_snapshot = build_context_snapshot(context)

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

    {new_messages, learning_response} =
      if response == nil do
        {[user_message], nil}
      else
        assistant_message = %{
          id: generate_message_id(),
          role: "assistant",
          content: response,
          timestamp: System.system_time(:millisecond),
          processing_method: processing_method
        }

        {[user_message, assistant_message], response}
      end

    Enum.each(new_messages, fn msg ->
      analysis_for_msg =
        if msg.role == "user", do: Map.get(context, :analysis_model), else: nil

      Brain.Graph.Writer.write_message(conversation_id, msg, analysis_for_msg)
    end)

    updated_conversation =
      conversation
      |> Map.put(:memory, conversation.memory ++ new_messages)
      |> Map.put(:active_context, context_snapshot)
      |> Map.put(:last_activity, System.system_time(:millisecond))

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

    user_id = Keyword.get(opts, :user_id)
    entities = Map.get(context, :entities, [])
    extract_and_store_beliefs(input, entities, user_id, conversation_id)
    feed_entities_to_world(entities, world_id)

    if ml_config[:enabled] and entities != [] and Config.auto_extraction_enabled?() do
      Learner.learn_from_classical_extraction(state.persona.name, entities, input)
    end

    if learning_response != nil and processing_method != :response_deferred do
      Task.start(fn ->
        interpretation = build_interpretation_from_context(input, context)

        OutcomeLearner.learn_from_outcome(interpretation, learning_response,
          world_id: world_id,
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

    if length(updated_learning_queue) > length(state.learning_queue) do
      send(self(), :process_learning_queue)
    end

    enriched_result =
      %{
        response: response || "",
        context: context_snapshot,
        processing_method: processing_method
      }
      |> maybe_put_ouro_messages(ouro_messages)
      |> maybe_include_analysis(context, opts)

    {:reply, {:ok, enriched_result}, updated_state}
  rescue
    e ->
      Logger.error("Evaluation failed for conversation #{conversation_id}: #{Exception.message(e)}")
      {:reply, {:error, {:generation_failed, Exception.message(e)}}, state}
  end

  defp try_classical_nlp_first(persona, input, memory, opts) do
    previous_context = get_previous_context(memory)

    if FollowupDetector.is_followup?(input, previous_context) do
      Logger.info("Detected follow-up message", %{
        input: input,
        previous_intent: previous_context[:intent]
      })

      Progress.report(opts, :followup_detected, %{
        previous_intent: previous_context[:intent],
        previous_entities: length(previous_context[:entities] || [])
      })

      handle_followup_message(persona, input, previous_context)
    else
      process_new_message(persona, input, memory, opts)
    end
  end

  defp process_new_message(persona, input, memory, opts) do
    user_id = Keyword.get(opts, :user_id)
    world_id = Keyword.get(opts, :world_id, "default")
    Process.put(:current_world_id, world_id)

    if Config.enabled?() and SelfKnowledgeAnalyzer.is_self_knowledge_query?(input) do
      Progress.report(opts, :meta_cognitive_query, %{
        query_type: :self_knowledge
      })

      handle_meta_cognitive_query(persona, input, user_id, opts)
    else
      process_standard_message(persona, input, memory, opts)
    end
  end

  defp process_standard_message(persona, input, memory, opts) do
    analysis_model =
      run_analysis_pipeline(input, memory, opts)
      |> materialize_profiles()

    Logger.debug("Analysis pipeline complete", %{
      strategy: analysis_model.overall_strategy,
      chunks: length(analysis_model.chunks),
      prompts: analysis_model.suggested_prompts
    })

    Progress.report(opts, :response_gate_start, %{})

    gate_opts = Keyword.put(opts, :conversation_memory, memory)

    case ResponseGate.evaluate(analysis_model, memory, gate_opts) do
      {:defer, reason} ->
        Logger.info("Response deferred by ResponseGate", reason)
        Progress.report(opts, :response_gate_complete, %{decision: :defer, reason: reason[:reason]})

        Progress.report(opts, :response_generated, %{
          response_type: :deferred,
          strategy: :response_optional,
          reason: reason[:reason]
        })

        context = extract_context_from_analysis(analysis_model)
        {nil, :response_deferred, Map.put(context, :defer_reason, reason)}

      {:optional, confidence, reason} ->
        Progress.report(opts, :response_gate_complete, %{
          decision: :optional,
          confidence: confidence,
          reason: reason[:reason]
        })

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
          Logger.debug("Response optional but proceeding", %{
            confidence: confidence,
            threshold: defer_threshold,
            reason: reason[:reason]
          })

          proceed_with_standard_response(persona, input, memory, analysis_model, opts)
        end

      {:respond, reason} ->
        Progress.report(opts, :response_gate_complete, %{decision: :respond, reason: reason[:reason]})
        proceed_with_standard_response(persona, input, memory, analysis_model, opts)
    end
  end

  defp proceed_with_standard_response(persona, input, memory, analysis_model, opts) do
    case analysis_model.overall_strategy do
      :needs_clarification ->
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
        Progress.report(opts, :response_generated, %{
          response_type: :acknowledgment,
          strategy: :defer_to_user
        })

        context = extract_context_from_analysis(analysis_model)
        {simple_acknowledgment(persona), :not_addressed, context}

      :cannot_respond ->
        Progress.report(opts, :response_generated, %{
          response_type: :fallback,
          strategy: :cannot_respond
        })

        context = extract_context_from_analysis(analysis_model)
        {simple_fallback_response(persona, input), :cannot_respond, context}

      _ ->
        try_nlp_with_analysis(persona, input, memory, analysis_model, opts)
    end
  end

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
    analysis_model = run_analysis_pipeline(input, [], []) |> materialize_profiles()

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
        EntityExtractor.extract_entities(input)
      end

    Logger.info("Extracted entities from follow-up", %{
      entities_count: length(entities),
      entities: Enum.map(entities, & &1.entity_type),
      used_pipeline: best_analysis != nil
    })

    carried = FollowupDetector.get_carried_context(input, previous_context)
    merged_context = FollowupDetector.merge_with_previous(carried, entities)

    Logger.info("Merged context", %{
      intent: merged_context.intent,
      all_filled: merged_context.all_required_filled,
      missing: merged_context.missing_slots
    })

    if merged_context.all_required_filled do
      response = generate_intent_response(merged_context, persona)

      context = %{
        intent: merged_context.intent,
        entities: merged_context.entities,
        slots: merged_context.slots,
        missing_slots: [],
        speech_act: Map.get(merged_context, :speech_act)
      }

      {response, :followup_completed, context}
    else
      prompt = generate_followup_clarification(merged_context)

      context = %{
        intent: merged_context.intent,
        entities: merged_context.entities,
        slots: merged_context.slots,
        missing_slots: merged_context.missing_slots,
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
        entities: union_chunk_entities(analysis_model.analyses),
        slots: (best_analysis.slots || %{}) |> extract_filled_slots(),
        missing_slots: best_analysis.missing_context || [],
        speech_act: extract_speech_act_info(best_analysis.speech_act),
        chunks: build_chunk_summaries(analysis_model.analyses),
        analysis_model: analysis_model
      }
    else
      %{}
    end
  end

  defp normalize_chunk_entity(e) do
    %{
      entity_type: e[:entity_type] || e["entity_type"] || e["type"],
      value: e[:value] || e["value"] || e["name"],
      confidence: e[:confidence] || e["confidence"] || 0.8
    }
  end

  defp normalize_chunk_entities(entities) when is_list(entities) do
    Enum.map(entities, &normalize_chunk_entity/1)
  end

  defp normalize_chunk_entities(_), do: []

  defp union_chunk_entities(analyses) when is_list(analyses) do
    analyses
    |> Enum.flat_map(fn analysis -> normalize_chunk_entities(analysis.entities || []) end)
    |> dedup_entities()
  end

  defp union_chunk_entities(_), do: []

  defp dedup_entities(entities) when is_list(entities) do
    entities
    |> Enum.reduce({[], MapSet.new()}, fn entity, {acc, seen} ->
      key = entity_dedup_key(entity)

      if MapSet.member?(seen, key) do
        {acc, seen}
      else
        {[entity | acc], MapSet.put(seen, key)}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp entity_dedup_key(%{entity_type: type, value: value}) do
    {normalize_dedup_term(type), normalize_dedup_term(value)}
  end

  defp normalize_dedup_term(nil), do: nil
  defp normalize_dedup_term(term) when is_binary(term), do: String.downcase(String.trim(term))
  defp normalize_dedup_term(term), do: term |> to_string() |> String.downcase() |> String.trim()

  defp build_chunk_summaries(analyses) when is_list(analyses) do
    Enum.map(analyses, fn analysis ->
      %{
        chunk_index: Map.get(analysis, :chunk_index),
        text: Map.get(analysis, :text),
        intent: Map.get(analysis, :intent),
        entities: normalize_chunk_entities(Map.get(analysis, :entities, [])),
        speech_act: extract_speech_act_info(Map.get(analysis, :speech_act)),
        response_strategy: Map.get(analysis, :response_strategy)
      }
    end)
  end

  defp build_chunk_summaries(_), do: []

  defp extract_speech_act_info(nil) do
    nil
  end

  defp extract_speech_act_info(speech_act) when is_map(speech_act) do
    %{
      category: Map.get(speech_act, :category),
      sub_type: Map.get(speech_act, :sub_type),
      confidence: Map.get(speech_act, :confidence),
      is_question: Map.get(speech_act, :is_question, false)
    }
  end

  defp extract_speech_act_info(_) do
    nil
  end

  defp extract_filled_slots(slots) when is_map(slots) do
    case Map.get(slots, :filled_slots) do
      nil -> slots
      filled -> filled
    end
  end

  defp extract_filled_slots(_) do
    %{}
  end

  defp generate_intent_response(context, _persona) do
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

  defp slots_to_entities(_) do
    []
  end

  defp generate_followup_clarification(context) do
    alias Brain.Response.Synthesizer

    case context.missing_slots do
      [] ->
        Synthesizer.get_generic_clarification()

      [slot | _] ->
        SlotDetector.get_clarification_prompt(slot, context.intent)
    end
  end

  defp run_analysis_pipeline(input, memory, opts) do
    history =
      memory
      |> Enum.filter(fn m ->
        Map.has_key?(m, :entities) or Map.has_key?(m, :context)
      end)
      |> Enum.take(5)
      |> Enum.map(fn m ->
        case Map.get(m, :context) do
          nil ->
            %{
              entities: Map.get(m, :entities, %{}),
              intent: Map.get(m, :intent),
              timestamp: Map.get(m, :timestamp, 0)
            }

          context ->
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

    Pipeline.process(input, pipeline_opts)
  end

  defp safe_pubsub_broadcast(topic, event, payload) when is_binary(topic) and is_binary(event) do
    if Process.whereis(ChatWeb.PubSub) != nil do
      PubSub.broadcast(ChatWeb.PubSub, topic, {String.to_atom(event), payload})
    end

    :ok
  rescue
    _ -> :ok
  end

  defp build_entities_map(entities) when is_list(entities) do
    Enum.reduce(entities, %{}, fn entity, acc ->
      entity_type = entity[:entity_type]
      entity_value = entity[:value]

      if entity_type do
        Map.put(acc, entity_type, entity_value)
      else
        acc
      end
    end)
  end

  defp build_entities_map(entities) when is_map(entities) do
    entities
  end

  defp build_entities_map(_) do
    %{}
  end

  defp try_nlp_with_analysis(persona, input, _memory, analysis_model, opts) do
    intent_bearing =
      Enum.filter(analysis_model.analyses, fn analysis ->
        analysis.intent != nil and analysis.intent != ""
      end)

    substantive_analyses =
      Enum.filter(intent_bearing, fn analysis ->
        not greeting_or_farewell?(analysis) and
          analysis.speech_act != nil and
          analysis.speech_act.category in [:directive, :assertive]
      end)

    intent_analysis =
      case substantive_analyses do
        [] -> List.first(intent_bearing)
        analyses -> ChunkPriority.select_primary(analyses)
      end

    analysis_intent = intent_analysis && intent_analysis.intent

    best_analysis =
      if intent_analysis do
        intent_analysis
      else
        respondable =
          Enum.filter(analysis_model.analyses, fn a ->
            a.response_strategy in [:can_respond, :hedged_response]
          end)

        case respondable do
          [] -> ChunkPriority.select_primary(analysis_model.analyses)
          list -> ChunkPriority.select_primary(list)
        end
      end

    analysis_entities =
      if best_analysis do
        normalize_chunk_entities(best_analysis.entities || [])
      else
        []
      end

    Logger.debug("Entity selection for intent", %{
      intent: analysis_intent,
      entities: Enum.map(analysis_entities, & &1[:entity_type]),
      chunk_index: best_analysis && Map.get(best_analysis, :chunk_index)
    })

    slots_info =
      if best_analysis do
        Map.get(best_analysis, :slots)
      else
        nil
      end

    missing_slots =
      if best_analysis do
        Map.get(best_analysis, :missing_context, [])
      else
        []
      end

    world_id = Keyword.get(opts, :world_id, "default")

    analysis_intent_confidence = get_analysis_intent_confidence(intent_analysis)

    disambiguation_opts =
      if best_analysis do
        base = [
          discourse: Map.get(best_analysis, :discourse),
          speech_act: Map.get(best_analysis, :speech_act),
          world_id: world_id,
          reuse_entities: analysis_entities
        ]

        if analysis_intent && analysis_intent != "" do
          base ++ [reuse_intent: {analysis_intent, analysis_intent_confidence}]
        else
          base
        end
      else
        [world_id: world_id]
      end

    all_analysis_entities = union_chunk_entities(analysis_model.analyses)

    num_chunks = length(analysis_model.analyses)

    Progress.report(opts, :nlp_pipeline_start, %{
      num_chunks: num_chunks,
      analysis_intent: analysis_intent,
      analysis_entity_count: length(analysis_entities)
    })

    {intent, entities, method, nlp_info} =
      if num_chunks > 1 do
        Logger.debug("Multi-chunk input: skipping NLPPipeline, unioning per-chunk entities", %{
          num_chunks: num_chunks,
          primary_entity_count: length(analysis_entities),
          union_entity_count: length(all_analysis_entities),
          selected_intent: analysis_intent
        })

        Progress.report(opts, :nlp_pipeline_complete, %{
          method: :analysis_only,
          reason: :multi_chunk,
          intent: analysis_intent,
          entities_count: length(all_analysis_entities)
        })

        {analysis_intent, all_analysis_entities, :analysis_only, %{intent: nil, confidence: nil}}
      else
        case NLPPipeline.process(input, disambiguation_opts) do
          {:ok, %{confidence: conf, intent: nlp_intent, entities: nlp_entities}} ->
            {intent, method} =
              reconcile_intents(
                analysis_intent, analysis_intent_confidence,
                nlp_intent, conf,
                input, analysis_entities
              )

            entities =
              merge_entities(analysis_entities, nlp_entities, all_analysis_entities, intent)

            Progress.report(opts, :nlp_pipeline_complete, %{
              method: method,
              nlp_confidence: conf,
              nlp_intent: nlp_intent,
              analysis_intent: analysis_intent,
              analysis_confidence: analysis_intent_confidence,
              final_intent: intent,
              entities_count: length(entities),
              reused_analysis: analysis_intent != nil and analysis_intent != ""
            })

            {intent, entities, method, %{intent: nlp_intent, confidence: conf}}

          {:error, reason} ->
            Logger.warning("NLP pipeline failed", %{reason: reason})

            Progress.report(opts, :nlp_pipeline_complete, %{
              method: :analysis_only,
              reason: :pipeline_error,
              intent: analysis_intent,
              entities_count: length(analysis_entities)
            })

            {analysis_intent, analysis_entities, :analysis_only, %{intent: nil, confidence: nil}}
        end
      end

    Logger.info("Processing with analysis", %{
      analysis_intent: analysis_intent,
      final_intent: intent,
      method: method,
      entities_count: length(entities),
      num_chunks: num_chunks
    })

    events =
      if best_analysis do
        Map.get(best_analysis, :events, [])
      else
        []
      end

    ConsistencyChecker.check_and_report(%{
      text: input,
      final_intent: intent,
      analysis_intent: analysis_intent,
      analysis_confidence: best_analysis && best_analysis.confidence,
      nlp_intent: nlp_info.intent,
      nlp_confidence: nlp_info.confidence,
      events: events
    }, opts)

    speech_act_info =
      if(best_analysis) do
        extract_speech_act_info(best_analysis.speech_act)
      else
        nil
      end

    tier3_context = extract_tier3_context(analysis_model)

    context = %{
      intent: intent,
      entities: entities,
      slots: extract_filled_slots(slots_info),
      missing_slots: missing_slots,
      speech_act: speech_act_info,
      chunks: build_chunk_summaries(analysis_model.analyses),
      tier3_models: tier3_context,
      analysis_model: analysis_model
    }

    analysis_for_learning = %{
      entities: entities,
      speech_act: speech_act_info,
      intent: intent
    }

    Learner.learn_from_conversation(persona.name, input, analysis_for_learning)

    Progress.report(opts, :learning_complete, %{
      intent: intent,
      entities_count: length(entities)
    })

    conversation_id = Keyword.get(opts, :conversation_id)
    track_stance(conversation_id, intent, best_analysis, opts)

    {response, response_type} =
      generate_analysis_response_with_type(intent, entities, analysis_model, persona, input, opts)

    Progress.report(opts, :response_generated, %{
      response_type: response_type,
      strategy: :can_respond,
      method: method,
      intent: intent,
      entities_count: length(entities),
      response_path: build_response_path(method, response_type, intent, entities)
    })

    {response, method, context}
  end

  defp get_analysis_intent_confidence(nil), do: 0.0
  defp get_analysis_intent_confidence(analysis) do
    case analysis.speech_act do
      %{intent_confidence: c} when is_number(c) -> c
      _ -> analysis.confidence || 0.0
    end
  end

  defp reconcile_intents(nil, _analysis_conf, nlp_intent, nlp_conf, _text, _entities) do
    method = if NLPPipeline.should_use_classical_result?(nlp_conf),
      do: :classical, else: :classical_low_confidence
    {nlp_intent, method}
  end

  defp reconcile_intents(analysis_intent, _analysis_conf, nil, _nlp_conf, _text, _entities) do
    {analysis_intent, :analysis_only}
  end

  defp reconcile_intents(analysis_intent, _analysis_conf, nlp_intent, _nlp_conf, _text, _entities)
       when analysis_intent == nlp_intent do
    {analysis_intent, :analysis_enhanced}
  end

  defp reconcile_intents(analysis_intent, analysis_conf, nlp_intent, nlp_conf, _text, _entities) do
    confidence_based_fallback(analysis_intent, analysis_conf, nlp_intent, nlp_conf)
  end

  defp confidence_based_fallback(analysis_intent, analysis_conf, nlp_intent, nlp_conf) do
    if analysis_conf >= nlp_conf do
      {analysis_intent, :analysis_enhanced}
    else
      {nlp_intent, :classical_preferred}
    end
  end

  defp merge_entities(analysis_entities, nlp_entities, all_analysis_entities, intent) do
    analysis_types = Enum.map(analysis_entities, & &1.entity_type) |> MapSet.new()

    allowed_types =
      if intent do
        SlotDetector.get_entity_types_for_intent(intent)
      else
        MapSet.new()
      end

    all_entities_for_conflict_check =
      if all_analysis_entities != [] do
        all_analysis_entities
      else
        analysis_entities
      end

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
        same_type = MapSet.member?(analysis_types, e_type)
        value_conflict = MapSet.member?(analysis_values, e_value) and not same_type

        type_not_allowed =
          MapSet.size(allowed_types) > 0 and not MapSet.member?(allowed_types, e_type)

        same_type or value_conflict or type_not_allowed
      end)

    analysis_entities ++ unique_nlp
  end

  defp extract_tier3_context(analysis_model) do
    analyses = Map.get(analysis_model, :analyses, [])

    %{
      event_frames: Enum.flat_map(analyses, &Map.get(&1, :event_frames, [])),
      srl_frames: Enum.flat_map(analyses, &Map.get(&1, :srl_frames, [])),
      poincare_available: Brain.ML.Poincare.Embeddings.ready?(),
      kg_lstm_available: Brain.ML.KnowledgeGraph.TripleScorer.ready?(),
      stance_tracker_available: Brain.Epistemic.StanceTracker.ready?()
    }
  rescue
    _ -> %{}
  end

  defp track_stance(nil, _intent, _analysis, _opts), do: :ok
  defp track_stance(_conv_id, nil, _analysis, _opts), do: :ok

  defp track_stance(conversation_id, intent, analysis, _opts) do
    if Brain.Epistemic.StanceTracker.ready?() do
      sentiment = (analysis && analysis.sentiment) || %{}
      label = Map.get(sentiment, :label, :neutral)
      score = Map.get(sentiment, :confidence, 0.5)

      position = case label do
        :positive -> score
        :negative -> -score
        _ -> 0.0
      end

      topic = intent_to_topic(intent)

      if topic != nil do
        Brain.Epistemic.StanceTracker.record_stance(
          conversation_id, topic, position, :user
        )
      end
    end

    :ok
  rescue
    _ -> :ok
  end

  defp intent_to_topic(nil), do: nil
  defp intent_to_topic(intent) when is_binary(intent) do
    case String.split(intent, ".") do
      [domain | _] when domain != "" -> domain
      _ -> nil
    end
  end
  defp intent_to_topic(_), do: nil

  defp build_clarification_response(prompts, _persona) do
    alias Brain.Response.Synthesizer

    case prompts do
      [] ->
        Synthesizer.get_generic_clarification()

      [single_prompt] ->
        single_prompt

      multiple ->
        first = List.first(multiple)
        rest_count = length(multiple) - 1

        "#{first} (I also have #{rest_count} more question#{if rest_count > 1 do
          "s"
        else
          ""
        end})"
    end
  end

  defp build_clarification_addendum(prompts) do
    alias Brain.Response.Synthesizer

    case prompts do
      [] ->
        ""

      [single_prompt] ->
        transition = Synthesizer.get_transition_phrase(:additional_info)

        "#{transition} #{String.downcase(String.first(single_prompt))}#{String.slice(single_prompt, 1..-1//1)}"

      [first | _rest] ->
        transition = Synthesizer.get_transition_phrase(:additional_info)
        "#{transition} #{String.downcase(String.first(first))}#{String.slice(first, 1..-1//1)}"
    end
  end

  defp simple_acknowledgment(_persona) do
    Brain.Response.Synthesizer.get_defer_response()
  end

  defp generate_analysis_response_with_type(
         intent,
         entities,
         analysis_model,
         _persona,
         query_text,
         opts
       ) do
    unified_context =
      Brain.Response.ContextBuilder.build(analysis_model, opts)

    gen_opts = %{
      user_id: Keyword.get(opts, :user_id),
      conversation_id: Keyword.get(opts, :conversation_id),
      unified_context: unified_context,
      dry_run_ouro: Keyword.get(opts, :dry_run_ouro, false)
    }

    case Generator.generate_via_synthesis(analysis_model, intent, entities, query_text, gen_opts) do
      {{:ouro_dry_run, _messages} = response, :ouro_dry_run} ->
        {response, :ouro_dry_run}

      {response, method} when is_binary(response) ->
        {response, method}

      {nil, :silence_preferred} ->
        {nil, :silence_preferred}

      {:error, reason} ->
        Logger.warning("Synthesis failed, falling back: #{inspect(reason)}")
        {Brain.Response.Synthesizer.get_cannot_respond_response(), :synthesis_fallback}
    end
  end

  defp simple_fallback_response(_persona, _input) do
    Brain.Response.Synthesizer.get_cannot_respond_response()
  end

  defp build_response_path(method, response_type, intent, entities) do
    steps = []

    steps =
      steps ++
        [
          %{
            step: 1,
            name: "Brain.evaluate",
            status: :completed,
            detail: "Received user input"
          }
        ]

    analysis_detail =
      case method do
        :analysis_only -> "Multi-chunk: used per-chunk analysis only"
        :analysis_enhanced -> "Single-chunk: NLPPipeline enhanced analysis"
        :classical_low_confidence -> "NLP had low confidence, used analysis"
        _ -> "Standard analysis pipeline"
      end

    steps =
      steps ++
        [
          %{
            step: 2,
            name: "try_nlp_with_analysis",
            status: :completed,
            detail: analysis_detail,
            method: method
          }
        ]

    steps =
      steps ++
        [
          %{
            step: 3,
            name: "Intent Classification",
            status:
              if(intent) do
                :completed
              else
                :skipped
              end,
            detail:
              if(intent) do
                "Classified as: #{intent}"
              else
                "No intent detected"
              end,
            intent: intent
          }
        ]

    entity_count = length(entities)

    steps =
      steps ++
        [
          %{
            step: 4,
            name: "Entity Extraction",
            status:
              if(entity_count > 0) do
                :completed
              else
                :skipped
              end,
            detail: "Extracted #{entity_count} entities",
            entities:
              Enum.map(entities, fn e ->
                %{type: e[:entity_type], value: e[:value]}
              end)
          }
        ]

    {gen_steps, gen_status} =
      case response_type do
        :synthesized ->
          {[
             %{
               handler: :synthesizer,
               tried: true,
               selected: true,
               reason: "Synthesized from domain knowledge (priv/knowledge/domains/)"
             },
             %{
               handler: :memory_augmented,
               tried: false,
               selected: false,
               reason: "Skipped (synthesized)"
             },
             %{handler: :template, tried: false, selected: false, reason: "Skipped (synthesized)"}
           ], "Generative synthesis from domain knowledge"}

        :memory_adapted ->
          {[
             %{
               handler: :synthesizer,
               tried: true,
               selected: false,
               reason: "No domain knowledge match"
             },
             %{
               handler: :memory_augmented,
               tried: true,
               selected: true,
               reason: "Adapted from similar past episodes"
             },
             %{
               handler: :template,
               tried: false,
               selected: false,
               reason: "Skipped (memory adapted)"
             }
           ], "Memory-adapted response from past episodes"}

        :special_handler ->
          {[
             %{
               handler: :synthesizer,
               tried: true,
               selected: false,
               reason: "No domain knowledge match"
             },
             %{
               handler: :memory_augmented,
               tried: true,
               selected: false,
               reason: "No similar episodes"
             },
             %{
               handler: :special_handler,
               tried: true,
               selected: true,
               reason: "Special handler (code/factual) matched"
             }
           ], "Special handler response (code/factual)"}

        :quality_improved ->
          {[
             %{
               handler: :quality_check,
               tried: true,
               selected: true,
               reason: "Response quality improved"
             },
             %{
               handler: :refinement,
               tried: true,
               selected: true,
               reason: "Poor quality detected, response enhanced"
             }
           ], "Quality-improved response"}

        :domain ->
          {[
             %{
               handler: :domain,
               tried: true,
               selected: true,
               reason: "Domain handler matched intent"
             },
             %{
               handler: :memory_augmented,
               tried: false,
               selected: false,
               reason: "Skipped (domain handled)"
             },
             %{
               handler: :template,
               tried: false,
               selected: false,
               reason: "Skipped (domain handled)"
             }
           ], "Domain-specific handler"}

        :memory_augmented ->
          {[
             %{
               handler: :domain,
               tried: true,
               selected: false,
               reason: "No domain handler for intent"
             },
             %{
               handler: :memory_augmented,
               tried: true,
               selected: true,
               reason: "Similar episodes found in Memory.Store"
             },
             %{
               handler: :template,
               tried: false,
               selected: false,
               reason: "Skipped (memory handled)"
             }
           ], "Memory-augmented response"}

        :template ->
          {[
             %{
               handler: :synthesizer,
               tried: true,
               selected: false,
               reason: "No domain knowledge match"
             },
             %{
               handler: :memory_augmented,
               tried: true,
               selected: false,
               reason: "No similar episodes found"
             },
             %{
               handler: :template,
               tried: true,
               selected: true,
               reason: "Template found in TemplateStore"
             }
           ], "Template-based response"}

        :conditional_template ->
          {[
             %{
               handler: :synthesizer,
               tried: true,
               selected: false,
               reason: "No domain knowledge match"
             },
             %{
               handler: :conditional_template,
               tried: true,
               selected: true,
               reason: "Condition matched, semantic ranking applied"
             }
           ], "Conditional template with semantic ranking"}

        :blended ->
          {[
             %{
               handler: :synthesizer,
               tried: true,
               selected: false,
               reason: "No domain knowledge match"
             },
             %{
               handler: :conditional_template,
               tried: true,
               selected: false,
               reason: "No matching conditions"
             },
             %{
               handler: :template_blender,
               tried: true,
               selected: true,
               reason: "Blended chunks from multiple templates"
             }
           ], "Template blending"}

        :smalltalk ->
          {[
             %{
               handler: :expressive,
               tried: true,
               selected: true,
               reason: "Expressive speech act (greeting/farewell/etc)"
             }
           ], "Expressive/smalltalk response"}

        :expressive ->
          {[
             %{handler: :expressive, tried: true, selected: true, reason: "Expressive speech act"}
           ], "Expressive response"}

        :fallback ->
          {[
             %{
               handler: :synthesizer,
               tried: true,
               selected: false,
               reason: "No domain knowledge match"
             },
             %{
               handler: :memory_augmented,
               tried: true,
               selected: false,
               reason: "No similar episodes found"
             },
             %{
               handler: :template,
               tried: true,
               selected: false,
               reason: "No template for intent"
             },
             %{handler: :fallback, tried: true, selected: true, reason: "All handlers exhausted"}
           ], "Fallback response"}

        other ->
          {[
             %{
               handler: :unknown,
               tried: true,
               selected: true,
               reason: "Untracked response type: #{inspect(other)}"
             }
           ], "Unknown (#{inspect(other)})"}
      end

    steps =
      steps ++
        [
          %{
            step: 5,
            name: "Response Generation",
            status: :completed,
            detail: gen_status,
            response_type: response_type,
            handlers_tried: gen_steps
          }
        ]

    %{
      steps: steps,
      final_handler: response_type,
      method: method,
      intent: intent,
      entity_count: entity_count
    }
  end

  defp generate_conversation_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp generate_message_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp create_learning_summary(input, response) do
    "User: #{String.slice(input, 0, 50)}... | Assistant: #{String.slice(response, 0, 50)}..."
  end

  defp store_in_cognitive_memory(entries) do
    if Process.whereis(Brain.Memory.Store) != nil do
      Enum.each(entries, fn entry ->
        tags = extract_tags_for_memory(entry.input)
        world_id = Map.get(entry, :world_id, "default")

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
    case Brain.ML.NLPPipeline.process(input) do
      {:ok, %{intent: intent}} when is_binary(intent) and intent != "unknown" ->
        [intent]

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp handle_meta_cognitive_query(_persona, input, user_id, _opts) do
    Logger.info("Handling meta-cognitive query", %{input: input, user_id: user_id})
    assessment = SelfKnowledgeAnalyzer.build_self_knowledge_assessment(user_id)

    response =
      Synthesizer.synthesize_self_knowledge_response(assessment,
        context: %{
          relationship_duration: :new,
          user_initiated: true
        }
      )

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
    user_id = user_id || "anonymous"

    if Config.auto_extraction_enabled?() do
      Enum.each(entities, fn entity ->
        entity_type = entity[:entity_type]
        entity_value = entity[:value]

        if entity_type && entity_value do
          if is_user_fact?(entity_type) do
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

      extract_self_referential_facts(input, user_id, conversation_id)
    end
  rescue
    e ->
      Logger.warning("Failed to extract beliefs: #{inspect(e)}")
  end

  defp feed_entities_to_world(entities, world_id) when is_list(entities) do
    promotable_types = ~w(person location city country organization company place)
    now = DateTime.utc_now()

    Enum.each(entities, fn entity ->
      value = entity[:value] || entity["value"]
      type = entity[:entity_type] || entity["entity_type"] || "unknown"
      type_str = to_string(type) |> String.downcase()

      if value &&
           (type_str in promotable_types or type_str == "unknown") &&
           String.length(to_string(value)) >= 2 do
        candidate = %{
          value: to_string(value),
          inferred_type: type_str,
          confidence: entity[:confidence] || entity["confidence"] || 0.5,
          discovered_at: now,
          occurrences: 1
        }

        try do
          WorldManager.add_candidate(world_id, candidate)
        rescue
          e ->
            Logger.debug("Failed to add entity candidate to world: #{inspect(e)}")
        end
      end
    end)

    :ok
  end

  defp feed_entities_to_world(_, _), do: :ok

  defp is_user_fact?(entity_type) do
    entity_type_str = to_string(entity_type) |> String.downcase()

    case Brain.ML.MicroClassifiers.classify(:user_fact_type, entity_type_str) do
      {:ok, "user_specific", score} when score > 0.3 -> true
      _ -> false
    end
  end

  defp normalize_predicate(predicate) when is_atom(predicate) do
    predicate
  end

  defp normalize_predicate(predicate) when is_binary(predicate) do
    predicate
    |> String.downcase()
    |> String.replace([" ", "-"], "_")
    |> String.to_atom()
  end

  defp normalize_predicate(_) do
    :unknown
  end

  defp extract_self_referential_facts(input, user_id, conversation_id) do
    expanded = Tokenizer.expand_contractions(input)
    tokens = Tokenizer.tokenize_normalized(expanded)
    extract_location_facts(tokens, expanded, user_id, conversation_id)
    extract_name_facts(tokens, expanded, user_id, conversation_id)
    extract_preference_facts(tokens, expanded, user_id, conversation_id)
    extract_work_facts(tokens, expanded, user_id, conversation_id)
  end

  defp extract_location_facts(tokens, _input, user_id, conversation_id) do
    cond do
      has_sequence?(tokens, ["i", "am", "from"]) ->
        value = extract_after_sequence(tokens, ["from"])
        store_fact_if_valid(user_id, :location, value, conversation_id)

      has_sequence?(tokens, ["i", "live", "in"]) ->
        value = extract_after_sequence(tokens, ["in"])
        store_fact_if_valid(user_id, :location, value, conversation_id)

      true ->
        :ok
    end
  end

  defp extract_name_facts(tokens, _input, user_id, conversation_id) do
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
    sequence_len = length(sequence)

    tokens
    |> Enum.chunk_every(sequence_len, 1, :discard)
    |> Enum.any?(&(&1 == sequence))
  end

  defp extract_after_sequence(tokens, marker_sequence) do
    marker_len = length(marker_sequence)

    case find_sequence_index(tokens, marker_sequence) do
      nil ->
        nil

      idx ->
        tokens
        |> Enum.drop(idx + marker_len)
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
      if chunk == sequence do
        idx
      else
        nil
      end
    end)
  end

  defp store_fact_if_valid(user_id, predicate, value, conversation_id) do
    clean_value =
      if value do
        String.trim(value)
      else
        ""
      end

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

  defp materialize_profiles(analysis_model) do
    updated_analyses =
      Enum.map(analysis_model.analyses, fn analysis ->
        if analysis.profile != nil do
          analysis
        else
          {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)
          profile = ChunkProfile.materialize(analysis, feature_vector)

          intent =
            case analysis.intent do
              nil -> profile.derived_label
              "" -> profile.derived_label
              existing -> existing
            end

          %{analysis | profile: profile, feature_vector: feature_vector, intent: intent}
        end
      end)

    %{analysis_model | analyses: updated_analyses}
  rescue
    e ->
      Logger.warning("Profile materialization failed: #{Exception.message(e)}")
      analysis_model
  end

  defp greeting_or_farewell?(analysis) do
    cond do
      analysis.profile != nil ->
        analysis.profile.speech_act_subtype in [:greeting, :farewell]

      true ->
        intent = to_string(analysis.intent || "")
        String.starts_with?(intent, "greeting") or String.starts_with?(intent, "farewell")
    end
  end

  defp build_interpretation_from_context(input, context) do
    intent = Map.get(context, :intent)
    activation = Map.get(context, :activation, 0.7)
    source = Map.get(context, :source, :pipeline)
    entities = Map.get(context, :entities, [])

    source_atom =
      case source do
        :heuristic_match -> :heuristic
        :memory_match -> :memory_match
        :pattern_recognition -> :pattern_recognition
        _ -> :model
      end

    Interpretation.new(intent, input, activation, source_atom)
    |> Interpretation.with_entities(entities)
  end
end
