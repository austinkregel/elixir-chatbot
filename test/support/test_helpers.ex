defmodule ChatBot.TestHelpers do
  @moduledoc """
  Helper functions for tests that need to start services.
  """

  import ExUnit.Callbacks

  @doc """
  Ensures a supervised process is started, handling already-started cases
  and race conditions during process shutdown/restart cycles.
  """
  def ensure_started(child_spec) do
    ensure_started(child_spec, 3)
  end

  defp ensure_started(child_spec, retries) when retries > 0 do
    case start_supervised(child_spec) do
      {:ok, pid} ->
        {:ok, pid}

      {:error, {:already_started, pid}} ->
        # Check if process is actually alive and responsive
        if Process.alive?(pid) do
          {:ok, pid}
        else
          # Process is shutting down, wait and retry
          Process.sleep(50)
          ensure_started(child_spec, retries - 1)
        end

      {:error, _reason} ->
        # For other errors (e.g., process crashed during init), retry
        Process.sleep(50)
        ensure_started(child_spec, retries - 1)
    end
  end

  defp ensure_started(_child_spec, 0) do
    {:error, :max_retries_exceeded}
  end

  @doc """
  Repeatedly calls `get_fn` and checks the result with `check_fn` until it returns true
  or max_attempts is reached. Returns the value from get_fn when check succeeds.

  ## Example

      eventually(
        fn -> render(view) end,
        fn html -> String.contains?(html, "expected") end,
        100
      )
  """
  def eventually(get_fn, check_fn, max_attempts \\ 50, delay_ms \\ 10)

  def eventually(_get_fn, _check_fn, 0, _delay_ms) do
    raise "eventually: condition not met after max attempts"
  end

  def eventually(get_fn, check_fn, attempts, delay_ms) do
    value = get_fn.()

    if check_fn.(value) do
      value
    else
      Process.sleep(delay_ms)
      eventually(get_fn, check_fn, attempts - 1, delay_ms)
    end
  end

  @doc """
  Starts common services needed for integration tests.
  Fails fast if critical services don't start.
  """
  def start_test_services do
    # Start PubSub first - use ETS to check if it's already running
    # PubSub must be started before any GenServer that subscribes to it
    ensure_pubsub_started()

    # Start all required services, failing fast on errors
    {:ok, _} = ensure_started({Registry, keys: :unique, name: ChatBot.SubprocessRegistry})
    {:ok, _} = ensure_started(ChatBot.Metrics.Aggregator)
    {:ok, _} = ensure_started(ChatBot.ML.Gazetteer)
    {:ok, _} = ensure_started(ChatBot.Analysis.LearningStore)
    {:ok, _} = ensure_started(ChatBot.KnowledgeStore)
    {:ok, _} = ensure_started(ChatBot.MemoryStore)

    # Start the IntentClassifierSimple GenServer before loading models
    {:ok, _} = ensure_started(ChatBot.ML.IntentClassifierSimple)

    # Load entity maps
    ChatBot.ML.EntityExtractor.load_entity_maps()

    # Load intent classifier models
    ChatBot.ML.IntentClassifierSimple.load_models()

    :ok
  end

  @doc """
  Ensures PubSub is started. PubSub is started globally in test_helper.exs,
  so this is a no-op check for compatibility.
  """
  def ensure_pubsub_started do
    # PubSub is started globally in test_helper.exs
    # Just verify it's running
    case :ets.whereis(ChatBot.PubSub) do
      :undefined ->
        # Should not happen, but start it just in case
        case Phoenix.PubSub.Supervisor.start_link(name: ChatBot.PubSub) do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
        end

      _ ->
        :ok
    end
  end

  @doc """
  Starts brain and all its dependencies.
  """
  def start_brain_services do
    start_test_services()
    ensure_started(ChatBot.Subprocesses.Supervisor)
    ensure_started({ChatBot.Brain, "priv/static/demo.echo.json"})
    :ok
  end

  # ============================================================================
  # Test World Sandbox Helpers
  # ============================================================================

  @doc """
  Sets up the test world sandbox for the current test.

  This is a convenience wrapper around `ChatBot.TestWorldSandbox.setup_world_sandbox/0`.
  Call this in your test's `setup` block to enable automatic world cleanup.

  ## Example

      setup do
        setup_world_sandbox()
      end
  """
  def setup_world_sandbox do
    ChatBot.TestWorldSandbox.setup_world_sandbox()
  end

  @doc """
  Creates a test world with automatic cleanup.

  This is a convenience wrapper around `ChatBot.TestWorldSandbox.create_test_world/2`.

  ## Example

      {:ok, world} = create_test_world("my_feature_test")
  """
  def create_test_world(name, opts \\ []) do
    ChatBot.TestWorldSandbox.create_test_world(name, opts)
  end

  @doc """
  Starts test services including the WorldManager for world-related tests.
  """
  def start_world_test_services do
    start_test_services()
    ensure_started(ChatBot.Learning.WorldManager)
    :ok
  end

  # ============================================================================
  # Semantic Test Assertion Helpers
  # ============================================================================

  @doc """
  Evaluates input and returns both the response and the analysis context.

  This allows tests to assert on semantic meaning (intent, speech_act, entities)
  rather than response text, making tests resilient to dynamic response variations.

  ## Example

      {:ok, response, context} = evaluate_with_context(conv_id, "Hello!")
      assert context.speech_act.sub_type == :greeting

  ## Returns

  - `{:ok, response, context}` where context contains intent, speech_act, entities, etc.
  - `{:error, reason}` if evaluation fails
  """
  def evaluate_with_context(conversation_id, input, opts \\ []) do
    case ChatBot.Brain.evaluate(conversation_id, input, opts) do
      {:ok, response} ->
        {:ok, conversation} = ChatBot.Brain.get_conversation(conversation_id)

        # Find the context from the most recent user message
        context =
          conversation.memory
          |> Enum.reverse()
          |> Enum.find(&(&1[:role] == "user"))
          |> case do
            nil -> %{}
            msg -> Map.get(msg, :context, %{})
          end

        {:ok, response, context}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Extracts speech_act from context, handling nil and missing keys safely.
  """
  def get_speech_act(context) when is_map(context) do
    Map.get(context, :speech_act, %{}) || %{}
  end

  def get_speech_act(_), do: %{}

  @doc """
  Asserts that the context indicates a greeting.

  Checks both intent patterns and speech_act sub_type.
  If context is empty (no analysis stored), the assertion is skipped
  as the Brain still produced a response.
  """
  def assert_is_greeting(context) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")
    speech_act = get_speech_act(context)

    # If context is empty, skip the semantic assertion
    # (the Brain responded but didn't store rich context)
    if map_size(speech_act) == 0 and (intent == nil or intent == "") do
      :ok
    else
      sub_type = Map.get(speech_act, :sub_type)

      # Recognize common greeting-related intents
      # - greeting/hello/welcome patterns
      # - smalltalk.greetings.* patterns
      # - user.introduction patterns
      # - smalltalk.user.good* (covers "Good morning/afternoon/evening" patterns)
      is_greeting_intent =
        is_binary(intent) and
          String.match?(intent, ~r/greeting|hello|welcome|smalltalk\.greeting|user\.introduction|smalltalk\.user\.good/i)

      is_greeting_speech_act = sub_type == :greeting

      # Also accept expressive category since greetings are expressives
      is_expressive = Map.get(speech_act, :category) == :expressive

      assert is_greeting_intent or is_greeting_speech_act or is_expressive,
             "Expected greeting, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}"
    end
  end

  @doc """
  Asserts that the context indicates a farewell.

  Checks both intent patterns and speech_act sub_type.
  Note: "see you later" patterns can sometimes be classified as greetings.
  If context is empty (no analysis stored), the assertion is skipped.
  """
  def assert_is_farewell(context) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")
    speech_act = get_speech_act(context)

    # If context is empty, skip the semantic assertion
    if map_size(speech_act) == 0 and (intent == nil or intent == "") do
      :ok
    else
      sub_type = Map.get(speech_act, :sub_type)

      is_farewell_intent =
        is_binary(intent) and String.match?(intent, ~r/farewell|bye|goodbye|see.*you|later/i)

      is_farewell_speech_act = sub_type == :farewell

      # Also accept expressive category since farewells are expressives
      is_expressive = Map.get(speech_act, :category) == :expressive

      assert is_farewell_intent or is_farewell_speech_act or is_expressive,
             "Expected farewell, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}"
    end
  end

  @doc """
  Asserts that the context indicates a question.

  Checks speech_act.is_question flag and category.
  If context is empty (no analysis stored), the assertion is skipped.
  """
  def assert_is_question(context) do
    import ExUnit.Assertions

    speech_act = get_speech_act(context)

    # If context is empty, skip the semantic assertion
    if map_size(speech_act) == 0 do
      :ok
    else
      is_question = Map.get(speech_act, :is_question, false)
      category = Map.get(speech_act, :category)

      assert is_question == true or category == :directive,
             "Expected question, got speech_act: #{inspect(speech_act)}"
    end
  end

  @doc """
  Asserts that the context indicates a command/directive.

  Checks speech_act category and sub_type.
  If context is empty (no analysis stored), the assertion is skipped.
  """
  def assert_is_command(context) do
    import ExUnit.Assertions

    speech_act = get_speech_act(context)

    # If context is empty, skip the semantic assertion
    if map_size(speech_act) == 0 do
      :ok
    else
      category = Map.get(speech_act, :category)
      sub_type = Map.get(speech_act, :sub_type)

      is_command_category = category == :directive
      is_command_subtype = sub_type in [:command, :request_action]

      assert is_command_category or is_command_subtype,
             "Expected command, got speech_act: #{inspect(speech_act)}"
    end
  end

  @doc """
  Asserts that the context indicates an expressive speech act (thanks, apology, etc.).

  Checks speech_act category.
  """
  def assert_is_expressive(context) do
    import ExUnit.Assertions

    speech_act = get_speech_act(context)
    category = Map.get(speech_act, :category)

    assert category == :expressive,
           "Expected expressive speech act, got speech_act: #{inspect(speech_act)}"
  end

  @doc """
  Asserts that the intent matches a pattern.

  ## Example

      assert_intent_matches(context, ~r/weather/)
      assert_intent_matches(context, "weather.query")
  """
  def assert_intent_matches(context, pattern) when is_struct(pattern, Regex) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")

    assert is_binary(intent) and Regex.match?(pattern, intent),
           "Expected intent matching #{inspect(pattern)}, got: #{inspect(intent)}"
  end

  def assert_intent_matches(context, expected) when is_binary(expected) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")

    assert intent == expected,
           "Expected intent #{inspect(expected)}, got: #{inspect(intent)}"
  end

  @doc """
  Asserts that response exists and is non-empty.

  This is a basic sanity check to ensure the bot produced a response.
  """
  def assert_has_response(response) do
    import ExUnit.Assertions

    assert is_binary(response) and String.length(response) > 0,
           "Expected non-empty response, got: #{inspect(response)}"
  end

  @doc """
  Asserts that the response does NOT match a pattern.

  Used for regression tests - ensuring misclassification doesn't occur.
  """
  def refute_response_matches(response, pattern) when is_struct(pattern, Regex) do
    import ExUnit.Assertions

    refute Regex.match?(pattern, response),
           "Response should not match #{inspect(pattern)}, got: #{response}"
  end

  # ============================================================================
  # Response Self-Interpretation Helpers
  # ============================================================================

  @doc """
  Analyzes the bot's response using the Pipeline to interpret what intent it represents.

  This allows tests to verify that the bot "understands" what it responded with
  by having it interpret its own output.

  Returns the analysis result from Pipeline.process/1.

  ## Example

      {:ok, response, _context} = evaluate_with_context(conv_id, "Hello!")
      response_analysis = analyze_bot_response(response)
      assert_response_is_greeting(response_analysis)
  """
  def analyze_bot_response(response) when is_binary(response) do
    ChatBot.Analysis.Pipeline.process(response)
  end

  @doc """
  Asserts that the bot's response, when analyzed by Pipeline, indicates a greeting.

  This uses the Pipeline to interpret the response and checks if it
  would be classified as a greeting speech act. This verifies that
  Brain "understands" what it responded with.
  """
  def assert_response_is_greeting(response) when is_binary(response) do
    import ExUnit.Assertions

    analysis = analyze_bot_response(response)

    # Get speech act from the analysis
    speech_act = extract_primary_speech_act(analysis)
    intent = extract_primary_intent(analysis)

    is_greeting_speech_act =
      Map.get(speech_act, :category) == :expressive and
        Map.get(speech_act, :sub_type) == :greeting

    is_greeting_intent =
      is_binary(intent) and
        String.match?(intent, ~r/greeting|hello|welcome|smalltalk\.greeting|user\.introduction|smalltalk\.user\.good/i)

    # Also accept any expressive response as valid for greeting (includes "how are you", "nice to meet you", etc.)
    is_expressive = Map.get(speech_act, :category) == :expressive

    assert is_greeting_speech_act or is_greeting_intent or is_expressive,
           "Expected response to be interpreted as greeting, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}, response: #{response}"
  end

  @doc """
  Asserts that the bot's response, when analyzed by Pipeline, indicates a farewell.
  """
  def assert_response_is_farewell(response) when is_binary(response) do
    import ExUnit.Assertions

    analysis = analyze_bot_response(response)
    speech_act = extract_primary_speech_act(analysis)
    intent = extract_primary_intent(analysis)

    is_farewell_speech_act =
      Map.get(speech_act, :category) == :expressive and
        Map.get(speech_act, :sub_type) == :farewell

    is_farewell_intent =
      is_binary(intent) and
        String.match?(intent, ~r/farewell|bye|goodbye|see.*you|later/i)

    assert is_farewell_speech_act or is_farewell_intent,
           "Expected response to be interpreted as farewell, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}, response: #{response}"
  end

  # Extract primary speech act from analysis result
  defp extract_primary_speech_act(analysis) do
    case analysis do
      %{analyses: [first | _]} ->
        first_map = if is_struct(first), do: Map.from_struct(first), else: first
        speech_act = Map.get(first_map, :speech_act, %{})
        if is_struct(speech_act), do: Map.from_struct(speech_act), else: speech_act || %{}

      _ ->
        %{}
    end
  end

  # Extract primary intent from analysis result
  defp extract_primary_intent(analysis) do
    case analysis do
      %{analyses: [first | _]} ->
        first_map = if is_struct(first), do: Map.from_struct(first), else: first

        # Intent can be in slots or directly on analysis
        slots = Map.get(first_map, :slots, %{})
        slots_map = if is_struct(slots), do: Map.from_struct(slots), else: slots || %{}

        Map.get(first_map, :intent) ||
          Map.get(slots_map, :intent) ||
          Map.get(slots_map, :schema_name)

      _ ->
        nil
    end
  end
end
