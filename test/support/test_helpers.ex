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

      {:error, reason} ->
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
end
