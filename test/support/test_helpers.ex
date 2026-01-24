defmodule ChatBot.TestHelpers do
  @moduledoc """
  Helper functions for tests that need to start services.
  """

  import ExUnit.Callbacks

  @doc """
  Ensures a supervised process is started, handling already-started cases.
  """
  def ensure_started(child_spec) do
    case start_supervised(child_spec) do
      {:ok, pid} -> {:ok, pid}
      {:error, {:already_started, pid}} -> {:ok, pid}
      {:error, reason} -> {:error, reason}
    end
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
  """
  def start_test_services do
    ensure_started({Phoenix.PubSub, name: ChatBot.PubSub})
    ensure_started({Registry, keys: :unique, name: ChatBot.SubprocessRegistry})
    # Start the Metrics Aggregator (for telemetry collection)
    ensure_started(ChatBot.Metrics.Aggregator)
    ensure_started(ChatBot.ML.Gazetteer)
    ensure_started(ChatBot.Analysis.LearningStore)
    ensure_started(ChatBot.KnowledgeStore)
    ensure_started(ChatBot.MemoryStore)

    # Load entity maps
    ChatBot.ML.EntityExtractor.load_entity_maps()

    # Load intent classifier models
    ChatBot.ML.IntentClassifierSimple.load_models()

    :ok
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
end
