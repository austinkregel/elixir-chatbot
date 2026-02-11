defmodule Brain.ML.TrainingExampleBuffer do
  @moduledoc """
  ETS-backed buffer for training examples generated from conversation outcomes.

  Accumulates {text, intent} examples and flushes them as incremental updates
  to the IntentClassifierSimple when the buffer reaches a threshold.

  - Flushes at 50+ examples
  - Minimum 1 hour between flushes
  - Uses IntentClassifierSimple.incremental_update/2 for the actual model update
  """

  use GenServer
  require Logger

  @ets_table :training_example_buffer
  @flush_threshold 50
  @min_flush_interval_ms 60 * 60 * 1000
  @check_interval_ms 5 * 60 * 1000

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Adds a training example to the buffer."
  def add_example(text, intent, name \\ __MODULE__) when is_binary(text) and is_binary(intent) do
    GenServer.cast(name, {:add_example, text, intent})
  end

  @doc "Returns current buffer stats."
  def stats(name \\ __MODULE__) do
    GenServer.call(name, :stats)
  end

  @doc "Force flush the buffer regardless of thresholds."
  def flush(name \\ __MODULE__) do
    GenServer.call(name, :flush)
  end

  @doc "Checks if the GenServer is ready."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @impl true
  def init(_opts) do
    :ets.new(@ets_table, [:named_table, :bag, :public])

    Process.send_after(self(), :check_flush, @check_interval_ms)

    {:ok,
     %{
       total_added: 0,
       total_flushed: 0,
       last_flush: nil,
       flush_count: 0
     }}
  end

  @impl true
  def handle_cast({:add_example, text, intent}, state) do
    :ets.insert(@ets_table, {intent, text, System.system_time(:millisecond)})
    {:noreply, %{state | total_added: state.total_added + 1}}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    buffer_size = :ets.info(@ets_table, :size)

    stats =
      Map.merge(state, %{
        buffer_size: buffer_size,
        flush_threshold: @flush_threshold
      })

    {:reply, stats, state}
  end

  def handle_call(:flush, _from, state) do
    {count, new_state} = do_flush(state, force: true)
    {:reply, {:ok, count}, new_state}
  end

  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_info(:check_flush, state) do
    buffer_size = :ets.info(@ets_table, :size)
    now = System.system_time(:millisecond)

    state =
      if buffer_size >= @flush_threshold and flush_interval_elapsed?(state.last_flush, now) do
        {_count, new_state} = do_flush(state)
        new_state
      else
        state
      end

    Process.send_after(self(), :check_flush, @check_interval_ms)
    {:noreply, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # --- Private ---

  defp do_flush(state, opts \\ []) do
    force = Keyword.get(opts, :force, false)
    now = System.system_time(:millisecond)

    if not force and not flush_interval_elapsed?(state.last_flush, now) do
      {0, state}
    else
      examples =
        :ets.tab2list(@ets_table)
        |> Enum.map(fn {intent, text, _ts} -> {text, intent} end)

      if examples == [] do
        {0, state}
      else
        :ets.delete_all_objects(@ets_table)

        # Send to IntentClassifierSimple for incremental update
        case Brain.ML.IntentClassifierSimple.incremental_update(examples) do
          {:ok, count} ->
            Logger.info("TrainingExampleBuffer: flushed #{length(examples)} examples",
              incremental_count: count
            )

          {:error, reason} ->
            Logger.warning("TrainingExampleBuffer: flush failed: #{inspect(reason)}")
        end

        new_state = %{
          state
          | total_flushed: state.total_flushed + length(examples),
            last_flush: now,
            flush_count: state.flush_count + 1
        }

        {length(examples), new_state}
      end
    end
  rescue
    e ->
      Logger.warning("TrainingExampleBuffer flush error: #{Exception.message(e)}")
      {0, state}
  end

  defp flush_interval_elapsed?(nil, _now), do: true

  defp flush_interval_elapsed?(last_flush, now) do
    now - last_flush >= @min_flush_interval_ms
  end
end
