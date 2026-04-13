defmodule Brain.Analysis.OutcomeLearner.Store do
  @moduledoc """
  ETS-backed store for pattern success counts used by OutcomeLearner.
  Replaces Process.get/put to ensure counts persist across process boundaries.
  """
  use GenServer

  @table :outcome_learner_pattern_counts

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_count(key) do
    try do
      GenServer.call(__MODULE__, {:get_count, key}, 1_000)
    catch
      :exit, _ -> 0
    end
  end

  def increment(key) do
    GenServer.cast(__MODULE__, {:increment, key})
  end

  @impl true
  def init(_opts) do
    table = :ets.new(@table, [:set, :public, :named_table, read_concurrency: true])
    {:ok, %{table: table}}
  end

  @impl true
  def handle_call({:get_count, key}, _from, state) do
    count =
      case :ets.lookup(@table, key) do
        [{^key, c}] -> c
        [] -> 0
      end

    {:reply, count, state}
  end

  @impl true
  def handle_cast({:increment, key}, state) do
    :ets.update_counter(@table, key, {2, 1}, {key, 0})
    {:noreply, state}
  end
end
