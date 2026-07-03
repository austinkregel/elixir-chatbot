defmodule Brain.Epistemic.JTMS.Supervisor do
  @moduledoc """
  Dynamic supervisor of per-world JTMS webs. Each agent's mind-world gets its own
  `Brain.Epistemic.JTMS` process (isolated truth-maintenance network), started
  lazily via `Brain.Epistemic.JTMS.ensure/1` and addressed through
  `Brain.Epistemic.JTMSRegistry`.

  Mirrors `Brain.Subprocesses.Supervisor`.
  """
  use DynamicSupervisor

  def start_link(opts \\ []) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts), do: DynamicSupervisor.init(strategy: :one_for_one)
end
