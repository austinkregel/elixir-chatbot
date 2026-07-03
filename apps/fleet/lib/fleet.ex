defmodule Fleet do
  @moduledoc """
  The Admiral-facing facade for the crew and the command channel.

  Commission ensigns, wire the chain of command, and issue orders. Chain wiring
  (`assign_co/2`) persists the `COMMANDS` edge to the Apache AGE `command_graph`
  and **surfaces** any persistence error (no graceful degradation).
  """

  alias Fleet.{CrewSupervisor, Ensign, Order, Comms, CommandGraph}

  @doc """
  Commissions an ensign for a soul. `opts` pass through to
  `Fleet.Ensign.start_link/1` — e.g. `:world_id`, `:tick_interval`, `:soul`,
  `:co`, `:reports`, `:grant` (`%{authorities: [...]}`), `:grants`.
  Returns `{:ok, pid, agent_id}`.
  """
  def commission(soul_id, opts \\ []) when is_binary(soul_id) do
    CrewSupervisor.start_ensign(Keyword.put(opts, :soul_id, soul_id))
  end

  @doc """
  Wire `co_id` as the commanding officer of `sub_id`: updates the in-process
  chain (both ensigns) and persists the `COMMANDS` edge. Returns `:ok` or
  `{:error, reason}` — a persistence failure is surfaced, not swallowed.
  """
  def assign_co(sub_id, sub_id) when is_binary(sub_id), do: {:error, :self_command}

  def assign_co(sub_id, co_id) when is_binary(sub_id) and is_binary(co_id) do
    Ensign.set_co(sub_id, co_id)
    Ensign.add_report(co_id, sub_id)
    Fleet.Telemetry.emit_event(co_id, :co_assigned, %{}, %{report: sub_id})

    case CommandGraph.establish_command(co_id, sub_id) do
      {:ok, _edge} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Issue an ORDER to an ensign as the Admiral. The ACK returns to the caller as
  `{:ack, ack}`. Options: `:authorities` (grant scope, default the order's
  requirement), `:world_id`, `:dry_run`, `:provenance`, `:from`, `:priority`, `:id`.
  """
  def order(agent_id, directive, opts \\ []) when is_binary(directive) do
    world_id = Keyword.get(opts, :world_id, "default")
    authorities = Keyword.get(opts, :authorities, [:cognition, {:world, world_id}])

    grant =
      case Keyword.get(opts, :provenance) do
        nil -> %{authorities: authorities}
        p -> %{authorities: authorities, provenance: p}
      end

    order = %Order{
      id: Keyword.get(opts, :id, gen_id()),
      from: Keyword.get(opts, :from, :admiral),
      directive: directive,
      grant: grant,
      world_id: world_id,
      dry_run: Keyword.get(opts, :dry_run, false),
      priority: Keyword.get(opts, :priority, "normal"),
      issued_at: System.monotonic_time(:millisecond)
    }

    Comms.order(agent_id, order)
  end

  @doc "Have `co_id` issue an ORDER to its report `report_id`."
  def issue_order(co_id, report_id, directive, opts \\ []),
    do: Ensign.issue_order(co_id, report_id, directive, opts)

  @doc "Have `co_id` relieve its report `report_id` of duty."
  def relieve(co_id, report_id), do: Ensign.relieve_report(co_id, report_id)

  @doc "Have `co_id` reinstate its report `report_id`."
  def reinstate(co_id, report_id), do: Ensign.reinstate_report(co_id, report_id)

  @doc """
  Admiral relieves a top-level agent (`co == nil`) of duty. Sends the RELIEVE
  from the calling process, which the runtime attributes as `:admiral`.
  """
  def relieve(agent_id) when is_binary(agent_id),
    do: Fleet.Comms.signal(agent_id, Fleet.Signal.new(:relieve, reason: "relieved by Admiral"))

  @doc "Admiral reinstates a top-level agent (`co == nil`)."
  def reinstate(agent_id) when is_binary(agent_id),
    do: Fleet.Comms.signal(agent_id, Fleet.Signal.new(:reinstate, reason: "reinstated by Admiral"))

  @doc "Have an ensign send a SITREP up to its CO."
  def sitrep(agent_id, body \\ %{}), do: Ensign.sitrep(agent_id, body)

  @doc """
  Hail an ensign — converse with it without giving an order. The reply arrives
  asynchronously as `{:hail_reply, %{agent_id:, question:, answer: | error:}}` in
  the calling process's mailbox. Confers no authority and creates no assignment.
  """
  def hail(agent_id, question) when is_binary(question), do: Ensign.hail(agent_id, question)

  @doc """
  Synchronous convenience for iex/tests: hail and block for the reply (or time
  out). Returns `{:ok, reply}` | `{:error, :timeout}`.
  """
  def hail_sync(agent_id, question, timeout \\ 120_000) when is_binary(question) do
    hail(agent_id, question)

    receive do
      {:hail_reply, %{agent_id: ^agent_id} = reply} -> {:ok, reply}
    after
      timeout -> {:error, :timeout}
    end
  end

  @doc """
  Route a model turn through the tool gate for `agent_id`. The model PROPOSES a
  capability (a `Fleet.Proposal`); the harness authorises it against the agent's
  own order-conferred grant and, only if permitted, dispatches it — framing the
  result as data and auditing every step. The model never holds the trigger.
  """
  def propose(agent_id, model_output) when is_binary(model_output),
    do: Ensign.propose(agent_id, model_output)

  @doc "Retires an ensign by pid."
  def retire(pid) when is_pid(pid), do: CrewSupervisor.retire(pid)

  @doc "Lists the commissioned crew (supervisor children)."
  def list, do: CrewSupervisor.list()

  @doc "True once the ensign has hydrated its soul."
  def ready?(agent_id), do: Ensign.ready?(agent_id)

  @doc "A public snapshot of an ensign's state."
  def status(agent_id), do: Ensign.status(agent_id)

  @doc """
  The live crew roster: each running ensign's `status/1` map enriched with its
  `:pid`, ordered by agent_id. Dying/unresponsive agents are skipped.
  """
  def roster do
    Registry.select(Fleet.Registry, [{{{:ensign, :"$1"}, :"$2", :_}, [], [{{:"$1", :"$2"}}]}])
    |> Enum.map(fn {agent_id, pid} ->
      try do
        Map.put(status(agent_id), :pid, pid)
      catch
        :exit, _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
    |> Enum.sort_by(& &1.agent_id)
  end

  defp gen_id, do: :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
end
